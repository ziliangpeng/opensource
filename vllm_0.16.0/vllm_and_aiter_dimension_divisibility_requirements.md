# vLLM 與 AITER 的維度整除性要求

## Scope

這篇筆記整理 vLLM 與 AMD AITER（AI Tensor Engine for ROCm）中，幾類最常見的維度整除性（divisibility）要求與錯誤來源。

重點是把幾種不同層次的限制放在同一張地圖裡：

- tensor parallel 的模型切分限制
- AITER MLA / MoE backend 的使用條件
- 量化 GEMM kernel 的 block / group 對齊
- fused MoE 與 all-gather 等分散式優化路徑的 shape 要求

這篇是偏 **運維 / 配置 / 常見報錯對照** 的總整理；如果要看其中某些約束在 vLLM `v0.16.0` 程式碼裡到底哪些是硬檢查、哪些只是 fallback 條件，另見：

- [[ai/llm/inference/frameworks/vllm_0.16.0/rocm_aiter_heads_experts_and_fallback_constraints|rocm_aiter_heads_experts_and_fallback_constraints]]

---

## TL;DR

可以先把這些整除性要求分成四大類：

1. **模型並行切分限制**
   - 例如 `num_attention_heads % TP == 0`
   - 某些情況下 vocab size 也必須能被 worker / GPU 數整除
2. **AITER backend 使用條件**
   - 例如 AITER MLA 需要 `--block-size 1`
   - 某些 backend 只接受特定 per-rank head shape
3. **量化 kernel / GEMM block 對齊**
   - 例如 int8 group size、`BLOCK_SIZE_K`
4. **分散式優化路徑的 shape 條件**
   - 例如 fused MoE、all-gather 優化是否能啟用

---

## 1) 張量並行（Tensor Parallelism）的整除性

### Attention heads 必須能被 TP 整除

vLLM 的 tensor parallel 實作有最基本的一條限制：

- attention heads 總數必須能被 `tensor_parallel_size` 整除
- 否則會報類似：
  - `Total number of attention heads (64) must be divisible by tensor parallel size (3)`

### 實務含義

像：

- Qwen3-30B-A3B-AWQ 有 32 個 attention heads

那麼：

- `TP=3、5、6、7` 這種無法整除的切法就會失敗

這是 **TP 切 head 維度** 的基礎限制，不是 AITER 特有問題。

### Vocabulary size 的整除性

另外一類比較老牌、但實際上常見的問題是：

- vocabulary size 需能被 worker / GPU 數整除

常見例子：

- `vocab=65025` 配 `4 workers`
- `vocab=49954` 配 `16 GPUs`

都可能在初始化時失敗。

### 如果 GPU 數不好整除

官方一般建議：

- 如果 GPU 數無法和模型切分需求相容，考慮改用 **pipeline parallelism (PP)**
- 因為 PP 對不均勻切分更友好

---

## 2) AITER MoE / MLA 的要求

### 什麼情況會走 AITER 路徑

在 AMD Instinct：

- MI300X
- MI325X
- MI350X
- MI355X

上啟用 `VLLM_ROCM_USE_AITER=1` 時：

- MoE（例如 Mixtral）
- MLA（例如 DeepSeek-V3 / R1）

都可能走到 AITER backend。

### AITER MLA 需要显式 `--block-size 1`

对 `VLLM_ROCM_USE_AITER_MLA` 而言，一个非常重要的实务限制是：

- **必须明确设置 `--block-size 1`**
- 否则 vLLM 会直接报错，而不是帮你偷偷改成 1

这跟很多人直觉以为「backend 既然只支援某个 block size，系统应该会自动补」不一样。

还要特别注意，这条限制：

- **只针对 AITER MLA 路径**
- 不是 AITER 所有 attention / kernel 路径的通用要求
- 这里的 `block_size` 指的是 **vLLM KV cache paging 的 block 大小**，不是 thread block size，也不是 GEMM 的 `BLOCK_SIZE_K`

### 为什么是 1

更准确的理解是：

- MLA 的 KV cache 是 **latent KV**，每个 token 本来就更接近“一条 latent”的存储粒度
- 这让 token-level paging（`block_size = 1`）在数据结构上天然更顺
- 同时当前 AITER MLA asm kernel 也只把 `block_size=1` 这一条 hot path 实现并验证到了可用状态

也就是说，这更像：

- **MLA latent KV 结构 + 当前 asm kernel 实现白名单** 的共同结果

而不是：

- AMD 硬件本身要求所有 AITER attention 都只能用 1

更详细的解释见：
- [[ai/llm/inference/frameworks/vllm_0.16.0/code_deep_dive/attention_backends/rocm_aiter_attention_backend_flow#15-为什么-aiter-mla-要求---block-size-1|rocm_aiter_attention_backend_flow]]

### block size 与 `cp_kv_cache_interleave_size`

另外，engine argument 层还有一组关系：

- `block_size` 必须能被 `cp_kv_cache_interleave_size` 整除
- 而且 `block_size >= cp_kv_cache_interleave_size`

这类限制比较像 **engine / cache layout 层级** 的相容条件。

### 但 `block_size=1` 也会带来很真实的系统代价

这点很重要：

- `block_size=1` 对 AITER MLA 来说是当前 hot path
- **但它并不免费**

比较常见的副作用包括：

- GPU memory utilization 变差
- APC / prefix caching 退化
- KV cache offloading 适配受限
- block table / scheduler metadata 成本上升
- 超长序列下更容易暴露 offset / indexing 边界问题

也就是说，这个设置更像：

- **用系统层代价，换 MLA decode kernel 的高吞吐**

所以对 DeepSeek / Kimi 这类 MLA 模型，它往往仍是净收益；但这不等于它是一个没有副作用的普适最优值。

更详细的 trade-off 解释见：
- [[ai/llm/inference/frameworks/vllm_0.16.0/code_deep_dive/attention_backends/rocm_aiter_attention_backend_flow#15-为什么-aiter-mla-要求---block-size-1|rocm_aiter_attention_backend_flow]]

### 与 per-rank head 白名单的关系

如果是 DeepSeek-style MLA，除了 `--block-size 1` 外，還要注意另一組限制：

- `ROCM_AITER_MLA` 在 vLLM `v0.16.0` 中對 per-rank heads 有白名單：`16 / 128`

詳見：
- [[ai/llm/inference/frameworks/vllm_0.16.0/rocm_aiter_heads_experts_and_fallback_constraints|rocm_aiter_heads_experts_and_fallback_constraints]]

也就是說，MLA 能不能走 AITER，至少要同時考慮：

- TP 切完後的 heads/rank
- block size
- backend 是否啟用

---

## 3) AITER 量化 GEMM 的限制

### AITER int8 scaled MM 支援組合有限

`AiterInt8ScaledMMLinearKernel` 不是任意 int8 量化配置都支援。

目前較明確的限制包括：

- 只支援 symmetric quantization
- 常見支援組合主要落在：
  - per-tensor × per-tensor
  - per-token × per-channel

這代表如果量化格式或 scale shape 不在 backend 支援集合內，可能不是「跑得慢」，而是根本不會進該 kernel。

### per-token-group int8 的 group size 整除性

在 int8 工具層裡，一個很直接的要求是：

```python
assert x.shape[-1] % group_size == 0
```

也就是：

- 輸入最後一維必須能被 `group_size` 整除

### `size_k must divisible by BLOCK_SIZE_K`

使用 Marlin / AWQ / 某些量化 kernel 搭配 TP 時，常見錯誤包括：

- `size_k must divisible by BLOCK_SIZE_K`

這通常表示：

- 原始 K 維度本來合法
- 但經過 TP 切分後，每 rank 的 K 維度不再是 kernel 需要的 `BLOCK_SIZE_K`（常見如 `64 / 128`）倍數

### 重點理解

這類錯誤往往不是「模型理論上不能切」，而是：

- **切分後的子問題 shape 不再落在量化 kernel 支援的 tile / block 集合內**

---

## 4) Fused MoE 與 All-Gather 的維度對齊

### fused_moe 的 K 維對齊

fused_moe 的輸入可以概念化成：

- `(*, K)`

其中：

- `K` 是 feature 維度

內部 tiled GEMM 常要求：

- `K` 對 `BLOCK_SIZE_K` 對齊

這與前面量化 kernel 的 `size_k` 問題是同一類型：

- 問題常不是語義層，而是 **kernel tile shape 相容性**

### all-gather 優化路徑的整除性

vLLM 分散式層某些 all-gather 優化，預設只會對這類張量啟用：

- shape / size 可以被 `all_gather_group.world_size` 整除

如果不滿足：

- 可能需要停用那條優化路徑
- 或退回較泛化的實作

這類限制常比 attention head / TP 的硬限制更隱性，因為它們不一定第一時間以「模型不合法」出現，而可能是：

- 某條 fast path 沒啟用
- 或某個 custom all-gather path 報 shape mismatch

---

## 5) 常見除錯建議

### A. 先檢查 TP 相關核心維度

選 TP size 時，優先確認這些量是否能整除：

- `num_attention_heads`
- `num_kv_heads`
- `intermediate_size`
- `vocab_size`

如果這些量和 TP 不相容：

- 改 TP
- 或改用 PP

### B. 跑 DeepSeek-V3 / R1 + AITER MLA 時

常見保守配置是：

- `--block-size 1`
- `--tensor-parallel-size 8`

因為這通常能同時滿足：

- AITER MLA 的 block size 條件
- heads/rank 落到常見合法值 `16`

### C. 遇到 `size_k must divisible by BLOCK_SIZE_K`

優先檢查：

- 量化 kernel 的 `group_size / block_size`
- TP 切分後的 `K` 維度
- 是否某個 quant backend（Marlin / AWQ / a8w8）只接受特定 block 形狀

### D. 不要把所有整除性錯誤都歸為同一類

最好分清楚你撞到的是哪一層：

1. **vLLM 並行配置硬限制**
2. **AITER backend 使用條件**
3. **量化 kernel tile / block 限制**
4. **分散式優化 fast path 條件**

這樣 debug 速度會快很多。

---

## Related

- [[ai/llm/inference/frameworks/vllm_0.16.0/rocm_aiter_heads_experts_and_fallback_constraints|rocm_aiter_heads_experts_and_fallback_constraints]]
- [[ai/llm/inference/frameworks/vllm_0.16.0/code_deep_dive/attention_backends/rocm_aiter_attention_backend_flow|rocm_aiter_attention_backend_flow]]
- [[ai/llm/inference/frameworks/vllm_0.16.0/design/attention_backends|vLLM attention backends]]
- [[ai/llm/inference/frameworks/vllm_0.16.0/design/fused_moe|vLLM fused MoE kernels]]
- [[ai/llm/attention/kernel_families/rocm_aiter_attention|ROCm AITER attention]]
- [[ai/llm/parallelism/tensor_parallelism_evolution|tensor_parallelism_evolution]]
