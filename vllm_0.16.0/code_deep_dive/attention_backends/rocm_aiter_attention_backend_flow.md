# ROCm AITER Attention Backend Flow (v0.16.0)

## Scope

这篇笔记只讲 **ROCm AITER attention / MLA backend 自身的 flow**：

- backend 被选中之后的实现入口
- metadata builder 怎么准备 decode 侧数据
- 实际 attention kernel 调用到哪里
- 这条 backend 自己的硬限制有哪些

不展开 MoE / grouped top-k / expert parallel 的 cross-cutting 约束；那些内容见：
- [[ai/llm/inference/frameworks/vllm_0.16.0/rocm_aiter_heads_experts_and_fallback_constraints|rocm_aiter_heads_experts_and_fallback_constraints]]

Tag inspected:
- `v0.16.0`

---

## TL;DR

对 DeepSeek-style MLA 而言，`ROCM_AITER_MLA` 这条路径在 vLLM `v0.16.0` 里可以概括成：

1. backend 名称是 `ROCM_AITER_MLA`
2. kernel block size 只支持 `1`
3. metadata builder 把 decode 侧 paged KV 索引整理成 AITER 需要的 `indptr / indices / last_page_len / qo_indptr`
4. decode 核心调用落到 `rocm_aiter_ops.mla_decode_fwd(...)`
5. prefill 侧沿用 `flash_attn_varlen_func`
6. 这条 backend 明确要求 **per-rank `num_heads` 只能是 `16` 或 `128`**

---

## 1) Backend 入口

文件：
- `vllm/v1/attention/backends/mla/rocm_aiter_mla.py`

backend 类：

```python
class AiterMLABackend(MLACommonBackend):
```

名称：

```python
get_name() -> "ROCM_AITER_MLA"
```

支持的 kernel block size：

```python
def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
    return [1]
```

### 含义

这不是通用 ROCm attention backend，而是 **MLA 专用 AITER backend**。而且它在 vLLM 这一层明确只接受 **kernel block size = 1**。

## 1.5) 为什么 AITER MLA 要求 `--block-size 1`

这个限制有几个很重要的边界：

- 它是 **AITER MLA 路径** 的要求
- 不是“AITER 全家桶所有算子都要求 block size = 1”
- 这里的 `--block-size` 指的是 **vLLM KV cache paging 的 block 大小（以 token 为单位）**
- 它不是 CUDA/HIP 的 thread block size，也不是 GEMM 的 `BLOCK_SIZE_K`

### 先讲最直接的事实

在 vLLM `v0.16.0` 这条 `ROCM_AITER_MLA` backend 里：

- backend 白名单只接受 `kernel block size = 1`
- ROCm 文档也要求 **必须显式设置 `--block-size 1`**
- vLLM 不会帮用户 silently 降成 1

所以从使用者角度看，最直接的结论就是：

- **这不是“推荐值”，而是当前实现的硬要求**

### 为什么 MLA 天然更适合 token-level paging

MLA（尤其 DeepSeek-V3 / R1 这类实现）与普通 MHA/GQA 最大的结构差别之一是：

- KV cache 被压成一条 **latent 向量**
- 每个 token 实际只存一条 latent，而不是传统 MHA 那种 `num_kv_heads × head_dim` 形态

这带来一个很关键的结果：

- 对 MLA 来说，“每个 token 一条 latent” 本来就让 **token 成为自然的最小 paging 粒度**
- 因此 `block_size = 1` 在 MLA 路径上不是一个很反直觉的设定，而更像是：
  - **直接把 paging 粒度和 latent KV 的天然粒度对齐**

也就是说，普通 MHA 常见的 `block_size = 16 / 32 / 64`，本质上是为了把多个 token 的 K/V 打包成更适合其 kernel/data layout 的块；而 MLA 的数据结构并不天然享受同样的“多 token 打包收益”。

### AITER MLA asm kernel 当前就是按 block_size=1 写死的 hot path

更关键的是：即便从抽象上讲未来也许可以支持更大 block size，**当前 AITER MLA asm kernel 并没有把那些变体产品化出来**。

更准确的说法应该是：

- AITER MLA decode 当前的手写 assembly kernel
- 它的 tile 假设、register 配置、shared/LDS 布局
- 都是围绕 `block_size = 1` 的 KV paging 方式组织的

所以这不是“backend 理论上支持多种 block size，只是推荐 1”；而是更接近：

- **当前只实现了 block_size=1 的 hot path**
- 其他 block size 没有对应的 AITER MLA asm 变体

### 为什么 vLLM 不直接帮用户自动改成 1

一个容易误解的点是：

- 既然 backend 只能吃 1，为什么不在用户忘了设置时自动兜底？

更合理的解释是：

- 如果系统 silently 把用户配置改掉，用户会误以为 `block_size=16` 这种配置在 AITER MLA 上也属于受支持情况
- 这会让“到底是走到了预期 hot path，还是走慢路 / 错路”变得不透明

因此：

- **显式报错** 比 **偷偷改配置** 更符合可调试性

### 和标准 AITER MHA 路径要分开看

还要特别避免一个混淆：

- AITER MLA 要求 `--block-size 1`
- **不等于** 标准 transformer MHA 用的 AITER unified attention 也要 `1`

两者的 block size 约束不是一回事，因为：

- 一个服务的是 MLA latent KV cache + 特定 asm decode 路径
- 另一个服务的是普通 attention 数据布局

所以应该理解成：

- **`--block-size 1` 是 MLA-specific 的实现层限制**
- 不是整个 AITER attention 栈的统一要求

### 更准确的一句话总结

最稳妥的总结是：

> `--block-size 1` 之所以在 AITER MLA 路径里是硬要求，不是因为 AMD 硬件只能这样，而是因为 MLA 的 latent KV cache 天然适合 token-level paging，而当前 AITER MLA asm kernel 也只把 `block_size=1` 这一条 hot path 实现并验证到了可用状态。

### 代价：`block_size=1` 也会引入一整串系统级瓶颈

这里必须补一个很重要的现实：

- `block_size=1` 虽然是 **当前 AITER MLA 的 hot path**
- 但它并不是“没有副作用的完美选择”

它会在 kernel 之外的系统层带来明显 trade-off，尤其体现在：

- GPU memory utilization
- APC / prefix caching
- KV cache offloading
- scheduler / block table metadata 成本
- 超长序列下的 offset / indexing 风险

#### 1) GPU 记忆体利用率会变差

因为 token-level paging 把 block 粒度压到每 token 一页：

- 失去了多 token 打包进同一个 page/block 的空间效率
- 某些实现文档明确把 `page_size = 1` 描述成会回到“原始表示”，从而明显浪费 GPU memory

也就是说，MLA 在“单 token 一条 latent”这件事上虽然天然适合 token 粒度，但 **这不等于 block_size=1 在系统角度没有空间代价**。

#### 2) APC / Prefix Caching 往往会退化

prefix cache / APC 通常以 **block 为单位** 做：

- hash
- lookup
- prefix 命中判断

如果 `block_size = 1`：

- 每个 token 都变成一个单独 cache entry
- metadata 数量膨胀
- 查找、匹配、维护成本都会上升

所以对那种高度依赖共享 system prompt / 重复前缀的 workload：

- **AITER MLA kernel 本身可能更快**
- 但 APC 的退化可能会吃掉一部分收益

#### 3) KV cache offloading 会受很大限制，甚至无法使用

这是一个很实用的部署层痛点：

- 在 MLA + `block_size=1` 组合下，KV cache offloading 的适配明显更差
- 某些公开论文甚至直接把 MLA 路径总结成：
  - 要求 `block_size = 1`
  - 且不能像 GQA 模型那样自然享受 KV cache offloading

这对：

- 长 context
- 大 batch
- 多租户 serving

都是硬伤，因为这些场景本来最依赖 cache capacity 与 cache mobility。

#### 4) block table / scheduler 成本会显著上升

`block_size = 1` 还有一个经常被低估的副作用：

- 每条序列的 block table 长度基本等于 token 数
- paged attention 的 indirection lookup 次数与 metadata 传输量随之增长
- scheduler 如果以 block 为分配 / 释放单位，就会增加 allocation / free / page-table 更新次数

这类开销在：

- 高并发
- 长序列
- 高频抢占 / 回收

的 serving 场景里会被放大。

#### 5) 超长序列下更容易撞到 indexing / offset 边界问题

因为 block 粒度更细：

- offset 增长更快
- block table 更长
- 某些实现中的 uint32 offset / indexing bug 会更早暴露

所以 `block_size=1` 不只是“有点慢”，在超长序列下还可能放大 correctness / overflow 风险。

### 真实权衡：为什么它依然常常值得开

这就是 AITER MLA 路径最核心的 trade-off：

- **系统层更差**：memory、APC、offloading、scheduler 都更痛
- **kernel 层更快**：MLA decode asm path 在热点模型上通常明显更快

所以在 DeepSeek-V3 / R1 / Kimi 这一类 MLA 模型上，现实往往不是“block_size=1 没成本”，而是：

- **尽管 block_size=1 有系统代价，AITER MLA decode 的 kernel 收益仍常常足够大，整体上仍是净收益**

但这件事不能推广到所有架构：

- 对 MLA 模型，它往往是值得的折衷
- 对标准 GQA / MHA 模型，类似代价未必能被相同程度的 kernel 收益抵消

### 更稳妥的实践结论

如果把这件事总结成一句面向部署的判断，可以写成：

> `block_size=1` 在 AITER MLA 路径里是当前最优 decode hot path，但它把一部分成本从 kernel 内部转移到了系统层：prefix caching、offloading、scheduler 和 block metadata 都会变差。因此它更像“MLA-specific 的净收益折衷”，而不是一个无代价的普适最优设置。

---

## 2) Metadata builder：把 paged KV 整理成 AITER decode 侧需要的格式

还是同一文件里的：

```python
class AiterMLAMetadataBuilder(MLACommonMetadataBuilder[AiterMLAMetadata]):
```

关键点：

- `paged_kv_last_page_len` 被初始化成全 1
- `_build_decode(...)` 里从 `block_table_tensor` 和 `seq_lens_device` 生成：
  - `paged_kv_indptr`
  - `paged_kv_indices`
  - `paged_kv_last_page_len`
  - `qo_indptr`
- 注释明确写了：
  - kernel block size is always 1
  - each page has exactly 1 token（这里说的是 kernel 视角，不是整个 vLLM paged KV cache 的一般概念）

### 这里真正发生了什么

AITER MLA decode 期望的是一套很像 sparse/paged attention 的 CSR 风格索引元数据。vLLM 的 metadata builder 负责把自己的 block table / sequence length 信息重排成这套格式。

---

## 3) 实现类初始化时的 backend-specific 硬限制

`AiterMLAImpl.__init__(...)` 里最关键的约束：

```python
assert num_heads == 16 or num_heads == 128
```

同时还不支持：

```python
alibi_slopes, sliding_window, logits_soft_cap
```

也就是：

```python
raise NotImplementedError(
    "Aiter MLA does not support one of the following: "
    "alibi_slopes, sliding_window, logits_soft_cap"
)
```

### 含义

这条 backend 的限制不只是“ROCm 上能不能跑”。还包括：

- per-rank head 数必须满足 AITER MLA 自己的硬编码约束
- 某些 attention feature 在这条 backend 上根本没实现

---

## 4) Prefill vs decode：这条 backend 不是“所有 attention 都走同一个 kernel”

在初始化里：

```python
from aiter import flash_attn_varlen_func
self.flash_attn_varlen_func = flash_attn_varlen_func
```

而 decode 侧在 `forward_mqa(...)` 中使用的则是 AITER 专用 decode op。

这说明：

- **prefill**：沿用 AITER 的 `flash_attn_varlen_func`
- **decode**：走 `rocm_aiter_ops.mla_decode_fwd(...)`

所以这条 backend 内部本身就是 **prefill / decode 分裂实现**，不是一个统一 attention kernel 包打天下。

---

## 5) Decode 核心 kernel 调用

在 `forward_mqa(...)` 里，真正的 decode 计算落到：

```python
rocm_aiter_ops.mla_decode_fwd(...)
```

这就是 DeepSeek-style MLA 在这条 ROCm AITER backend 上最关键的 decode 内核入口。

### 可操作的 profiler 视角

如果你在 profiler / trace 里想确认是否真的走到了这条路径，可以重点盯：

- backend 名是否是 `ROCM_AITER_MLA`
- decode 期是否出现 `mla_decode_fwd`
- prefill 期是否走 `flash_attn_varlen_func`

---

## 6) 这篇笔记不该过度延伸到哪里

容易混在一起但最好分开的几件事：

### A. `num_heads % TP == 0`
这是 vLLM 模型并行配置层约束，不是 AITER backend 自己单独发明的规则。

### B. grouped top-k / expert grouping
这是 MoE router 路径问题，不属于 attention backend flow 本身。

### C. EP 下 experts 是否必须整除
这是 fused MoE / expert map 话题，也不属于 attention backend flow 本身。

这些内容见：
- [[ai/llm/inference/frameworks/vllm_0.16.0/rocm_aiter_heads_experts_and_fallback_constraints|rocm_aiter_heads_experts_and_fallback_constraints]]

---

## Practical takeaway

如果某个 DeepSeek MLA 模型在 ROCm + AITER 路径出问题，先分三层判断：

1. **是不是 attention backend 选错了**
2. **是不是这条 backend 自己的硬限制没满足**
   - per-rank heads 不是 `16/128`
   - 开了 unsupported feature
3. **是不是别的系统层约束在出问题**
   - TP 切分
   - MoE router / grouped top-k
   - EP / expert map

不要把所有“整除性限制”都归咎到 AITER attention backend 本身。

---

## Related

- [[ai/llm/inference/frameworks/vllm_0.16.0/deepseek_v3_rocm_attention_backend_kernel_map|deepseek_v3_rocm_attention_backend_kernel_map]]
- [[ai/llm/inference/frameworks/vllm_0.16.0/rocm_aiter_heads_experts_and_fallback_constraints|rocm_aiter_heads_experts_and_fallback_constraints]]
- [[ai/llm/inference/frameworks/vllm_0.16.0/design/attention_backends|attention_backends]]
- [[ai/llm/attention/kernel_families/rocm_aiter_attention|ROCm AITER attention]]
