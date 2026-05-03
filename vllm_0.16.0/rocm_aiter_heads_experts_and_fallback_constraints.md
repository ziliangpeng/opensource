# vLLM v0.16.0 + ROCm AITER — heads / experts 整除性限制：哪些是硬限制，哪些只是优化路径条件

## Scope

这篇笔记专门澄清 vLLM `v0.16.0` 在 ROCm + AITER 路径里常见的“整除性限制（divisibility constraints）”到底分成哪几类：

- **配置层硬限制（hard constraints）**
- **特定 backend / kernel 的硬限制**
- **只影响是否能走某条优化路径、失败时会 fallback 的条件**
- **更底层 kernel runtime 才暴露出来的 tile / block 约束**

重点不是泛泛解释 GPU kernel 为什么喜欢整除，而是把 **vLLM 代码里真的显式检查了什么** 和 **大家常把什么说得过头** 区分开。

Tag inspected:
- `v0.16.0`

---

## TL;DR

最重要的结论：

1. **`num_attention_heads % tensor_parallel_size == 0` 是 vLLM 配置层硬限制。**
2. **ROCm AITER MLA decode 路径额外要求 per-rank `num_heads` 只能是 `16` 或 `128`。**
3. **Fused MoE 层要求 `intermediate_size % tp_size == 0`。**
4. **Grouped TopK 的 `num_experts % num_expert_group == 0` 不是全局硬限制**；不满足时 vLLM 会 fallback 到非 grouped router。**不是一定报错。**
5. **Expert Parallel 默认支持 experts 不均匀分配**；不是一般性要求 `num_experts` 必须整除 EP size。只有某些模式（例如 EPLB）才要求均匀整除。
6. **BLOCK_SIZE_K / MFMA / tile 对齐类报错更像 backend-specific kernel runtime 限制**，不应一概表述成“vLLM 配置层统一硬限制”。

---

## 1) 配置层硬限制：attention heads 必须能被 TP 整除

`vllm/config/model.py` 里直接检查：

```python
if total_num_attention_heads % tensor_parallel_size != 0:
    raise ValueError(
        f"Total number of attention heads ({total_num_attention_heads})"
        " must be divisible by tensor parallel size "
        f"({tensor_parallel_size})."
    )
```

这说明：

- `num_heads % TP == 0` 不是经验规则，而是 **vLLM 显式验证**。
- 如果不满足，模型配置阶段就会失败，不会等到底层 kernel 再炸。

### 含义

这是 **tensor parallel（张量并行）切 attention heads** 的基本前提。Q heads 必须能均分到 TP ranks。

---

## 2) backend-specific 硬限制：ROCm AITER MLA 只接受 16 或 128 个 per-rank heads

`vllm/v1/attention/backends/mla/rocm_aiter_mla.py` 里：

```python
assert num_heads == 16 or num_heads == 128, (
    f"Aiter MLA only supports 16 or 128 number of heads.\n"
    f"Provided {num_heads} number of heads.\n"
    "Try adjusting tensor_parallel_size value."
)
```

这里的 `num_heads` 是 **本 rank 的 head 数（per rank）**，不是全模型总 head 数。

在 `vllm/model_executor/models/deepseek_v2.py` 的 attention 初始化也能看到：

```python
self.total_num_heads = num_heads
assert self.total_num_heads % tp_size == 0
self.num_heads = self.total_num_heads // tp_size
```

所以更准确的说法是：

- 先满足 **全局 `num_heads % TP == 0`**
- 再满足 **AITER MLA 的 per-rank `num_heads ∈ {16, 128}`**

### 实际含义

这不是“所有 ROCm attention 都这样”，而是 **AITER MLA 这条特定 backend path 的硬限制**。

### 为什么是 `16` 或 `128`，而不是 `32`

这里需要把 **代码里能直接证明的东西** 和 **来自 AITER / AMD 资料与 issue 讨论的解释** 分开看：

#### 代码里能直接证明的部分

vLLM `v0.16.0` 能直接证明的只有一件事：

- `ROCM_AITER_MLA` backend 把 per-rank `num_heads` 白名单硬编码成了 `{16, 128}`。
- 因而 `32`、`64`、`8` 都不会通过这条 backend 的初始化断言。

也就是说，从代码角度，**“32 不合法”是确定的；“为什么实现者只放行 16/128”则需要看更底层的 kernel 假设与外部资料。**

#### 更底层、但不是仅靠 vLLM 代码就能完全证明的解释

根据用户提供的 AITER / AMD 资料与 issue 讨论，一个更合理的解释框架是：

1. **MLA decode 并不是普通 MHA 的 head-batch 视角**。
   - DeepSeek V3 / R1 的 MLA 会把 KV 压到 shared latent，再在 decode kernel 里改写成更接近 absorbed-projection / MQA 风格的计算。
   - 在这种实现里，`num_heads` 更像是 kernel / GEMM 里的一个结构性维度，而不只是“多开几个 head batch”这么简单。

2. **AITER MLA decode kernel 不是“所有 MFMA shape 都写一遍”的通用实现**。
   - 即使 CDNA 的 MFMA 原语理论上能支持别的 M 尺寸，也不代表 AITER MLA decode 已经为 `32` 或 `64` 写好、调好、验证好专门 schedule。
   - 对 vLLM 使用者来说，真正重要的是：**这条 backend 当前实现只提供了 16 / 128 这两个合法点。**

3. **`32` 的问题更像是“会撞到 kernel 假设 / schedule 选择”，而不是一个抽象的数学上永远不可能。**
   - 换句话说，`32` 不是因为 attention 理论上不能算；而是因为 **AITER 当前这版 MLA decode kernel 没把 `32 heads/rank` 作为受支持形状发布出来**。
   - 这也是为什么最稳妥的表述是：
     - `16/128` 是 **实现白名单**
     - `32/64/8` 是 **当前未被这条 backend 支持的形状**

#### 对你写 KB 时的措辞建议

下面这种写法最稳：

> AITER MLA 对 per-rank heads 的 `16/128` 限制，直接来源于 backend / kernel 实现白名单。更底层的可能原因与 MLA decode 的 shape 改写、CDNA MFMA tile 选择、寄存器 / LDS 压力和 AITER 团队实际只为若干部署点做了 schedule tuning 有关；但这些是对实现动机的解释，不应误写成“vLLM 代码本身已经把所有微架构原因完整证明了”。

#### 实务含义

如果你遇到：

- TP4 下 DeepSeek 变成 `32 heads/rank`
- 或更小模型 / 更大 TP 变成 `8 heads/rank`

那对 vLLM `v0.16.0` 的 `ROCM_AITER_MLA` 路径来说，最直接的判断就是：

- **这条 backend 不支持**
- 解决办法通常是：
  - 调 TP 让 per-rank heads 变成 `16` 或 `128`
  - 或关闭 AITER MLA，改走 Triton MLA / 其他 backend

---

## 3) Fused MoE 的 TP 硬限制：expert intermediate size 必须能被 TP 整除

`vllm/model_executor/layers/fused_moe/layer.py`：

```python
assert intermediate_size % self.tp_size == 0
self.intermediate_size_per_partition = intermediate_size // self.tp_size
```

这条是明确的：

- 对于 fused MoE 实现，expert FFN 的 `intermediate_size` 必须能按 TP 切开。
- 这属于 **模型/层级配置约束**，不是运行时偶然碰到的 tile 问题。

---

## 4) Grouped TopK：这是“能不能走 grouped path”的条件，不是全局硬报错

很多总结会把这件事说成：

> `num_experts % num_expert_group == 0`，否则 grouped routing 不能正确工作。

这方向不算错，但 **如果写成“否则 vLLM 一定报错”就不准确**。

在 `vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py`：

```python
def valid_grouping() -> bool:
    num_experts = router_logits.shape[-1]
    if num_experts <= self.num_expert_group:
        return False
    return num_experts % self.num_expert_group == 0
```

但紧接着：

```python
if not valid_grouping():
    if self.e_score_correction_bias is not None:
        topk_weights, topk_ids = fused_topk_bias(...)
    else:
        topk_weights, topk_ids, token_expert_indices = fused_topk(...)
    return topk_weights, topk_ids
```

### 更准确的结论

- `num_experts % num_expert_group == 0` 是 **grouped top-k 路径的适用条件**。
- 不满足时，vLLM **fallback 到普通 top-k / bias top-k 路径**。
- 所以这不是“全局硬限制”，而是“**某条优化/特化 routing path 的前提**”。

### 对 DeepSeek 系模型的含义

像 DeepSeek v2/v3 这类模型经常带：

- `n_group`
- `topk_group`

这会影响 grouped routing 的可用性，但不能简单概括成“experts 必须被 group 数整除否则整个模型不能跑”。

---

## 5) Expert Parallel：默认支持不均匀分配，不是一般性要求 experts 整除 EP size

这是最容易被讲错的一点。

在 `vllm/model_executor/layers/fused_moe/layer.py` 的 `determine_expert_map(...)`：

```python
base_experts = global_num_experts // ep_size
remainder = global_num_experts % ep_size
local_num_experts = base_experts + 1 if ep_rank < remainder else base_experts
```

这说明默认逻辑是：

- 先均分 `global_num_experts // ep_size`
- 多出来的 `remainder` 分给前几个 EP rank
- **也就是支持 remainder-aware 的不均匀分配**

### 只有某些模式才要求均匀整除

例如 EPLB 打开时，才有：

```python
assert self.global_num_experts % self.ep_size == 0, (
    "EPLB currently only supports even distribution of experts across ranks."
)
```

### 更准确的结论

- **默认 EP：不要求 `num_experts % ep_size == 0`**
- **EPLB 等特定模式：要求均匀整除**

所以把它概括成：

> split experts 一般都要求 `num_experts % (TP×DP) == 0`

会说得太硬，而且不够贴近 vLLM 代码里真正起作用的量；在 fused MoE 层这里，最直接控制 expert placement 的是 **`ep_size`**。

---

## 6) AITER Fused MoE 的 expert map 约束存在，但它和“experts 必须整除某个数”不是同一回事

在 `vllm/model_executor/layers/fused_moe/layer.py`：

```python
if self.use_ep and self.rocm_aiter_fmoe_enabled:
    assert self.expert_mask is None or torch.all(
        (expert_mask == 0) | (expert_mask == 1)
    ), "Aiter Fused MoE kernel only supports expert_map with 0 and 1s."
```

这说明 AITER Fused MoE 对 expert map 的表达形式有自己的要求。

但这里的重点是：

- 它约束的是 **expert map / expert mask 的形式**
- 不是一个直接的“每卡 expert 数必须整除 8/16”这样的配置层规则

---

## 7) BLOCK_SIZE_K / MFMA / tile 对齐：更像底层 kernel runtime 约束，而不是 vLLM 配置层通用规则

在 fused MoE Triton kernel、各种 benchmark / tuning 代码里，确实能看到：

- `BLOCK_SIZE_M`
- `BLOCK_SIZE_N`
- `BLOCK_SIZE_K`
- `group_size`
- `block_k_diviable`

例如 `vllm/model_executor/layers/fused_moe/fused_moe.py` 里明确提到：

```python
padding ensures divisibility by BLOCK_SIZE_M
```

这说明对齐/整除确实重要。**但从 vLLM Python 配置层代码直接能确认的，主要是 token/block 处理会做 padding 或 backend-specific 分支。**

因此更稳妥的表述是：

- **底层 kernel（AITER / Triton / CUTLASS / quantized GEMM）常有 tile/block 对齐要求**
- 某些错误会表现成 `size_k must divisible by BLOCK_SIZE_K`
- 但这不等于“vLLM 整体有一条统一的 model-config 规则说 hidden/k/expert-count 必须满足某个固定公约数”

---

## 8) 哪些常见说法需要降级

### 可以直接说“对”的

- `num_attention_heads % TP == 0`
- AITER MLA per-rank `num_heads` 只能是 `16` 或 `128`
- fused MoE 的 `intermediate_size % tp_size == 0`

### 要改成“特定路径条件 / fallback 条件”的

- `num_experts % num_expert_group == 0`
  - 这是 grouped top-k 路径条件
  - 不满足时会 fallback，不是全局硬失败

### 要改成“只在某些模式下成立”的

- `num_experts % ep_size == 0`
  - 默认 EP 不需要
  - EPLB 等模式才需要

### 要改成“底层 kernel 约束，不宜上升成 vLLM 通用规则”的

- `BLOCK_SIZE_K / MFMA / tile divisibility`
- “每卡 expert 数必须对齐 8 或 16”这类说法

---

## 9) 调参 / debug 实务顺序

如果在 ROCm + AITER + MoE 路径碰到断言或奇怪 fallback，排查顺序最好是：

1. **先看配置层硬限制**
   - `num_heads % TP == 0`
   - `intermediate_size % TP == 0`
2. **再看 backend-specific 硬限制**
   - AITER MLA per-rank heads 是否是 `16` 或 `128`
3. **再看是否只是 grouped / EP 优化路径失效**
   - `num_experts % num_expert_group == 0` 是否满足
   - 是否因为不满足而 fallback
4. **最后再怀疑更底层 kernel tile / quantization / backend runtime 约束**
   - `BLOCK_SIZE_K`
   - quant block shape
   - AITER / Triton / CUTLASS 专有要求

---

## Related

- [[ai/llm/inference/frameworks/vllm_0.16.0/deepseek_v3_rocm_attention_backend_kernel_map|deepseek_v3_rocm_attention_backend_kernel_map]]
- [[ai/llm/inference/frameworks/vllm_0.16.0/code_deep_dive/attention_backends/rocm_aiter_attention_backend_flow|rocm_aiter_attention_backend_flow]]
- [[ai/llm/inference/frameworks/vllm_0.16.0/design/attention_backends|vLLM attention backends]]
- [[ai/llm/inference/frameworks/vllm_0.16.0/design/fused_moe|vLLM fused MoE kernels]]
- [[ai/llm/attention/kernel_families/rocm_aiter_attention|ROCm AITER attention]]
- [[ai/llm/parallelism/wide_expert_parallelism_moe_inference|Wide Expert Parallelism for MoE Model Inference]]
