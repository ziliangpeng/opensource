# GPU Model Runner V1 vs V2 (vLLM v0.16.0)

## TL;DR
V2 is a **major refactor** of the GPU model execution pipeline that removes V1’s persistent‑batch reordering bookkeeping and shifts more work onto GPU‑resident data structures (block tables, input assembly, sampling), improving scalability and simplifying maintenance. V2 is still **experimental** as of v0.16.0.

---

## Timeline (verified dates + links)

### V1 runner (initial)
- **2025‑01‑27** — vLLM V1 alpha announced (V1 engine + V1 model runner introduced).
  - Blog: https://blog.vllm.ai/2025/01/27/v1-alpha-release.html
- **2025‑03‑18** — V1 engine enabled by default (v0.8.0).
  - Release note: https://github.com/vllm-project/vllm/releases/tag/v0.8.0

### V2 runner (experimental)
- **2025‑12‑03** — GPU Model Runner V2 introduced as **Experimental** (v0.12.0).
  - Release note: https://github.com/vllm-project/vllm/releases/tag/v0.12.0
  - PR: https://github.com/vllm-project/vllm/pull/25266

### V0 engine removal (context)
- **2025‑10‑02** — V0 engine fully removed; V1 is the only engine (v0.11.0).
  - Release note: https://github.com/vllm-project/vllm/releases/tag/v0.11.0

### V0 deprecation RFC (context)
- **2025‑05‑22** (RFC timestamp on GitHub) — V0 deprecation proposal; notes V1 is default since v0.8.0.
  - RFC: https://github.com/vllm-project/vllm/issues/18571

---

## Architecture differences (V1 vs V2)

### V1 runner
- Uses **persistent batch** with CPU‑side request state + reordering.
- Heavy bookkeeping to keep request state, block tables, and sampling state consistent.
- Block tables and slot mapping primarily managed on CPU, copied to GPU each step.
- Sampler is legacy implementation (includes historical hacks for certain settings).

**File (V1):**
- `vllm/v1/worker/gpu_model_runner.py`

### V2 runner
- **Eliminates persistent batch reordering**; assigns fixed indices per request.
- Separates:
  - **Persistent request state** (CPU + optional UVA mapping)
  - **Step‑level InputBatch** built on GPU using Triton kernels
- **GPU‑persistent block tables**; CPU only sends diffs.
- **Triton‑native sampler**, removes legacy hacks, improves logprob efficiency.
- Simplifies DP + CUDA graph handling; better fit for multimodal + structured outputs.

**File (V2):**
- `vllm/v1/worker/gpu/model_runner.py`

---

## Why V2 was needed (design motivations)
- **Reduce complexity:** V1 reordering + bookkeeping became brittle with more features.
- **Performance scaling:** CPU overhead and CPU→GPU copies became a bottleneck at large batch sizes / long context.
- **Extensibility:** New features (structured outputs, VLM, advanced decoding) required heavy patching in V1.

V2 shifts key data‑flow to GPU and removes redundant state to simplify future development.

---

## Summary (motivation → core change → result)
**Motivation：** V1 的 persistent batch + reordering 带来了复杂的 CPU 侧 bookkeeping 和大量 CPU→GPU 数据搬运，随着功能扩展（多模态、structured outputs、spec decode）维护成本和性能瓶颈都显著上升。  
**Core change：** 把“批次构造/关键状态”从 CPU 迁移到 GPU：固定 request index + GPU‑persistent block tables + GPU 侧 Triton 组装 InputBatch。  
**Result：** 代码路径更简洁、可维护性更高，CPU 开销与同步减少；在高并发/长上下文场景下扩展性更好（具体收益依赖模型与配置，需用基准验证）。

---

## Key mechanisms in V2
- **GPU‑persistent block tables** (plus optional UVA variants)
- **GPU‑built InputBatch** via Triton kernels
- **Triton sampler** for efficient logprob + top‑k
- Cleaner CUDA graph integration

---

## Reordering vs. gather（关键澄清）
**V1 的 reordering**：
- 在 CPU 上对“持久批次”做全量重排，让活跃请求变成连续前缀。
- 必须同步重排多个张量（token_ids、positions、block tables、lengths…），bookkeeping 复杂。

**V2 的 gather**：
- 不是搬动持久状态本体，而是在 GPU 上**按需 gather**出“本步需要的子集”。
- 本质是并行重排/选择，但**范围更小、代价更低**。
- 结果是：仍有“压缩/聚合”步骤，但不再是 V1 那种全量重排成本。

---

## Enable V2 (experimental)
V2 is still experimental in v0.16.0. It’s typically enabled via environment flag (exact variable may change; check latest docs/CLI):
- Commonly seen: `VLLM_USE_V2_MODEL_RUNNER=1`

---

## References (all links)
- V1 alpha blog: https://blog.vllm.ai/2025/01/27/v1-alpha-release.html
- v0.8.0 release (V1 default): https://github.com/vllm-project/vllm/releases/tag/v0.8.0
- v0.11.0 release (V0 removed): https://github.com/vllm-project/vllm/releases/tag/v0.11.0
- v0.12.0 release (V2 experimental): https://github.com/vllm-project/vllm/releases/tag/v0.12.0
- V0 deprecation RFC: https://github.com/vllm-project/vllm/issues/18571
- Model Runner V2 PR: https://github.com/vllm-project/vllm/pull/25266
