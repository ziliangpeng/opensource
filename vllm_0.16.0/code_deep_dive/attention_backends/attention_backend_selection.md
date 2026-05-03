# vLLM v0.16.0 — Attention Backend Selection (First Pass)

## Scope
This note summarizes how vLLM v0.16.0 selects attention backends, what “explicit backend” means, and how platform selection + backend validation interact. It includes the key questions we discussed.

## 1) What does “explicitly set backend” mean?
It means `vllm_config.attention_config.backend` is **not None**, typically set via CLI or config.

**CLI:**
- `--attention-backend <BACKEND_NAME>`

**Config:**
- `AttentionConfig(backend=...)`

The backend enum value is parsed from string in `AttentionConfig` and stored in `vllm_config.attention_config.backend`.

## 2) Where does backend selection happen?
**Call path** (standard decoder attention):
1. `Attention.__init__` → calls `get_attn_backend(...)` if no backend passed in
2. `get_attn_backend(...)` builds an `AttentionSelectorConfig`
3. `_cached_get_attn_backend` delegates to `current_platform.get_attn_backend_cls(...)`
4. The selected backend class may enforce a required KV cache layout

**Key selector code:**
```py
# vllm/v1/attention/selector.py
vllm_config = get_current_vllm_config()
backend_enum = vllm_config.attention_config.backend
...
attention_cls = current_platform.get_attn_backend_cls(
    backend_enum,
    attn_selector_config=attn_selector_config,
)
```

## 3) Explicit backend: what happens if it’s invalid?
**Rule:** Explicit backend → validate → **fail fast** (no fallback).

CUDA implementation:
```py
# vllm/platforms/cuda.py
backend_class = selected_backend.get_class()
invalid_reasons = backend_class.validate_configuration(...)
if invalid_reasons:
    raise ValueError(...)
```

So if a backend is explicitly set but incompatible with the platform/config, vLLM raises a `ValueError` and stops.

## 4) Auto-selection algorithm (platform‑specific)
### CUDA (most common)
**Flow** (`vllm/platforms/cuda.py`):
1) If `selected_backend` is set → validate it via `backend_class.validate_configuration(...)` → use it or error.
2) If not set → build a priority list via `_get_backend_priorities(use_mla, device_capability)`.
3) For each backend in order → `validate_configuration(...)` → collect valid/invalid.
4) Choose the **highest‑priority valid** backend; if none valid → error with reasons.

**Priority lists (CUDA, v0.16.0)**
- **Non‑MLA, SM10 (Blackwell):** `FLASHINFER → FLASH_ATTN → TRITON_ATTN → FLEX_ATTENTION`
- **Non‑MLA, other SMs:** `FLASH_ATTN → FLASHINFER → TRITON_ATTN → FLEX_ATTENTION`
- **MLA, SM10 (Blackwell):** `FLASHINFER_MLA → CUTLASS_MLA → FLASH_ATTN_MLA → FLASHMLA → TRITON_MLA → FLASHMLA_SPARSE`
- **MLA, other SMs:** `FLASH_ATTN_MLA → FLASHMLA → FLASHINFER_MLA → TRITON_MLA → FLASHMLA_SPARSE`

**Qualification checks (CUDA)**
Each backend’s `validate_configuration(...)` enforces:
- `head_size`, `dtype`, `kv_cache_dtype`, `block_size`
- `use_mla`, `has_sink`, `use_sparse`, `use_mm_prefix`, `attn_type`
- `compute capability`
- backend‑specific combination constraints

**Rationale for the Blackwell vs non‑Blackwell priority swap (sourced)**
- vLLM’s Blackwell InferenceMAX blog explicitly states that on Blackwell it will prefer FlashInfer (TRT‑LLM kernels) and fall back to FlashAttention when needed.
- The GPT‑OSS/Blackwell blog states FlashInfer is used to maximize Blackwell tensor‑core utilization.
- v0.16.0 release notes call out “FlashInfer MLA is now the default MLA backend on Blackwell, with TRTLLM as default prefill.”

**Rationale for MLA ordering (sourced)**
- PR **#32615** (and the v0.16.0 release notes) explicitly make **FlashInfer MLA** the default MLA backend on Blackwell, and TRT‑LLM the default prefill backend. This is the source of the Blackwell MLA ordering in v0.16.0.

**Version note**
- In **v0.16.0**, `_get_backend_priorities` takes only `(use_mla, device_capability)` and has **no** `num_heads`‑dependent sparse ordering. Any `num_heads`/sparse reordering is **main‑branch only**, not part of the v0.16.0 tag.

**Compact selection flow (CUDA vs ROCm)**
```
CUDA:
  if explicit_backend:
      validate() → use | error
  else:
      list = priorities(use_mla, SM)
      for b in list: if validate(b) → pick first
      else error

ROCm:
  if use_sparse:
      require block_size==1 & not fp8 KV → ROCM_AITER_MLA_SPARSE
  elif use_mla:
      if explicit_backend:
          allow ROCM_AITER_MLA | ROCM_AITER_TRITON_MLA | TRITON_MLA (block_size!=1)
      else:
          if aiter_mla_enabled OR block_size==1 → ROCM_AITER_MLA
          else → TRITON_MLA
  else:
      if explicit_backend: allow FLEX/TRITON/ROCM_ATTN/ROCM_AITER_*
      else:
          1) AITER + UNIFIED → ROCM_AITER_UNIFIED_ATTN
          2) AITER + MHA + gfx9 → ROCM_AITER_FA
          3) use_prefill_decode_attention → ROCM_ATTN
          4) AITER + gfx9 + (MHA not False) → ROCM_AITER_FA
          5) else → TRITON_ATTN
```

---

### ROCm
ROCm does **not** use the CUDA priority table; it uses explicit branching (`vllm/platforms/rocm.py`).

**If `use_sparse`:**
- Rejects FP8 KV cache dtype
- Requires `block_size == 1`
- Selects `ROCM_AITER_MLA_SPARSE`

**If `use_mla`:**
- If `selected_backend` is **None**:
  - If `rocm_aiter_ops.is_mla_enabled()` **or** `block_size == 1` → `ROCM_AITER_MLA`
  - Else → `TRITON_MLA`
- If `selected_backend` is set:
  - `TRITON_MLA` allowed **only** when `block_size != 1`
  - `ROCM_AITER_MLA` or `ROCM_AITER_TRITON_MLA` allowed
  - Anything else → error

**If non‑MLA:**
Selection order when `selected_backend` is None:
1) If `VLLM_ROCM_USE_AITER` **and** `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION` → `ROCM_AITER_UNIFIED_ATTN`
2) If `VLLM_ROCM_USE_AITER` **and** `VLLM_ROCM_USE_AITER_MHA` **and** `gfx9` → `ROCM_AITER_FA`
3) If `attention_config.use_prefill_decode_attention` → `ROCM_ATTN`
4) If `VLLM_ROCM_USE_AITER` **and** `gfx9` **and** `VLLM_ROCM_USE_AITER_MHA` is not False → `ROCM_AITER_FA`
5) Else → `TRITON_ATTN`

If `selected_backend` is set in non‑MLA mode, ROCm accepts only:
- `FLEX_ATTENTION`, `TRITON_ATTN`, `ROCM_ATTN`, `ROCM_AITER_FA`, `ROCM_AITER_UNIFIED_ATTN`
Anything else → error.

### XPU
- Forces KV cache layout to `NHD`
- MLA → `TRITON_MLA`
- Non‑MLA → prefers `FLASH_ATTN` unless dtype/override forces `TRITON_ATTN`

### CPU
- Always `CPU_ATTN`
- MLA and sparse attention are not supported

## 5) Backend vs Platform vs Kernel — hierarchy
**Hierarchy:**
1) **Platform** decides which backend class to use (CUDA/ROCm/XPU/CPU).
2) **Backend class** (a *family* of implementations) defines capabilities + validation and selects impl + metadata builder.
3) **Kernel implementation** is the actual attention kernel used by that backend.

## 6) After backend selection: cudagraph mode vs kernel choice
- **Engine core / model runner** resolves **CUDAGraph mode** based on backend *builder* support (not a specific kernel path). This happens in `gpu_model_runner._check_and_update_cudagraph_mode()`.
- **GPU worker / model runner** constructs and captures the actual CUDA graph during warmup and replays it during execution.
- **Kernel path selection** (e.g., FA2 vs FA3, TRTLLM vs native FlashInfer) happens inside the **backend implementation** in the GPU worker context, not in the engine core.

**Important nuance:**
- Some backends are **platform‑specific** (`ROCM_*`, `CPU_ATTN`), others are **shared** (e.g., `TRITON_ATTN` can be selected by CUDA/ROCm/XPU).
- A backend can have **multiple internal kernel paths** (e.g., FlashAttention FA2/FA3; FlashInfer native vs TRTLLM).

## 6) Explicit backend options (enum members)
From `AttentionBackendEnum`:
- `FLASH_ATTN`, `FLASH_ATTN_DIFFKV`, `TRITON_ATTN`, `FLASHINFER`
- MLA: `FLASHINFER_MLA`, `TRITON_MLA`, `CUTLASS_MLA`, `FLASHMLA`, `FLASHMLA_SPARSE`, `FLASH_ATTN_MLA`
- ROCm: `ROCM_ATTN`, `ROCM_AITER_FA`, `ROCM_AITER_UNIFIED_ATTN`, `ROCM_AITER_MLA`, `ROCM_AITER_TRITON_MLA`, `ROCM_AITER_MLA_SPARSE`
- Other: `FLEX_ATTENTION`, `TREE_ATTN`, `CPU_ATTN`, `NO_ATTENTION`, `CUSTOM`
- `TORCH_SDPA` is only used for ViT attention selection

## 7) How backends validate compatibility
Every backend runs `validate_configuration(...)` which checks:
- head size, dtype, KV cache dtype
- block size
- support for sinks, MLA, sparse, MM prefix, attention type
- compute capability
- backend‑specific combination constraints

---

## Open Questions
- Why does `ROCM_AITER_MLA_SPARSE` require `block_size == 1`?
- Why is `TRITON_MLA` on ROCm only allowed when `block_size != 1` (per current selection logic)?

## Sources
- `vllm/v1/attention/selector.py`
- `vllm/platforms/cuda.py`
- `vllm/platforms/rocm.py`
- `vllm/platforms/xpu.py`
- `vllm/platforms/cpu.py`
- `vllm/v1/attention/backends/registry.py`
- `vllm/v1/attention/backend.py`
- `vllm/config/attention.py`
- `vllm/engine/arg_utils.py`
- https://docs.vllm.ai/en/v0.16.0/design/attention_backends/
- https://blog.vllm.ai/2025/10/09/blackwell-inferencemax.html
- https://blog.vllm.ai/2026/02/01/gpt-oss-optimizations.html
- https://github.com/vllm-project/vllm/releases/tag/v0.16.0
- https://github.com/vllm-project/vllm/pull/32615
