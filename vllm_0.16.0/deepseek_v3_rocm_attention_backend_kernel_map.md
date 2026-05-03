# DeepSeek V3 on ROCm (vLLM v0.16.0): Attention Backend → Kernel Map

## Scope
This note maps DeepSeek V3 (MLA path) backend options on ROCm to concrete implementation functions in vLLM `v0.16.0`.

Reference tag inspected: `v0.16.0`.

---

## Backend options (ROCm + MLA)

From `vllm/platforms/rocm.py` + backend registry, MLA-relevant options include:
- `ROCM_AITER_MLA`
- `ROCM_AITER_TRITON_MLA` (runtime name: `AITER_TRITON_MLA`)
- `TRITON_MLA`
- `ROCM_AITER_MLA_SPARSE` (when sparse MLA is selected)

---

## 1) ROCM_AITER_MLA

Implementation files:
- `vllm/v1/attention/backends/mla/rocm_aiter_mla.py`
- shared MLA logic in `vllm/model_executor/layers/attention/mla_attention.py`

### Decode-side kernel/function
- `rocm_aiter_ops.mla_decode_fwd(...)`
  - called in `AiterMLAImpl.forward_mqa(...)`

### Prefill-side function path
Prefill backend is selected in MLA common code; possible runtime calls include:
- FlashAttention prefill: `flash_attn_varlen_func(...)`
- FlashInfer prefill: `prefill.prefill_main.run(...)`
- cuDNN prefill: `cudnn_batch_prefill_with_kv_cache(...)`
- TRTLLM ragged deepseek prefill: `_run_prefill_*_trtllm_ragged` path

---

## 2) ROCM_AITER_TRITON_MLA (AITER_TRITON_MLA)

Implementation file:
- `vllm/v1/attention/backends/mla/aiter_triton_mla.py`

This backend inherits from `AiterMLAImpl` and overrides prefill varlen attention function source.

### Decode-side kernel/function
- Still uses: `rocm_aiter_ops.mla_decode_fwd(...)`

### Prefill-side function difference
- Overrides `flash_attn_varlen_func` to:
  - `aiter.ops.triton.mha.flash_attn_varlen_func`

So this is not identical to `TRITON_MLA`; it is an AITER-centered implementation with Triton MHA varlen prefill hook.

---

## 3) TRITON_MLA

Implementation file:
- `vllm/v1/attention/backends/mla/triton_mla.py`

### Decode-side kernel/function
- `decode_attention_fwd(...)`
  - called in `TritonMLAImpl.forward_mqa(...)`

### Notes
- This is the vLLM Triton MLA implementation path (not AITER MLA decode path).
- Includes explicit constraints (e.g., FP8 KV cache not supported in this implementation path).

---

## 4) ROCM_AITER_MLA_SPARSE

Implementation file:
- `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py`

### Decode-side kernel/function
- `rocm_aiter_ops.mla_decode_fwd(...)` appears in sparse MLA flow as well.

Sparse variant adds sparse-indexing/selection logic around MLA decode behavior.

---

## Why names are confusing

- `TRITON_MLA` and `AITER_TRITON_MLA` both contain “Triton”, but they are different stacks.
- `AITER_TRITON_MLA` = AITER MLA base with Triton varlen prefill function hook.
- `TRITON_MLA` = vLLM Triton decode attention implementation (`decode_attention_fwd`).

---

## Practical verification checklist

When running DeepSeek V3 on ROCm, verify actual path via:
1. startup backend logs (`Using ... backend`),
2. attention backend name at runtime,
3. profiler symbols / call trace for:
   - `mla_decode_fwd` vs `decode_attention_fwd`,
   - `flash_attn_varlen_func` source (AITER Triton vs other path).
