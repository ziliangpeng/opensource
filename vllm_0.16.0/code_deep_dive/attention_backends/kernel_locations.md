# vLLM v0.16.0 — Attention Backend Kernel Locations

This note lists where each backend’s kernels live (external package vs vLLM in‑tree), plus the **real external dependency** (Python package when applicable). Scope: **all attention backends in v0.16.0** (from `vllm/v1/attention/backends/registry.py`).

## FLASH_ATTN
- **External dependency:** `flash-attn` (Python package: `flash_attn`)
- **Kernel repo:** https://github.com/Dao-AILab/flash-attention
- **Entry points:** `vllm/v1/attention/backends/flash_attn.py`, `.../fa_utils.py`

## FLASH_ATTN_DIFFKV
- **External dependency:** `flash-attn` (Python package: `flash_attn`)
- **Kernel repo:** https://github.com/Dao-AILab/flash-attention
- **Entry point:** `vllm/v1/attention/backends/flash_attn_diffkv.py`

## FLASH_ATTN_MLA
- **External dependency:** `flash-attn` (Python package: `flash_attn`)
- **Kernel repo:** https://github.com/Dao-AILab/flash-attention
- **Entry point:** `vllm/v1/attention/backends/mla/flashattn_mla.py`

## FLASHINFER
- **External dependency:** `flashinfer` (Python package: `flashinfer`)
- **Kernel repo:** https://github.com/flashinfer-ai/flashinfer
- **Entry point:** `vllm/v1/attention/backends/flashinfer.py`
- **TRT‑LLM kernels (via FlashInfer on Blackwell):** https://github.com/NVIDIA/TensorRT-LLM

## FLASHINFER_MLA
- **External dependency:** `flashinfer` (Python package: `flashinfer`)
- **Kernel repo:** https://github.com/flashinfer-ai/flashinfer
- **Entry point:** `vllm/v1/attention/backends/mla/flashinfer_mla.py`

## TRITON_ATTN
- **External dependency:** none (in‑tree Triton kernel)
- **Entry point:** `vllm/v1/attention/backends/triton_attn.py`

## TRITON_MLA
- **External dependency:** none (in‑tree Triton kernel)
- **Entry point:** `vllm/v1/attention/backends/mla/triton_mla.py`

## ROCM_ATTN
- **External dependency:** none (in‑tree ROCm kernel)
- **Entry point:** `vllm/v1/attention/backends/rocm_attn.py`

## ROCM_AITER_FA
- **External dependency:** AMD AITER (ROCm) — repo: https://github.com/ROCm/aiter
- **Note:** vLLM links AITER C++ kernels via its ROCm extension (`vllm._aiter_ops`/`vllm._rocm_C`); there is no standalone `aiter` pip import.
- **Entry point:** `vllm/v1/attention/backends/rocm_aiter_fa.py`

## ROCM_AITER_UNIFIED_ATTN
- **External dependency:** AMD AITER (ROCm) — repo: https://github.com/ROCm/aiter
- **Note:** vLLM links AITER C++ kernels via its ROCm extension (`vllm._aiter_ops`/`vllm._rocm_C`); there is no standalone `aiter` pip import.
- **Entry point:** `vllm/v1/attention/backends/rocm_aiter_unified_attn.py`

## ROCM_AITER_MLA
- **External dependency:** AMD AITER (ROCm) — repo: https://github.com/ROCm/aiter
- **Note:** vLLM links AITER C++ kernels via its ROCm extension (`vllm._aiter_ops`/`vllm._rocm_C`); there is no standalone `aiter` pip import.
- **Entry point:** `vllm/v1/attention/backends/mla/rocm_aiter_mla.py`

## ROCM_AITER_TRITON_MLA
- **External dependency:** AMD AITER (ROCm) — repo: https://github.com/ROCm/aiter
- **Note:** vLLM links AITER C++ kernels via its ROCm extension (`vllm._aiter_ops`/`vllm._rocm_C`); there is no standalone `aiter` pip import.
- **Entry point:** `vllm/v1/attention/backends/mla/aiter_triton_mla.py`

## ROCM_AITER_MLA_SPARSE
- **External dependency:** AMD AITER (ROCm) — repo: https://github.com/ROCm/aiter
- **Note:** vLLM links AITER C++ kernels via its ROCm extension (`vllm._aiter_ops`/`vllm._rocm_C`); there is no standalone `aiter` pip import.
- **Entry point:** `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py`

## FLASHMLA
- **External dependency:** FlashMLA (repo: https://github.com/deepseek-ai/FlashMLA)
- **Entry point:** `vllm/v1/attention/backends/mla/flashmla.py`

## FLASHMLA_SPARSE
- **External dependency:** FlashMLA (repo: https://github.com/deepseek-ai/FlashMLA)
- **Entry point:** `vllm/v1/attention/backends/mla/flashmla_sparse.py`

## CUTLASS_MLA
- **External dependency:** NVIDIA CUTLASS (repo: https://github.com/NVIDIA/cutlass)
- **Entry point:** `vllm/v1/attention/backends/mla/cutlass_mla.py`

## FLEX_ATTENTION
- **External dependency:** PyTorch (flex attention API in core PyTorch)
- **Entry point:** `vllm/v1/attention/backends/flex_attention.py`

## CPU_ATTN
- **External dependency:** none (in‑tree CPU backend)
- **Entry point:** `vllm/v1/attention/backends/cpu_attn.py`

## TREE_ATTN
- **External dependency:** none (in‑tree backend)
- **Entry point:** `vllm/v1/attention/backends/tree_attn.py`

## NO_ATTENTION
- **External dependency:** none (in‑tree backend)
- **Entry point:** `vllm/v1/attention/backends/no_attention.py`

## TORCH_SDPA (ViT only)
- **External dependency:** PyTorch (SDPA in core PyTorch)

## CUSTOM
- **External dependency:** user‑provided module via `register_backend()`
