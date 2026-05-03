# vLLM v0.16.0 — Attention Backend Flows

This folder holds per‑backend flow docs (selection → metadata → kernel path) for vLLM v0.16.0.

## Backends (one file each)
- [[ai/llm/inference/frameworks/vllm_0.16.0/code_deep_dive/attention_backends/flash_attention_backend_flow]]
- [[ai/llm/inference/frameworks/vllm_0.16.0/code_deep_dive/attention_backends/flashinfer_backend_flow]]
- [[ai/llm/inference/frameworks/vllm_0.16.0/code_deep_dive/attention_backends/triton_attention_backend_flow]]
- [[ai/llm/inference/frameworks/vllm_0.16.0/code_deep_dive/attention_backends/rocm_aiter_attention_backend_flow]]

## Scope
- Backend‑specific constraints & validation
- Metadata builder behavior
- Kernel path selection inside the backend
- CUDA‑graph support implications
