# Vllm 0.16.0

## Scope
This directory contains 8 documents related to **ai/llm/inference/frameworks/vllm_0.16.0**.

## Documents
### code_structure.md
Analyzing version **v0.16.0** (source available in `code/` submodule).

### deepseek_v3_rocm_attention_backend_kernel_map.md
This note maps DeepSeek V3 (MLA path) backend options on ROCm to concrete implementation functions in vLLM `v0.16.0`.

### mixed_prefill_decode_kernel_path.md
This note documents how vLLM `v0.16.0` executes a batch containing both prefill and decode requests.

### next_deep_dives_2_2_2.md
- **Why it matters (2):** This is the core performance path; backend choice directly affects throughput/latency and KV‑append behavior.

### repo_structure.md
Analyzing version **v0.16.0** (source available in `code/` submodule).

### rocm_aiter_heads_experts_and_fallback_constraints.md
Code-verified note separating true hard constraints from grouped-topk / EP fallback conditions in the ROCm AITER path.

### vllm_and_aiter_dimension_divisibility_requirements.md
Operational summary of divisibility requirements across TP, AITER MLA/MoE, quantized GEMM kernels, and distributed fast paths.

### todo.md
- **Why it matters (2):** Core performance path; backend choice drives throughput/latency and KV‑append behavior.
