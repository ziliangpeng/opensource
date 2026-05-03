# Design

## Scope
This directory contains 8 documents related to **ai/llm/inference/frameworks/vllm_0.16.0/design**.

## Documents
### arch_overview.md
Based on `docs/design/arch_overview.md` in the vLLM v0.16.0 source.

### attention_backends.md
Based on `docs/design/attention_backends.md` in the vLLM v0.16.0 source.

### cuda_graphs.md
Based on `docs/design/cuda_graphs.md` in the vLLM v0.16.0 source.

### fused_moe.md
Based on `docs/design/fused_moe_modular_kernel.md`, `docs/design/moe_kernel_features.md`, and `vllm/model_executor/layers/fused_moe/` in the vLLM v0.16.0 source.

### huggingface_integration.md
Based on `docs/design/huggingface_integration.md` and `vllm/model_executor/models/transformers/` in the vLLM v0.16.0 source.

### paged_attention.md
Based on `docs/design/paged_attention.md` and the original vLLM paper (SOSP 2023). Note that the vLLM doc itself is marked as historical — the original custom kernel no longer runs in modern vLLM. This doc covers both the original design and the evolution to today.

### prefix_caching.md
Based on `docs/design/prefix_caching.md` in the vLLM v0.16.0 source.

### torch_compile.md
Based on `docs/design/torch_compile.md` in the vLLM v0.16.0 source.
