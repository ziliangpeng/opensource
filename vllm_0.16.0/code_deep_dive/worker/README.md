# Worker

## Scope
This directory contains 2 documents related to **ai/llm/inference/frameworks/vllm_0.16.0/code_deep_dive/worker**.

## Documents
### gpu_worker_model_runner.md
This note covers the **GPU Worker → V1 GPUModelRunner** execution path in vLLM v0.16.0, specifically the flow **after scheduler output** arrives at the worker and **before kernels execute**, including KV connector hooks and CUDA graph dispatch. It assumes the earlier doc *scheduler-output-to-kernel.md* for the sched...

### model_runner_v1_v2.md
V2 is a **major refactor** of the GPU model execution pipeline that removes V1’s persistent‑batch reordering bookkeeping and shifts more work onto GPU‑resident data structures (block tables, input assembly, sampling), improving scalability and simplifying maintenance. V2 is still **experimental** as of v0.16.0.
