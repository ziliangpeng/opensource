# vLLM v0.16.0 — Code Deep Dive

## Scope
Code-level walkthroughs for vLLM execution path and major subsystems.

## Structure Map
- top-level flows: request lifecycle, scheduler-output, sampling, dp attention
- submodules: `api_server/`, `engine_core/`, `worker/`, `kv_transfer/`, `attention_backends/`

## Key Docs
- [[ai/llm/inference/frameworks/vllm_0.16.0/code_deep_dive/request_lifecycle]]
- [[ai/llm/inference/frameworks/vllm_0.16.0/code_deep_dive/engine_core/scheduler]]
- [[ai/llm/inference/frameworks/vllm_0.16.0/code_deep_dive/worker/model_runner_v1_v2]]

## Status
High-value docs exist; navigation is now explicit.

## Next
Keep module README links updated whenever adding new deep-dive files.
