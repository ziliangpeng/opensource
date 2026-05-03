# Engine Core

## Scope
This directory contains 3 documents related to **ai/llm/inference/frameworks/vllm_0.16.0/code_deep_dive/engine_core**.

## Documents
### engine_core.md
This note covers the EngineCore process: its components, busy loop, threading model, and how it coordinates scheduling with GPU execution.

### kv_cache_manager.md
This note covers the KV cache manager: how it allocates and frees blocks, how prefix caching works, and how different attention types each handle their own caching strategy.

### scheduler.md
This note covers the scheduler: how it decides which requests to run, how many tokens each gets, how it handles memory pressure, and how it produces the `SchedulerOutput` consumed by the executor.
