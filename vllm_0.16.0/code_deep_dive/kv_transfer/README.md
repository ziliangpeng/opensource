# Kv Transfer

## Scope
This directory contains 4 documents related to **ai/llm/inference/frameworks/vllm_0.16.0/code_deep_dive/kv_transfer**.

## Documents
### kv_connector.md
The KV Connector is vLLM's abstraction for **moving KV cache data in and out of the local GPU**. It sits alongside the KVCacheManager (which handles GPU-local prefix caching) and provides a second tier of KV cache access.

### kv_offloading.md
vLLM's built-in KV offloading system spills KV cache blocks from GPU to CPU memory and loads them back on prefix cache miss. It's a **Tier 2 cache** sitting behind the GPU prefix cache (Tier 1): when a request's prefix isn't found in GPU memory, the scheduler checks if it was previously offloaded to CPU before falli...

### lmcache_connector.md
LMCache is an external KV cache store that extends vLLM's GPU prefix cache to a second tier (CPU, disk, or remote). vLLM has **two** LMCache connectors with fundamentally different loading strategies:

### prefill_decode_disaggregation.md
Framework-specific deep dive into how vLLM 0.16.0 implements prefill-decode disaggregation. For the cross-system concept overview, start with [[../../../../prefill_decode_disaggregation|the inference-level PD disaggregation note]].
