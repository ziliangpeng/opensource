# SGLang — Architecture Deep-Dives

Unversioned reference for SGLang (sgl-project/sglang), focusing on subsystems relevant to Character.AI's production deployment (Gemma 4 31B on AMD MI325X).

## Notes

- [`project_family_and_omni.md`](./project_family_and_omni.md) — How SGLang grew into a project family (education → tooling → orchestration → engine → modality → operators), and why SGLang-Omni exists: speech/omni serving is a multi-stage heterogeneous pipeline, not one AR loop.
- [`notes/hicache-write-back-vs-write-through.md`](./notes/hicache-write-back-vs-write-through.md) — HiCache write_back vs write_through: mechanism, throughput tax, retention, and why vLLM/LMCache can't do write_back (architectural analysis with vLLM PR #43946 and TensorRT-LLM comparison).
- [`notes/hicache-swa-device-only.md`](./notes/hicache-swa-device-only.md) — Why HiCache only offloads FULL KV (not SWA): bounded vs unbounded per-conversation KV, restore asymmetry, and the implementation bug that made SWA host backup useless.

## Context

These notes were written while reviewing Mike Martin's SGLang fork (`character-tech/sglang-internal`, branch `cai-mixed-split-dispatch`, 94 commits) for Gemma 4 31B inference on AMD MI325X. The fork achieves 4.3× per-pod throughput vs production vLLM.

## Related

- [SGLang HiCache design doc](https://docs.sglang.io/docs/advanced_features/hicache_design)
- [SGLang PR #5543](https://github.com/sgl-project/sglang/pull/5543) — write_back first added (2025-04-20)
- [vLLM PR #43946](https://github.com/vllm-project/vllm/pull/43946) — eviction-triggered store (open, unmerged)
- [vLLM RFC #19854](https://github.com/vllm-project/vllm/issues/19854) — KV cache offloading discussion
- [TensorRT-LLM kv-cache-reuse docs](https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html)
