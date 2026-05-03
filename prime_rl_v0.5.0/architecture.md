# Architecture (placeholder)

> To be filled in after first pass through `src/prime_rl/`.

Goal: one diagram + ~1 page of prose covering:

- Process topology: trainer process(es), orchestrator, inference (vLLM) replicas, prefill vs decode pools.
- Async boundaries: where the rollout/train loop is decoupled, what's the queue/buffer between them.
- Weight sync path: trainer → transport → inference.
- Rollout path: orchestrator → inference → environment (verifiers) → rollout buffer → trainer.
- Config flow: how `configs/` schema is composed and propagated to each component.

Pin every code reference to commit `63331ad8b17048b6d5f2051b2bf159e1392924b7`.
