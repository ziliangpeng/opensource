# prime-rl v0.5.0

Code-reading notes for [PrimeIntellect-ai/prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) — async RL training framework, FSDP2 for training + vLLM for inference, designed for 1000+ GPU scale.

## Pinned version

- **Tag**: `v0.5.0`
- **Released**: 2026-03-30
- **Commit SHA**: [`63331ad8b17048b6d5f2051b2bf159e1392924b7`](https://github.com/PrimeIntellect-ai/prime-rl/tree/v0.5.0)
- **License**: Apache-2.0

When linking source from these notes, use:
`https://github.com/PrimeIntellect-ai/prime-rl/blob/63331ad8b17048b6d5f2051b2bf159e1392924b7/<path>`

## Headline features (per the v0.5.0 release notes)

- **Disaggregated prefill–decode inference** with multi-replica support and vLLM router integration ([#2030](https://github.com/PrimeIntellect-ai/prime-rl/pull/2030)).
- Fully async RL loop: trainer (FSDP2) and inference (vLLM) decoupled.
- FP8 inference, PD disaggregation, EP and CP parallelism.
- Native [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) environment integration via Environments Hub.
- End-to-end post-training: SFT + RL + evals.
- Slurm and Kubernetes deployment.
- Multimodal (VLM) support — Qwen3-VL etc.

## Top-level layout (`src/prime_rl/`)

| Dir | Purpose (placeholder — to be verified) |
|---|---|
| `configs/` | Pydantic config schema for trainer / orchestrator / inference |
| `entrypoints/` | CLI entry points (training, SFT, eval, …) |
| `inference/` | vLLM-side wrappers — rollout generation, PD-disagg routing |
| `orchestrator/` | Async RL orchestrator: schedules rollouts, weight transfers |
| `trainer/` | FSDP2 training loop, model-specific stacks (`trainer/models/`) for MoE EP/CP |
| `transport/` | Weight / KV / activation transport between trainer and inference |
| `templates/` | Prompt / chat templates |
| `utils/` | Shared utilities |

Verified subsystems will move out of the placeholder table into dedicated notes.

## Reading roadmap (planned)

1. **`architecture.md`** — component map: how trainer / orchestrator / inference / transport fit together; where async boundaries live.
2. **`notes/orchestrator.md`** — the async RL loop core: rollout scheduling, off-policy correction, weight sync cadence.
3. **`notes/transport.md`** — weight transfer trainer → inference; KV/activation movement.
4. **`notes/trainer_fsdp2.md`** — FSDP2 setup, EP for MoE, CP for long context.
5. **`notes/inference_pd_disagg.md`** — PD disaggregation, vLLM router integration (the v0.5.0 headline).
6. **`notes/verifiers_integration.md`** — how environments plug in via the verifiers package.

## Supported model families (from upstream README)

GLM-5, Qwen3 MoE, Qwen3.5 MoE, Qwen3-VL, MiniMax M2, Nemotron H, Trinity (afmoe), GLM-4 / GLM-4.5 MoE / INTELLECT-3, GPT-OSS (HF MoE). Custom optimized stacks live under `src/prime_rl/trainer/models/`.

## Status

Skeleton — actual reading TBD.
