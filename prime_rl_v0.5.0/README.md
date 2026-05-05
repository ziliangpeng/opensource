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

## Top-level layout (`src/prime_rl/`, verified at SHA `63331ad8`)

```
entrypoints/   rl.py, sft.py, inference.py          ← 3 CLI entry points
orchestrator/  orchestrator.py, scheduler.py, buffer.py,
               advantage.py, trajectories.py, envs.py,
               env_server/, vf_utils.py, filters.py  ← rollout scheduling + advantage + envs
trainer/       model.py, optim.py, weights.py, batch.py,
               parallel_dims.py, world.py,
               rl/, sft/, models/                    ← FSDP2 training + custom model stacks
inference/     server.py, vllm/                      ← vLLM wrapper
transport/     base.py, zmq.py, filesystem.py        ← two transports: ZMQ and filesystem
configs/       rl.py, sft.py, trainer.py,
               orchestrator.py, inference.py,
               env_server.py, shared.py              ← Pydantic schemas
utils/         cp.py, elastic.py, monitor/,
               act_offloading.py, async_utils.py,
               heartbeat.py, metrics_server.py, …
```

Key signals from this layout:
- **Three entry points** → multi-process design: trainer, orchestrator, inference run separately.
- **Two transport backends** (ZMQ + filesystem) → likely ZMQ for hot path (weights / rollouts) and filesystem for checkpoints / cold artifacts. To confirm.
- **`trainer/rl/` + `trainer/sft/`** → trainer is parameterized by training mode; the RL-specific bits live under `rl/`.
- **`orchestrator/env_server/`** → environments served as a separate process (matches the verifiers integration story).

## Reading roadmap

### Step 1 — Process topology (next session, start here)
**Target files**:
- `src/prime_rl/entrypoints/rl.py`
- `src/prime_rl/entrypoints/inference.py`
- `src/prime_rl/entrypoints/sft.py`
- `src/prime_rl/transport/base.py`, `transport/zmq.py`, `transport/filesystem.py`

**Question to answer**: For a single `rl` training run, how many processes get spawned, where do they live (single node? multi node?), and what flows over each transport?

**Output**: `notes/process_topology.md` with a process diagram and a table of (transport, sender, receiver, payload).

### Step 2 — `architecture.md`
Once topology is clear, fill in the placeholder `architecture.md` at the project root: process map + async boundaries + data flow + config flow.

### Step 3 — Orchestrator deep-dive
**Target**: `src/prime_rl/orchestrator/orchestrator.py` + `scheduler.py` + `buffer.py` + `advantage.py` + `trajectories.py`.
Goal: understand the async RL loop core — rollout scheduling, off-policy correction (look in `advantage.py` / `filters.py`), buffer semantics, weight sync cadence.
**Output**: `notes/orchestrator.md`.

### Step 4 — Transport
**Target**: `src/prime_rl/transport/{base,zmq,filesystem,types}.py` + caller sites (grep for imports).
Goal: weight format, who pushes vs pulls, sync vs async, what happens on failure.
**Output**: `notes/transport.md`.

### Step 5 — Trainer (FSDP2 + EP + CP)
**Target**: `src/prime_rl/trainer/{model,parallel_dims,world,weights}.py` + `trainer/rl/` + `trainer/models/` (custom MoE stacks).
Goal: how FSDP2 is wired, where EP for MoE plugs in, where CP for long context plugs in, custom kernel hookups (quack-kernels etc.).
**Output**: `notes/trainer_fsdp2.md`.

### Step 6 — Inference: PD disaggregation (v0.5.0 headline)
**Target**: `src/prime_rl/inference/{server,vllm/}` + the v0.5.0 PR [#2030](https://github.com/PrimeIntellect-ai/prime-rl/pull/2030).
Goal: how prefill and decode workers are separated, vLLM router integration, multi-replica scaling.
**Output**: `notes/inference_pd_disagg.md`.

### Step 7 — Verifiers / environment integration
**Target**: `src/prime_rl/orchestrator/{envs,env_server,vf_utils}.py` + the [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) repo's plug-in surface.
Goal: how environments register, how rollouts call them, agentic vs SWE env shapes.
**Output**: `notes/verifiers_integration.md`.

### Side roadmap (open questions to track)
- Where does FP8 inference get configured? (likely overlap with vLLM bits.)
- AMD ROCm path: does prime-rl assume CUDA-only, or does it work on the AITER/ROCm vLLM stack we use? Grep for `cuda` vs `rocm` / `hip`.
- Slurm vs k8s deployment paths: skim `k8s/` and any Slurm scripts under `scripts/`.

## Supported model families (from upstream README)

GLM-5, Qwen3 MoE, Qwen3.5 MoE, Qwen3-VL, MiniMax M2, Nemotron H, Trinity (afmoe), GLM-4 / GLM-4.5 MoE / INTELLECT-3, GPT-OSS (HF MoE). Custom optimized stacks live under `src/prime_rl/trainer/models/`.

## Status

Skeleton — actual reading TBD.
