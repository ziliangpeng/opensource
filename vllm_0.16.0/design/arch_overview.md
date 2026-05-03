# vLLM Architecture Overview

Based on `docs/design/arch_overview.md` in the vLLM v0.16.0 source.

## Entrypoints

Two ways to use vLLM:

- **`LLM` class** — offline/batch inference, Python API, synchronous. Use when you want to run inference in a script without a server.
- **`vllm serve`** — launches an OpenAI-compatible HTTP server, async, supports streaming. (`python -m vllm.entrypoints.openai.api_server` is deprecated.)

## V1 Process Architecture

vLLM V1 runs as multiple OS processes. This is important for understanding CPU resource requirements and how the system scales.

| Process | Count | Role |
|---------|-------|------|
| API Server | `A` (default = DP) | HTTP handling, tokenization, multimodal preprocessing |
| Engine Core | `DP` (default 1) | Scheduler + KV cache manager, runs a busy loop |
| GPU Worker | `N` = DP × TP | One per GPU, runs model forward passes |
| DP Coordinator | 1 if DP > 1, else 0 | Load balancing + MoE forward pass sync across DP ranks |
| **Total** | **`A + DP + N` (+ 1 if DP > 1)** | |

Processes communicate via **ZMQ sockets**. API servers connect to all engine cores in a many-to-many topology — any API server can route requests to any engine core.

### Examples

- **4 GPUs, TP=4**: 1 API server + 1 engine core + 4 GPU workers = **6 processes**
- **8 GPUs, TP=2, DP=4**: 4 API servers + 4 engine cores + 8 GPU workers + 1 DP coordinator = **17 processes**

The engine core runs a **busy loop** and is very sensitive to CPU starvation. Minimum recommended: 2 + N physical cores for N GPUs (not vCPUs — physical cores).

## Class Hierarchy

Within each GPU worker process, there are two nested classes:

```
Worker (OS process, one per GPU)
  └── Model Runner (Python class)
        └── Model (torch.nn.Module)
```

- **Worker** — an OS process spawned by the executor via `multiprocessing`. Identified by `rank` (global orchestration) and `local_rank` (device assignment). One per GPU.
- **Model Runner** — a Python class inside the worker process. Manages the forward pass: prepares input tensors, captures CUDA graphs, calls the model.
- **Model** — a `torch.nn.Module` subclass. The actual PyTorch neural network with weights and computation.

So: worker is a process, model runner is a Python class, model is a PyTorch class. All three live in the same GPU worker process.

## Three Key Design Decisions

### 1. Single `VllmConfig` object passed everywhere

All configuration lives in one `VllmConfig` object that flows through the entire class hierarchy. Adding a new feature only requires adding a field to `VllmConfig` — no need to change constructors at every layer. It acts as engine-level global state.

### 2. Uniform model constructor signature

All models implement:

```python
def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
```

Keyword-only to prevent accidental misuse. The `prefix` argument (e.g., `"vision"`, `"language"`) enables:
- Non-uniform quantization (different parts of the model quantized differently)
- Sub-model initialization for VLMs (vision encoder + language decoder composed uniformly)

### 3. Sharding and quantization happen at initialization, not after

For a 405B model (810GB weights) on 16 H100 80GB GPUs: if you loaded the full model then sharded, you'd need 810GB on each GPU temporarily. Instead, each layer only creates its own weight shard during initialization — much lower peak memory. Same logic applies to quantization.

This is why vLLM can serve very large models without requiring intermediate full-model materialization on each GPU.
