# vLLM v0.16.0: From SchedulerOutput to GPU Kernels

A code-level walkthrough of the full execution path from the scheduler handing off a batch to actual GPU kernel execution across single and multi-GPU deployments.

---

## Why This Path Matters

The scheduler decides *who runs* — but there's a long journey from that decision to electrons moving in silicon. Understanding this path answers:
- What processes are involved and how they communicate?
- Which rank does sampling, and why?
- Where do NCCL collectives vs. Python RPCs happen?
- What does the kernel actually receive?

---

## Process Map

In multi-GPU V1 engine, there are three distinct process tiers:

```
┌─────────────────────────────────────────────────┐
│  API Server Process                             │
│  entrypoints/openai/api_server.py               │
│  → AsyncLLM (v1/engine/async_llm.py)           │
│     communicates via ZMQ (ROUTER/PUSH/PULL)    │
└────────────────┬────────────────────────────────┘
                 │ ZMQ (msgpack EngineCoreRequest)
                 ▼
┌─────────────────────────────────────────────────┐
│  EngineCore Process                             │
│  v1/engine/core.py                              │
│  Contains: Scheduler + MultiprocExecutor        │
│  communicates via collective_rpc (shared mem)  │
└────────────────┬────────────────────────────────┘
                 │ collective_rpc (MessageQueue shm)
        ┌────────┼──────────────────┐
        ▼        ▼                  ▼
┌────────────┐ ┌────────────┐  ┌────────────┐
│  Worker 0  │ │  Worker 1  │  │  Worker N  │
│  (rank 0)  │ │  (rank 1)  │  │  (rank N)  │
│  GPU:0     │ │  GPU:1     │  │  GPU:N     │
└────────────┘ └────────────┘  └────────────┘
    ←── NCCL collectives during forward pass ──→
```

---

## Step-by-Step Execution Path

### Step 1 — `EngineCore.step()` kicks off the work
`vllm/v1/engine/core.py:EngineCore.step()`

```python
scheduler_output = self.scheduler.schedule()
future = self.model_executor.execute_model(scheduler_output, non_block=True)
model_output = future.result()
engine_core_outputs = self.scheduler.update_from_output(scheduler_output, model_output)
```

EngineCore is **coordinator only** — it schedules, dispatches, and processes final tokens. It does no forward math.

---

### Step 2 — `MultiprocExecutor.execute_model()` fan-outs to workers
`vllm/v1/executor/multiproc_executor.py:MultiprocExecutor.execute_model()`

```python
def execute_model(self, scheduler_output, non_block=False):
    return self.collective_rpc(
        "execute_model",
        args=(scheduler_output,),
        unique_reply_rank=self.output_rank,  # ← only this rank replies
        non_block=non_block,
    )
```

`collective_rpc` broadcasts the method + args via **shared-memory MessageQueue** (`rpc_broadcast_mq`) to every worker process simultaneously. All workers execute it — but only `output_rank` sends a response back.

#### IPC mechanism
- **Broadcast**: `rpc_broadcast_mq.enqueue((method, args, kwargs, output_rank))`
- **Response**: only `response_mqs[output_rank]` is dequeued by EngineCore
- Not ZMQ, not HTTP — custom shared-memory ring buffer for low latency

---

### Step 3 — Every worker executes its slice of the forward pass
`vllm/v1/worker/gpu_worker.py:GPUWorker.execute_model()`

Each worker:
1. Unpacks `SchedulerOutput` (request metadata, token IDs, KV block assignments)
2. Prepares input tensors on its GPU
3. Calls `self.model_runner.execute_model(scheduler_output)`

The model runner (`gpu_model_runner.py`) constructs batched input tensors, sets up attention metadata (paged KV block indices, query/seq lengths), and runs `model.forward(...)`.

---

### Step 4 — Distributed forward: TP/PP/EP collectives via NCCL
During `model.forward()`, per-layer NCCL operations happen automatically inside the model layers:

| Parallelism | What happens | NCCL op |
|---|---|---|
| **Tensor Parallel (TP)** | Each rank holds a shard of Q/K/V weight matrices; attention + FFN outputs are combined | `all_reduce` after each layer's linear |
| **Pipeline Parallel (PP)** | Each stage holds a subset of layers; activations are passed forward | `send/recv` hidden states between PP stages |
| **Expert Parallel (EP / MoE)** | Experts are sharded; tokens are routed to expert-owning ranks | `all_to_all` for token dispatch/combine |

These collectives are embedded in layer implementations (`vllm/model_executor/layers/`), not in the scheduler or executor.

**TP logit gathering** (key for sampling):
`vllm/model_executor/layers/logits_processor.py`
```python
logits = tensor_model_parallel_gather(logits)
# Non-TP-rank-0 workers get None here
# Only TP rank 0 has the complete logits
```

**PP last-stage gate:**
Only workers on the last PP stage compute LM head → logits. Earlier PP stages return intermediate hidden states and exit early.

---

### Step 5 — Attention backend dispatches to CUDA kernels

Attention is the inner loop. The backend is selected at startup:

| Backend | File | Kernel |
|---|---|---|
| FlashAttention (default CUDA) | `v1/attention/backends/flash_attn.py` | `vllm_flash_attn` (Dao-AILab) |
| FlashInfer | `v1/attention/backends/flashinfer.py` | FlashInfer paged kernels |
| Triton | `v1/attention/backends/triton_attn.py` | Triton custom ops |
| ROCm AITER | `v1/attention/backends/rocm_aiter_fa.py` | ROCm-optimized attention |
| Classic PagedAttn | `csrc/attention/paged_attention_v1.cu` / `v2.cu` | vLLM custom CUDA kernel |

In FlashAttention path:
```python
# flash_attn.py:FlashAttentionImpl.forward()
reshape_and_cache_flash(key, value, kv_cache, slot_mapping, ...)  # write new KV to paged cache
flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, ...)  # actual attention compute
```

Paged KV cache: physical GPU memory is divided into fixed-size blocks (e.g., 16 tokens/block). The scheduler's `SchedulerOutput` carries the block table mapping logical positions to physical blocks. The model runner converts this into `slot_mapping` tensors that `reshape_and_cache_flash` uses to write new KV entries.

---

### Step 6 — Sampling: fully on `output_rank`'s GPU
`vllm/v1/sample/sampler.py:Sampler.sample()`
`vllm/v1/sample/ops/topk_topp_sampler.py`

After logits are complete on `output_rank`:

```python
# All on GPU tensors:
logits = apply_temperature(logits, temperatures)      # logits.div_() in-place GPU
logits = apply_top_k_top_p(logits, top_k, top_p)    # topk_topp_sampler GPU kernel
sampled_token_ids = sample_from_logits(logits)        # multinomial / argmax GPU
```

Core ops use FlashInfer / Triton / torch GPU primitives — **RNG, top-k, multinomial sampling are all GPU operations**.

`output_rank = world_size - tensor_parallel_size * prefill_context_parallel_size`
= TP rank 0 of the last PP stage

---

### Step 7 — Results return to EngineCore
`vllm/v1/worker/gpu_model_runner.py` builds `ModelRunnerOutput` containing:
- `sampled_token_ids: List[List[int]]` (CPU-copied from GPU, pinned memory)
- `logprobs` (if requested)

This is returned through `response_mqs[output_rank]` → EngineCore receives it, updates `SequenceGroup` states, and streams token deltas back to the API server via ZMQ.

---

## IPC Summary

| Boundary | Protocol | Direction |
|---|---|---|
| API Server ↔ EngineCore | ZMQ ROUTER/DEALER (requests) + PUSH/PULL (outputs) | Bidirectional |
| EngineCore ↔ Workers | Shared-memory MessageQueue (`collective_rpc`) | Broadcast in, one reply out |
| Worker ↔ Worker (GPU math) | NCCL (TP all_reduce / PP send-recv / EP all_to_all) | Peer-to-peer GPU |

---

## Multi-GPU Rank Roles

| Rank | Role during forward |
|---|---|
| All TP ranks | Hold weight shards, execute partial matmuls, all_reduce to combine |
| Non-last PP stages | Execute subset of layers, send hidden states to next stage |
| Last PP stage, non-TP-0 | Execute final layers, partial logits — **discarded** |
| `output_rank` (last PP + TP-0) | Full logits → sampling → returns `ModelRunnerOutput` |

---

## Myth vs Fact

| Myth | Fact |
|---|---|
| EngineCore does sampling on CPU | EngineCore never touches logits. Sampling is fully on `output_rank`'s GPU. |
| Each worker returns a result | Only `output_rank` returns `ModelRunnerOutput` to EngineCore. Others execute silently. |
| TP workers independently sample | TP non-0 ranks get `None` from logit gather — only TP rank 0 has full logits. |
| Sampling uses CPU RNG/top-k | RNG, top-k, multinomial are all GPU ops (FlashInfer/Triton/torch CUDA). |
| EngineCore-to-Worker uses ZMQ | EngineCore↔Worker uses shared-mem MessageQueue. ZMQ is only API↔EngineCore. |

---

## Key File Index

| Component | File |
|---|---|
| EngineCore step/dispatch | `vllm/v1/engine/core.py` |
| Executor fan-out | `vllm/v1/executor/multiproc_executor.py` |
| GPU worker entry | `vllm/v1/worker/gpu_worker.py` |
| Model runner + batching | `vllm/v1/worker/gpu_model_runner.py` |
| Sampler | `vllm/v1/sample/sampler.py` |
| Top-k/top-p GPU ops | `vllm/v1/sample/ops/topk_topp_sampler.py` |
| Logit gathering (TP) | `vllm/model_executor/layers/logits_processor.py` |
| FlashAttention backend | `vllm/v1/attention/backends/flash_attn.py` |
| FlashInfer backend | `vllm/v1/attention/backends/flashinfer.py` |
| Paged attention CUDA | `csrc/attention/paged_attention_v1.cu` / `v2.cu` |
| Attention backend registry | `vllm/v1/attention/backends/registry.py` |
