# vLLM v0.16.0: Token Sampling on GPU — Deep Dive

A code-level reference for how vLLM v0.16.0 implements token sampling entirely on the GPU, covering historical context, the exact call chain, kernel selection logic, and RNG design.

---

## Historical Arc: From CPU Bottleneck to Full GPU Sampling

### Early vLLM (2023–2024, v0.5.x and earlier)
- Sampling was partially on GPU but with a severe CPU bottleneck.
- Logits were frequently copied from GPU → CPU (`.cpu()`) and sampled using high-level PyTorch APIs (`torch.topk`, `torch.multinomial`, `numpy.random`).
- Every token step incurred a device sync, causing the GPU to idle waiting for CPU sampling to complete.
- Issue #3384 (March 2024) explicitly flagged: *"Sampling is very slow, causing a CPU bottleneck."*

### v0.6.0 (September 2024)
- Began reducing CPU overhead; introduced multi-step decoding to allow async sampling.

### V1 Engine Alpha (January 27, 2025) — key turning point
- `Sampler` fully restructured to live on the GPU Worker side.
- FlashInfer integration (PR #7137 and related) brought sorting-free GPU sampling kernels.
- FlashInfer's Dual Pivot Rejection Sampling introduced: no full vocabulary sort needed.

### v0.16.0 (February 25, 2026) — fully mature
- V1 engine is the only engine.
- Sampling is 100% on `output_rank`'s GPU.
- Triton-based top-k/top-p kernels added February 17, 2026 (commit `c656ba3b`).
- Zero unnecessary CPU-GPU synchronization in the hot path.

---

## Where Sampling Happens

Sampling runs **exclusively on `output_rank`** — the worker at TP rank 0 of the last PP stage. All other workers execute their forward slice and return nothing.

`output_rank = world_size - tensor_parallel_size * prefill_context_parallel_size`

Logits stay on GPU from logit gather all the way through to `sampled_token_ids`. Only the final small token-ID tensor is returned to EngineCore.

---

## Call Chain (all on `output_rank`'s GPU)

```
GPUWorker.sample_tokens()                    # gpu_worker.py
  → GPUModelRunner.sample_tokens()           # gpu/model_runner.py
    → self.sampler(logits, sampling_metadata)
      → Sampler.forward() / .sample()        # v1/sample/sampler.py  (nn.Module)
          1. logits = logits.to(torch.float32)        # GPU cast
          2. apply_temperature(logits, temps)          # logits.div_() in-place GPU
          3. apply_penalties / bad_words / processors  # in-place GPU
          4. greedy path: logits.argmax(dim=-1)        # GPU, temperature=0 requests
          5. TopKTopPSampler.__call__(logits, k, p)   # v1/sample/ops/topk_topp_sampler.py
        → SamplerOutput(sampled_token_ids=...)
```

---

## Kernel Selection: FlashInfer → Triton → PyTorch

`vllm/v1/sample/ops/topk_topp_sampler.py:TopKTopPSampler.__call__()`

Three backends in priority order:

### 1. FlashInfer (default when available)
- Activated when `VLLM_USE_FLASHINFER_SAMPLER=1` and FlashInfer is installed.
- Uses **Dual Pivot Rejection Sampling** — sorting-free algorithm.
- Works via parallel prefix sums instead of sorting the full vocabulary.
- Extremely fast on large vocab (128k+) with heterogeneous per-request top-k/top-p.
```python
flashinfer_sample(logits.contiguous(), k, p, generators)
```

### 2. Triton kernel (fallback, batch ≥ 8)
- Added February 17, 2026 (commit `c656ba3b`).
- Custom Triton kernel `apply_top_k_top_p_triton`.
- Better than PyTorch native for medium-to-large batch sizes.

### 3. PyTorch native (final fallback)
- `logits.softmax(dim=-1)` → `torch.multinomial(probs, ...)`
- RNG via `torch.empty_like(probs).exponential_()` per generator.
- Fully on GPU, but higher overhead than FlashInfer/Triton.
- Used for small batches or when FlashInfer/Triton are unavailable.

---

## RNG Design

```python
# sampling_metadata.generators:
{request_id: torch.Generator(device='cuda')}
```

- Each request gets its own `torch.Generator` seeded on the GPU device.
- Guarantees per-request reproducibility without cross-request interference.
- No CPU RNG involved in the hot path.
- Preserves determinism across TP ranks since only `output_rank` samples.

---

## Why GPU Sampling Is Hard (and Worth It)

### CPU was simple but costly
- `torch.topk` + `torch.multinomial` = easy to write, trivial to debug.
- But: every step required GPU→CPU copy + synchronization = GPU idle time.

### GPU challenges
| Challenge | Solution in vLLM |
|---|---|
| Heterogeneous per-request params (different top-k, top-p, temp per request) | Batched kernel with per-row parameters |
| Sorting 128k+ vocab is slow on GPU | FlashInfer Dual Pivot Rejection Sampling (sorting-free) |
| Per-request seeded RNG | `torch.Generator(device='cuda')` per request |
| Warp divergence from different sampling paths | Fused kernels with masked ops |
| Cross-platform (NVIDIA/AMD/CPU) | FlashInfer for CUDA, Triton for portability, PyTorch fallback |
| Numerical stability + determinism | float32 cast before sampling, explicit seed management |

### Performance impact
- Eliminated GPU→CPU copy + sync per token.
- Full overlap: sampling happens while previous iteration's output is being streamed.
- Throughput improvement: 50%+ at large batch sizes vs. CPU sampling path.

---

## Adjacent Topics (not core sampling)

### Structured output / guided decoding
- Grammar state machine (xgrammar) can run on CPU to compute a **bitmask** of allowed tokens.
- But the actual masking is applied on GPU: `logits.masked_fill_(~bitmask, -inf)`.
- Condition: `--guided-decoding-backend xgrammar` or when `grammar_output` is non-None.

### Logprobs
- `Sampler.compute_logprobs(logits.log_softmax(...))` runs on GPU.
- Final `LogprobsTensors` packaging involves a CPU copy (pinned memory) for IPC return.
- Condition: `max_num_logprobs > 0`.

### Speculative decoding
- Draft token sampling and verification both run on the GPU Worker.
- `model_executor.take_draft_token_ids()` also uses `unique_reply_rank`.

---

## Key Files

| File | Role |
|---|---|
| `vllm/v1/worker/gpu_worker.py` | `GPUWorker.sample_tokens()` entry point |
| `vllm/v1/worker/gpu/model_runner.py` | Passes logits to sampler |
| `vllm/v1/sample/sampler.py` | Core `Sampler` nn.Module: temperature, penalties, dispatch |
| `vllm/v1/sample/ops/topk_topp_sampler.py` | Kernel selection: FlashInfer → Triton → PyTorch |
| `vllm/model_executor/layers/logits_processor.py` | TP logit gather before sampling |

---

## See Also
- [[ai/llm/inference/frameworks/vllm_0.16.0/code_deep_dive/scheduler_output_to_kernel]] — full execution path from scheduler to GPU kernels
