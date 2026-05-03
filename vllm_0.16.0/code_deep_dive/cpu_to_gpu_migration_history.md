# vLLM: CPU → GPU Operation Migration History

A code-verified reference for major operations that moved from CPU-side implementations to GPU-native paths across vLLM's evolution (2023 → v0.16.0, Feb 2026). The central milestone is the **V1 engine** (alpha Jan 2025, default from ~v0.8, sole engine in v0.16.0), which targeted near-zero CPU overhead in the execution loop.

> **Note on sourcing:** Claims marked ✅ are directly verified in v0.16.0 source. Claims marked ⚠️ are directionally accurate but nuanced by code inspection.

---

## 1. Token Sampling

**(1) What it does:** Selects next tokens from logits via top-k/top-p filtering, temperature scaling, and multinomial sampling.

**(2) Why originally on CPU:** Simple to implement with `torch.topk` + `torch.multinomial` + Python RNG. No custom kernels needed.

**(3) GPU implementation difficulty:**
- Vocabulary is 128k+ tokens; each request has different top-k, top-p, temperature parameters → needs per-row batched GPU kernels.
- RNG must be per-request seeded on GPU (`torch.Generator(device='cuda')`).
- Sorting 128k vocab is slow on GPU → FlashInfer uses sorting-free Dual Pivot Rejection Sampling.
- Must work across NVIDIA/AMD/CPU backends.

**(4) When / trigger:** V0 bottleneck documented in issue #3384 (March 2024). V1 engine alpha (Jan 27, 2025) restructured sampler entirely to GPU Worker side. FlashInfer sampling integration added via PR #7137 and related.

**(5) Performance impact:** Eliminated GPU→CPU copy + device sync per token. 50%+ throughput improvement at large batch sizes vs. CPU sampling path.

**Code evidence (v0.16.0):** ✅
- `vllm/v1/sample/sampler.py`: `compute_logprobs = logits.log_softmax(dim=-1)` — pure GPU
- `vllm/v1/sample/ops/topk_topp_sampler.py`: FlashInfer → PyTorch native fallback; no `.cpu()` in hot path
- FlashInfer path: `flashinfer.sampling.top_k_top_p_sampling_from_logits(...)` — GPU kernel
- Fallback: `torch.multinomial(renorm_probs, ...)` — GPU tensor

**Kernel priority (v0.16.0):** FlashInfer (opt-in via `VLLM_USE_FLASHINFER_SAMPLER=1`) → PyTorch native GPU multinomial. Note: FlashInfer is **not** the default — users must opt in explicitly.

---

## 2. Logprobs Computation

**(1) What it does:** Computes log probabilities over vocabulary for the sampled token and optionally top-k tokens, returned to users or used in guided decoding.

**(2) Why originally on CPU:** Copy full logits to CPU, run `log_softmax` + `topk` in Python — straightforward.

**(3) GPU implementation difficulty:**
- Full-vocab softmax on 128k+ logits is memory-intensive; need "selective top-k only" materialization.
- Must fuse with sampling pipeline without extra device syncs.
- `torch.compile`-based custom kernel avoids extra copies.

**(4) When / trigger:** V1 engine alpha (Jan 2025). Issue #5907 reported GPU OOM + CPU saturation under high logprob requests.

**(5) Performance impact:** GPU→CPU transfer reduced 10x+ (top-k tokens only vs full vocab). 20-40% throughput improvement for high-logprobs workloads.

**Code evidence (v0.16.0):** ✅
- `vllm/v1/sample/ops/logprobs.py`: `batched_count_greater_than` — `@torch.compile` GPU kernel
- `vllm/v1/sample/sampler.py`: `compute_logprobs = logits.log_softmax(dim=-1, dtype=torch.float32)` — GPU, no `.cpu()`

---

## 3. Attention Metadata Preparation

**(1) What it does:** Constructs per-step batch metadata for PagedAttention: `block_table`, `slot_mapping`, `position_ids`, `seq_lens`, `query_start_loc`, etc.

**(2) Why originally on CPU:** Simple Python list comprehensions + loops rebuilding from scratch every step.

**(3) GPU implementation difficulty:**
- Dynamic batch composition changes every step; must handle chunked prefill, prefix caching, TP/PP, and speculative decode simultaneously.
- Needs incremental diff updates rather than full rebuild.
- Zero-copy staging via pinned CPU buffers + async H2D copy.

**(4) When / trigger:** V1 engine (Jan 2025), "execution loop zero CPU overhead" design goal.

**(5) Performance impact:** Eliminates metadata rebuild bottleneck for small/fast models where GPU compute is not the bottleneck. Better cudagraph compatibility.

**Code evidence (v0.16.0):** ⚠️ Nuanced
- `vllm/v1/worker/gpu_model_runner.py` uses **pinned CPU buffers** (`.cpu` numpy-backed views, `pin_memory=True`) as staging areas.
- Pattern: write incrementally to `self.seq_lens.cpu[...]`, `self.slot_mapping.cpu[...]` — then async H2D copy to GPU tensor.
- Not "fully GPU" — CPU still writes metadata, but via pinned memory with non-blocking transfers, minimizing sync overhead.
- True zero-CPU-overhead applies only to the GPU forward + sampling path, not metadata prep.

---

## 4. KV Cache Management & Preemption

**(1) What it does:** Block-level allocation/deallocation of paged KV cache; preemption when memory is full.

**(2) Why originally on CPU:** Python dict-based allocator + CPU-coordinated GPU↔CPU swap for preempted sequences.

**(3) GPU implementation difficulty:**
- Block table must be updated atomically with scheduling decisions.
- Preemption via recompute (re-run prefill) is simpler than GPU↔CPU swap in V1.
- Offloading (GPU→CPU/NVMe) now managed through `kv_offload/` subsystem with LRU/ARC managers.

**(4) When / trigger:** V1 engine (Jan 2025). V1 largely **removed** GPU↔CPU full swap in favor of **recompute-based preemption** and a dedicated KV offload layer.

**(5) Performance impact:** Fewer preemption stalls; KV offload enables larger effective cache via CPU/NVMe backing.

**Code evidence (v0.16.0):** ✅
- `vllm/v1/core/sched/scheduler.py`: preemption via `recompute_kv_load_failures` flag — no GPU↔CPU tensor swap path found in V1 scheduler
- `vllm/v1/kv_offload/`: full offload subsystem with `abstract.py`, `lru_manager.py`, `arc_manager.py`, `backends/cpu.py`
- `cpu_worker.py`: `cpu_kvcache_space_bytes` — CPU KV cache only for CPU inference mode, not standard GPU preemption

---

## 5. Speculative Decoding Verification (Rejection Sampling)

**(1) What it does:** Verifies draft model's speculative tokens against target model probabilities using rejection sampling.

**(2) Why originally on CPU:** Draft-accept logic was CPU-coordinated in V0 for simplicity.

**(3) GPU implementation difficulty:**
- Must fuse with main sampler (shared GPU RNG, same top-k rejection logic).
- Needs cross-TP-rank consistency.
- Structured output compatibility requires grammar bitmask integration.

**(4) When / trigger:** V1 engine full integration (2025). Unified Parallel Drafting PR #32887 (early 2026).

**(5) Performance impact:** Verification without extra CPU sync; speculative decoding speedup from ~1.5x to 2.5x+ (especially for n-gram / MTP speculative strategies).

**Code evidence (v0.16.0):** ⚠️ Mixed
- `vllm/v1/sample/rejection_sampler.py`: forward pass largely on GPU (tensors on `logit_start_indices.device`)
- BUT: line 235: `output_token_ids_np = output_token_ids.cpu().numpy()` — final accepted tokens are CPU-copied for output
- Line 274: `num_draft_tokens = torch.tensor(metadata.num_draft_tokens, device="cpu")` — some metadata still CPU
- Core rejection math is on GPU; output packaging has necessary CPU copies

---

## 6. Multimodal Input Preprocessing

**(1) What it does:** Converts image/video/audio inputs to encoder embeddings cached for reuse.

**(2) Why originally on CPU:** Python Pillow/torchvision preprocessing, simple blocking calls.

**(3) GPU implementation difficulty:**
- Vision encoder runs on GPU; need async pipeline so preprocessing doesn't block decoding.
- Encoder output caching (`encoder_cache`) avoids recomputing embeddings for repeated images.
- Zero-copy IPC between preprocessor and worker.

**(4) When / trigger:** V1 engine (Jan 2025); encoder_runner separated into dedicated module.

**(5) Performance impact:** GPU idle time reduction during VLM serving; throughput improvement especially for repeated-image workloads.

**Code evidence (v0.16.0):** ✅
- `vllm/v1/worker/gpu/mm/encoder_runner.py`: `encoder_cache: dict[str, torch.Tensor]` — GPU tensor cache per mm hash
- `vllm/v1/worker/worker_base.py`: `_apply_mm_cache` fetches pre-computed features; `mm_receiver_cache` for IPC
- `vllm/v1/worker/gpu/buffer_utils.py:32`: `out.copy_(tmp, non_blocking=True)` — async copy pattern

---

## Common Pattern Across All Migrations

| Stage | Pattern |
|---|---|
| V0 (2023–2024) | CPU computation, GPU→CPU copies, device syncs, Python loops |
| V1 alpha (Jan 2025) | Restructured to GPU Worker; pinned-memory staging; async H2D |
| V1 mature (v0.16.0) | GPU-native kernels (FlashInfer/Triton/torch.compile); EngineCore CPU-free hot path |

**What CPU still does in v0.16.0:**
- Scheduling decisions (request ordering, block allocation accounting)
- Metadata staging (pinned CPU buffers for H2D transfers)
- Output packaging (small token-ID tensors copied to CPU for IPC)
- Grammar state machines for structured output (bitmask generation)

**What is fully GPU in v0.16.0:**
- All logit operations (temperature, penalties, top-k/p, sampling)
- Logprobs computation
- Attention forward pass + KV cache writes
- Speculative rejection sampling math
- Vision encoder forward pass

---

## Note on Grok-Reported Commit IDs

Two commit IDs mentioned in secondary sources (`bbf81f9a` for prefix caching, `c656ba3b` for Triton top-k) were **not found** in the v0.16.0 tag. These may refer to commits on branches not included in the shallow clone, or may be hallucinated. The operational claims they supported are directionally correct based on code inspection, but the specific commit attribution is unverified.

---

## See Also
- [[ai/llm/inference/frameworks/vllm_0.16.0/code_deep_dive/scheduler_output_to_kernel]] — full execution path from scheduler dispatch to GPU kernels
- [[ai/llm/inference/frameworks/vllm_0.16.0/code_deep_dive/token_sampling_gpu_deep_dive]] — deep dive on GPU sampling implementation and history
