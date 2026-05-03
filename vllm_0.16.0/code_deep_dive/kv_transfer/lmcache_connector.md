# LMCache Connector — Code Deep Dive

## Overview

LMCache is an external KV cache store that extends vLLM's GPU prefix cache to a second tier (CPU, disk, or remote). vLLM has **two** LMCache connectors with fundamentally different loading strategies:

| | `LMCacheConnectorV1` | `LMCacheMPConnector` |
|---|---|---|
| **Loading** | Synchronous (blocks forward) | Asynchronous (multi-process) |
| **Process model** | Single process | Separate LMCache worker process |
| **`is_async` return** | Always `False` | `True` when tokens to load |
| **Request during load** | IN the forward batch | WAITING_FOR_REMOTE_KVS |
| **Best for** | Local CPU store, small prefixes | Remote stores, large prefixes |

## LMCacheConnectorV1 (Sync)

### Why Sync?

The original connector was designed sync for three reasons:

1. **Simplicity** — no state machine for in-flight transfers. `retrieve()` returns, KV is done.
2. **Target workload** — LMCache started as a CPU-side prefix cache. CPU→GPU PCIe for typical prefixes (hundreds of tokens) is microseconds to low milliseconds.
3. **Correctness** — since `is_async=False`, the scheduler treats external tokens as already computed and schedules the request into the forward batch immediately. KV MUST be in GPU before attention runs.

### Loading Path

```
Scheduler:
  get_num_new_matched_tokens(req, local_computed)
    → lookup_client.lookup(token_ids)        # check LMCache for hits
    → returns (num_hit_tokens, False)         # False = sync

  update_state_after_alloc(req, ext_tokens)
    → marks load_spec.can_load = True

  build_connector_meta(scheduler_output)
    → creates ReqMeta with load_spec for each request
    → includes slot_mapping (GPU block offsets for where to write)

Worker (start_load_kv, BLOCKS HERE):
  for each request with load_spec:
    slot_mapping = request.slot_mapping.cuda()   # ← CPU→GPU copy of mapping
    lmcache_engine.retrieve(
        tokens[:cached],
        token_mask,
        kvcaches=kvcaches,                       # ← direct write into GPU paged KV
        slot_mapping=slot_mapping[:cached],
    )
    # Blocking! Returns only when all KV is in GPU.

  # THEN forward() runs for the entire batch including this request
```

**The latency concern**: Every request with an LMCache hit delays the entire batch's forward pass. For a large prefix (thousands of tokens from CPU/remote), this adds meaningful latency.

### Saving Path

```
Worker (wait_for_save, also blocks):
  for each request with save_spec:
    lmcache_engine.store(
        token_ids,
        mask=store_mask,
        kvcaches=kvcaches,
        slot_mapping=slot_mapping,
        offset=skip_leading_tokens,
    )
```

Saves happen in `wait_for_save()` after the forward pass. Also blocking — the step doesn't complete until all stores finish.

### Layerwise Mode

When `use_layerwise=True`, LMCache uses Python generators to pipeline load/save with the forward pass:

```
start_load_kv():
  layerwise_retriever = lmcache_engine.retrieve_layer(...)
  next(layerwise_retriever)  # prefetch first 2 layers
  next(layerwise_retriever)

forward() → attention layer i:
  wait_for_layer_load(layer_i):
    next(layerwise_retriever)  # blocks until layer i ready

  save_kv_layer(layer_i):
    next(layerwise_storer)     # start saving layer i
```

Better than fully blocking, but each attention layer still waits for its KV load.

## LMCacheMPConnector (Async)

### Key Difference

```python
# LMCacheMPConnector.get_num_new_matched_tokens():
return need_to_load, need_to_load > 0   # True = async!
```

When `is_async=True`:
- Request enters `WAITING_FOR_REMOTE_KVS` — NOT in the forward batch
- KV loads happen in a **separate process** via `LMCacheMPWorkerAdapter`
- Forward pass continues with other requests unblocked

### Loading Path

```
Worker (start_load_kv):
  # Record CUDA event for synchronization
  with torch.cuda.stream(torch.cuda.current_stream()):
      event = torch.cuda.Event(interprocess=True)
      event.record()

  # Submit to separate process — returns immediately
  worker_adapter.batched_submit_retrieve_requests(request_ids, ops, event)

  # Forward pass runs for OTHER requests
```

```
Worker (wait_for_save):
  # Same pattern for stores
  event = torch.cuda.Event(interprocess=True)
  event.record()
  worker_adapter.batched_submit_store_requests(request_ids, ops, event)
```

Completion is reported via `get_finished()` across steps, just like NIXL.

### Scheduler-Side Differences

The MP connector also uses async lookup:

```python
# Submit lookup (non-blocking)
scheduler_adapter.maybe_submit_lookup_request(req_id, block_hashes)

# Check result (may return None if not ready)
ret = scheduler_adapter.check_lookup_result(req_id)
if ret is None:
    return None, True  # tell scheduler to try again later
```

This means even the cache hit check doesn't block the scheduler.

## Request Lifecycle Comparison

### LMCacheConnectorV1 (sync)
```
Step N:
  Scheduler: get_num_new_matched_tokens → (1000, False)
             allocate_slots (request scheduled for forward)
  Worker:    start_load_kv → BLOCKS for ~5ms copying 1000 tokens from CPU
             forward()     → runs with this request + all others
             wait_for_save → BLOCKS storing new KV back
```

### LMCacheMPConnector (async)
```
Step N:
  Scheduler: get_num_new_matched_tokens → (1000, True)
             request → WAITING_FOR_REMOTE_KVS
  Worker:    start_load_kv → submits to MP worker, returns immediately
             forward()     → runs OTHER requests only

Step N+K:
  Worker:    get_finished() → req_id done
  Scheduler: moves request to schedulable
  
Step N+K+1:
  Request scheduled normally, KV already in GPU
```

## Chunking and Deduplication

LMCache operates on **chunks** (default 256 tokens), not individual blocks:
- Store: only saves when token count crosses a chunk boundary
- Load: retrieves chunk-aligned prefixes
- `skip_leading_tokens`: avoids re-storing already-saved chunks
- `discard_partial_chunks`: whether to save the last incomplete chunk

The `block_size_factor` handles the mismatch between vLLM's block size and LMCache's chunk size.

## Token Masking

LMCache uses a `token_mask` to handle the overlap between GPU prefix cache and LMCache:

```python
token_mask = torch.ones(len(tokens), dtype=torch.bool)
masked_token_count = (
    vllm_cached_tokens // chunk_size * chunk_size
)
token_mask[:masked_token_count] = False  # don't load what GPU already has
```

Only the tokens NOT already in GPU are actually loaded from LMCache.

## Multimodal Support

LMCache handles multimodal inputs by hashing image/media features into token IDs:
```python
apply_mm_hashes_to_token_ids(token_ids_tensor, mm_hashes, mm_positions)
```
This allows prefix caching to work across requests with identical media inputs.

## P/D Disaggregation via LMCache

LMCache also supports P/D disaggregation through `DisaggSpec`:
- Prefill instance stores KV to LMCache
- Decode instance loads KV from LMCache
- Uses `kv_role`: `"kv_producer"` vs `"kv_consumer"`
- Producer skips loading, consumer skips saving

## Key Configuration

- `LMCACHE_CONFIG_FILE` — path to LMCache config
- `enable_async_loading` — async lookup on scheduler side (not GPU transfer)
- `use_layerwise` — layerwise pipeline for V1 connector
- `enable_blending` — blend partial cache hits
- `save_decode_cache` — whether to cache decode-phase KV
- `chunk_size` — LMCache chunk size (default 256)
- `discard_partial_chunks` / `save_unfull_chunk` — partial chunk handling
- `skip_last_n_tokens` — skip caching last N tokens
- `priority_limit` — skip saving for low-priority requests

## Source Files

```
vllm/distributed/kv_transfer/kv_connector/v1/
├── lmcache_connector.py              — V1 wrapper (sync, delegates to impl)
├── lmcache_mp_connector.py           — Multi-process connector (async)
└── lmcache_integration/
    ├── vllm_v1_adapter.py            — LMCacheConnectorV1Impl (main logic)
    ├── multi_process_adapter.py      — MP adapter
    └── utils.py                      — Helpers (config, MLA detection, MM hashing)
```

## MP Connector History & Stability

### Timeline
- **Created**: Oct 31, 2025 (first commit), merged Nov 12, 2025 (PR #27902 by Yihua Cheng/ApostaC, UChicago)
- **First release**: vLLM v0.11.1 (Nov 18, 2025) — squeezed in after last RC, PR was tagged `[WIP]`
- **Stabilized**: ~Jan 2026 (no functional changes between v0.15.0 and v0.16.0)
- **Total commits**: 10 (vs 18 for V1 connector)

### Known Issues (as of Feb 2026)
- **Breaks with vLLM async scheduling** (LMCache #2356, Jan 2026) — duplicate request IDs in `get_finished()` crash scheduler. Workaround: `--no-async-scheduling`
- **Version compatibility pain** (LMCache #1768) — tight coupling with vLLM internals
- **Still labeled "Experimental"** in LMCache docs

### Production Status
- **V1 connector (sync)**: Used in vLLM Production Stack, Red Hat llm-d, Ceph integration
- **MP connector (async)**: Experimental, not widely deployed yet
- **llm-d benchmarks** (16x H100, Qwen3-32B): LMCache CPU offload gives ~22% throughput improvement, ~28% TTFT reduction when KV exceeds GPU memory

## LMCache Repo vs vLLM-Bundled Code

The LMCache repo (`github.com/LMCache/LMCache`, `dev` branch) evolves significantly faster than the copy in vLLM. As of Feb 2026:

### Key Architectural Changes in Latest LMCache

**1. Token-Based Hashing (moved to server side)**
- vLLM 0.16.0: MP connector passes block hashes (from vLLM's KVCacheManager) → `LoadStoreOp.block_hashes: list[bytes]`
- Latest LMCache: passes raw token IDs → `LoadStoreOp.token_ids: list[int]` with `start`/`end`
- Server computes hashes itself using `TokenHasher` (blake3 default, vLLM hash compat)
- Decouples from vLLM's hash implementation changes across versions

**2. Session Management (new)**
- `Session` per request: tracks accumulated tokens, rolling chunk hashes (incremental)
- `SessionManager`: thread-safe, TTL-based expiry (10 min default)
- Server calls `end_session(request_id)` on completion

**3. Storage Manager Refactor**
- Reservation-based writes: `reserve_write()` / `finish_write()`
- Async prefetch for lookups: `submit_prefetch_task()` / `query_prefetch_status()`
- Context manager for reads: `read_prefetched_results()`

**4. Async Dedup Fix**
- Worker adapter has explicit dedup logic (`previously_finished` set)
- Addresses the async scheduling crash (#2356)

**5. Request Telemetry**
- New `RequestTelemetryFactory` for P/D disaggregation monitoring

### Code Size Comparison
| File | vLLM 0.16.0 | Latest LMCache |
|---|---|---|
| MP adapter | 395 lines | 577 lines (+46%) |
| V1 adapter | 1,431 lines | 1,629 lines (+14%) |
| Plus new files | — | session.py, token_hasher.py, storage_manager/ |

## CUDA Kernels & Hardware Support

LMCache uses custom CUDA kernels for performance-critical operations:

- **`mem_kernels.cu`** — fused multi-layer KV transfer between GPU paged memory and contiguous buffers. Called by `lmc_ops.multi_layer_kv_transfer()` in the MP server's store/retrieve
- **`pos_kernels.cu`** — positional encoding ops
- **`ac_enc.cu` / `ac_dec.cu`** — arithmetic coding (CacheGen compression)
- **`mem_alloc.cpp`** — custom memory allocator
- **C++ storage manager** — with TTL locks

### AMD ROCm Support
- **Officially supported** for core store/retrieve (MI300X targeted)
- `mem_kernels.cu` has `#ifdef USE_ROCM` for HIP compatibility
- HIPify pipeline in setup.py: `BUILD_WITH_HIP=1 pip install -e .`
- Pre-built Docker images on AMD Infinity Hub
- ⚠️ CacheGen compression kernels (`ac_enc/dec.cu`) lack ROCm guards — likely NVIDIA-only

## Open Questions / Future Work

- V1 sync connector could benefit from CUDA stream overlap (load on separate stream, sync before attention)
- MP connector adds process overhead — is there a middle ground?
- `get_finished()` in V1 connector returns `(None, None)` — async saving not tracked, relies on blocking `wait_for_save()`
- The `async_loading` config flag only affects scheduler-side lookup, not the actual GPU transfer — naming is confusing
- LMCache repo code diverging from vLLM-bundled copy — sync cadence unclear
