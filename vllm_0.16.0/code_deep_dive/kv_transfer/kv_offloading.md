# KV Offloading — Code Deep Dive

## Overview

vLLM's built-in KV offloading system spills KV cache blocks from GPU to CPU memory and loads them back on prefix cache miss. It's a **Tier 2 cache** sitting behind the GPU prefix cache (Tier 1): when a request's prefix isn't found in GPU memory, the scheduler checks if it was previously offloaded to CPU before falling back to recomputation.

This is entirely separate from LMCache-based offloading. LMCache connectors (`LMCacheConnectorV1`, `LMCacheMPConnector`) have their own storage manager, eviction logic, and CUDA kernels. The system described here is **vLLM-native** — no external dependencies, purpose-built for the `OffloadingConnector`.

### History

Introduced by Or Ozeri (IBM) in a 5-commit series on **Sep 18-19, 2025**, first released in **vLLM v0.11.0**. Before this, CPU offloading required LMCache as an external dependency. As of v0.16.0 the API is still marked experimental (~25 follow-up commits in 5 months, including 4 bug fixes).

## Architecture

```
OffloadingConnector (KVConnectorBase_V1)
  ├── OffloadingConnectorScheduler          ← scheduler process
  │     └── OffloadingManager               ← tracks what's in CPU, eviction
  │           └── Backend (CPUBackend)       ← CPU block allocation bookkeeping
  │
  └── OffloadingConnectorWorker             ← GPU worker process
        └── OffloadingWorker                ← dispatches to handlers by transfer type
              └── SingleDirectionHandler[]  ← async CUDA stream GPU↔CPU copies
                    └── ops.swap_blocks()   ← custom CUDA kernel
```

The OffloadingManager and everything under `v1/kv_offload/` is a **sub-component of the OffloadingConnector** — not a general-purpose abstraction. No other connector uses it. Each connector (LMCache, NIXL, etc.) manages its own external storage independently.

### Source Files

```
vllm/distributed/kv_transfer/kv_connector/v1/
  └── offloading_connector.py       — OffloadingConnector, scheduler + worker halves

vllm/v1/kv_offload/
  ├── abstract.py                   — OffloadingManager ABC, LoadStoreSpec, events
  ├── lru_manager.py                — LRUOffloadingManager
  ├── arc_manager.py                — ARCOffloadingManager
  ├── backend.py                    — Backend ABC, BlockStatus
  ├── backends/cpu.py               — CPUBackend, CPUBlockStatus
  ├── mediums.py                    — GPULoadStoreSpec, CPULoadStoreSpec
  ├── spec.py                       — OffloadingSpec ABC
  ├── cpu.py                        — CPUOffloadingSpec (wires manager + handlers)
  ├── factory.py                    — OffloadingSpecFactory (registry)
  └── worker/
      ├── worker.py                 — OffloadingWorker, OffloadingHandler ABC
      └── cpu_gpu.py                — SingleDirectionOffloadingHandler, CpuGpuOffloadingHandlers

vllm/v1/worker/
  └── kv_connector_model_runner_mixin.py  — cross-layer uniform KV cache layout
```

## Scheduler Side

### OffloadingManager (abstract)

Runs in the scheduler process. Tracks which block hashes exist in CPU memory and manages eviction. Six operations:

| Method | Purpose |
|--------|---------|
| `lookup(block_hashes)` | Scan left-to-right, count consecutive hits. Returns count or `None` (retry later) |
| `prepare_load(block_hashes)` | Pin blocks (`ref_cnt++`), return `LoadStoreSpec` for worker |
| `touch(block_hashes)` | Mark as recently used (LRU freshness) |
| `complete_load(block_hashes)` | Unpin blocks (`ref_cnt--`) after transfer completes |
| `prepare_store(block_hashes)` | Allocate space, evict if needed, return `PrepareStoreOutput` |
| `complete_store(block_hashes)` | Mark blocks ready (`ref_cnt`: -1 → 0), now loadable |

**ref_cnt encoding:**

- **-1** = allocated but not yet written (store in progress, not readable)
- **0** = ready and evictable
- **>0** = pinned for active load(s), not evictable

### LRUOffloadingManager

`OrderedDict[BlockHash, BlockStatus]`. `touch()` calls `move_to_end()`. Eviction pops from the front (oldest), skipping blocks with `ref_cnt > 0`.

### ARCOffloadingManager

Implements ARC (Adaptive Replacement Cache) — self-tunes between recency and frequency:

```
T1 (OrderedDict) — recent blocks (accessed once)
T2 (OrderedDict) — frequent blocks (accessed 2+ times)
B1 (OrderedDict) — ghost list for evicted T1 entries (metadata only)
B2 (OrderedDict) — ghost list for evicted T2 entries
target_t1_size (float) — adaptive partition boundary
```

**Adaptation rules:**

- `touch()`: T1 block accessed again → promote to T2
- Ghost hit on B1 → "evicting recents too aggressively" → increase `target_t1_size`
- Ghost hit on B2 → "evicting frequents too aggressively" → decrease `target_t1_size`
- Eviction: if `len(T1) >= target_t1_size`, evict from T1 (→ B1); else evict from T2 (→ B2)
- Ghost lists bounded to `cache_capacity` entries each

For LLM workloads: some prompts share common prefixes (recency), others reuse the same system prompts (frequency). ARC adapts to whichever pattern dominates.

### CPUBackend

Simple free-list allocator for CPU block IDs:

- Sequential allocation: blocks get IDs 0, 1, 2...
- Freed blocks go to `allocated_blocks_free_list` for reuse
- `get_num_free_blocks() = free_list.len + (total - high_water_mark)`

`BlockStatus` is a `ctypes.Structure` (not a Python dataclass) — presumably for compact memory usage since there can be millions of blocks.

### Block Size Factor

The offloaded block size can differ from the GPU block size. E.g., GPU `block_size=16`, offloaded `block_size=64` → `block_size_factor=4`. Each CPU block maps to 4 GPU blocks. This reduces bookkeeping overhead and improves transfer efficiency (fewer, larger transfers).

The scheduler converts between the two granularities via `_get_block_hashes()`, which steps through the request's block hashes at `block_size_factor` stride.

### OffloadingConnectorScheduler

Bridges the OffloadingManager to the KV connector interface:

**`get_num_new_matched_tokens(request, num_computed_tokens)`:**
1. Compute block hashes at offloaded block size granularity
2. `manager.touch(all_hashes)` — keep them fresh
3. `manager.lookup(hashes from computed onward)` — count consecutive CPU hits
4. If `< 1 offloaded block` of hits, return 0 (not worth loading)
5. If hits overlap with blocks already being loaded (`_blocks_being_loaded`), return `None` (retry later to avoid redundant loads)
6. Return `(num_hit_tokens, True)` — True means async loading

**`update_state_after_alloc(request, blocks, num_external_tokens)`:**
1. `manager.prepare_load(hit_block_hashes)` — pin blocks, get `CPULoadStoreSpec`
2. Build `TransferSpec = (CPULoadStoreSpec, GPULoadStoreSpec)`
3. Queue in `_reqs_to_load`

**`_get_reqs_to_store(scheduler_output)`:**

For every active request (new + cached), checks if new full blocks exist since last store:
1. `manager.prepare_store(new_block_hashes)` — allocate CPU space, evict if needed
2. Filter: skip blocks already stored (dedup)
3. Build `TransferSpec = (GPULoadStoreSpec, CPULoadStoreSpec)`

**`request_finished(request, block_ids)`:**

Returns `(True, None)` if async stores are still in flight for this request — tells the scheduler to delay freeing GPU blocks until the store completes.

## Worker Side

### OffloadingWorker

A dispatcher that routes `TransferSpec` jobs to the right `OffloadingHandler` based on `(src_medium, dst_medium)`:

- `("GPU", "CPU")` → gpu_to_cpu_handler
- `("CPU", "GPU")` → cpu_to_gpu_handler

### SingleDirectionOffloadingHandler

Where the actual GPU↔CPU copies happen. Each direction has its own instance.

**CUDA stream management:**

- Each transfer gets its **own CUDA stream** (pooled for reuse)
- Transfers execute **in submission order** — each stream `wait_event`s the previous transfer's end event
- For GPU→CPU: also `wait_stream(current_stream)` to wait for model computation
- CUDA events with timing enabled track start/end for bandwidth metrics

**Transfer mechanics:**

1. `expand_block_ids()` — converts offloaded block IDs to kernel-level block IDs (applying `block_size_factor`)
2. Build `src_to_dst` mapping tensor (numpy → torch)
3. Call `ops.swap_blocks(src_tensor, dst_tensor, block_size_bytes, src_to_dst)` — custom CUDA kernel that copies blocks between tensors

**Completion polling:**

`get_finished()` checks the front of a deque: `end_event.query()` is a non-blocking check. Returns `TransferResult` with timing and size for metrics.

### CpuGpuOffloadingHandlers

Allocates **CPU-side KV tensors** and creates two `SingleDirectionOffloadingHandler` instances. Key complexity: probing the attention backend to match tensor layouts:

1. Calls `attn_backend.get_kv_cache_shape(...)` with test values to discover the dimension layout
2. Detects whether the backend uses `(2, num_blocks, ...)` (split K/V) or `(num_blocks, ...)` format
3. Detects cross-layer tensors (extra `num_layers` dimension)
4. Calls `attn_backend.get_kv_cache_stride_order(...)` to handle non-standard dimension orderings
5. Finds the kernel block size by locating the block_size dimension in the test shape
6. Allocates CPU tensors matching the GPU layout, using **pinned memory** for faster PCIe transfers

## Cross-Layer Uniform KV Cache Layout

The `OffloadingConnector` sets `prefer_cross_layer_blocks = True`, triggering a special memory layout:

```
Normal:       layer_0: [num_blocks, block_size, num_heads, head_dim]
              layer_1: [num_blocks, block_size, num_heads, head_dim]

Cross-layer:  all: [num_blocks, num_layers, block_size, num_heads, head_dim]
              (dimension order determined by attention backend's stride order)
```

The benefit: offloading block N copies **all layers at once** in one contiguous transfer, instead of N separate small copies. `use_uniform_kv_cache()` in the model runner mixin checks three conditions:

1. Single KV cache group (all layers same attention type)
2. Connector returns `prefer_cross_layer_blocks = True`
3. Attention backend supports a `num_layers` stride dimension

NIXL connector also optionally uses this layout (for FLASH_ATTN and FLASHINFER backends).

## End-to-End Flow

### Storing (GPU → CPU)

```
1. Scheduler: _get_reqs_to_store()
   - For each request, check if new full blocks since last store
   - manager.prepare_store(block_hashes) → allocate CPU blocks, evict LRU/ARC if needed
   - Build TransferSpec = (GPULoadStoreSpec, CPULoadStoreSpec)

2. Worker: prepare_store_kv() — DEFERS submission to next step
   (avoids competing with token sampling for PCIe bandwidth)

3. Next step: start_kv_transfers() — submits deferred stores
   Handler kicks off GPU→CPU copy on dedicated CUDA stream

4. Worker: get_finished() polls CUDA events → "finished_sending"

5. Scheduler: manager.complete_store(block_hashes) → blocks become ready (ref_cnt: -1 → 0)
```

**Deferred stores**: GPU→CPU copies are intentionally delayed one step. At the end of a step, the GPU is busy with token sampling and output transfer over PCIe. By deferring to the start of the next step, the store runs during scheduler planning time, overlapping PCIe transfers with CPU work.

### Loading (CPU → GPU)

```
1. Scheduler: get_num_new_matched_tokens()
   - manager.touch(all_block_hashes) — keep fresh
   - manager.lookup(from computed onward) — count consecutive CPU hits
   - Return (num_hit_tokens, True=async)

2. Scheduler: update_state_after_alloc()
   - manager.prepare_load(block_hashes) — pin blocks (ref_cnt++)
   - Request enters WAITING_FOR_REMOTE_KVS

3. Worker: start_kv_transfers() → submit CPU→GPU load
   Request is NOT in the forward batch — GPU stays busy with other work

4. Worker: get_finished() → "finished_recving"

5. Scheduler: manager.complete_load(block_hashes) → unpin (ref_cnt--)
   Request moves to schedulable queue
```

### Dedup with GPU Prefix Cache

When both GPU prefix caching and CPU offloading are enabled, the same block could exist in both. The scheduler tracks `_blocks_being_loaded` — if a new request's hit blocks overlap with an in-flight CPU→GPU load, it returns `None` (retry later). After the load completes, the GPU prefix cache has the blocks, and subsequent requests hit Tier 1 directly.

## Configuration

```bash
vllm serve $MODEL \
  --kv-transfer-config '{
    "kv_connector": "OffloadingConnector",
    "kv_connector_extra_config": {
      "cpu_bytes_to_use": "100e9",
      "block_size": 64,
      "eviction_policy": "arc"
    }
  }' \
  --enable-prefix-caching
```

| Config key | Default | Description |
|-----------|---------|-------------|
| `cpu_bytes_to_use` | (required) | CPU memory budget in bytes |
| `block_size` | GPU block_size | Offloaded block size in tokens (must be multiple of GPU block_size) |
| `eviction_policy` | `"lru"` | `"lru"` or `"arc"` |
| `spec_name` | `"CPUOffloadingSpec"` | Spec class name (for custom backends) |
| `spec_module_path` | None | Python module path for custom spec class |

## Relationship to GPU Prefix Cache LRU

The GPU BlockPool (`FreeKVCacheBlockQueue`) and the OffloadingManager both use LRU-style eviction, but they are **not duplicates** — they manage different memory pools at different tiers:

```
Request arrives, needs KV for prefix
  │
  ▼
Tier 1: GPU prefix cache (BlockPool LRU)       ← fast, limited (GPU memory)
  miss? ──▶ Tier 2: CPU offload (OffloadingManager LRU/ARC)  ← slower, larger (CPU memory)
              miss? ──▶ Recompute from scratch                ← slowest
```

Block lifecycle across the two tiers:

1. **Computed on GPU** → lives in GPU BlockPool
2. **Offloaded to CPU** (proactively, every step for full blocks) → exists in both GPU and CPU
3. **Evicted from GPU** (BlockPool LRU pops it when GPU needs space) → still survives in CPU
4. **Later request hits CPU** → loaded back to GPU
5. **Eventually evicted from CPU** (OffloadingManager LRU/ARC) → gone, must recompute

The two caches operate independently — GPU LRU decides what stays in GPU, CPU LRU/ARC decides what stays in CPU. The only coordination is the `touch()` call: when the scheduler checks a request's prefix, it calls `manager.touch(block_hashes)` for ALL the request's blocks, including ones that hit GPU. This keeps CPU blocks fresh even when they're being served from GPU Tier 1, preventing unnecessary CPU eviction.

## Design Observations

1. **Pluggable at every layer**: Backend (CPU/NVMe?), Manager (LRU/ARC), Spec (via factory registry), Handler (via type dispatch). Adding NVMe offloading would require a new Backend + Handler without touching the manager or connector.

2. **No per-layer hooks**: Unlike P/D connectors which hook into each attention layer's forward pass (`save_kv_layer` / `wait_for_layer_load`), the offloading connector's per-layer methods are **no-ops**. All transfers happen at whole-block granularity via the cross-layer layout.

3. **Deferred stores avoid PCIe contention**: The one-step deferral means a block isn't in CPU until at least 2 steps after it becomes full — acceptable since blocks that just became full are unlikely to be evicted from GPU immediately anyway.

4. **Not a general abstraction**: The OffloadingManager is internal to the OffloadingConnector. LMCache, NIXL, etc. each have their own storage management. If you're using LMCache for CPU offloading, none of this code is involved.
