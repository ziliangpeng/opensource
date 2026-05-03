# vLLM KV Cache Manager Deep Dive (`v0.16.0`)

This note covers the KV cache manager: how it allocates and frees blocks, how prefix caching works, and how different attention types each handle their own caching strategy.

## 1) Component hierarchy

```
KVCacheManager                          ← scheduler-facing API
  └── KVCacheCoordinator                ← orchestrates multiple attention types
        ├── BlockPool                   ← physical block allocation + prefix cache lookup
        │     ├── KVCacheBlock[]        ← block metadata array (ref count, hash, linked-list pointers)
        │     ├── FreeKVCacheBlockQueue ← doubly-linked list for LRU eviction
        │     └── BlockHashToBlockMap   ← hash table for prefix cache
        └── SingleTypeKVCacheManager[]  ← per-attention-type logic
              ├── FullAttentionManager
              ├── SlidingWindowManager
              ├── ChunkedLocalAttentionManager
              ├── MambaManager
              ├── CrossAttentionManager
              └── SinkFullAttentionManager
```

Key design: there is **one shared BlockPool** across all attention types, but each attention type has its own `SingleTypeKVCacheManager` that knows how to scan for cache hits and which blocks to skip.

Key source files:

- `vllm/v1/core/kv_cache_manager.py` — top-level API
- `vllm/v1/core/kv_cache_coordinator.py` — multi-group orchestration
- `vllm/v1/core/block_pool.py` — physical block management
- `vllm/v1/core/single_type_kv_cache_manager.py` — per-attention-type managers
- `vllm/v1/core/kv_cache_utils.py` — data structures, hashing, config generation
- `vllm/v1/kv_cache_interface.py` — KVCacheSpec definitions and KVCacheConfig

## 2) The block: core unit

Each `KVCacheBlock` (`kv_cache_utils.py`) is a lightweight metadata object:

```python
@dataclass
class KVCacheBlock:
    block_id: int              # physical index in GPU memory (0..N-1)
    ref_cnt: int = 0           # how many requests use this block
    _block_hash: bytes | None  # set when block is full & content-addressed
    prev_free_block: KVCacheBlock | None  # doubly-linked free list pointers
    next_free_block: KVCacheBlock | None
    is_null: bool = False      # True only for the null_block sentinel
```

A block represents `block_size` tokens' worth of KV cache (typically 16 tokens). The actual GPU memory is a contiguous tensor; `block_id` is an index into it.

**The null block** (`block_id=0`) is a special sentinel used as a placeholder in sliding window and Mamba managers where early positions fall outside the attention window.

## 3) The free queue: LRU eviction

`FreeKVCacheBlockQueue` (`kv_cache_utils.py`) is a doubly-linked list of `KVCacheBlock` objects with sentinel head/tail nodes. All operations are O(1):

- **Front = LRU (evict first)**. When the system needs a new block, it pops from the front.
- **Back = MRU (evict last)**. When a request finishes, its freed blocks go to the back.
- **O(1) remove of any node** via the doubly-linked pointers.

The eviction order within a freed request is subtle: blocks are freed in **reverse** order (tail blocks first), so blocks covering the most recently generated tokens enter the free queue closer to the back. This means blocks at the beginning of a prompt survive longer in the cache — good for prefix caching since prompts often share prefixes.

At initialization, all blocks (0..N-1) are in the free queue. Block 0 is immediately popped and designated as the null block.

## 4) Prefix caching: the block hash chain

vLLM uses a **Merkle chain** hash scheme for prefix caching.

### How hashes are computed

Each block's hash depends on the previous block's hash plus its own token IDs (`kv_cache_utils.py:hash_block_tokens`):

```
block_0_hash = hash(NONE_HASH, tokens[0:16], extra_keys)
block_1_hash = hash(block_0_hash, tokens[16:32], extra_keys)
block_2_hash = hash(block_1_hash, tokens[32:48], extra_keys)
...
```

`NONE_HASH` is a random 32-byte seed initialized once at process startup.

The chain property means: if two requests share the first 32 tokens, their `block_0_hash` and `block_1_hash` will be identical. But if they diverge at token 33, `block_2_hash` will differ even if the next 16 tokens happen to be the same — because the hash encodes the **full prefix**, not just the block's local contents.

**Extra keys** include: multimodal input hashes and positions, LoRA adapter names, cache salt, and prompt embeddings. This prevents cross-contamination between different LoRA adapters or multimodal contexts.

Block hashing happens in the **input thread** during request preprocessing (`get_request_block_hasher` returns a closure used by the input thread), not in the scheduler's busy loop.

### The lookup table

`BlockHashToBlockMap` (`block_pool.py`) maps `BlockHashWithGroupId → KVCacheBlock`. It uses a union type to avoid dict allocation for the common case:

- Single block per hash → stored directly as `KVCacheBlock`
- Multiple blocks per hash (duplicates from non-deduplicating allocation) → stored as `dict[block_id → KVCacheBlock]`

`BlockHashWithGroupId` is the block hash bytes with a 4-byte group ID suffix appended. This distinguishes the same logical prefix in different KV cache groups (e.g., full attention group 0 vs sliding window group 1).

### Lazy eviction

When a request finishes, its blocks are freed (ref_cnt decremented) but **retain their hash in the lookup table**. They enter the free queue and remain findable for prefix cache hits. A block is only evicted from the hash table when it's later popped from the free queue for reuse (`_maybe_evict_cached_block` in `block_pool.py`). This maximizes cache hit rates.

## 5) Cache hit scanning: different strategies per attention type

Different attention types need different scanning strategies in `find_longest_cache_hit`:

### FullAttentionManager — left-to-right

```
block_hashes: [h0, h1, h2, h3, h4, h5]
               ✓   ✓   ✓   ✗
                        ↑ stop at first miss
Result: [b0, b1, b2]  →  48 tokens cached
```

Full attention is prefix-based: the chain hash guarantees that if block 2 is in cache, blocks 0 and 1 must also be. So scanning left-to-right and stopping at the first miss is correct.

With EAGLE speculative decoding enabled, the last matched block is dropped to force recomputation (needed for the draft head's hidden states).

The result is also aligned to `alignment_tokens` by popping trailing blocks.

### SlidingWindowManager — right-to-left

```
block_hashes: [h0, h1, h2, h3, h4, h5]  (window covers last 3 blocks)
                             ✓   ✓   ✓
                             ↑ scan from right
Result: [null, null, null, b3, b4, b5]
```

Sliding window only needs the most recent tokens within the window. The scan starts from the right and stops after finding `sliding_window_contiguous_blocks` consecutive hits (= `ceil((window-1) / block_size)`). Positions outside the window are filled with `null_block`.

### MambaManager — right-to-left, first hit only

```
block_hashes: [h0, h1, h2, h3, h4, h5]
                                      ✓
                                      ↑ return on first hit
Result: [null, null, null, null, null, b5]
```

Mamba (SSM) only needs the state from the last block. Everything else is null-padded. The scan returns immediately on the first hit.

### ChunkedLocalAttentionManager (LLaMA 4)

Computes the chunk boundary: `local_start = (max_length // chunk_size) * chunk_size`. Everything before `local_start` is null-padded. From `local_start` onward, does a left-to-right scan like full attention.

### CrossAttentionManager

Does not participate in prefix caching at all. `find_longest_cache_hit` raises `NotImplementedError`. Cross-attention blocks (for encoder-decoder models) are allocated through a separate `num_encoder_tokens` parameter.

### SinkFullAttentionManager

Extends FullAttentionManager. In `__init__`, pre-allocates `sink_len // block_size` blocks permanently from the free queue — these are reserved as attention sink blocks for streaming LLM inference.

## 6) The coordinator: orchestrating multiple attention types

The coordinator sits between KVCacheManager and the per-type managers. Three variants:

### KVCacheCoordinatorNoPrefixCache

For `enable_caching=False`. `find_longest_cache_hit` always returns empty blocks and 0.

### UnitaryKVCacheCoordinator

The common case: a single KV cache group (standard GPT, LLaMA, DeepSeek). Delegates directly to one manager. Supports context parallelism (DCP/PCP), which multiplies the effective `block_size` — each physical block covers more tokens since each rank holds only a portion of the KV heads.

### HybridKVCacheCoordinator

For models with multiple attention types (e.g., LLaMA 4 has full attention + chunked local attention layers). This is the complex case.

**The problem**: full attention might find 100 blocks of cache hits, but the other attention type only finds 60 blocks. The cache hit must be the **minimum** across all groups, and it must be aligned to `lcm(all block sizes)`.

**Fixed-point iteration algorithm**:

```
1. Start with max possible hit length
2. For each attention type group:
   a. Find cache hits up to current hit length
   b. If this group finds fewer hits, reduce hit length
3. If hit length changed, go back to step 2
4. Converge when no group reduces the length
```

For the common "simple hybrid" case (1 full attention group + 1 other), this converges in one iteration. Full attention groups cache their results between iterations (downward-closed property: if N blocks hit, M < N blocks also hit).

**Constraints**: does not support context parallelism (DCP/PCP) or KV cache events (these assume a single group).

**Coordinator selection** (`get_kv_cache_coordinator`):

```python
if not enable_caching:        → KVCacheCoordinatorNoPrefixCache
elif len(groups) == 1:        → UnitaryKVCacheCoordinator
else:                         → HybridKVCacheCoordinator
```

## 7) The BlockPool: physical block management

`BlockPool` (`block_pool.py`) owns the physical blocks and the prefix cache lookup table.

### Core operations

**`get_cached_block(block_hash, kv_cache_group_ids)`**: looks up the hash in the lookup table for ALL group IDs simultaneously. Returns a list of blocks (one per group) or `None` if any group misses. This ensures a cache hit is valid across all groups.

**`get_new_blocks(num_blocks)`**: pops `num_blocks` from the LRU end of the free queue. For each block, calls `_maybe_evict_cached_block` (removes its hash from the lookup table if present). Sets ref_cnt to 1.

**`touch(blocks)`**: for cache-hit blocks that are in the free queue (ref_cnt == 0), removes them from the free queue and increments ref_cnt. This "claims" them so they aren't evicted.

**`free_blocks(ordered_blocks)`**: decrements ref_cnt for each block. If ref_cnt reaches 0 (and not null), appends to the free queue tail. Called with blocks in reverse order for optimal LRU behavior.

**`cache_full_blocks(request, blocks, num_cached_blocks, num_full_blocks, ...)`**: for each newly full block (from `num_cached_blocks` to `num_full_blocks`), computes its `BlockHashWithGroupId`, sets `block.block_hash`, and inserts into the lookup table. Also emits `BlockStored` KV events if events are enabled.

**`reset_prefix_cache()`**: only works when all blocks are free (just the null block is "used"). Replaces the lookup table with a new empty one and clears all block hashes.

**`evict_blocks(block_ids)`**: externally driven eviction (e.g., from KV connector when remote blocks are invalidated). Only removes from the hash table; does not return blocks to the free queue.

## 8) KVCacheManager: the scheduler-facing API

`KVCacheManager` (`kv_cache_manager.py`) is the interface the scheduler uses. It wraps the coordinator and converts results to `KVCacheBlocks`.

### `KVCacheBlocks` (dataclass)

```python
@dataclass
class KVCacheBlocks:
    blocks: tuple[Sequence[KVCacheBlock], ...]  # one sequence per kv_cache_group
```

- `get_block_ids()` → `tuple[list[int], ...]` — physical block IDs for the model runner
- `get_unhashed_block_ids()` → blocks without a hash (used for P/D transfer)
- `__add__` — concatenates two KVCacheBlocks group-wise

### `get_computed_blocks(request)` → `(KVCacheBlocks, num_computed_tokens)`

Called for waiting requests being scheduled for the first time:

1. Returns empty if prefix caching is disabled or `request.skip_reading_prefix_cache` is set.
2. Sets `max_cache_hit_length = request.num_tokens - 1` (must leave at least 1 token to compute for logits).
3. Delegates to `coordinator.find_longest_cache_hit(request.block_hashes, max_cache_hit_length)`.
4. Records prefix cache stats (queries/hits) if logging is enabled.

### `allocate_slots(request, num_new_tokens, ...)` → `KVCacheBlocks | None`

The core allocation method, called by the scheduler for every scheduled request:

1. Compute token counts: `num_local_computed_tokens`, `total_computed_tokens`, `num_tokens_need_slot`.
2. Call `coordinator.remove_skipped_blocks()` — frees blocks outside the attention window (sliding window / Mamba).
3. Call `coordinator.get_num_blocks_to_allocate()` — check if enough free blocks exist.
4. **If not enough → return `None`**. The scheduler will preempt another request.
5. Call `coordinator.allocate_new_computed_blocks()` — claim prefix-cached blocks (`touch`: remove from free queue, increment ref_cnt).
6. Call `coordinator.allocate_new_blocks()` — get fresh blocks from the free queue.
7. Call `coordinator.cache_blocks()` — register newly full blocks in the hash table (unless `delay_cache_blocks=True` for P/D transfer).
8. Return `KVCacheBlocks` wrapping the new blocks.

### `free(request)`

Called when a request finishes. Delegates to `coordinator.free(request.request_id)`, which calls each manager's `free`, which frees blocks in reverse order (tail first → back of free queue → survives longer for prefix cache).

### `cache_blocks(request, num_computed_tokens)`

Called explicitly after P/D KV transfer completes (when `delay_cache_blocks` was True during allocation). Registers blocks in the prefix cache hash table.

## 9) Memory reclamation for bounded attention

For sliding window and Mamba models, memory would grow with context length unless blocks are actively freed. This happens in `remove_skipped_blocks()` (`single_type_kv_cache_manager.py`), called at the start of each `allocate_slots`.

**SlidingWindowManager**: `num_skipped_tokens = max(0, num_computed - sliding_window + 1)`
- E.g., window=4096, computed=8192 → skip first 4097 tokens → free those blocks.

**MambaManager**: `num_skipped_tokens = num_computed - 1`
- Only needs the last state. Almost everything is freed.

**ChunkedLocalAttentionManager**: `num_skipped = (computed // chunk_size) * chunk_size`
- Entire previous chunks are freed.

**FullAttentionManager**: `num_skipped = 0` — never skips.

The freed positions are replaced with `null_block` in the request's block list, keeping the block table append-only (important for GPU-side indexing).

## 10) KV cache groups and specs

`KVCacheConfig` (`kv_cache_interface.py`) describes the shape of the KV cache:

```python
@dataclass(frozen=True)
class KVCacheConfig:
    num_blocks: int                           # total physical blocks
    kv_cache_tensors: list[KVCacheTensor]     # GPU tensor layout
    kv_cache_groups: list[KVCacheGroupSpec]    # one per attention type
```

Each `KVCacheGroupSpec` holds a `KVCacheSpec` subclass:

| Spec class | Attention type | Memory bound | Notes |
|---|---|---|---|
| `FullAttentionSpec` | Standard transformer | Full context length | Also used for MLA |
| `SlidingWindowSpec` | Sliding window | Window size + batch | |
| `ChunkedLocalAttentionSpec` | LLaMA 4 local | Chunk size + batch | |
| `MambaSpec` | SSM state | 1 state per request | Not token KV |
| `CrossAttentionSpec` | Encoder-decoder | Max encoder length | Whisper, T5 |
| `SinkFullAttentionSpec` | Streaming LLM | Full + sink blocks | |

Group construction logic (`kv_cache_utils.py:get_kv_cache_groups`):

1. All same spec → one group (most models).
2. Same attention type but different hidden sizes → `UniformTypeKVCacheSpecs` wrapper.
3. Multiple attention types → multiple groups, with page sizes unified via `unify_kv_cache_spec_page_size` (scales up block_size of smaller-page specs until all groups have the same `page_size_bytes`).

When hybrid memory allocator is disabled, `SlidingWindowSpec` and `ChunkedLocalAttentionSpec` are converted to `FullAttentionSpec` (`unify_hybrid_kv_cache_specs`), giving up the memory savings but simplifying the system.

## 11) Encoder cache manager

`EncoderCacheManager` (`encoder_cache_manager.py`) is a **separate subsystem** for multimodal encoder outputs (image embeddings for LLaVA, etc.). It is not part of the KV block system.

### Design

- **Reference counting by `mm_hash`**: multiple requests sharing the same image share the same encoder output.
- **LRU eviction**: when no requests reference an encoder output, it moves to a `freeable` ordered dict. When the cache is full and a new allocation is needed, the oldest unreferenced entries are evicted.
- **Two-phase lifecycle**: allocated when the request is scheduled → freed when the encoder output's placeholder tokens have all been consumed by the decoder (`_free_encoder_inputs` in the scheduler).

### Key operations

- `check_and_update_cache(request, input_id)` → True if the encoder output is already cached. If it was unreferenced, reclaims it.
- `can_allocate(request, input_id, budget, ...)` → checks compute budget and cache space, evicts if needed.
- `allocate(request, input_id)` → decrements free slots, adds request to reference set.
- `free_encoder_input(request, input_id)` → removes request from reference set; if no more references, moves to `freeable`.
- `get_freed_mm_hashes()` → returns hashes of evicted entries (for workers to free GPU memory).

### EncoderDecoderCacheManager

Simplified variant for encoder-decoder models (Whisper). Does not cache across requests — each request's encoder output is allocated fresh and freed after one step. Uses a delayed-free scheme: outputs are freed one step after they're used, ensuring the model has finished with them.

## 12) Integration points

### P/D disaggregation

- `allocate_slots` accepts `delay_cache_blocks=True` to allocate blocks without registering them in the prefix cache.
- After KV transfer completes, `cache_blocks` is called to register the blocks.
- `get_unhashed_block_ids()` on `KVCacheBlocks` returns the blocks that need remote KV data.
- `num_external_computed_tokens` tracks tokens whose KV was computed on the prefill node.

### EAGLE speculative decoding

- `num_lookahead_tokens` reserves extra block space for draft tokens.
- Draft tokens are NOT cached — `cache_blocks` is capped at `request.num_tokens` which excludes draft tokens.
- `find_longest_cache_hit` with EAGLE drops the last matched block to force recomputation for the draft head.

### Context parallelism (DCP/PCP)

- `dcp_world_size * pcp_world_size` multiplies the effective `block_size` — each physical block covers more tokens since each rank only holds a portion.
- Only supported by `UnitaryKVCacheCoordinator` (single KV cache group).

### Metrics collection

`KVCacheMetricsCollector` (`kv_cache_metrics.py`) samples a subset of blocks (default 1%) and tracks:

- Block lifetime (birth to eviction)
- Idle time (last access to eviction)
- Reuse gaps (between consecutive accesses)

Hooked into `BlockPool.get_new_blocks` (allocation), `BlockPool.touch` (cache hit), and `BlockPool._maybe_evict_cached_block` (eviction). Events are drained by the scheduler's `make_stats()`.
