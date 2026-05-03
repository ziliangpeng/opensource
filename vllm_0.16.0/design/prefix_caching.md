# Prefix Caching

Based on `docs/design/prefix_caching.md` in the vLLM v0.16.0 source.

## Relationship to PagedAttention

PagedAttention and prefix caching are two separate layers built on the same block abstraction. It helps to keep their responsibilities distinct:

**PagedAttention** (see [[ai/llm/inference/frameworks/vllm_0.16.0/design/paged_attention]]) is about **allocating GPU memory**. The GPU holds a pool of fixed-size slots, each identified by an integer `block_id`. The attention kernel fetches KV data from `kv_cache[block_id, token_offset, head, dim]`. PagedAttention's job is to manage which slots are free, allocate slots to requests, and build the block table handed to the kernel. It knows nothing about what content is in each slot.

**Prefix caching** is about **reusing slots across requests**. It maintains a hash → slot_id mapping. When a new request arrives, its prefix blocks are hashed and looked up in this map. On a hit, the matched slot IDs are handed directly to the block table — skipping the forward pass for those tokens entirely. Prefix caching sits on top of PagedAttention's slot pool; it doesn't change how slots are allocated or how the kernel works.

The split in code: `KVCacheBlock`, `FreeKVCacheBlockQueue`, and `KVCacheBlocks` are the block management layer (PagedAttention's concern). `BlockHashToBlockMap`, `KVCacheCoordinator`, and `SingleTypeKVCacheManager` are the caching layer. `BlockPool` bundles both because eviction must atomically update the free list and remove the hash — see [[ai/llm/inference/frameworks/vllm_0.16.0/design/prefix_caching#Changing the Eviction Algorithm]].

## Core Idea

When multiple requests share the same prefix (system prompt, few-shot examples, shared document), the KV cache for those tokens gets recomputed from scratch without prefix caching. Prefix caching avoids this — it stores computed KV blocks and reuses them on cache hit. It doesn't change model outputs and is described as "almost a free lunch."

## Block Hashing

Each full block is identified by a hash computed from three components:

```
block_hash = hash(parent_block_hash, curr_block_token_ids, extra_keys)
```

- **Parent hash** — chains the entire prefix history; two blocks with identical tokens but different preceding contexts get different hashes
- **Token IDs** — the actual tokens in this block (reduces collision probability)
- **Extra keys** — LoRA adapter name, multimodal content hash, cache salt (for multi-tenant isolation)

**Only full blocks are cached** — a partially filled trailing block is not eligible.

### Hash Algorithm

As of v0.11, the default is `sha256` (collision-safe). Configurable via `--prefix-caching-hash-algo`:

| Algorithm | Serialization | Notes |
|-----------|--------------|-------|
| `sha256` (default) | pickle | Safe, but not reproducible across Python/vLLM versions |
| `sha256_cbor` | cbor2 | Reproducible across environments, cross-language compatible |
| `xxhash` | pickle | Faster, non-cryptographic — collision risk in multi-tenant use |
| `xxhash_cbor` | cbor2 | Fast + reproducible |

### Multimodal Prefix Caching

Images are replaced by placeholder tokens during tokenization. Since two requests with the same placeholders but different images must not share cache, the image's content hash is included as an `extra_key` in the block hash.

### Cache Salt (Security)

In multi-tenant environments, an attacker can probe cache hit latency to infer what other users have cached. Per-request `cache_salt` is injected into the first block's hash, so only requests with the same salt share cached blocks.

## Data Structures

### `BlockHash` and `BlockHashWithGroupId`
**File:** `vllm/v1/core/kv_cache_utils.py`

```python
# A block's content hash (bytes produced by sha256/xxhash)
BlockHash = NewType("BlockHash", bytes)

# BlockHash + KV cache group ID packed into bytes
BlockHashWithGroupId = NewType("BlockHashWithGroupId", bytes)

# For external use — bytes or int (backward compat with older int hashes)
ExternalBlockHash: TypeAlias = bytes | int
```

### `KVCacheBlock` and `FreeKVCacheBlockQueue`

These are PagedAttention-layer data structures — see [[ai/llm/inference/frameworks/vllm_0.16.0/design/paged_attention]] for full definitions. The prefix-caching-relevant fields on `KVCacheBlock` are:

- `_block_hash` — set once when the block is full; `None` means not cached
- `reset_hash()` — called on eviction to clear the hash so the slot can be reused
- `prev_free_block` / `next_free_block` — the linked list pointers inside `FreeKVCacheBlockQueue` that enable O(1) removal on cache hit

### `BlockHashToBlockMap`
**File:** `vllm/v1/core/block_pool.py`

The prefix cache lookup table mapping hashes to physical block slots.

```python
class BlockHashToBlockMap:
    _cache: dict[BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]]

    def get_one_block(self, key: BlockHashWithGroupId) -> KVCacheBlock | None:
        """Cache hit lookup."""

    def insert(self, key: BlockHashWithGroupId, block: KVCacheBlock) -> None:
        """Register a newly cached block."""

    def pop(self, key: BlockHashWithGroupId, block_id: int) -> KVCacheBlock | None:
        """Remove on eviction."""
```

### `BlockPool`
**File:** `vllm/v1/core/block_pool.py`

Owns the GPU block pool and all allocation/eviction logic. Created once at startup.

```python
class BlockPool:
    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        hash_block_size: int,
        enable_kv_cache_events: bool = False,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        self.blocks: list[KVCacheBlock]              # all blocks, indexed by block_id
        self.free_block_queue: FreeKVCacheBlockQueue # LRU free list
        self.cached_block_hash_to_block: BlockHashToBlockMap  # prefix cache
        self.null_block: KVCacheBlock                # singleton placeholder block

    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int]
    ) -> list[KVCacheBlock] | None:
        """Prefix cache hit lookup. Returns None on miss."""

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """Allocate from free list. Evicts LRU cached blocks if needed."""

    def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
        """Mark blocks as in-use — remove from free queue, increment ref_cnt."""

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        """Return blocks to free queue tail (in reverse order)."""

    def cache_full_blocks(self, request, blocks, ...) -> None:
        """Register newly filled blocks in prefix cache."""

    def evict_blocks(self, block_ids: set[int]) -> None:
        """Explicitly evict blocks from prefix cache."""

    def reset_prefix_cache(self) -> bool:
        """Clear all cached hashes (used for RLHF / benchmarking resets)."""
```

### `KVCacheBlocks`

Also a PagedAttention-layer structure — see [[ai/llm/inference/frameworks/vllm_0.16.0/design/paged_attention]]. It wraps the block list per KV cache group and exposes `get_block_ids()` to extract integer slot IDs for building the block table tensor.

### `KVCacheManager`
**File:** `vllm/v1/core/kv_cache_manager.py`

The main public API used by the scheduler. Delegates to `KVCacheCoordinator` and `BlockPool`.

```python
class KVCacheManager:
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        hash_block_size: int,
        enable_caching: bool = True,
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
        ...
    ):
        self.block_pool: BlockPool
        self.prefix_cache_stats: PrefixCacheStats | None

    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
        """
        Look up prefix cache for a new request.
        Returns: (cached_blocks, num_computed_tokens)
        Calls coordinator.find_longest_cache_hit() internally.
        """

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        ...
    ) -> KVCacheBlocks | None:
        """
        Allocate blocks for a request's new tokens.
        Returns None if insufficient blocks available.

        Block layout after allocation:
        | cached (prefix hit) | new_computed | new | lookahead |
        """

    def free(self, request: Request) -> None:
        """Free all blocks for a completed request."""

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """Register newly filled blocks into the prefix cache."""

    @property
    def usage(self) -> float:
        """KV cache utilization (0.0–1.0)."""
```

### `KVCacheCoordinator` (and variants)
**File:** `vllm/v1/core/kv_cache_coordinator.py`

Coordinates across multiple KV cache groups (needed for hybrid models with different attention types). Has three concrete implementations:

- `KVCacheCoordinatorNoPrefixCache` — no caching
- `UnitaryKVCacheCoordinator` — single KV cache group (most models)
- `HybridKVCacheCoordinator` — multiple groups with different attention types (e.g., full attention + sliding window in the same model)

```python
class KVCacheCoordinator(ABC):
    block_pool: BlockPool
    single_type_managers: tuple[SingleTypeKVCacheManager, ...]

    @abstractmethod
    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """Find the longest matching prefix in cache across all KV groups."""
```

### `SingleTypeKVCacheManager` (and variants)
**File:** `vllm/v1/core/single_type_kv_cache_manager.py`

Attention-type-specific management. The `find_longest_cache_hit()` algorithm differs per attention type — e.g., sliding window attention scans right-to-left because recent blocks matter more.

```python
class SingleTypeKVCacheManager(ABC):
    block_size: int
    block_pool: BlockPool
    req_to_blocks: defaultdict[str, list[KVCacheBlock]]  # request_id → block list
    num_cached_block: dict[str, int]                     # cached block count per request

    @classmethod
    @abstractmethod
    def find_longest_cache_hit(cls, block_hashes, ...) -> tuple[list[KVCacheBlock], ...]:
        """Customized per attention type."""
```

Concrete implementations:
- `FullAttentionManager` — standard left-to-right prefix scan
- `SlidingWindowManager` — right-to-left scan (recent blocks prioritized)
- `ChunkedLocalAttentionManager`
- `MambaManager` — linear recurrent attention
- `CrossAttentionManager` — encoder-decoder
- `SinkFullAttentionManager` — full attention with sink tokens

Factory:
```python
def get_manager_for_kv_cache_spec(kv_cache_spec: KVCacheSpec, **kwargs) -> SingleTypeKVCacheManager:
    """Returns the appropriate manager subclass for the given attention type."""
```

## Operations

### New Request: Block Allocation

1. Scheduler calls `kv_cache_manager.get_computed_blocks(request)`:
   - Hashes prefix blocks of the prompt
   - Looks up `BlockHashToBlockMap` — returns all blocks with matching hashes
   - Returns `(cached_blocks, num_computed_tokens)`

2. Scheduler calls `kv_cache_manager.allocate_slots(...)`:
   - **Touch** cache-hit blocks: increment `ref_cnt`, remove from free queue (prevents eviction while in use)
   - **Allocate new blocks**: `popleft()` from free queue head (LRU first). If the popped block was cached, it gets evicted (`reset_hash()`, removed from `BlockHashToBlockMap`)
   - Any newly filled block is immediately inserted into `BlockHashToBlockMap` — available for other requests in the same batch

### Cache Hit: `touch()`
**File:** `vllm/v1/core/block_pool.py`

When prefix cache blocks are reused by a new request, `touch()` removes them from the free queue (so they can't be evicted) and increments their reference count.

```python
def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
    for block in blocks:
        # ref_cnt=0 means the block is currently in the free list
        if block.ref_cnt == 0 and not block.is_null:
            self.free_block_queue.remove(block)  # O(1) via doubly linked list
        block.ref_cnt += 1
```

### Free: `free_blocks()`
**File:** `vllm/v1/core/block_pool.py`

When a request completes, blocks are returned to the free queue **in reverse order** (last block first → appended to tail → survives longest). The intuition: deeper prefix blocks are more likely to be shared with future requests.

```python
def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
    blocks_list = list(ordered_blocks)
    for block in blocks_list:
        block.ref_cnt -= 1
    self.free_block_queue.append_n(
        [block for block in blocks_list if block.ref_cnt == 0 and not block.is_null]
    )
```

Blocks only re-enter the free queue when `ref_cnt` reaches 0 — multiple requests can share the same cached block, and it stays pinned until all of them finish.

### Eviction: `get_new_blocks()` + `_maybe_evict_cached_block()`
**File:** `vllm/v1/core/block_pool.py`

Eviction happens implicitly during `get_new_blocks()` — no background eviction thread, no separate eviction pass. When the free queue head is a cached block, popping it triggers eviction.

```python
def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
    blocks = self.free_block_queue.popleft_n(num_blocks)
    for block in blocks:
        if self.enable_caching:
            self._maybe_evict_cached_block(block)
        block.ref_cnt += 1
    return blocks
```

`_maybe_evict_cached_block()` does the actual cleanup — removes from `BlockHashToBlockMap` and clears the block's hash so it can be reused:

```python
def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
    block_hash = block.block_hash
    if block_hash is None:
        return False  # not a cached block, nothing to do

    if self.cached_block_hash_to_block.pop(block_hash, block.block_id) is None:
        return False  # wasn't in the hash table

    block.reset_hash()  # clears _block_hash → None, block is now blank/reusable

    if self.enable_kv_cache_events:
        self.kv_event_queue.append(BlockRemoved(...))
    return True
```

### Full Lifecycle

```
Cache hit:    touch()       → remove from free queue, ref_cnt++
Request done: free_blocks() → ref_cnt--, if 0: append to free queue tail (MRU)
Need memory:  get_new_blocks() → popleft_n() from head (LRU) → _maybe_evict_cached_block()
```

The free queue *is* the eviction queue. Cached blocks with no active users sit in the free list at MRU positions. As memory pressure grows, `popleft_n()` naturally reaches them from the LRU end.

## Changing the Eviction Algorithm

The eviction policy is mostly encapsulated in `FreeKVCacheBlockQueue`. `BlockPool` only interacts with it through five methods:

```
popleft_n()  — pick eviction victims
remove()     — pin a block on cache hit (remove from eviction candidates)
append()     — return one block to free pool
append_n()   — return many blocks (request completion)
```

Replacing `FreeKVCacheBlockQueue` with a different class that implements the same interface (e.g., a min-heap for LFU) would leave `BlockPool` untouched. The swap point is `BlockPool.__init__` where the queue is instantiated.

### Complications

**1. Linked list pointers live inside `KVCacheBlock`**

`prev_free_block` and `next_free_block` are fields on the block dataclass itself — this is an optimization enabling O(1) `remove()` without a separate node object. A heap-based policy couldn't reuse these fields.

**2. The "deeper = more valuable" heuristic is in the calling convention, not the queue**

In `free_blocks()`, the caller (KVCacheManager) passes blocks in reverse order so deeper prefix blocks land at the tail (MRU). This policy is not inside `FreeKVCacheBlockQueue` — it's in the calling convention. A scoring-based eviction policy would need to reconsider this.

**3. No per-block metadata for richer policies**

`KVCacheBlock` has no frequency counter (needed for LFU), no priority field (needed for "never evict system prompt blocks"), and no cost annotation. Any richer policy would require adding fields to `KVCacheBlock`.

### Difficulty Summary

| Policy change | Effort |
|---|---|
| LRU variant (CLOCK, SLRU) | Moderate — replace queue class, keep interface |
| LFU | Larger — add frequency counter to `KVCacheBlock`, replace queue |
| Priority-aware (e.g., pin system prompt blocks) | Larger — add priority field to `KVCacheBlock`, change queue and caller convention |

There has been community interest in priority-based eviction (e.g., never evict blocks shared across thousands of requests for a common system prompt), but it hasn't landed in vLLM yet.

## Duplicate Blocks (V1 Limitation)

In V1, the block table is **append-only** — once a block ID is assigned to a position, it can't be swapped. In V0, when a newly computed block was identical to an existing cached block, vLLM would free the duplicate and reuse the cached one. V1 cannot do this, so duplicate blocks for the same hash can temporarily coexist. They're cleaned up when the request completes.

## Hash Computation
**File:** `vllm/v1/core/kv_cache_utils.py`

```python
def hash_block_tokens(
    hash_function: Callable[[Any], bytes],
    parent_block_hash: BlockHash | None,
    curr_block_token_ids: Sequence[int],
    extra_keys: tuple[Any, ...] | None = None,
) -> BlockHash:
    """
    Computes block hash. Components:
    - parent_block_hash: chained prefix hash (None for first block, uses NONE_HASH seed)
    - curr_block_token_ids: tuple of token IDs in this block
    - extra_keys: LoRA name, image hash, cache_salt, etc.
    """
```

For hybrid models where different KV cache groups have different block sizes, `BlockHashListWithBlockSize` lazily converts hashes between block sizes by concatenating consecutive hashes.
