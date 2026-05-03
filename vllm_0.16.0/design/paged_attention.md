# PagedAttention

Based on `docs/design/paged_attention.md` and the original vLLM paper (SOSP 2023). Note that the vLLM doc itself is marked as historical — the original custom kernel no longer runs in modern vLLM. This doc covers both the original design and the evolution to today.

## The Core Idea

Before vLLM, LLM serving systems pre-allocated contiguous memory for each sequence's KV cache based on the maximum possible sequence length. This caused 60–80% memory fragmentation and waste — a 2048-token max context reservation mostly sits empty for a 100-token generation. It also made sharing KV cache across requests (e.g., for common prefixes) impossible.

PagedAttention borrows the OS virtual memory paging concept: KV cache is split into fixed-size **blocks** (pages), each storing `BLOCK_SIZE` tokens for one head. A **block table** maps logical token positions to physical block indices. Sequences can now use non-contiguous physical memory, blocks can be shared across sequences, and unused slots within a block are the only waste (internal fragmentation limited to at most one block per sequence).

This is purely a memory management concept. The attention computation is a separate concern.

## The Original Kernel (Historical)

The 2023 paper introduced a custom CUDA kernel (`csrc/attention/attention_kernels.cu`) that could compute attention directly over the paged non-contiguous KV layout. Standard attention implementations at the time assumed contiguous memory, so a purpose-built kernel was necessary.

The kernel is a **decode-phase single-query attention kernel** — each forward pass generates one token, so `q` has shape `[num_seqs, num_heads, head_size]` (one query token per sequence). The KV cache has shapes:

```cpp
k_cache: [num_blocks, num_kv_heads, head_size/x, block_size, x]
v_cache: [num_blocks, num_kv_heads, head_size, block_size]
```

### Thread Hierarchy

The GPU thread hierarchy maps cleanly onto the attention computation:

| Unit | Size | Responsibility |
|------|------|---------------|
| Thread group | ~2 threads | One QK dot product (one query × one key token) |
| Warp | 32 threads | All key tokens in one KV block |
| Thread block | multiple warps | Full context for one (sequence, head) pair |
| Grid | `(num_heads, num_seqs, partitions)` | One thread block per cell |

### Computation Flow

1. Load Q token into **shared memory** (`q_vecs`) — accessed repeatedly across all K iterations, worth the cost
2. Iterate over K blocks — load key tokens into **register memory** (`k_vecs`) — each used once, no need for shared memory
3. Compute QK dot products with a cross-thread-group reduction (`Qk_dot::dot`)
4. **Two-pass softmax**:
   - Pass 1: store all QK scores as `logits[]` in shared memory, find `qk_max` via warp shuffle reductions
   - Pass 2: compute `exp(logits - qk_max)`, reduce for `exp_sum`, normalize
5. Load V blocks from HBM, compute weighted sum accumulation (`accs[]`) in registers
6. Reduce `accs` across warps, write output to global memory

### IO Limitation

The original kernel is **not IO-aware** in the FlashAttention sense. It makes two separate passes over K from HBM (once for QK, implicitly via the block iteration), and materializes all logits in shared memory before computing the V-weighted sum. FlashAttention's online softmax fuses QK + softmax + AV into a single tiled pass that reads each KV element from HBM exactly once. Early benchmarks showed the original paged kernel was ~2-3x slower than FlashAttention 2.

## The Evolution: Paging + IO-Aware Attention

The key insight that resolved this: **paging (memory management) and attention computation (kernel efficiency) are separable concerns**.

FlashAttention 2.5+ and FlashAttention 3, as well as FlashInfer, added native support for paged KV cache by accepting a `block_table` parameter directly:

```python
# FlashAttention API with paged KV cache
flash_attn_with_kvcache(
    q,
    k_cache,          # shape: (num_blocks, block_size, num_kv_heads, head_size)
    v_cache,
    block_table=...,  # int32 tensor: logical token → physical block mapping
    cache_seqlens=...,
)
```

FlashInfer provides dedicated wrappers built around paged KV cache from the ground up:
- `BatchDecodeWithPagedKVCacheWrapper` — for the decode phase
- `BatchPrefillWithPagedKVCacheWrapper` — for the prefill phase

These kernels use online softmax (running max + running sum) with tiling, touching each KV element from HBM only once — the same IO-optimal property as standard FlashAttention — while operating directly on the non-contiguous paged layout via block table lookup.

## How Modern vLLM Uses This

vLLM uses a **pluggable attention backend** architecture. The memory management layer (PagedAttention paging, block table maintenance, KV cache allocation, block swapping) is fully owned by vLLM's scheduler and cache engine regardless of which backend runs.

Before each forward pass, the CPU constructs an `AttentionMetadata` object that bundles the block table, sequence lengths, and other per-batch information. This gets passed down through the model runner to the attention backend. At forward pass time, vLLM passes the `attn_metadata.block_table` and paged `kv_cache` tensors directly into the chosen backend — no copying into contiguous buffers.

Backend selection (`--attention-backend` or auto):

| Hardware | Default backend | Notes |
|----------|----------------|-------|
| Hopper / Ada (SM 8.9–9.0) | FlashAttention | FA3 on Hopper |
| Ampere (SM 8.0–8.6) | FlashAttention | FA2 |
| Blackwell (SM 10.x) | FlashInfer | TRT-LLM backend |
| ROCm (AMD) | aiter / Triton | Completely separate stack — see below |

FlashInfer is NVIDIA-only. It is particularly optimized for high-concurrency decode scenarios (prefetch, cascading softmax for prefix sharing), making it the preferred backend when serving many simultaneous requests on NVIDIA hardware.

Source files for the modern integration:
- `vllm/v1/attention/backends/flash_attn.py`
- `vllm/v1/attention/backends/flashinfer.py`
- `vllm/v1/attention/backends/rocm_attn.py`, `rocm_aiter_fa.py` — AMD backends
- `csrc/attention/paged_attention_v1.cu`, `paged_attention_v2.cu` — original kernel, still accessible via `--attention-backend=XFORMERS` (legacy)

## Kernel Layer: vLLM's FlashAttention Fork and AMD's aiter

### NVIDIA: `vllm.vllm_flash_attn` (a maintained fork)

vLLM does not use the upstream `flash-attn` pip package. Instead it maintains its own fork at `https://github.com/vllm-project/flash-attention`, fetched at build time via CMake `FetchContent` and compiled into two bundled shared libraries:

- `_vllm_fa2_C.abi3.so` — FA2, used on Ampere / Ada
- `_vllm_fa3_C.abi3.so` — FA3, used on Hopper

These are imported as `vllm.vllm_flash_attn` and called by the FlashAttention backend.

**Why a fork and not upstream?** The fork contains both build changes and real kernel modifications:

Build / packaging (~60% of the delta):
- CMake integration — upstream FA uses pure setuptools, incompatible with vLLM's cmake build
- Package renamed `flash_attn` → `vllm_flash_attn` to avoid conflicts with user-installed upstream
- Compile-time kernel pruning via `-DFLASHATTENTION_DISABLE_DROPOUT`, `-DFLASHATTENTION_DISABLE_UNEVEN_K` etc., reducing the number of compiled kernel variants to shorten build time
- `vllm_flash_attn/` Python wrapper directory with vLLM-specific API surface

Kernel-level changes (~40% of the delta):
- **Attention sinks** — the most significant delta. vLLM modified the FA2 and FA3 CUDA forward pass kernels to support attention sinks (`s_aux` parameter), allowing certain sink tokens to always be visible in sliding window attention. Models like GPT-OSS-120B require this. Upstream Dao-AILab has not merged this yet (FA4 is starting to include it).
- **FP8 KV cache descale** — `q_descale`/`k_descale`/`v_descale` parameters to support fused FP8 quantized KV cache
- **Performance patches** — tile size adjustments, SM90 FP8 two-level accumulation for long-context precision, varlen sort/swizzle, `num_splits` support

The fork periodically merges upstream Dao-AILab changes (weekly syncs) but maintains its own divergent patches. There is an open issue (#31877, Jan 2025) discussing upstreaming changes back to reduce maintenance overhead. Long-term the fork may converge back to upstream as FA4 natively includes attention sinks.

### AMD/ROCm: `aiter` library — a completely separate ecosystem

AMD does not use vLLM's FlashAttention fork at all. The ROCm path is built on AMD's own `aiter` library (`vllm._aiter_ops`) and Triton, with 5+ distinct backends registered:

| Backend | Library | Notes |
|---|---|---|
| `RocmAttentionBackend` | Triton + PagedAttention | Standard attention |
| `AiterFlashAttentionBackend` | AMD aiter | High-performance FA equivalent |
| `AiterMLABackend` | AMD aiter | Multi-head Latent Attention (DeepSeek) |
| `AiterTritonMLABackend` | aiter + Triton | Hybrid MLA |
| `ROCMAiterMLASparseBackend` | AMD aiter | Sparse MLA |

`aiter` is AMD's equivalent of FlashInfer — a purpose-built high-performance kernel library for MI-series GPUs (HIP/ROCm). FlashInfer itself is NVIDIA-only.

### The original vLLM kernel (legacy)

`csrc/attention/paged_attention_v1.cu` and `paged_attention_v2.cu` are the 2023 paper's hand-written CUDA kernels. Still in the repo, still functional, but only used when explicitly selecting `--attention-backend=XFORMERS`. v2 adds multi-partition support for long sequences (splits context across partitions, merges partial softmax results via `merge_attn_states.cu`). Effectively the museum exhibit of vLLM history.

## Implementation Details: Hashing, KV Sizes, and CPU/GPU Split

### Block Hash

For prefix caching, each block is identified by a **64-bit integer hash** (Python's built-in `hash()`), not a string. The hash is computed as:

```python
hash(tuple(token_ids_in_block) + (parent_block_hash,))
```

The parent hash chaining is the key design: a block's hash encodes its entire prefix, not just its own 16 tokens. Two blocks with identical token content but different preceding contexts get different hashes. This is what makes prefix caching semantically correct — cache hits only occur when the full prefix matches.

### KV Size Per Token

KV cache size depends on the model's number of KV heads and head dimension. For Llama-3.1-8B (GQA, 8 KV heads, head dim 128, BF16):

- Per token: 8 heads × 128 dims × 2 bytes × 2 (K+V) = **4 KB**
- Per block (16 tokens): **64 KB**
- Per block across all 32 layers: **2 MB**

Models using MLA (e.g., DeepSeek V3) are dramatically smaller because KV is projected into a low-rank compressed latent vector before caching, reducing per-token KV size by ~10x.

### CPU/GPU Split

The "pointer" to a block is an **integer block index** (0 to num_gpu_blocks-1), not a raw GPU memory address. The kernel computes the actual address internally. This split of responsibilities keeps the bookkeeping clean:

**CPU owns:**
- `BlockPool` / `FreeKVCacheBlockQueue` — tracks which integer block IDs are free/in-use (free list)
- Hash → block_id dict (`BlockHashToBlockMap`) — Python dictionary for prefix cache lookup
- Block table construction (`KVCacheBlocks`) — builds the logical→physical mapping array before each forward pass

**GPU owns:**
- Pre-allocated KV cache tensor per layer: `[num_blocks, block_size, num_kv_heads, head_size]` — the memory pool
- Block table tensor — integer array copied from CPU each forward pass, shape `[num_seqs, max_blocks_per_seq]`

### Data Structures

#### `KVCacheBlock`
**File:** `vllm/v1/core/kv_cache_utils.py`

The core unit — one Python object per GPU slot, pre-allocated at startup (no runtime allocation). `block_id` is the immutable slot index used to index into the GPU tensor. `ref_cnt` tracks how many active requests are using this slot. The `block_hash` and `reset_hash()` fields are specific to prefix caching (see [[ai/llm/inference/frameworks/vllm_0.16.0/design/prefix_caching]]).

```python
@dataclass
class KVCacheBlock:
    block_id: int                              # immutable slot index in GPU pool (0 to num_gpu_blocks-1)
    ref_cnt: int = 0                           # how many requests are currently using this block
    _block_hash: BlockHashWithGroupId | None = None  # set when full; None = not cached
    is_null: bool = False                      # placeholder block (e.g., outside sliding window)

    # Doubly linked list pointers — only manipulated by FreeKVCacheBlockQueue
    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None

    def reset_hash(self):
        """Called on eviction — clears hash so block can be reused."""
        self._block_hash = None
```

#### `FreeKVCacheBlockQueue`
**File:** `vllm/v1/core/kv_cache_utils.py`

A doubly linked list of free (unallocated) blocks in LRU order. Head = least recently used (evicted first), tail = most recently used. The linked list pointers live directly inside `KVCacheBlock` — no separate node objects, giving O(1) removal from anywhere in the list.

```python
class FreeKVCacheBlockQueue:
    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        self.num_free_blocks: int
        self.fake_free_list_head: KVCacheBlock  # sentinel
        self.fake_free_list_tail: KVCacheBlock  # sentinel

    def popleft(self) -> KVCacheBlock:
        """Pop LRU block (allocation / eviction)."""

    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        """Pop n LRU blocks at once."""

    def remove(self, block: KVCacheBlock) -> None:
        """O(1) removal — used when a block is pinned on cache hit."""

    def append(self, block: KVCacheBlock) -> None:
        """Return block to tail (MRU position) — used on free."""

    def append_n(self, blocks: list[KVCacheBlock]) -> None: ...
```

#### `KVCacheBlocks`
**File:** `vllm/v1/core/kv_cache_manager.py`

The interface object passed between the scheduler and the KV cache manager. Wraps the block list per KV cache group. Its primary job is `get_block_ids()` — extracting the integer slot IDs needed to build the block table tensor that goes to the attention kernel.

```python
@dataclass
class KVCacheBlocks:
    blocks: tuple[Sequence[KVCacheBlock], ...]  # indexed by [group_id][block_index]

    def get_block_ids(self, allow_none: bool = False) -> tuple[list[int], ...] | None:
        """Extract integer block IDs to build the block table."""

    def new_empty(self) -> "KVCacheBlocks":
        """Create an empty KVCacheBlocks with same group structure."""
```

### Request Lifecycle

```
New request arrives
  → CPU: hash prefix blocks, look up hash→block_id dict
  → CPU: reuse existing block IDs for cache hits (no recompute)
  → CPU: allocate new block IDs from free list for cache misses
  → CPU: build block_table = [block_id_0, block_id_1, ...]
  → CPU→GPU: copy block_table as part of attn_metadata
  → GPU: attention kernel reads block_table[seq][step] → block_id
              → indexes kv_cache[block_id, token_offset, head, dim]
```

All bookkeeping lives on CPU. The GPU just holds the data pool and receives a fresh block table each forward pass.

## Summary

PagedAttention solved the memory fragmentation problem so definitively that the paged KV cache interface has become a standard that all major attention kernels now support. The ecosystem convergence means modern vLLM gets both:

- **Memory efficiency** from paging: near-zero fragmentation, KV block sharing across requests (prefix caching)
- **Compute efficiency** from FlashAttention/FlashInfer: IO-optimal single-pass fused attention kernels
