# HiCache: Why SWA KV stays on GPU only

**What:** The design decision behind `SGLANG_HICACHE_FULL_ONLY` — only FULL attention KV is offloaded to host/disk; SWA (sliding window attention) KV stays on GPU.
**Why:** This is the most counterintuitive decision in HiCache: SWA occupies 71.4% of device KV bytes, yet it's the part that's NOT offloaded. The reasoning reveals the core principle of hierarchical cache design.

## The apparent paradox

SWA occupies 71.4% of per-token KV bytes. It looks like the part that most needs offloading. But it's the part that stays on GPU. Why?

## Per-token KV byte calculation (Gemma 4 31B)

Gemma 4 31B uses YOCO KV sharing — half the layers share KV from an earlier same-type layer and allocate zero KV themselves. So the actual KV-owning layers are 25 SWA + 5 FULL = 30 (not 50+10).

```
SWA per-token = 25 layers × 4 kv_heads × 256 head_dim × 2 (K+V) = 51,200 bytes
FULL per-token =  5 layers × 4 kv_heads × 512 head_dim × 2 (K+V) = 20,480 bytes
Total per-token = 71,680 bytes

SWA fraction = 51,200 / 71,680 = 71.4%
FULL fraction = 20,480 / 71,680 = 28.6%
```

SWA is 71.4% because of 25 vs 5 layers (5:1 count advantage). Each FULL layer stores 2× the bytes of a SWA layer (head_dim 512 vs 256), but SWA has 5× more layers.

## The core insight: offload's purpose = future restore value

Host tier buys not "how much GPU memory we save now" but "the value of future restores." SWA and FULL have a fundamental asymmetry here:

| | SWA | FULL |
|---|---|---|
| Per-conversation KV size | **Bounded** — sliding window=1024, always only last 1024 tokens' KV. ~50MB/conversation, doesn't grow with length | **Unbounded** — grows linearly with conversation length. 15k tokens = 293MB, 50k = 977MB |
| Restore requirement | Only needs trailing window (last 1024 tokens) | Needs entire prefix (from first token to match point) |
| Device hold cost | Low — 1024 tokens/conversation, nearly free to hold on GPU | High — grows with conversation length, quickly exceeds HBM |
| Offload ROI | **Near zero** — 71% host bytes for near-zero restore value | **High** — linearly growing history, restore saves significant recompute |

**Ideal division of labor:** SWA window (small, bounded) pinned on GPU (cheap); FULL history (large, linearly growing) offloaded to host (expensive but worth it). Next turn returns → host restores FULL + GPU supplies SWA window → full-length match. This is the complete logic of `FULL_ONLY` design.

## Crossover calculation

Per-conversation SWA cap = 25 layers × 1024 tokens × bytes/token. FULL = 5 layers × L tokens × bytes/token (where L = conversation length). L > 2560 tokens → FULL exceeds SWA. Production multi-turn chains far exceed this, so while SWA is 71.4% instantaneously, the **growth term** is always FULL.

## SWA bounded is not automatic

SWA's bounded property requires SWA-aware allocator/radix to actively free/tombstone out-of-window blocks. SGLang's SWA radix cache does this. vLLM's equivalent is `SlidingWindowManager.remove_skipped_blocks()`. Without hybrid management, SWA would allocate like FULL and not be bounded.

## Implementation reality: SWA host backup was broken

Even setting aside economics, SWA host backup was broken in implementation. Commit `27827664a` (Mike, `SGLANG_HICACHE_FULL_ONLY`):

- Per-component eviction dropped interior nodes' SWA before the full-KV write_back backup ran
- SWA match validator saw host chains with no SWA → collapsed best valid match to root
- Host hits never fired on production traffic
- **Measured: 136M device vs 69k host cached tokens in 10min at 4% split**

So removing SWA host pool was "pure win": fixed the implementation bug AND aligned with the economically optimal division of labor.

Theoretically correct approach: back only each node's trailing window slice. But in a radix tree, every interior node needs its own window slot, value decays fast — not worth the engineering cost. So the answer is both: implementation bug (explains zero hits) + economics (explains why even if fixed, backing full SWA history is wrong).

## What if GPU doesn't have enough HBM for SWA?

It won't run out. SWA per-conversation is bounded at ~1024 tokens (~50MB) regardless of conversation length. The SWA pool doesn't grow with conversation length. With 1.76M SWA tokens deployed (fp8), that's ~1700 concurrent conversations' SWA windows — more than enough for 10 pods.

## References

- Commit `27827664a` — `SGLANG_HICACHE_FULL_ONLY` (Mike, 2026-08-16)
- [SGLang HiCache design doc](https://docs.sglang.io/docs/advanced_features/hicache_design)
- [SGLang Unified Radix Cache blog](https://www.lmsys.org/blog/2026-08-11-unified-radix-cache/) — YOCO KV sharing, hybrid SWA architecture
