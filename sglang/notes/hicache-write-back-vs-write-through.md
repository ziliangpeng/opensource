# HiCache: write_back vs write_through — and why vLLM can't do write_back

**What:** SGLang HiCache write policies, throughput implications, and cross-framework architectural comparison.
**Why:** The write_back vs write_through distinction is the single most important design decision in HiCache. It determines whether host memory adds real capacity or just mirrors GPU. vLLM/LMCache only support write_through, which the user identified as a fundamental limitation ~9 months before this analysis.

## write_through vs write_back: mechanism

### write_through (vLLM, LMCache, SGLang default)

Every step that produces new KV on GPU **immediately copies it to host**. GPU and host content stay synchronized. Host = mirror of GPU.

```
Each step:
  GPU HBM ──── DMA copy ──→ Host RAM
  (attention compute + DMA copy share HBM bandwidth)
  Host = GPU mirror (every page has a copy)
```

### write_back (SGLang HiCache, TensorRT-LLM)

GPU KV operates normally with **zero host interaction**. Only when GPU HBM is full and eviction is needed, the evicted KV is DMA'd to host before the GPU slot is freed. Host = evicted pages only (disjoint from GPU).

```
Normal step (HBM has room):
  GPU HBM (no DMA, zero overhead)

Eviction step (HBM full):
  GPU HBM ──── DMA copy ──→ Host RAM (evicted page)
  GPU slot freed for new KV
  Host ∩ GPU ≈ ∅ (disjoint by construction)
```

## Throughput tax

Decode attention is memory-bound (HBM bandwidth limited). write_through's per-step DMA competes for the same HBM bandwidth:

| Config | Throughput | Tax | Why |
|--------|-----------|-----|-----|
| No HiCache (baseline) | 30.5 req/s | — | — |
| **write_back + direct** | **30.5 req/s (= baseline)** | **0%** | DMA only on eviction (rare); normal steps have zero overhead |
| write_through + kernel | ~18 req/s | −41% | GPU kernel does D→H copy, competes for compute + bandwidth |
| write_through + direct | ~3 req/s | −90% | SDMA copies KV every step, PCIe/HBM read port saturated |

**Why direct (−90%) is worse than kernel (−41%):** Counterintuitive. `write_through + direct` copies the entire batch's new KV via SDMA every step. SDMA doesn't compete for GPU compute, but it reads HBM (competing with attention for read ports) and constantly saturates PCIe. `write_through + kernel` competes for compute, but the GPU kernel can overlap some operations with the attention kernel via scheduling. Net effect: direct's PCIe/HBM read port contention is more damaging than kernel's compute contention.

## Retention: why write_through doesn't add capacity (when host < GPU)

| | write_through | write_back |
|---|---|---|
| Host content | GPU's mirror — every GPU page has a host copy | Only evicted pages — GPU doesn't have these |
| Host ∩ GPU | ≈ GPU's full content (superset or subset) | ≈ ∅ (disjoint) |
| Effective capacity | max(GPU, host) (because overlap) | GPU + host (because disjoint) |
| When host < GPU pool | Host = GPU's LRU subset → **zero retention gain** | Host = what GPU doesn't have → **real retention gain** |
| When host ≥ GPU pool | Host ⊇ GPU → eviction keeps page on host → has retention | Same, but with zero throughput tax |

**Critical nuance:** write_through's "zero retention gain" is **sizing-dependent, not policy-inherent**. Upstream SGLang's default `hicache_ratio=2.0` (host ⊇ device) gives write_through real retention. CAI's "zero retention" result is because 80G host < device FULL pool capacity. write_back has retention gain at **any** sizing (disjoint by construction).

## write_back is upstream SGLang (not CAI custom)

write_back was added to SGLang in April 2025 (PR #5543). AMD support landed July 2026 (PR #28534). CAI's work was: choosing to use write_back, making it work under hybrid SWA (conditional reprefill cap, concurrency fixes, no re-entrant eviction), and cherry-picking 7 upstream write_back fixes.

## Why vLLM/LMCache can't do write_back

### One-sentence answer

"Radix tree vs block manager" is the surface cause. The real divide is: **SGLang makes eviction a first-class event with owner, metadata, and lock protection; vLLM makes eviction an anonymous block reclaim, then outsources offload to a connector API that cannot observe eviction.**

### Three concrete gaps

| Gap | SGLang | vLLM | Impact |
|-----|--------|------|--------|
| **1. Eviction hook** | Scheduler-internal: eviction = tree node operation | None — BlockPool has no listener API | **Root cause**. Without a hook, only eager store is possible |
| **2. Tier location tracking** | Node records current tier (L1/L2/L3) | Block only has hash, no "where am I now" metadata | Even with a hook, can't know which block is where |
| **3. Overwrite race** | Node lock / protected state during transfer | Evicted block immediately reusable; lazy D2H must finish before overwrite | Deepest correctness challenge |

### vLLM's connector API is request-lifecycle-driven, not block-lifecycle-driven

vLLM's KVConnector API triggers offload on request scheduling events (lookup before scheduling, save after computation). The connector cannot see block-level eviction events. So vLLM can only copy KV to host when "seeing new block produced" — which is write_through / eager store. Result: host = GPU mirror, no retention gain when host ≈ GPU size.

This is exactly the user's concern from ~9 months ago: "Can you use HBM and CPU RAM in combined instead of duplicate?" LMCache CTO's answer: "really, really difficult." The analysis confirms: from LMCache's position (external plugin), it IS nearly impossible — the connector API doesn't expose eviction events. But it's not architecturally impossible.

### vLLM PR #43946: someone proved it works

[vLLM PR #43946](https://github.com/vllm-project/vllm/pull/43946) (2026-05-29, yinlin09): "kv_offload: eviction-triggered store for OffloadingConnector" — this IS write_back for vLLM.

- Adds `BlockPool.register_eviction_listener(fn)` — acknowledges vLLM had no eviction hook
- Listener receives (block_id, BlockHash) before `reset_hash()`
- Connector scheduler batches pending evictions into TransferJobs
- Race: D2H gather on same compute stream as `model.forward()`, relies on stream ordering (author notes GPU may need additional work)

**Benchmark results** (480B MoE, 96×16K prefix, working set > 2× HBM):

| Config | Throughput | Cache hit |
|--------|-----------|-----------|
| No offload | ~270 tok/s | — |
| Eager offload (write_through) | **worse than no offload** | ~0% host hit |
| **Eviction-triggered (write_back)** | **421 tok/s (+56%)** | **88.1%** |

The PR author directly confirms the user's observation: under eager offload, "host pool gets ~0 hits, just eats D→H write churn."

**Status:** open, needs-rebase, zero reviewer engagement since May 2026, single kv-group only (no hybrid attention), TPU-only tested.

### vLLM's design philosophy: "recompute > data movement"

vLLM v1 even deleted V0's swap-based preemption in favor of recompute-only. [RFC #38260](https://github.com/vllm-project/vllm/issues/38260) (multi-tier offloading) explicitly states "Always cascade to all tiers: when a block is confirmed in the primary tier, it is asynchronously pushed to every secondary tier" — i.e., vLLM's official tiering roadmap continues to bet on write_through. SGLang originated from RadixAttention, where the tree is designed for KV lifecycle management; HiCache is a natural extension.

### LMCache CTO's "really difficult" is accurate — from their position

From LMCache's seat (external plugin): nearly impossible. KVConnector API has no eviction events, LMCache is outside the engine, physically cannot observe eviction. Doing write_back requires changing vLLM core (exactly what PR #43946 does), which a plugin vendor cannot unilaterally ship.

From vLLM core's seat: moderate engineering effort + real correctness risk, not architecturally impossible. PR #43946 proves it with a contained diff and +56% throughput.

### Other frameworks with write_back

| Framework | write_back? | Notes |
|-----------|------------|-------|
| **TensorRT-LLM** | **Yes — production, long-standing** | Official docs: "Reusable blocks that are needed for higher priority tasks are copied to a buffer in host memory instead of being evicted." Primary pool (GPU) / secondary pool (host, `hostCacheSize`). Host only holds evicted blocks — host ∩ device ≈ ∅, same as HiCache write_back. Near-zero cost on Grace-Hopper. |
| **NVIDIA Dynamo (KVBM)** | Yes (eviction-driven tier demotion) | Multi-tier G1-G4, tier demotion is eviction-driven (medium confidence) |
| **vllm-marconi-offload** (third-party PyPI) | Yes | "demote-on-store: when HBM evicts a block, the manager demotes it to L1 instead of dropping it" |
| vLLM native (OffloadingConnector) | No (write_through only) | RFC #38260 continues write_through cascade |
| LMCache | No (write_through only) | No eviction-triggered offload in repo |
| HF TGI | No | No hierarchical KV offload |

## What it would take to add write_back to vLLM

PR #43946 is a ready-made checklist:

1. **`BlockPool`** — add eviction listener API (core change, small but sensitive)
2. **`OffloadingConnectorScheduler`** — collect evictions, create synthetic transfer jobs, bypass per-request bookkeeping (eviction doesn't belong to any request)
3. **Worker side** — stream ordering / staging buffer for D2H vs forward race on GPU
4. **Unsolved hard problems:** hybrid KV (sliding window / Mamba, multi kv-group), MLA, TP>1 canonical layout, eviction burst backpressure (D2H queue when hundreds of blocks evicted at once)

**Scale estimate:** contained MVP ~few hundred lines; production-grade (hybrid models + GPU race + TP) is real work.

## References

- [SGLang HiCache design doc](https://docs.sglang.io/docs/advanced_features/hicache_design)
- [SGLang PR #5543](https://github.com/sgl-project/sglang/pull/5543) — write_back first added (2025-04-20)
- [vLLM PR #43946](https://github.com/vllm-project/vllm/pull/43946) — eviction-triggered store (open, unmerged)
- [vLLM RFC #19854](https://github.com/vllm-project/vllm/issues/19854) — KV cache offloading discussion, includes "soon-to-be-evicted" design discussion
- [vLLM RFC #38260](https://github.com/vllm-project/vllm/issues/38260) — multi-tier, "always cascade" = write_through
- [TensorRT-LLM kv-cache-reuse docs](https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html)
- [LMCache architecture](https://docs.lmcache.ai/developer_guide/architecture.html)
- SGLang HiCache write_back timeline: PR #5543 (2025-04) → #28534 AMD enable (2026-07) → #31845 unified tree fix → #33777 host-pressure reclaim
