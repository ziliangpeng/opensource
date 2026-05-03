# KV Connector — Code Deep Dive

## Overview

The KV Connector is vLLM's abstraction for **moving KV cache data in and out of the local GPU**. It sits alongside the KVCacheManager (which handles GPU-local prefix caching) and provides a second tier of KV cache access.

Two major use cases:
1. **Prefill/Decode Disaggregation (P/D)** — one instance prefills, another decodes, KV transferred via RDMA/network
2. **KV Cache Offloading** — spilling KV cache to CPU/disk and reloading on prefix cache miss

## Relationship to Prefix Cache

The connector does NOT replace the GPU prefix cache — they are complementary:

```
Request arrives with prompt tokens
        │
        ▼
┌─────────────────────────┐
│   KVCacheManager         │  ← Tier 1: GPU prefix cache
│   (BlockPool hash table) │     Zero-cost hits, no copies.
│   get_computed_blocks()  │
└────────┬────────────────┘
         │ cache miss beyond GPU hits
         ▼
┌─────────────────────────┐
│   KVConnector            │  ← Tier 2: external KV
│   get_num_new_matched   │     OffloadingConnector: CPU/disk
│   _tokens()             │     LMCache: external KV store
└─────────────────────────┘     NixlConnector: remote GPU (P/D)
```

The scheduler calls them sequentially:
1. `kv_cache_manager.get_computed_blocks()` → local GPU prefix cache hits
2. `connector.get_num_new_matched_tokens(req, local_hits)` → additional tokens from external store

**Always enable prefix caching** (`--enable-prefix-caching`) even with connectors — it's the fast path. The connector only handles what falls out of GPU.

## Architecture: Split Scheduler/Worker Design

Every connector is instantiated **twice** — once per role:

```
KVConnectorBase_V1
  ├── Role: SCHEDULER  (runs in scheduler process)
  │     • get_num_new_matched_tokens()  — how many tokens exist externally?
  │     • update_state_after_alloc()    — record block mapping after allocation
  │     • build_connector_meta()        — package metadata for workers
  │     • request_finished()            — should blocks be freed now or sent first?
  │
  └── Role: WORKER  (runs in each GPU worker)
        • register_kv_caches()          — register GPU memory regions
        • start_load_kv()               — begin async/sync reads
        • wait_for_layer_load()         — block until layer i is loaded
        • save_kv_layer()               — async save from attention layer
        • wait_for_save()               — block until saves complete
        • get_finished()                — report completed async transfers
```

Communication between the two halves flows through `KVConnectorMetadata` — built by the scheduler side and shipped to workers via `SchedulerOutput`.

## Sync vs Async Loading

The second return value of `get_num_new_matched_tokens()` is critical:

- **`False` (sync)** — the request IS scheduled in the forward batch. KV must be loaded in `start_load_kv()` before forward begins. Simpler but blocks the entire batch.
- **`True` (async)** — the request enters `WAITING_FOR_REMOTE_KVS` state. Not in the forward batch. KV loads in background across steps. Higher throughput, more complex state management.

## Integration with the Scheduler

The scheduling loop hooks into the connector at four points:

### 1. Cache Hit Check
```python
# In scheduler._schedule_waiting_requests()
ext_tokens, load_kv_async = connector.get_num_new_matched_tokens(req, local_computed)
```

### 2. Block Allocation
```python
new_blocks = kv_cache_manager.allocate_slots(
    request, num_new_tokens,
    num_external_computed_tokens=ext_tokens,
    delay_cache_blocks=load_kv_async,  # don't hash blocks if async loading
)
connector.update_state_after_alloc(request, blocks, ext_tokens)
```

### 3. Build Step Metadata
```python
meta = connector.build_connector_meta(scheduler_output)
scheduler_output.kv_connector_metadata = meta  # shipped to workers
```

### 4. Request Completion
```python
delay_free, kv_transfer_params = connector.request_finished(req, block_ids)
# delay_free=True → blocks kept alive until get_finished() confirms transfer done
# kv_transfer_params → forwarded to decode instance (P/D case)
```

## Worker-Side Lifecycle (per step)

```
GPUModelRunner.execute_model():
  ├── pre_forward()
  │     ├── handle_preemptions()          — cancel in-flight saves for preempted reqs
  │     ├── bind_connector_metadata()     — set metadata from scheduler
  │     └── start_load_kv()              — kick off reads (sync or async)
  │
  ├── model.forward()                    — attention layers call:
  │     ├── wait_for_layer_load(layer_i) — ensure layer i KV is ready
  │     └── save_kv_layer(layer_i, ...)  — async save layer i
  │
  └── post_forward()
        ├── wait_for_save()              — ensure all saves complete
        ├── get_finished()               — (done_sending, done_recving) sets
        ├── get_block_ids_with_load_errors() — failed block IDs
        └── clear_connector_metadata()
```

## Async Loading Flow (NIXL, LMCacheMPConnector, OffloadingConnector)

```
Step N:
  Scheduler:
    1. Request arrives, get_num_new_matched_tokens() → (tokens, async=True)
    2. Blocks allocated, but num_new_tokens = 0 (no compute scheduled)
    3. Request enters WAITING_FOR_REMOTE_KVS state

  Worker:
    pre_forward():  start_load_kv() → kicks off async transfers
    forward():      runs OTHER requests (the waiting one is NOT in this batch)
    post_forward(): get_finished() → not done yet

Step N+1, N+2...:
    Transfers continue in background, other requests execute normally

Step N+K:
    post_forward(): get_finished() → reports req_id done!

  Scheduler:
    _update_from_kv_xfer_finished()
    → req.num_computed_tokens = all prompt tokens
    → moves to schedulable queue

Step N+K+1:
    Request scheduled normally with KV already in GPU
```

**Key insight**: the waiting request does NOT block the forward pass. The GPU stays busy with other work. Trade-off: higher throughput but higher TTFT for the transferred request.

Potential concerns with async loading:
- **Memory bandwidth contention** — RDMA/PCIe transfers overlap with running forward pass
- **Block allocation waste** — blocks reserved but empty during transfer
- **Latency** — request sits idle for multiple steps

## Registered Connectors

| Connector | Transport | Sync/Async | Use Case |
|---|---|---|---|
| **NixlConnector** | NIXL (RDMA/UCX) | Async | P/D disaggregation over network |
| **LMCacheConnectorV1** | LMCache library | **Sync** | External KV cache store (local) |
| **LMCacheMPConnector** | LMCache multi-process | Async | External KV cache store (remote/large) |
| **P2pNcclConnector** | NCCL P2P | — | Same-machine GPU transfers |
| **OffloadingConnector** | CPU/Disk | Async | KV cache spill/reload (L2 cache) |
| **MooncakeConnector** | Mooncake | — | P/D via Mooncake transfer engine |
| **MoRIIOConnector** | MoRIIO | — | P/D via MoRIIO |
| **MultiConnector** | Composite | — | Wraps multiple connectors (e.g., NIXL + Offloading) |
| **DecodeBenchConnector** | None | — | Benchmarking decode without real transfers |

## Error Handling

- Failed transfers → block IDs marked invalid via `get_block_ids_with_load_errors()`
- Scheduler calls `_update_requests_with_invalid_blocks()` to roll back `num_computed_tokens`
- Affected requests recompute from the longest valid prefix
- P-side timeout → if D doesn't pull within `VLLM_NIXL_ABORT_REQUEST_TIMEOUT`, blocks freed anyway

## Key Data Structures

- **`KVConnectorMetadata`** — abstract, each connector defines its own (e.g., `NixlConnectorMetadata` with `reqs_to_recv`, `reqs_to_save`, `reqs_to_send`)
- **`KVConnectorOutput`** — worker→scheduler feedback: `finished_sending`, `finished_recving`, `invalid_block_ids`, stats
- **`kv_transfer_params`** — per-request dict passed through the engine API: `{remote_block_ids, remote_engine_id, remote_host, remote_port, do_remote_prefill, do_remote_decode}`

## Source Files

```
vllm/distributed/kv_transfer/kv_connector/
├── v1/
│   ├── base.py                    — KVConnectorBase_V1 (abstract base class)
│   ├── nixl_connector.py          — NIXL RDMA connector (P/D)
│   ├── offloading_connector.py    — CPU/disk offloading connector
│   ├── lmcache_connector.py       — LMCache v1 wrapper (sync)
│   ├── lmcache_mp_connector.py    — LMCache multi-process (async)
│   ├── multi_connector.py         — Composite connector
│   ├── example_connector.py       — Reference implementation
│   ├── metrics.py                 — Connector stats/prometheus
│   └── lmcache_integration/       — Native LMCache adapter code
│       ├── vllm_v1_adapter.py     — LMCacheConnectorV1Impl
│       └── multi_process_adapter.py
├── factory.py                     — KVConnectorFactory (registry + creation)
└── utils.py                       — Shared utilities

vllm/v1/worker/
├── gpu/kv_connector.py            — GPUModelRunner KVConnector wrapper
└── kv_connector_model_runner_mixin.py — Mixin for model runners
```
