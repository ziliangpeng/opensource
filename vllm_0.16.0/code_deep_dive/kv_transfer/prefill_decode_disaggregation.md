# Prefill-Decode Disaggregation in vLLM 0.16.0

## Note Role
This file is the vLLM-implementation deep dive for PD disaggregation.
For the general architecture patterns and cross-system comparison, see:
- [[../../../../prefill_decode_disaggregation|Prefill-Decode Disaggregation]]

## Why Disaggregate?

Prefill (processing the full prompt) is **compute-bound** — big matrix multiplications across all input tokens. Decode (generating tokens one at a time) is **memory-bandwidth-bound** — tiny batches, limited by how fast you can read KV caches. When both run on the same GPU, they interfere: prefill spikes stall decode latency (TPOT), and decode's small batches waste compute capacity during generation.

Disaggregation separates them onto dedicated instances:
- **P instance** (producer): runs prefill only, computes KV caches, transfers them out
- **D instance** (consumer): receives KV caches, skips prefill, runs decode only

This lets you optimize each independently — P instances can use higher batch sizes and different GPU utilization targets, D instances can maximize decode throughput. You can also scale them at different ratios (xPyD: x prefill instances, y decode instances) based on workload characteristics.

---

## Architecture Overview

PD disaggregation in vLLM 0.16.0 involves three layers:

```
┌─────────────────────────────────────────────────────────┐
│                   External Proxy/Router                  │
│         (routes requests to P and D instances)           │
└──────────┬──────────────────────────────┬───────────────┘
           │                              │
           ▼                              ▼
┌─────────────────────┐      ┌─────────────────────────┐
│   P Instance (vLLM) │      │     D Instance (vLLM)    │
│  kv_role=kv_producer │─────▶│   kv_role=kv_consumer    │
│  prefill + KV send   │ KV  │   KV recv + decode       │
└─────────────────────┘transfer└─────────────────────────┘
```

### The Three Software Layers

1. **KV Connector Framework** (`kv_connector/v1/base.py`) — The plumbing. Defines the `KVConnectorBase_V1` interface with scheduler-side and worker-side methods. All PD implementations plug into this.

2. **Disagg Serving API** (`entrypoints/serve/disagg/`) — A newer token-in/token-out HTTP API (`/inference/v1/generate`) designed for external coordinators. Accepts raw token IDs, returns token IDs + `kv_transfer_params`.

3. **Connector Implementations** — The actual KV transfer mechanisms (P2P NCCL, NIXL, LMCache, etc.)

---

## Configuration

All PD configuration flows through `KVTransferConfig` (in `config/kv_transfer.py`):

```python
@config
class KVTransferConfig:
    kv_connector: str | None = None       # e.g. "P2pNcclConnector", "NixlConnector"
    kv_role: KVRole | None = None         # "kv_producer" | "kv_consumer" | "kv_both"
    kv_buffer_size: float = 1e9           # bytes, GPU buffer for receiving KV
    kv_buffer_device: str = "cuda"        # "cuda" or "cpu"
    kv_port: int = 14579                  # port for KV connector communication
    kv_connector_extra_config: dict = {}  # connector-specific config
    kv_load_failure_policy: str = "recompute"  # "recompute" or "fail"
    engine_id: str | None = None          # auto-generated UUID
```

Key properties:
- `is_kv_producer` → True when role is `"kv_producer"` or `"kv_both"`
- `is_kv_consumer` → True when role is `"kv_consumer"` or `"kv_both"`

Note: when vLLM starts with `kv_connector` set but no explicit `kv_role`, it errors. When used as a standalone (non-disagg) KV cache (offloading, LMCache for prefix caching), `_post_init_kv_transfer_config` in `VllmConfig` forces `kv_role = "kv_both"`.

---

## Implementation 1: P2P NCCL Connector

**Files:**
- `kv_connector/v1/p2p/p2p_nccl_connector.py` — Connector implementing the V1 interface
- `kv_connector/v1/p2p/p2p_nccl_engine.py` — ZMQ + NCCL communication engine
- `kv_connector/v1/p2p/tensor_memory_pool.py` — CPU spillover buffer (buddy allocator)
- `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_proxy_p2p_nccl_xpyd.py` — External proxy

### Three Actors

#### 1. The Proxy (external, not part of vLLM)

A Quart (async Flask) HTTP server that serves as the client-facing entry point.

**Runs:**
- **HTTP server** on port 10001 — receives `/v1/completions` and `/v1/chat/completions`
- **ZMQ ROUTER socket** on port 30001 — receives heartbeat pings from P/D instances

**Service discovery:** Every P and D instance runs a `ping()` thread that sends its identity to the proxy every 3 seconds via ZMQ DEALER→ROUTER:
```python
data = {
    "type": "P" or "D",
    "http_address": "10.0.1.2:20001",   # vLLM OpenAI-compatible API
    "zmq_address": "10.0.1.2:21001",    # KV transfer ZMQ address
}
```
The proxy maintains two dicts (`prefill_instances`, `decode_instances`) keyed by http_address. Stale entries (no ping in 5s) are evicted.

**Request routing flow:**
```
Client → Proxy(:10001)
  1. Pick 1P + 1D via round-robin
  2. Encode routing into request_id:
     "___prefill_addr_{P_zmq}___decode_addr_{D_zmq}_{uuid}"
  3. Forward to P with max_tokens=1 (prefill-only), WAIT for completion
  4. Forward original request to D, STREAM response back to client
```

The critical insight: **the request_id itself carries the P↔D routing addresses**. Both P and D parse it with regex to find each other's ZMQ address. No separate routing metadata or service mesh needed.

#### 2. P2pNcclEngine (runs inside each vLLM instance)

Created by `P2pNcclConnector` on the worker side. Each instance runs:

| Thread | Socket | Purpose |
|--------|--------|---------|
| `listen_for_requests` | ZMQ ROUTER on `kv_port + rank` | Accept NCCL setup (NEW), KV push (PUT), KV pull (GET) |
| `ping` | ZMQ DEALER → proxy:30001 | Service discovery heartbeat (every 3s) |
| `send_async` (PUT_ASYNC only) | — | Drains send queue, sends KV via NCCL |

#### 3. P2pNcclConnector (scheduler + worker halves)

Implements `KVConnectorBase_V1`. Unlike NIXL (which has separate `NixlConnectorScheduler` / `NixlConnectorWorker` classes), P2P NCCL uses a single class where `is_producer` flag switches behavior.

### NCCL Connection Establishment

On the first request between a P↔D pair, the initiating side (P for PUT/PUT_ASYNC, D for GET) calls `create_connect()`:

```
Initiator (e.g. P)                    Receiver (e.g. D, listener thread)
     │                                         │
     │ create ZMQ DEALER socket                │
     │ connect to D's zmq_address              │
     │                                         │
     │ generate NCCL unique_id                 │
     │───{cmd:"NEW", unique_id}──────────────▶│
     │                                         │ ncclCommInitRank(world=2, rank=1)
     │ ncclCommInitRank(world=2, rank=0)       │
     │                                         │
     │       NCCL group established            │
     │     (cached in self.comms[addr])        │
```

Each P↔D pair forms an independent **2-rank NCCL group**. This means:
- No global coordination needed
- Adding/removing P or D instances doesn't require restarting the cluster
- Just know the peer's ZMQ address (embedded in the request_id)

**GPU memory cost:** Each NCCL group uses ~52MB with `NCCL_MAX_NCHANNELS=8` or ~100MB with 16 channels. For large xPyD (e.g. DeepSeek's 96P144D), this approach doesn't scale — hence NIXL.

### KV Transfer (PUT_ASYNC — the recommended mode)

```
P worker thread          P send_async thread       D listener thread        D worker thread
     │                        │                         │                        │
 save_kv_layer()              │                         │                        │
  for each layer:             │                         │                        │
   extract KV blocks          │                         │                        │
   ──queue item──────▶ popleft()                        │                        │
     │                  send {cmd:"PUT",                │                        │
     │                   tensor_id, shape,              │                        │
     │                   dtype} via ZMQ ──────────────▶ │                        │
     │                         │              recv "0" ◀─ alloc GPU tensor       │
     │                  ncclSend() ══════════════════▶ ncclRecv()               │
     │                         │                        store in recv_store      │
     │                         │                        (or spill to CPU pool)   │
     │                         │                         │                        │
     │                         │                         │                start_load_kv()
     │                         │                         │                 recv_tensor()
     │                         │                         │                  wait on recv_store_cv
     │                         │                         │                  inject into KV cache
```

The `tensor_id` format is `{request_id}#{layer_name}`, so each attention layer's KV cache is transferred separately.

### Three Transfer Modes

| Mode | Direction | Blocking? | Performance | How it works |
|------|-----------|-----------|-------------|-------------|
| **PUT** | P→D (sync) | Blocks P's main thread | Worst | P sends metadata via ZMQ, then ncclSend. Blocks until complete. |
| **PUT_ASYNC** | P→D (async) | Non-blocking for P | Best | P queues items, dedicated thread drains queue and sends. |
| **GET** | D pulls from P | Blocks D's recv path | Middle | P stores KV in buffer. D sends GET request, P responds with metadata, then ncclSend. |

### Buffer Overflow and CPU Spillover

D's GPU buffer (`kv_buffer_size`, typically 5-10% of GPU memory) creates a tension:
- **Too small:** KV spills to `TensorMemoryPool` (CPU memory, buddy-allocator inspired). Transfer happens at PCIe speed (~21 GB/s for PCIe 4.0), adding latency.
- **Too large:** Reduces GPU memory available for decode's own KV cache, shrinking batch size and hurting throughput.

When the GPU buffer overflows on D's listener thread:
```python
if self.buffer_size + tensor_size > self.buffer_size_threshold:
    addr = self.pool.store_tensor(tensor)  # spill to CPU
    tensor = (addr, tensor.dtype, tensor.shape)  # store as reference
```

When `recv_tensor()` later retrieves it:
```python
if isinstance(tensor, tuple):
    addr, dtype, shape = tensor
    tensor = self.pool.load_tensor(addr, dtype, shape, self.device)  # copy back to GPU
```

### Chunked Prefill Handling

The P2P connector handles chunked prefill (when the prompt is too long to prefill in one step):

```python
# In build_connector_meta, producer side:
if num_tokens < len(prompt_token_ids):
    # Not done yet — accumulate block_ids, wait for next chunk
    self.chunked_prefill[req_id] = (block_ids, prompt_token_ids)
    continue
# All chunks done — send the full KV
meta.add_request(request_id=req_id, ...)
self.chunked_prefill.pop(req_id, None)
```

KV is only transferred after **all chunks complete** prefill on P.

---

## Implementation 2: NIXL Connector

**File:** `kv_connector/v1/nixl_connector.py` (2775 lines — the most complex connector)

NIXL (NVIDIA Interconnect eXchange Library) is the production-grade approach, designed for large-scale deployments where P2P NCCL's per-pair NCCL groups don't scale.

### Key Differences from P2P NCCL

| Aspect | P2P NCCL | NIXL |
|--------|----------|------|
| Transport | NCCL (GPU direct) | UCX / RDMA (configurable backends) |
| Connection model | 2-rank NCCL group per P↔D pair | NIXL agent registration + RDMA read |
| Scaling | O(P×D) NCCL groups, ~52-100MB each | Lightweight agent metadata exchange |
| Transfer direction | P pushes to D (PUT) or D pulls (GET) | D reads from P's memory (RDMA read) |
| Routing | request_id encodes ZMQ addresses | `kv_transfer_params` carries engine_id, host, port, block_ids |
| Proxy | Required (external script) | External coordinator sets `kv_transfer_params` |
| Compatibility | Trust-based (same config assumed) | Explicit compat hash checked during handshake |
| Block size | Must match P and D | Supports heterogeneous block sizes (postprocess on receive) |
| TP support | Symmetric only | Heterogeneous TP (e.g., P with TP=4, D with TP=2) |

### Architecture: Three-Class Split

Unlike P2P NCCL's single connector class, NIXL has explicit separation:

```
NixlConnector (KVConnectorBase_V1)
  ├── NixlConnectorScheduler  — runs in scheduler process
  │     ├── ZMQ handshake listener thread
  │     ├── tracks _reqs_need_recv, _reqs_need_save, _reqs_need_send
  │     └── builds NixlConnectorMetadata
  └── NixlConnectorWorker     — runs in each TP worker
        ├── nixl_wrapper (NIXL agent)
        ├── ThreadPoolExecutor for background handshakes
        ├── _ready_requests queue
        └── RDMA transfer execution
```

### Handshake Protocol

NIXL uses ZMQ for handshake only (not for data transfer):

**Scheduler side** runs a `_nixl_handshake_listener` thread:
- Binds ZMQ ROUTER on `VLLM_NIXL_SIDE_CHANNEL_HOST:VLLM_NIXL_SIDE_CHANNEL_PORT`
- Serves `NixlHandshakePayload` containing:
  - `compatibility_hash` — derived from model config, dtype, KV layout, attention backend, block size
  - `agent_metadata_bytes` — serialized `NixlAgentMetadata` with engine_id, block info, KV cache base addresses, device_id
  - `connector_version` — protocol version (currently v2)

**Worker side** initiates handshake (`_nixl_handshake`):
- Connects ZMQ REQ to remote's side channel
- Sends `GET_META_MSG` for each remote TP rank needed
- Validates compatibility hash (prevents mismatched configs from silently corrupting data)
- Registers remote agent in NIXL wrapper (enables RDMA transfers)
- Handshake is done **once per engine pair**, in a background thread

### Transfer Flow (D pulls from P via RDMA)

Unlike P2P NCCL where P pushes KV to D, NIXL uses a **pull model**:

```
1. P finishes prefill for request R
2. P's scheduler calls request_finished():
   - Delays block freeing (delay_free_blocks = True)
   - Returns kv_transfer_params:
       {do_remote_prefill: True,
        remote_block_ids: [...],
        remote_engine_id: "uuid",
        remote_request_id: "R",
        remote_host: "10.0.1.2",
        remote_port: 5557,
        tp_size: 4}
3. External coordinator receives this in the API response
4. Coordinator sends request to D with kv_transfer_params attached

5. D's scheduler sees kv_transfer_params.do_remote_prefill = True
6. get_num_new_matched_tokens() returns (num_prompt_tokens - computed, is_async=True)
7. update_state_after_alloc() queues the request in _reqs_need_recv
8. build_connector_meta() creates NixlConnectorMetadata with recv list
9. D's worker initiates NIXL handshake with P (if first time)
10. D reads P's KV blocks via RDMA into its own allocated blocks
11. D sends notification to P that read is complete
12. P frees the blocks
```

### Block Ownership and Lifetime

A critical design concern: P must **not free its KV blocks** until D has finished reading them via RDMA.

P's `request_finished()` returns `delay_free_blocks = True`, which tells the scheduler to hold the blocks. It also sets an expiration timer (`VLLM_NIXL_ABORT_REQUEST_TIMEOUT`). The blocks are freed when:
- D sends a NIXL notification confirming the read is done, OR
- The timeout expires (D is presumed dead/slow, blocks freed anyway)

For heterogeneous TP, P must wait for **all** assigned D TP workers to finish reading:
```python
self.consumer_notification_counts_by_req = defaultdict[ReqId, int](int)
```

### Failure Handling

NIXL has explicit failure policies via `kv_load_failure_policy`:
- **"recompute"** (default): If KV transfer fails, the request is rescheduled to recompute prefill locally on D
- **"fail"**: The request immediately fails with an error

Failed requests are tracked in `_failed_recv_reqs`, and invalid blocks from failed NIXL ops in `_invalid_block_ids`.

---

## The Disagg Serving API

**Files:**
- `entrypoints/serve/disagg/api_router.py` — FastAPI router
- `entrypoints/serve/disagg/serving.py` — `ServingTokens` handler
- `entrypoints/serve/disagg/protocol.py` — `GenerateRequest` / `GenerateResponse`

This is a newer, cleaner API designed for sophisticated external coordinators (like `llm-d` from Red Hat, or vLLM Production Stack).

### Endpoint: `POST /inference/v1/generate`

**Request (`GenerateRequest`):**
```python
class GenerateRequest(BaseModel):
    request_id: str           # caller-controlled, used for tracking
    token_ids: list[int]      # pre-tokenized input (no text!)
    sampling_params: SamplingParams
    kv_transfer_params: dict | None  # routing info from coordinator
    model: str | None
    priority: int = 0
```

**Response (`GenerateResponse`):**
```python
class GenerateResponse(BaseModel):
    request_id: str
    choices: list[GenerateResponseChoice]  # each has token_ids, logprobs
    kv_transfer_params: dict | None        # P returns this for coordinator
    prompt_logprobs: list | None
```

### Why Token IDs Instead of Text?

The coordinator tokenizes once and sends raw token IDs to both P and D. This avoids:
- Double tokenization overhead
- Tokenizer inconsistencies between instances
- Allows the coordinator to work at the token level for routing decisions

### `force_no_detokenize` Mode

When `ServingTokens` is created with `force_no_detokenize=True` (via `--tokens-only` flag), the engine skips detokenization entirely. Pure token-in/token-out pipeline — no string processing overhead.

### Abort Endpoint

When `--tokens-only` is set, an additional endpoint is registered:
```
POST /abort_requests
Body: {"request_ids": ["id1", "id2"]}
```
The coordinator uses this to cancel in-flight prefill requests on P if D is unavailable or the client disconnects.

---

## How KV Transfer Params Flow Through the Engine

The `kv_transfer_params` dict is the glue between the external coordinator and vLLM's internal KV transfer machinery.

### Inbound (request arrives at D with params from coordinator):

```
API request with kv_transfer_params
  → EngineCore.add_request() (core.py:311)
      validates kv_connector exists
  → Scheduler.get_num_new_matched_tokens()
      connector reads params, returns external token count
  → Scheduler.update_state_after_alloc()
      connector queues the recv
  → Worker.start_load_kv()
      connector executes the KV transfer
```

### Outbound (P finishes prefill, returns params to coordinator):

```
Scheduler._free_request() returns kv_transfer_params
  → SchedulerOutput includes kv_transfer_params
  → output_processor propagates to RequestOutput
  → API response includes kv_transfer_params
  → Coordinator extracts and forwards to D
```

In `scheduler.py` (line ~1332-1404):
```python
kv_transfer_params = None
# ... on request finish:
kv_transfer_params = self._free_request(request)
# ... included in output if present:
if kv_transfer_params:
    output = EngineCoreOutput(..., kv_transfer_params=kv_transfer_params)
```

---

## Connector Comparison

| Connector | Transport | Async? | Scale | Use Case |
|-----------|-----------|--------|-------|----------|
| **P2pNcclConnector** | NCCL (GPU direct) | PUT_ASYNC | Small-medium xPyD | Same-rack, few instances |
| **NixlConnector** | UCX/RDMA | Yes (D-pull) | Large xPyD | Production, cross-rack, heterogeneous TP |
| **LMCacheConnectorV1** | LMCache store | Sync (blocks forward) | Any | When you want content-addressable caching |
| **LMCacheMPConnector** | LMCache (multi-process) | Async (CUDA events) | Any | LMCache without blocking forward |
| **OffloadingConnector** | CPU/disk | N/A | Single instance | Not PD — local KV offload to save GPU memory |

---

## Deployment Patterns

### P2P NCCL: Simple xPyD

```bash
# Proxy
python disagg_proxy_p2p_nccl_xpyd.py &  # :10001 HTTP, :30001 ZMQ

# Prefill instance
vllm serve $MODEL --port 20001 \
  --kv-transfer-config '{"kv_connector":"P2pNcclConnector",
    "kv_role":"kv_producer","kv_buffer_size":"1e1","kv_port":"21001",
    "kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20001"}}'

# Decode instance
vllm serve $MODEL --port 20002 \
  --kv-transfer-config '{"kv_connector":"P2pNcclConnector",
    "kv_role":"kv_consumer","kv_buffer_size":"8e9","kv_port":"22001",
    "kv_connector_extra_config":{"proxy_ip":"10.0.1.1","proxy_port":"30001","http_port":"20002"}}'
```

Note: P's `kv_buffer_size` is tiny (`1e1` = 10 bytes, doesn't receive KV). D's is large (`8e9` = 8GB, ~10% of 80GB A800).

### NIXL: Production Scale

```bash
# Prefill instance (uses side channel for handshake)
vllm serve $MODEL --port 8000 \
  --kv-transfer-config '{"kv_connector":"NixlConnector",
    "kv_role":"kv_producer"}'

# Decode instance
vllm serve $MODEL --port 8001 \
  --kv-transfer-config '{"kv_connector":"NixlConnector",
    "kv_role":"kv_consumer"}'

# External coordinator (e.g., llm-d, vLLM Production Stack)
# sends requests to P, gets kv_transfer_params back, forwards to D
```

---

## Key Design Decisions and Trade-offs

### 1. Request ID as Routing (P2P NCCL) vs. Structured Params (NIXL)

P2P NCCL embeds routing in the request_id string (`___prefill_addr_...___decode_addr_...`). Clever and simple, but brittle — regex parsing, tightly coupled to proxy logic. NIXL uses proper `kv_transfer_params` dicts with typed fields. More complex but composable with any coordinator.

### 2. Push (P→D) vs. Pull (D←P)

P2P NCCL pushes KV from P to D. D must have buffer space ready. If buffer overflows, KV spills to CPU.

NIXL lets D pull via RDMA read. P holds blocks until D confirms read. More complex block lifetime management, but D controls the timing (better for scheduling).

### 3. GPU Memory Budget

Both approaches face the same fundamental tension: memory used for KV transfer buffers reduces memory available for serving. P2P NCCL recommends 5-10% of GPU memory. NIXL avoids buffers entirely on the fast path (RDMA reads directly into allocated KV blocks), but needs host buffers for non-RDMA accelerators.

### 4. Failure Recovery

P2P NCCL: if D's buffer overflows, KV spills to CPU (graceful degradation). If NCCL connection dies, no recovery — request fails.

NIXL: explicit `kv_load_failure_policy`. "recompute" reschedules the request for local prefill on D. Timeout-based block freeing on P prevents resource leaks from dead D instances.

### 5. Compatibility Checking

P2P NCCL: none. Assumes P and D have identical configs. Silent corruption if they don't.

NIXL: compatibility hash checked during handshake. Mismatched model, dtype, KV layout, or attention backend → explicit error. Can be disabled with `enforce_handshake_compat: false`.

---

## Network Latency Requirements for PD

### The Core Constraint

```
KV_transfer_time < prefill_time_if_D_did_it_locally
```

If KV transfer takes longer than D just recomputing prefill itself, disaggregation is pointless (or worse).

### KV Cache Size per Request

```
KV size = num_layers × 2 (K+V) × num_tokens × num_heads × head_dim × dtype_bytes
```

| Model | 1K tokens (fp16) | 4K tokens (fp16) | Notes |
|-------|-----------------|-----------------|-------|
| Llama-3.1-8B | 128 MB | 512 MB | 32L × 2 × 8 heads × 128 dim |
| Llama-3.1-70B | 320 MB | 1.28 GB | 80L × 2 × 8 heads × 128 dim |
| DeepSeek-V3 (MLA) | Much smaller | Much smaller | Compressed latent representation — PD-friendly |

### Transfer Times by Interconnect

| Interconnect | Bandwidth | 128MB (8B, 1K) | 320MB (70B, 1K) |
|-------------|-----------|----------------|-----------------|
| NVLink (intra-node) | ~600 GB/s | 0.2 ms | 0.5 ms |
| InfiniBand NDR | ~50 GB/s | 2.5 ms | 6.4 ms |
| InfiniBand HDR | ~25 GB/s | 5 ms | 13 ms |
| PCIe 4.0 (CPU spill) | ~21 GB/s | 6 ms | 15 ms |
| 100GbE RDMA (RoCE) | ~12 GB/s | 10.7 ms | 26.7 ms |
| 25GbE TCP | ~3 GB/s | 43 ms | 107 ms |

For reference, prefill time for 1K tokens on an H100: ~20-50ms for 8B, ~200-500ms for 70B.

### Practical Guidance

- **Same-node (NVLink/PCIe):** Always viable. P2P NCCL works great here.
- **Same-rack InfiniBand (HDR/NDR):** The production sweet spot. NIXL with RDMA handles this well. Transfer latency is comparable to or less than prefill time.
- **Cross-rack RDMA:** Works but tight for short prompts. Fine for long-context workloads where prefill is expensive anyway.
- **Regular Ethernet (no RDMA):** Generally too slow for latency-sensitive PD. Plain TCP can't keep up for interactive use cases.

### Hidden Costs Beyond Raw Bandwidth

1. **NCCL group setup** (P2P NCCL): First transfer between a P↔D pair pays ~100ms+ for `ncclCommInitRank`. Amortized on subsequent requests.
2. **NIXL handshake:** ZMQ round-trip + NIXL agent registration. Also one-time per engine pair.
3. **Per-layer granularity:** P2P NCCL transfers KV per-layer (separate `ncclSend` per attention layer). More round-trips but enables pipelining.
4. **Contention:** Multiple requests sharing the same link. Buffer overflow → CPU spill (P2P NCCL) adds PCIe latency on top.
5. **Proxy serialization** (P2P NCCL): The proxy waits for P to complete before sending to D. That's P's full prefill time + transfer time added to TTFT.

### When PD Disaggregation Pays Off

- **Long-context workloads (32K+ tokens):** Prefill dominates, so even moderate transfer speeds pay for themselves.
- **High QPS with mixed prompt lengths:** P can batch long prefills while D maintains low decode latency.
- **Asymmetric scaling:** When you need more prefill capacity than decode (or vice versa).
- **Different GPU types:** P on compute-optimized GPUs, D on memory-bandwidth-optimized GPUs.

**Bottom line:** InfiniBand RDMA (≥HDR) between P and D is effectively the minimum for production PD deployments. NVLink for same-node. Below IB speeds, returns diminish rapidly for short-to-medium prompts.
