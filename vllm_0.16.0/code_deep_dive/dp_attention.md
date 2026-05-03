# vLLM v0.16.0 — DP Attention (Data-Parallel Attention) Deep Dive

> Scope: **vLLM v0.16.0 code only**. This document summarizes how “DP attention” works in vLLM, focusing on request routing, DP coordinator stats, and KV ownership.

## 1) What “DP attention” means in vLLM (code-first)
In vLLM v0.16.0, “DP attention” is **not** a special attention kernel. It is **data-parallel execution at the engine level**, where:
- Each **DP rank is a separate EngineCore process**.
- Requests are routed to a single rank; all attention + KV operations happen within that rank.
- KV cache lives **inside the chosen rank**, so KV reuse only happens if the same conversation is routed to the same rank.

### Code anchors
- `vllm/entrypoints/openai/engine/serving.py` → `_get_data_parallel_rank()`
- `vllm/v1/engine/__init__.py` → `EngineCoreRequest.data_parallel_rank`
- `vllm/v1/engine/core_client.py` → `DPLBAsyncMPClient.get_core_engine_for_request()`

---

## 2) Rank selection (header vs internal LB)
### 2.1 Header path (explicit routing)
The OpenAI server reads **`X-data-parallel-rank`** and passes it through. There is **no internal logic** beyond parsing the integer:

```python
@staticmethod
def _get_data_parallel_rank(raw_request: Request | None) -> int | None:
    if raw_request is None:
        return None

    rank_str = raw_request.headers.get("X-data-parallel-rank")
    if rank_str is None:
        return None

    try:
        return int(rank_str)
    except ValueError:
        return None
```

Files where this is used:
- `vllm/entrypoints/openai/completion/serving.py`
- `vllm/entrypoints/openai/chat_completion/serving.py`

Both pass `data_parallel_rank` into `input_processor.process_inputs(...)` and `engine_client.generate(...)`.

### 2.2 No header → internal load balancing
If `data_parallel_rank` is **None**, the client chooses a rank using queue stats:

```python
class DPLBAsyncMPClient(DPAsyncMPClient):
    def get_core_engine_for_request(self, request: EngineCoreRequest) -> EngineIdentity:
        if (eng_index := request.data_parallel_rank) is None:
            current_counts = self.lb_engines
            num_engines = len(current_counts)
            min_score = sys.maxsize
            eng_index = 0
            for i in range(num_engines):
                idx = (self.eng_start_index + i) % num_engines
                waiting, running = current_counts[idx]
                score = waiting * 4 + running
                if score < min_score:
                    min_score = score
                    eng_index = idx
            current_counts[eng_index][0] += self.client_count

        chosen_engine = self.core_engines[eng_index]
        self.reqs_in_flight[request.request_id] = chosen_engine
        return chosen_engine
```

**Key point:** this is **queue‑based LB**, not KV‑aware routing.

### 2.3 Rank affinity for KV reuse (why it matters)
- Each DP rank owns its **local KV cache**.
- If a multi‑turn chat request goes to a **different rank**, KV reuse is lost.
- To get KV hits, you must enforce **rank affinity** by supplying the header.

Typical pattern:
- Choose a stable mapping (e.g., `rank = hash(chat_id) % dp_size`).
- Inject `X-data-parallel-rank` on every turn of that chat.

---

## 3) DP coordination mechanics (wave + microbatch/padding)
DP ranks are replicas, but vLLM **intentionally synchronizes** some decisions across them for correctness and performance.

### 3.1 Wave coordination (run/pause sync)
Purpose: keep all DP ranks in the same **running vs paused** phase so scheduling and collectives don’t desync.

- The **DP Coordinator** tracks a global **wave number** and **running/paused** state.
- When a request arrives while engines are paused, a **START_DP_WAVE** broadcast wakes all ranks.

This coordination is about **phase alignment**, not routing.

### 3.2 Microbatch + DP padding synchronization
Purpose: ensure all DP ranks agree on **microbatching** and **token counts** so they execute compatible shapes (required for collectives and cudagraphs).

In `vllm/v1/worker/dp_utils.py`, `coordinate_batch_across_dp(...)`:
- All ranks report their token counts and whether they want microbatching.
- An **all‑reduce** synchronizes the decision.
- If microbatching happens, ranks **pad to the same token count**.

---

## 4) DP Coordinator: what it does (and does NOT do)
The DP Coordinator is **not a router**. It exists to **broadcast queue stats + wave state** so the API server can do load‑based routing and synchronize run/pause behavior.

### 4.1 Stats update path
`vllm/v1/engine/core_client.py` subscribes to coordinator updates and maintains `lb_engines`:

```python
# Update local load-balancing state.
counts, wave, running = msgspec.msgpack.decode(buf)
self.current_wave = wave
self.engines_running = running
if counts is not None:
    sliced_counts = counts[count_slice]
    self.lb_engines = sliced_counts
```

### 4.2 Coordinator responsibilities (from code)
`vllm/v1/engine/coordinator.py`:
- Collects per‑engine **waiting/running queue lengths**.
- Publishes stats to front‑ends.
- Tracks and broadcasts **wave state** (pause/resume coordination).

**Notably:** it does **not** perform request routing itself.

---

## 5) KV cache behavior under DP (local vs global reuse)
The KV cache is **local to each DP rank** (each EngineCore process). vLLM does **not** share KV cache across DP ranks.

### 5.1 Prefix cache is still per‑rank
Prefix caching uses block hashes derived from:
- token IDs
- optional **extra keys** (LoRA name, multimodal hashes, cache_salt, prompt embeds)

These hashes are computed in `vllm/v1/core/kv_cache_utils.py`, which means **prefix reuse happens only inside the rank that owns the cache**.

### 5.2 Practical effect
- If a multi‑turn chat is routed to a **different DP rank**, it loses KV/prefix reuse.
- **Sticky rank routing** is required for KV hits.

---

## 6) Practical implication: KV‑aware routing requires explicit rank
Because each DP rank owns its own KV cache, **KV reuse only happens with rank affinity**. vLLM does **not** implement KV‑aware routing; it only offers the header hook.

**Conclusion:** if you want KV hits, you must set **`X-data-parallel-rank`** at ingress (or via an external router) to enforce sticky rank routing.

---

## 7) Practical routing patterns (sticky rank)
### 7.1 Stateless hash routing
- Compute a deterministic rank per chat/session:
  - `rank = hash(chat_id) % dp_size`
- Inject `X-data-parallel-rank` on every turn.

**Pros:** simplest, no storage.
**Cons:** if `dp_size` changes, many sessions remap → KV misses.

### 7.2 Stateful affinity (map + TTL)
- Maintain a **chat_id → rank** map (e.g., Redis or LRU with TTL).
- On first request, choose a rank (hash or least‑loaded), store it.
- Reuse that rank for all subsequent turns until TTL expires.

**Pros:** stable across `dp_size` changes (within TTL).
**Cons:** needs storage + failure handling.

### 7.3 Failure handling
- If a rank dies → reassign (KV miss) or fail fast.
- Short TTLs reduce stale affinity when topology changes.

---

## 8) History & motivation (high-level)
**Verified sources explicitly using “DP attention”:**
- Ray Serve “Data parallel attention” pattern
  - https://docs.ray.io/en/latest/serve/llm/architecture/serving-patterns/data-parallel.html
- vLLM data-parallel deployment docs
  - https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/
- vLLM issue: “Plan to support DP attention for Deepseek models”
  - https://github.com/vllm-project/vllm/issues/12871
- vLLM RFC: “Data Parallel Attention and Expert Parallel MoEs”
  - https://github.com/vllm-project/vllm/issues/16037

**Motivation (consistent across sources):**
- MLA/MoE models (e.g., DeepSeek) suffer KV‑cache duplication under TP because MLA has few KV heads.
- DP attention shards **requests**, letting each DP rank keep its **own KV cache**, avoiding TP duplication.
- Works best with EP for MoE layers; requires coordination across ranks.

**Timeline (explicit-term sources):**
- **Dec 2024:** SGLang v0.4 blog introduces “Data Parallelism Attention” (not yet directly verified here).
- **Early 2025:** vLLM opens issue #12871 (DP attention for DeepSeek).
- **Apr 2025:** vLLM RFC #16037 formalizes DP attention + EP.
- **Mid‑2025:** vLLM + Ray docs publish the serving pattern.

---

## Sources (code)
- `vllm/entrypoints/openai/engine/serving.py`
- `vllm/entrypoints/openai/completion/serving.py`
- `vllm/entrypoints/openai/chat_completion/serving.py`
- `vllm/v1/engine/__init__.py`
- `vllm/v1/engine/core_client.py`
- `vllm/v1/engine/coordinator.py`
