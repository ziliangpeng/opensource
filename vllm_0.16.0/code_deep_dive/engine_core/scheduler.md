# vLLM Scheduler Deep Dive (`v0.16.0`)

This note covers the scheduler: how it decides which requests to run, how many tokens each gets, how it handles memory pressure, and how it produces the `SchedulerOutput` consumed by the executor.

Key source: `vllm/v1/core/sched/scheduler.py` (~2172 lines).

## 1) What the scheduler does

The scheduler runs once per busy loop iteration. Its job is to produce a `SchedulerOutput` — essentially a dictionary of `{req_id: num_tokens}` that tells the model runner exactly which requests to include in this forward pass and how many tokens to process for each.

The key design principle is stated in the code (line 322):

> There's no "decoding phase" nor "prefill phase". Each request just has `num_computed_tokens` and `num_tokens_with_spec`. The scheduler tries to assign tokens so each request's `num_computed_tokens` catches up to `num_tokens_with_spec`.

This unified view means the same scheduling algorithm naturally handles:

- **Full prefills**: a new request with 1000 prompt tokens gets `num_new_tokens = 1000`.
- **Chunked prefills**: the token budget caps `num_new_tokens` at, say, 512, so the request gets scheduled across two steps.
- **Decoding**: a running request with one new token gets `num_new_tokens = 1`.
- **Speculative decoding**: a request with 5 spec tokens gets `num_new_tokens = 6` (1 real + 5 spec).
- **Prefix cache hits**: if 800 of 1000 prompt tokens are cached, `num_new_tokens = 200`.

## 2) Data structures

### Two queues

```python
self.waiting: RequestQueue   # FCFS deque or priority heap
self.running: list[Request]  # plain list
```

`waiting` holds requests not yet assigned to the GPU. `running` holds requests actively generating tokens. There is no separate "swapped" queue — preempted requests go back into `waiting`.

### Request lifecycle through the scheduler

```
WAITING ──────────────→ RUNNING ──→ FINISHED_*
   ↑                       │
   └── PREEMPTED ←─────────┘
```

Special intermediate statuses that keep a request in `waiting`:

- `WAITING_FOR_FSM` — structured output grammar still compiling (async in input thread).
- `WAITING_FOR_REMOTE_KVS` — P/D disaggregation, waiting for remote KV transfer.
- `WAITING_FOR_STREAMING_REQ` — streaming input session, waiting for next input chunk.

### Request queue implementations (`request_queue.py`)

Two policies, selected via `scheduler_config.policy`:

- **`FCFSRequestQueue`**: inherits from `deque[Request]`. FIFO ordering. `add_request` appends to the right, `pop_request` pops from the left.
- **`PriorityRequestQueue`**: wraps a min-heap. Requests are ordered by `(priority, arrival_time)` — lower priority value = higher priority. `prepend_request` just does `heappush` (no concept of "front" in a heap).

### Token budgets

Two constraints control batch size:

- **`max_num_scheduled_tokens`** (from `scheduler_config.max_num_batched_tokens`) — total tokens across all requests in one step.
- **`max_num_running_reqs`** (from `scheduler_config.max_num_seqs`) — maximum concurrent requests.

Additionally, for multimodal models:

- **`max_num_encoder_input_tokens`** — budget for encoder (vision) tokens per step.

## 3) The `schedule()` method

This is the core algorithm (~575 lines, `scheduler.py:321–896`). It runs in two phases.

### Phase 1: Schedule RUNNING requests (lines 352–517)

Iterates through `self.running` and decides how many tokens each already-running request gets:

```python
req_index = 0
while req_index < len(self.running) and token_budget > 0:
    request = self.running[req_index]
    num_new_tokens = (
        request.num_tokens_with_spec
        + request.num_output_placeholders
        - request.num_computed_tokens
    )
    # ... cap by budget, max_model_len, long_prefill_threshold ...
    # ... allocate KV blocks ...
```

For a typical decode step, `num_new_tokens = 1`. For a chunked prefill continuation, it could be larger.

**When allocation fails — preemption** (lines 429–473):

If `kv_cache_manager.allocate_slots()` returns `None` (not enough free blocks), the scheduler enters a preemption loop:

```python
while True:
    new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens, ...)
    if new_blocks is not None:
        break  # success

    # Evict lowest-priority request
    if self.policy == SchedulingPolicy.PRIORITY:
        preempted_req = max(self.running, key=lambda r: (r.priority, r.arrival_time))
    else:  # FCFS
        preempted_req = self.running.pop()  # last = most recent

    self._preempt_request(preempted_req, ...)
    if preempted_req == request:
        break  # evicted ourselves, give up
```

Preemption is **full eviction**: all KV cache blocks are freed, `num_computed_tokens` resets to 0, and the request goes back to the front of `waiting`. On rescheduling, prefix caching can recover some of the freed blocks if they haven't been overwritten.

**Key detail**: if a request gets `num_new_tokens = 0` (e.g., all tokens already scheduled in a PP setup, or encoder budget exhausted), the scheduler does `continue` — not `break`. This means it **skips** that request but keeps trying lower-priority requests. This relaxes strict FCFS ordering for running requests.

### Phase 2: Schedule WAITING requests (lines 533–802)

Only runs if Phase 1 produced **zero preemptions**. This prevents oscillation: if we just evicted requests due to memory pressure, don't immediately try to add new ones.

```python
if not preempted_reqs:
    while self.waiting and token_budget > 0:
        if len(self.running) == self.max_num_running_reqs:
            break
        request = self.waiting.peek_request()
        # ... various skip checks ...
        # ... get cached blocks, allocate new blocks ...
        # ... move request from waiting to running ...
```

**Skip conditions** checked before scheduling a waiting request:

1. `WAITING_FOR_REMOTE_KVS` — check if KV transfer is done; if not, skip and re-queue.
2. `WAITING_FOR_FSM` — check if grammar is compiled; if not, skip and re-queue.
3. `WAITING_FOR_STREAMING_REQ` — skip and re-queue.
4. LoRA limit — if scheduling this request would exceed `max_loras`, skip.

Skipped requests are collected in a temporary `skipped_waiting_requests` queue and prepended back to the front of `waiting` at the end, preserving their ordering.

**Prefix cache lookup** (line 600–601):

```python
new_computed_blocks, num_new_local_computed_tokens = (
    self.kv_cache_manager.get_computed_blocks(request)
)
```

This checks how many of the request's prompt tokens are already in the KV cache (from prefix caching). These tokens don't need to be recomputed, reducing `num_new_tokens`.

**KV connector (P/D disaggregation)** (lines 605–626):

If a KV connector is configured, the scheduler also checks for externally cached tokens on a remote node. The total computed tokens = local cached + external cached. If loading is async, the request enters `WAITING_FOR_REMOTE_KVS`.

**Chunked prefill gating** (lines 659–665):

```python
if (not self.scheduler_config.enable_chunked_prefill
        and num_new_tokens > token_budget):
    break  # can't chunk, so stop scheduling entirely
```

If chunked prefill is disabled and the request doesn't fit in the remaining budget, scheduling stops. If enabled, the request is chunked to fit.

**Block allocation** (lines 713–722):

```python
new_blocks = self.kv_cache_manager.allocate_slots(
    request, num_new_tokens,
    num_new_computed_tokens=num_new_local_computed_tokens,
    new_computed_blocks=new_computed_blocks,
    num_lookahead_tokens=effective_lookahead_tokens,
    num_external_computed_tokens=num_external_computed_tokens,
    ...
)
```

If allocation fails for a waiting request, scheduling simply stops (`break`) — no preemption of other waiting requests.

### Post-scheduling (lines 804–896)

After both phases:

1. **Common prefix blocks** — computes the longest common prefix among all running requests (for cascade attention optimization).
2. **Builds `SchedulerOutput`** — packages new request data, cached request data, scheduled tokens, encoder inputs, finished request IDs.
3. **KV connector metadata** — builds opaque metadata for the connector (KV store plans, etc.).
4. **`_update_after_schedule()`** — advances `num_computed_tokens` for all scheduled requests, frees encoder inputs that have been fully consumed.

## 4) SchedulerOutput structure

The scheduler produces a `SchedulerOutput` (`output.py`) that tells the model runner everything it needs:

```python
@dataclass
class SchedulerOutput:
    scheduled_new_reqs: list[NewRequestData]      # first-time requests (full data)
    scheduled_cached_reqs: CachedRequestData      # returning requests (diff only)
    num_scheduled_tokens: dict[str, int]           # req_id → token count
    total_num_scheduled_tokens: int
    scheduled_spec_decode_tokens: dict[str, list[int]]
    scheduled_encoder_inputs: dict[str, list[int]]
    num_common_prefix_blocks: list[int]
    finished_req_ids: set[str]                     # for workers to free cached state
    free_encoder_mm_hashes: list[str]
    ...
```

**`NewRequestData`** contains full request info (prompt token IDs, multimodal features, sampling params, block IDs, num computed tokens, LoRA request). Sent once when a request first enters the running queue.

**`CachedRequestData`** contains only the diff for returning requests (new block IDs, new token IDs for PP, num computed tokens). This minimizes communication between the EngineCore and workers.

The `prev_step_scheduled_req_ids` optimization: for requests scheduled in consecutive steps, the scheduler skips sending `all_token_ids` (only needed when a request wasn't scheduled in the prior step and the worker needs the full context).

## 5) `update_from_output()` — post-GPU processing

After the model forward pass, `update_from_output()` processes the results (`scheduler.py:1239–1492`). This runs in the main thread, in the same busy loop iteration.

For each scheduled request:

1. **Speculative decoding reconciliation**: if spec tokens were used, compares generated tokens to drafts. Adjusts `num_computed_tokens` by the number of rejected tokens.
2. **Token appending and stop checking**: calls `_update_request_with_output()` which iterates through generated tokens, appending each and checking stop conditions.
3. **Stop handling**: if stopped, calls `_handle_stopped_request()` — either fully finishes the request or re-queues it (for streaming sessions).
4. **Resource cleanup**: `_free_request()` frees KV cache blocks, encoder cache, adds to `finished_req_ids`.
5. **Grammar advancement**: for structured output, calls `grammar.accept_tokens()` to advance the FSM state.
6. **Output construction**: builds `EngineCoreOutput` per request, grouped by `client_index` into `EngineCoreOutputs`.

After the loop, stopped requests are batch-removed from `self.running` (optimized: single-item removal avoids list comprehension).

## 6) Stop condition checking (`utils.py`)

```python
def check_stop(request: Request, max_model_len: int) -> bool:
    if request.num_output_tokens < sampling_params.min_tokens:
        return False
    if not sampling_params.ignore_eos and last_token_id == request.eos_token_id:
        request.status = RequestStatus.FINISHED_STOPPED
        return True
    if last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = RequestStatus.FINISHED_STOPPED
        return True
    if request.num_tokens >= max_model_len or request.num_output_tokens >= request.max_tokens:
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True
    return False
```

Checks in order: min tokens gate → EOS → stop token IDs → length caps.

**Important**: string-based stop conditions (`stop` strings in sampling params) are NOT checked here. Those require detokenization and are checked in the API server process.

## 7) Preemption details

Preemption (`_preempt_request`, line 898):

```python
def _preempt_request(self, request, timestamp):
    self.kv_cache_manager.free(request)
    self.encoder_cache_manager.free(request)
    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = 0
    request.num_preemptions += 1
    self.waiting.prepend_request(request)
```

Key characteristics:

- **Full eviction** — all blocks freed, computed token count reset to zero. No partial preemption or swap-to-CPU.
- **Prepended to waiting** — goes to the front of the queue so it will be rescheduled first.
- **Prefix cache recovery** — on rescheduling, `get_computed_blocks()` may find some blocks still in cache (not yet evicted by other requests), reducing recomputation.
- **No preemption during Phase 2** — if block allocation fails for a waiting request, scheduling simply stops. Only running requests can trigger preemption of other running requests.

## 8) AsyncScheduler

`AsyncScheduler` (`async_scheduler.py`) is a thin subclass (~60 lines) used when `max_concurrent_batches > 1` (pipelined execution). The key difference: scheduling happens **before** the previous step's results are known.

**`_update_after_schedule` override**: after scheduling, adds "output placeholders" to each decode request:

```python
request.num_output_placeholders += 1 + cur_num_spec_tokens
```

These placeholders represent tokens the model will generate but haven't been produced yet. The scheduler accounts for them when computing `num_new_tokens`, preventing double-scheduling.

**`_update_request_with_output` override**: when results arrive:

- Decrements `num_output_placeholders` by the number of actual tokens received.
- Calls `kv_cache_manager.cache_blocks()` to progressively cache blocks as tokens are confirmed (instead of caching all at once after scheduling).
- Handles `discard_latest_async_tokens` flag for forced preemption during `reset_prefix_cache`.

**Early termination optimization** (line 355–369 in `schedule()`): if a request's computed tokens + output placeholders would exceed `max_tokens`, the scheduler skips it to avoid scheduling an unnecessary extra step.

## 9) Encoder input scheduling

For multimodal models, the scheduler manages encoder (vision) inputs alongside decoder tokens (`_try_schedule_encoder_inputs`, line 1057–1213).

The logic determines which encoder inputs (images/video) need processing in the current step:

- An encoder input is needed if its placeholder tokens overlap with the tokens being computed in this step.
- The encoder has its own compute budget (`encoder_compute_budget`) and cache budget.
- If an encoder input can't be scheduled (budget or cache full), `num_new_tokens` is adjusted to stop just before that encoder input's placeholder position.
- The `encoder_cache_manager` tracks which encoder outputs are cached, preventing redundant computation.
- After an encoder output's placeholder tokens have all been processed by the decoder, `_free_encoder_inputs` releases it from the encoder cache.

## 10) Request add/finish/abort

**`add_request()`** (line 1642): handles both new requests and streaming session continuations. For new requests, adds to `waiting` queue and `self.requests` dict. For existing streaming sessions, either queues the next input chunk or commences it immediately.

**`finish_requests()`** (line 1664): handles external finish signals (client disconnect, API server abort, stop string detected). Batch-removes from both `running` and `waiting` queues, then frees resources for each request.

**`_free_request()`** (line 1716): frees encoder cache, adds to `finished_req_ids` (so workers know to clean up), and frees KV blocks (unless delayed for async KV transfer).

## 11) Structured output integration

The scheduler interacts with structured output (constrained decoding) at two points:

1. **Scheduling gate**: requests in `WAITING_FOR_FSM` are skipped until grammar compilation finishes (done async in the input thread).
2. **Grammar bitmask** (`get_grammar_bitmask`, line 1215): after scheduling but before sampling, collects requests that use structured output and produces a token bitmask via `structured_output_manager.grammar_bitmask()`. This bitmask is applied during sampling to restrict which tokens the model can output.
3. **Grammar advancement** (in `update_from_output`): after tokens are generated, `grammar.accept_tokens()` advances the FSM state.
4. **Draft token validation** (in `update_draft_token_ids`): for speculative decoding, `grammar.validate_tokens()` filters draft tokens that violate the grammar.

## 12) Class hierarchy

```
SchedulerInterface (ABC, interface.py)
└── Scheduler (scheduler.py)
    └── AsyncScheduler (async_scheduler.py)
```

`SchedulerInterface` defines the contract: `schedule()`, `update_from_output()`, `add_request()`, `finish_requests()`, `get_grammar_bitmask()`, etc. The base `Scheduler` implements the full algorithm. `AsyncScheduler` overrides only `_update_after_schedule` and `_update_request_with_output` for pipelined execution.
