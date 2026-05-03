# GPU Worker & Model Runner (vLLM v0.16.0, V1)

## Scope
This note covers the **GPU Worker → V1 GPUModelRunner** execution path in vLLM v0.16.0, specifically the flow **after scheduler output** arrives at the worker and **before kernels execute**, including KV connector hooks and CUDA graph dispatch. It assumes the earlier doc *scheduler-output-to-kernel.md* for the scheduler→kernel input translation.

> **V1 vs V2**: V1 model runner is `vllm/v1/worker/gpu_model_runner.py`. V2 lives in `vllm/v1/worker/gpu/model_runner.py`. `gpu_worker.py` selects based on `VLLM_USE_V2_MODEL_RUNNER`.

---

## Entry points (GPU Worker)
**File:** `vllm/v1/worker/gpu_worker.py`

### `Worker.execute_model(scheduler_output)`
- **Receives** `SchedulerOutput` from EngineCore.
- **Pipeline Parallel (PP):**
  - Non‑first rank receives `IntermediateTensors` from previous stage.
  - Non‑last rank returns `IntermediateTensors` instead of final output.
- **Delegation:** calls `self.model_runner.execute_model(scheduler_output, intermediate_tensors)`.

### Initialization path
- `init_device()` selects model runner (V1 vs V2) and initializes distributed environment.
- `initialize_from_config(kv_cache_config)` allocates KV cache and initializes KV transfer.
- `compile_or_warm_up_model()` runs dummy passes and captures CUDA graphs if enabled.

---

## V1 GPUModelRunner execution flow
**File:** `vllm/v1/worker/gpu_model_runner.py`

### High-level flow inside `GPUModelRunner.execute_model(...)`
1. **State updates**
   - `finish_requests`, `free_states`, `add_requests`, `update_requests`.
   - `block_tables.apply_staged_writes()` applies KV block table updates.
   - If no scheduled tokens → return `kv_connector.no_forward(...)` (fast exit).

2. **CUDA graph decision**
   - Uses a *cudagraph dispatcher* to select mode and padding size.
   - If eligible, runs captured graph; otherwise falls back to eager path.

3. **Input preparation**
   - `prepare_inputs(...)` builds **InputBatch**:
     - `input_ids`, `positions`, `query_start_loc`, `seq_lens`.
     - `block_tables` + `slot_mappings`.
     - Speculative decode expansions, LoRA activation, multimodal embeddings.

4. **KV Connector hooks** (if enabled)
   - `kv_connector.pre_forward(scheduler_output)`
   - Forward pass
   - `kv_connector.post_forward(scheduler_output, wait_for_save=True)`

5. **Forward pass**
   - `set_forward_context(...)` wraps execution.
   - Calls model forward with attention metadata + KV caches.

6. **Sampling + output packaging**
   - `sample_tokens(...)` → logits → sampling.
   - Builds `ModelRunnerOutput` (or `AsyncModelRunnerOutput`).
   - Includes `kv_connector_output` if available.

---

## Input batch + KV cache mechanics (V1)
- **Persistent batch**: request states are retained across iterations for efficiency.
- `prepare_inputs(...)` builds:
  - **Slot mapping** (token→KV slot index).
  - **Block tables** for KV cache groups.
  - **Attention metadata** via `build_attn_metadata(...)`.
- KV updates are done via **block table staged writes** and **slot mapping**; actual KV writes happen inside attention kernels.

---

## KV Connector integration (V1)
**Files:**
- `vllm/v1/worker/gpu/kv_connector.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/*`

**Behavior:**
- Initialized during `initialize_from_config(...)` via `ensure_kv_transfer_initialized(...)`.
- `pre_forward(...)`:
  - Applies preemption handling.
  - Binds `scheduler_output.kv_connector_metadata`.
  - Starts async remote KV load into local slots.
- `post_forward(...)`:
  - Waits for save operations.
  - Collects finished/invalid blocks + stats.
  - Returns `KVConnectorOutput` attached to `ModelRunnerOutput`.

If no transfer group is active, a **NO‑OP connector** is used.

---

## CUDA graph path (V1)
- **Capture** happens during warmup (`compile_or_warm_up_model → capture_model`).
- **Dispatch** uses a batch descriptor (num tokens, num reqs, uniform decode, LoRA, etc.).
- **Modes:** full vs piecewise (variable lengths). If not eligible, falls back to eager.
- Graph replay reduces Python overhead and can significantly improve decode throughput.

---

## Outputs + PP behavior
- Last PP rank returns `ModelRunnerOutput`.
- Non‑last rank returns `IntermediateTensors` → forwarded to next stage.
- Outputs include sampled token IDs, logprobs (optional), pooling outputs (if pooling model), and KV connector output (if active).

---

## Key design notes
- **Persistent batch** minimizes reallocation and improves throughput.
- **Slot‑mapped KV cache** enables paged attention with efficient reuse.
- **CUDA graphs** trade capture cost for faster steady‑state decode.
- **KV connector** enables disaggregated prefill/decode and remote KV scenarios with minimal changes to the forward path.

---

## Files to read (V1)
- `vllm/v1/worker/gpu_worker.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/worker/gpu/input_batch.py`
- `vllm/v1/worker/gpu/kv_connector.py`
- `vllm/v1/worker/gpu/cudagraph_utils.py`
- `vllm/v1/worker/gpu/attn_utils.py`

