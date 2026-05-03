# vLLM Code Structure (`vllm/`)

Analyzing version **v0.16.0** (source available in `code/` submodule).

~471K lines of Python, ~230 model implementations.

## Top-Level Layout

```
vllm/
├── entrypoints/       # User-facing APIs (HTTP server, offline LLM class)
├── v1/                # New architecture — THE active engine
├── engine/            # Legacy engine (being replaced by v1)
├── model_executor/    # Model definitions & weight loading (largest: ~500 files)
├── distributed/       # Tensor/pipeline/expert parallelism
├── config/            # Configuration classes
├── compilation/       # torch.compile integration, cudagraph
├── attention/         # Attention backends
├── lora/              # LoRA adapter support
├── multimodal/        # Vision/audio input handling
├── tool_parsers/      # Tool/function calling parsers
└── reasoning/         # Reasoning/chain-of-thought support
```

## Key Layers (top to bottom)

### 1. Entrypoints (`entrypoints/`)

- `api_server.py` — main FastAPI server (OpenAI-compatible)
- `llm.py` — offline/batch `LLM` class for programmatic use
- `openai/` — OpenAI-compatible API protocol handlers
- `launcher.py` — process launcher

### 2. Engine (`v1/engine/`) — the brain

- `async_llm.py` — async engine for the API server
- `llm_engine.py` — synchronous engine for offline use
- `core.py` — the **core engine loop** (scheduling + dispatching)
- `input_processor.py` — tokenization, multimodal preprocessing
- `detokenizer.py` — incremental detokenization of outputs
- `output_processor.py` — post-processing of model outputs

### 3. Scheduler (`v1/core/`)

- `sched/` — the scheduler (decides which requests to batch)
- `kv_cache_manager.py` — KV cache block allocation
- `block_pool.py` — physical block management

### 4. Executor (`v1/executor/`)

- `uniproc_executor.py` — single-GPU execution
- `multiproc_executor.py` — multi-GPU via multiprocessing
- `ray_executor.py` — multi-node via Ray

### 5. Worker (`v1/worker/`)

- `gpu_worker.py` — per-GPU worker process
- `gpu_model_runner.py` — runs the actual model forward pass
- `gpu_input_batch.py` — batching/padding input tensors
- `block_table.py` — maps logical to physical KV cache blocks

### 6. Model Executor (`model_executor/`)

- `models/` — ~230 model implementations (deepseek, llama, qwen, etc.)
- `layers/` — reusable layers (linear, attention, MoE, quantization)
- `model_loader/` — weight loading from HuggingFace checkpoints

## Request Flow

```
User HTTP request
  → entrypoints/api_server.py
    → v1/engine/async_llm.py (add_request)
      → v1/engine/core.py (schedule + execute)
        → v1/core/sched/ (pick requests for this batch)
        → v1/executor/ (dispatch to GPU workers)
          → v1/worker/gpu_worker.py (execute_model)
            → v1/worker/gpu_model_runner.py (forward pass)
              → model_executor/models/<model>.py (actual model)
        → v1/engine/output_processor.py (process logits → tokens)
      → v1/engine/detokenizer.py (tokens → text)
    → stream response back to user
```

## Notable Subsystems

- `v1/spec_decode/` — speculative decoding (draft model predicts, main model verifies)
- `compilation/` — torch.compile with custom backends, cudagraph capture
- `lora/` — dynamic LoRA adapter loading/swapping
- `distributed/` — tensor parallel, pipeline parallel, expert parallel for MoE
- `multimodal/` — vision encoders, audio processing

## Note

The codebase is in transition from the legacy `engine/` to `v1/`. The `v1/` architecture is the active one with cleaner separation between scheduling, execution, and model running.
