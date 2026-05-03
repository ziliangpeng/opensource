# vLLM OpenAI API Server Deep Dive (`v0.16.0`)

This note focuses on the API server process itself: startup, app wiring, request handling, streaming, and shutdown behavior.

## 1) Startup path

1. CLI entrypoint:
   - `vllm serve` reaches `ServeSubcommand.cmd`.
   - Single-server path runs `uvloop.run(run_server(args))` (`vllm/entrypoints/cli/serve.py:47`, `vllm/entrypoints/cli/serve.py:111`).
2. Server bootstrap:
   - `run_server()` decorates logs and calls `setup_server()` (`vllm/entrypoints/openai/api_server.py:450`, `vllm/entrypoints/openai/api_server.py:456`).
   - `setup_server()` validates parser args, binds socket before engine init, sets startup SIGTERM handler (`vllm/entrypoints/openai/api_server.py:419`, `vllm/entrypoints/openai/api_server.py:421`, `vllm/entrypoints/openai/api_server.py:434`).
3. Worker init:
   - `run_server_worker()` creates engine client context with `build_async_engine_client(...)` (`vllm/entrypoints/openai/api_server.py:460`, `vllm/entrypoints/openai/api_server.py:476`).
   - `build_async_engine_client_from_engine_args()` instantiates `AsyncLLM` and resets MM cache (`vllm/entrypoints/openai/api_server.py:106`, `vllm/entrypoints/openai/api_server.py:137`, `vllm/entrypoints/openai/api_server.py:150`).
4. App init:
   - Query supported tasks from engine (`vllm/entrypoints/openai/api_server.py:480`).
   - Build app and initialize shared state (`vllm/entrypoints/openai/api_server.py:483`, `vllm/entrypoints/openai/api_server.py:484`).
5. HTTP serving:
   - Calls `serve_http(...)` to launch Uvicorn + watchdog (`vllm/entrypoints/openai/api_server.py:491`, `vllm/entrypoints/launcher.py:27`, `vllm/entrypoints/launcher.py:82`).

## 2) FastAPI app composition

`build_app()` wires routers and middleware (`vllm/entrypoints/openai/api_server.py:158`):

- Router registration:
  - basic API routes (`vllm/entrypoints/openai/api_server.py:181`),
  - serve routes (`vllm/entrypoints/openai/api_server.py:185`),
  - models routes (`vllm/entrypoints/openai/api_server.py:189`),
  - sagemaker routes (`vllm/entrypoints/openai/api_server.py:195`),
  - feature routers by task (generate/transcription/realtime/pooling) (`vllm/entrypoints/openai/api_server.py:201`, `vllm/entrypoints/openai/api_server.py:208`, `vllm/entrypoints/openai/api_server.py:215`, `vllm/entrypoints/openai/api_server.py:222`).
- Core middleware:
  - CORS (`vllm/entrypoints/openai/api_server.py:228`),
  - auth if API key configured (`vllm/entrypoints/openai/api_server.py:240`),
  - request-id headers (optional) (`vllm/entrypoints/openai/api_server.py:245`),
  - scaling middleware (`vllm/entrypoints/openai/api_server.py:251`),
  - custom middleware from CLI (`vllm/entrypoints/openai/api_server.py:261`).

Auth nuance:
- Authentication middleware intentionally skips OPTIONS and non-`/v1` paths (`vllm/entrypoints/openai/server_utils.py:39`, `vllm/entrypoints/openai/server_utils.py:74`).

## 3) Shared app state initialization

`init_app_state()` builds the server-side dependency graph (`vllm/entrypoints/openai/api_server.py:277`):

- Sets engine client/config on `app.state` (`vllm/entrypoints/openai/api_server.py:308`).
- Resolves served model names and optional request logger (`vllm/entrypoints/openai/api_server.py:294`, `vllm/entrypoints/openai/api_server.py:299`).
- Builds `OpenAIServingModels`, initializes static LoRAs (`vllm/entrypoints/openai/api_server.py:322`, `vllm/entrypoints/openai/api_server.py:327`).
- Builds tokenization/templating serving object (`vllm/entrypoints/openai/api_server.py:328`).
- Initializes per-task serving state modules (`vllm/entrypoints/openai/api_server.py:338`, `vllm/entrypoints/openai/api_server.py:345`, `vllm/entrypoints/openai/api_server.py:354`, `vllm/entrypoints/openai/api_server.py:359`).
- Initializes server load tracking counters (`vllm/entrypoints/openai/api_server.py:364`).

## 4) Request handling pipeline (chat/completion)

For `POST /v1/chat/completions` (completion is parallel in structure):

1. Route layer:
   - Request is validated as JSON (`vllm/entrypoints/openai/chat_completion/api_router.py:36`, `vllm/entrypoints/openai/utils.py:43`).
   - Wrapped with cancellation and load-aware decorators (`vllm/entrypoints/openai/chat_completion/api_router.py:44`, `vllm/entrypoints/openai/chat_completion/api_router.py:45`).
2. Serving layer:
   - `create_chat_completion()` builds `EngineCoreRequest` and calls `engine_client.generate(...)` (`vllm/entrypoints/openai/chat_completion/serving.py:455`).
3. Stream response:
   - Route returns `StreamingResponse` for generator outputs (`vllm/entrypoints/openai/chat_completion/api_router.py:73`).
   - Stream generator iterates async results and emits SSE frames (`vllm/entrypoints/openai/chat_completion/serving.py:735`, `vllm/entrypoints/openai/chat_completion/serving.py:786`).

The completion endpoint follows the same pattern (`vllm/entrypoints/openai/completion/serving.py:226`, `vllm/entrypoints/openai/completion/serving.py:359`).

## 5) Load/cancellation behavior in API layer

- `with_cancellation` races handler execution against client-disconnect listener (`vllm/entrypoints/utils.py:57`, `vllm/entrypoints/utils.py:89`).
- `load_aware_call` increments/decrements `app.state.server_load_metrics`, including streaming background cleanup (`vllm/entrypoints/utils.py:106`, `vllm/entrypoints/utils.py:123`, `vllm/entrypoints/utils.py:130`).

## 6) Stop checking and request cancellation

Stop checking is split across two processes. The EngineCore (scheduler) handles token-level checks; the API server handles text-level checks.

### What the EngineCore checks (scheduler side)

`check_stop()` in `vllm/v1/core/sched/utils.py:40` runs after every forward step in the scheduler:

- **EOS token**: last generated token == `eos_token_id` → stop.
- **Stop token IDs**: last token in `sampling_params.stop_token_ids` → stop.
- **Max tokens / max model length**: output length or total length reached limit → stop.

These are all token-ID comparisons — no detokenization needed. The scheduler marks the request as finished and frees its KV cache blocks.

### What the API server checks (output_processor side)

`detokenizer.update()` in `vllm/v1/engine/output_processor.py:637` runs during output processing:

- **Stop strings**: text patterns like `"\nQ:"`, `"```"`, `"</tool_call>"`, etc.

This must happen in the API server because stop strings require detokenization first. The same string can be tokenized differently depending on context, and a stop string may span token boundaries. When a stop string is detected, the output processor sets `finish_reason=STOP` and returns the request ID in a `reqs_to_abort` list. The `output_handler` then sends an ABORT to the EngineCore over ZMQ (`vllm/v1/engine/async_llm.py:700`).

### Why stop strings exist (and when they matter)

For modern chat models, turn boundaries are special tokens (`<|im_end|>`, `<|eot_id|>`, etc.) that the model learns to emit during training. The EOS token check in the scheduler handles this — stop strings are not needed for normal chat.

Stop strings exist for non-chat and legacy use cases:

- **Raw completion API** (`/v1/completions`): no structured messages, no special tokens. Few-shot prompting with base models uses text conventions (e.g., `stop=["\nQ:"]`).
- **Older models**: pre-chat-template models (GPT-2, early LLaMA base) used text markers like `\n\nHuman:` for turn boundaries.
- **Custom structured output**: stopping at `}` for JSON, `\n\`\`\`` for code blocks, etc.
- **Tool/agent frameworks**: some use text markers like `</tool_call>` that aren't single tokens in the vocabulary.

### Cancellation flow (client disconnect)

1. `with_cancellation` decorator races the handler against a client-disconnect listener (`vllm/entrypoints/utils.py:57`).
2. On disconnect, `generate()` receives `CancelledError` (`vllm/v1/engine/async_llm.py:601`).
3. `abort()` sends an ABORT message over ZMQ to the EngineCore (`vllm/v1/engine/async_llm.py:603`).
4. EngineCore calls `scheduler.finish_requests(request_ids, FINISHED_ABORTED)` (`vllm/v1/engine/core.py:327`).
5. The request is excluded from the next scheduling step and its KV blocks are freed.

Note: this does not interrupt a GPU forward pass already in flight — CUDA kernels can't be stopped mid-execution. It prevents the request from being scheduled in future steps.

### Summary table

| Stop condition | Detected by | Needs detokenization? |
|---|---|---|
| EOS token | EngineCore (scheduler) | No |
| Stop token IDs | EngineCore (scheduler) | No |
| Max tokens / max model len | EngineCore (scheduler) | No |
| Stop strings (text) | API server (output_processor) | Yes |
| Client disconnect | API server (cancellation) | No |

## 7) Uvicorn, watchdog, and shutdown

- `serve_http()` logs routes, builds `uvicorn.Server`, and starts:
  - Uvicorn serve task,
  - watchdog task polling engine health (`vllm/entrypoints/launcher.py:38`, `vllm/entrypoints/launcher.py:83`, `vllm/entrypoints/launcher.py:128`).
- If engine is dead and keep-alive is disabled, watchdog requests server exit (`vllm/entrypoints/launcher.py:148`, `vllm/entrypoints/launcher.py:150`).
- Exception handlers are registered for runtime/engine exceptions to return 500 and trigger termination logic (`vllm/entrypoints/launcher.py:153`, `vllm/entrypoints/launcher.py:179`).

## 8) Multi-API-server mode (brief)

When `--api-server-count > 1`, `run_multi_api_server()` launches shared core engines and multiple API server worker processes (`vllm/entrypoints/cli/serve.py:217`, `vllm/entrypoints/cli/serve.py:248`, `vllm/entrypoints/cli/serve.py:272`).

## 9) Clarification: when ZMQ is used (and async vs sync)

This section clarifies behavior discussed during code reading.

- In v1, the engine client selector is:
  - `AsyncMPClient` (ZMQ + background process) for async+mp.
  - `SyncMPClient` (ZMQ + background process) for sync+mp.
  - `InprocClient` (no ZMQ, in-process) for sync+non-mp.
  - (`vllm/v1/engine/core_client.py:75`, `vllm/v1/engine/core_client.py:94`, `vllm/v1/engine/core_client.py:97`, `vllm/v1/engine/core_client.py:266`).
- Async without multiprocessing is currently unsupported (`vllm/v1/engine/core_client.py:83`).
- API server path uses `AsyncLLM`, which directly builds `make_async_mp_client(...)`, so API-server async path is ZMQ-based (`vllm/v1/engine/async_llm.py:148`).
- For sync library usage (`LLMEngine`), default from `from_engine_args` is `enable_multiprocessing=False`, so default is in-process unless explicitly enabled (or env turns it on) (`vllm/v1/engine/llm_engine.py:161`, `vllm/v1/engine/llm_engine.py:169`, `vllm/v1/engine/llm_engine.py:180`).

Practical takeaway:
- Async path: ZMQ.
- Sync path: default no ZMQ (in-proc), optional ZMQ when multiprocessing is enabled.

## 10) Clarification: token vs message granularity

The transport is message-oriented; token granularity is not fixed 1:1.

- `EngineCoreOutput` carries `new_token_ids: list[int]` for each request, so one request output can include multiple tokens (`vllm/v1/engine/__init__.py:137`).
- One `EngineCoreOutputs` message can include outputs for multiple requests (`outputs: list[...]`) (`vllm/v1/engine/__init__.py:188`, `vllm/v1/core/sched/scheduler.py:1461`).
- Multi-token-per-request in one step can happen in speculative decoding acceptance paths (scheduler processes variable-length `generated_token_ids`) (`vllm/v1/core/sched/scheduler.py:1298`, `vllm/v1/core/sched/scheduler.py:1305`).
- API-visible streaming can further aggregate due to `stream_interval` buffering (`vllm/config/scheduler.py:136`, `vllm/v1/engine/output_processor.py:285`).
- Delta outputs may also be merged when producer outpaces consumer (`vllm/v1/engine/output_processor.py:50`).

Practical takeaway:
- `1 token == 1 message` is only a common case, not a protocol guarantee.
