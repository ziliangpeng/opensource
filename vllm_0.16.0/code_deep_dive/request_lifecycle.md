# vLLM Request Lifecycle: Components and Processes (`v0.16.0`)

Scope for this note:
- OpenAI-compatible API server path.
- v1 engine path.
- Data parallel excluded for clarity.

## Process and component map

- API server process:
  - Runs FastAPI/Uvicorn.
  - Owns route handlers and OpenAI serving adapters.
  - Creates the engine client during startup (`vllm/entrypoints/openai/api_server.py:476`).
- Async engine frontend (same API process):
  - `AsyncLLM` object (`vllm/v1/engine/async_llm.py:71`).
  - Holds `AsyncMPClient` and `OutputProcessor` (`vllm/v1/engine/async_llm.py:148`).
  - Owns per-request output collector queues (`vllm/v1/engine/output_processor.py:45`).
- EngineCore process (separate process):
  - Launched by `launch_core_engines` via `CoreEngineProcManager` (`vllm/v1/engine/core_client.py:490`, `vllm/v1/engine/utils.py:907`).
  - Runs scheduler + execution busy loop (`vllm/v1/engine/core.py:1018`).
- Model worker processes (typically one per GPU rank under TP/PP):
  - Managed by executor inside EngineCore.
  - Do actual model forward/sample compute.

## IPC map (API process <-> EngineCore process)

- Request channel:
  - Frontend binds ZMQ `ROUTER` (`vllm/v1/engine/core_client.py:507`).
  - Engine connects as `DEALER` (`vllm/v1/engine/core.py:1156`).
- Output channel:
  - Engine sends on ZMQ `PUSH` (`vllm/v1/engine/core.py:1242`).
  - Frontend receives on ZMQ `PULL` (`vllm/v1/engine/core_client.py:510`).
- Payload format:
  - msgpack-encoded `EngineCoreOutputs` (`vllm/v1/engine/__init__.py:176`, `vllm/v1/engine/core.py:1279`).

## End-to-end lifecycle for one request

1. HTTP request reaches route handler and stream-mode path returns `StreamingResponse` (`vllm/entrypoints/openai/chat_completion/api_router.py:46`, `vllm/entrypoints/openai/chat_completion/api_router.py:73`).
2. OpenAI serving layer validates/transforms request and builds `EngineCoreRequest` (`vllm/entrypoints/openai/chat_completion/serving.py:455`).
3. Serving calls `engine_client.generate(...)` (`vllm/entrypoints/openai/chat_completion/serving.py:455`).
4. `AsyncLLM.generate()` registers request state and sends add-request to engine client (`vllm/v1/engine/async_llm.py:537`, `vllm/v1/engine/async_llm.py:571`).
5. `AsyncMPClient` serializes and sends ADD request via ZMQ (`vllm/v1/engine/core_client.py:913`, `vllm/v1/engine/core_client.py:922`).
6. Engine input thread receives/decode/enqueues request (`vllm/v1/engine/core.py:1139`, `vllm/v1/engine/core.py:1195`, `vllm/v1/engine/core.py:1218`).
7. Engine busy loop schedules and executes model work (`vllm/v1/engine/core.py:1018`, `vllm/v1/engine/core.py:1059`).
8. Scheduler builds `EngineCoreOutput` entries grouped per frontend client (`vllm/v1/core/sched/scheduler.py:1239`, `vllm/v1/core/sched/scheduler.py:1394`, `vllm/v1/core/sched/scheduler.py:1461`).
9. Engine output thread encodes and sends `EngineCoreOutputs` over ZMQ (`vllm/v1/engine/core.py:1220`, `vllm/v1/engine/core.py:1279`).
10. Frontend `AsyncMPClient` receives/decode outputs and puts them into async queue (`vllm/v1/engine/core_client.py:873`, `vllm/v1/engine/core_client.py:878`, `vllm/v1/engine/core_client.py:891`).
11. `AsyncLLM` output handler converts core outputs to `RequestOutput` and pushes per-request collectors (`vllm/v1/engine/async_llm.py:647`, `vllm/v1/engine/async_llm.py:681`).
12. Request coroutine drains collector and yields response chunks (`vllm/v1/engine/async_llm.py:586`, `vllm/v1/engine/async_llm.py:596`).
13. OpenAI stream generator emits SSE events (`data: ...`) to client (`vllm/entrypoints/openai/chat_completion/serving.py:735`, `vllm/entrypoints/openai/chat_completion/serving.py:786`).

## Streaming granularity (important)

- ZMQ message != one token.
- `EngineCoreOutput` has `new_token_ids: list[int]` (can be multiple tokens) (`vllm/v1/engine/__init__.py:137`).
- `EngineCoreOutputs.outputs` can carry multiple requests in one message (`vllm/v1/engine/__init__.py:188`).
- `stream_interval` can intentionally buffer tokens before user-visible stream output (`vllm/config/scheduler.py:136`, `vllm/v1/engine/output_processor.py:285`).
- Delta outputs can be merged when producer outpaces consumer (`vllm/v1/engine/output_processor.py:50`).

## Cancellation and failure path

- Route handlers are wrapped by cancellation-aware decorator (`vllm/entrypoints/utils.py:57`).
- On cancellation/disconnect, `AsyncLLM.generate()` aborts the in-flight engine request (`vllm/v1/engine/async_llm.py:601`, `vllm/v1/engine/async_llm.py:603`).
- Engine-dead sentinel is propagated as `EngineDeadError` (`vllm/v1/engine/core_client.py:436`, `vllm/v1/engine/core_client.py:437`).
