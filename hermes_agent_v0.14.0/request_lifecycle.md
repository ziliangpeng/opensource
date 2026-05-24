# Request Lifecycle — `hermes -z "hello"`

> **Phase 2** hot path trace: one-shot mode, simplest possible request.
>
> Source: upstream `v2026.5.16` (tag `v2026.5.16`).

## Overview

The full path from `hermes -z "hello"` to the response:

```
terminal
  ↓
hermes_cli/main.py:main()           # pyproject.toml entry: hermes = "hermes_cli.main:main"
  ↓   (fire.Fire dispatches -z)
hermes_cli/oneshot.py:run_oneshot() # suppress banner/spinner/logging, resolve model/provider
  ↓
hermes_cli/oneshot.py:_run_agent()  # build AIAgent + wire runtime credentials
  ↓
run_agent.py:AIAgent.chat()         # simple wrapper → run_conversation()
  ↓
run_agent.py:AIAgent.run_conversation()  # core agent loop
  ↓   (LLM call, no tool calls needed for "hello")
response → stdout
```

## Step-by-step

### Step 1: Entry point

```
hermes -z "hello"
  ↕  pyproject.toml: hermes = "hermes_cli.main:main"
hermes_cli/main.py:main(z="hello")
```

Fire parses `-z` as the `z` keyword argument to `main()`. When `z` is non-None, `main()` delegates to `run_oneshot()`.

### Step 2: run_oneshot() — quiet wrapper

**File:** `hermes_cli/oneshot.py` (350 LOC)

```python
def run_oneshot(prompt: str, model=None, provider=None, toolsets=None) -> int:
```

Does three things before touching the agent:

1. **`logging.disable(logging.CRITICAL)`** — silence all stderr logging for the duration
2. **`os.environ["HERMES_YOLO_MODE"] = "1"`** — auto-approve all shell/tool approvals (non-interactive = no user to prompt)
3. **Redirect stdout+stderr to `/dev/null`** — suppress banner, spinner, tool progress. Only the final response reaches real stdout.

Then calls `_run_agent(prompt)` inside the redirect.

### Step 3: _run_agent() — resolve model/provider + create AIAgent

**File:** `hermes_cli/oneshot.py` — `_run_agent()` (lines 218–336)

Resolution order for **model**:
1. `--model` CLI arg
2. `HERMES_INFERENCE_MODEL` env var
3. `config.yaml` → `model.default` or `model.model`

Resolution order for **provider**:
1. If `--model` was explicit (arg or env) without `--provider` → auto-detect via `detect_provider_for_model()`
2. Else → `config.yaml` → `model.provider` or `HERMES_INFERENCE_PROVIDER` env var → `resolve_runtime_provider()`

Key modules called during resolution:

| Module | Role |
|--------|------|
| `hermes_cli/config.py:load_config()` | Parse `~/.hermes/config.yaml` |
| `hermes_cli/models.py:detect_provider_for_model()` | Match model name to provider catalog |
| `hermes_cli/runtime_provider.py:resolve_runtime_provider()` | Resolve api_key, base_url, api_mode, credential_pool |
| `hermes_cli/tools_config.py:_get_platform_tools()` | Determine which toolsets are enabled for platform="cli" |

Also checks `DIRECT_ALIASES` (from `config.yaml` `model_aliases:`) before hitting the catalog — this lets custom endpoints (local servers, proxies) bypass provider detection.

Also creates a `SessionDB()` for the session store.

**AIAgent construction** (lines 305–328):

```python
agent = AIAgent(
    api_key=runtime["api_key"],
    base_url=runtime["base_url"],
    provider=runtime["provider"],
    api_mode=runtime["api_mode"],
    model=effective_model,
    enabled_toolsets=toolsets_list,
    quiet_mode=True,
    platform="cli",
    session_db=session_db,
    credential_pool=runtime.get("credential_pool"),
    clarify_callback=_oneshot_clarify_callback,  # returns "pick a default" in one-shot
)
```

Key differences from interactive mode:
- `quiet_mode=True`, `suppress_status_output=True` — no spinner or streaming callbacks
- `clarify_callback` — returns a synthetic "make the best assumption" instruction instead of stalling
- No `stream_delta_callback` or `tool_gen_callback` — streaming output is captured, not rendered
- No `HERMES_INTERACTIVE` env var — terminal tool won't prompt for sudo passwords

### Step 4: agent.chat() — simple wrapper

**File:** `run_agent.py` — `AIAgent.chat()` method

```python
def chat(self, message: str) -> str:
    """Simple interface — returns final response string."""
    result = self.run_conversation(user_message=message)
    if isinstance(result, dict):
        return result.get("final_response", "")
    return str(result)
```

`chat()` is intentionally simple: it's the API used by oneshot mode, cron jobs, and programmatic callers who don't need the full `run_conversation()` dict with `messages`, `errors`, `partial`, etc.

### Step 5: run_conversation() — the core agent loop

**File:** `run_agent.py` — 16,408 LOC. This is the single most important function in the repo.

Simplified loop:

```
run_conversation(user_message, system_message=None, conversation_history=None, task_id=None) → dict
```

**Phase A: Message assembly** (before the loop)

1. Build system prompt:
   - Load `AGENTS.md` (or `PRINCIPLES.md` / `CLAUDE.md` / `.cursorrules` from CWD)
   - Load MEMORY.md + USER.md (from `~/.hermes/memories/`)
   - Load preloaded skills (if any)
   - Compose into a system message
2. Build user message:
   - Current `user_message` (from caller)
   - Prepend `conversation_history` if resuming a session
3. Build tool schemas:
   - Call `model_tools.discover_builtin_tools()` which imports `tools/registry.py`
   - `registry` returns all registered tool schemas (name, description, JSON parameters)
   - Tools are filtered by `enabled_toolsets`

**Phase B: The loop**

```python
while (api_call_count < max_iterations and iteration_budget.remaining > 0) \
        or self._budget_grace_call:
    if self._interrupt_requested:
        break

    # 1. Make the LLM call
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tool_schemas,  # list of OpenAI-format tool definitions
    )

    # 2. Process the response
    if response.choices[0].finish_reason == "stop":
        # Normal text response — extract content, break loop
        return {"final_response": content, "messages": messages, ...}

    if response.tool_calls:
        for tool_call in response.tool_calls:
            result = model_tools.handle_function_call(
                tool_call.name, tool_call.args, task_id
            )
            # Inject result back as a tool-role message
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            })
        api_call_count += 1
        # Loop continues — next LLM call sees the tool result
```

**Key observations about the loop:**
- **Entirely synchronous** — no asyncio. The loop blocks on each LLM call and each tool execution.
- **No streaming in `run_conversation()` itself** — streaming deltas are handled via callbacks (`stream_delta_callback`) that fire during the LLM call for interactive display. The return value is always the full accumulated text.
- **Grace call** — `self._budget_grace_call` gives one extra API call when budget runs out, so the agent can produce a final summary.
- **Interrupt** — `self._interrupt_requested` is checked every iteration. Set by `agent.interrupt()` from signal handlers.

### Step 6: Response path

For `"hello"`, the model likely returns text directly (no tool calls). The loop exits after one iteration with `final_response`.

Back in `oneshot.py`:

```python
if response:
    real_stdout.write(response)
    if not response.endswith("\n"):
        real_stdout.write("\n")
    real_stdout.flush()
return 0
```

The user sees:
```
Hello! How can I help you today?
```

Exit code 0. No session_id line (unlike `cli.py -q` mode which prints `session_id: ...` to stderr).

## Files touched during this trace

| File | LOC | What it contributes |
|------|-----|-------------------|
| `hermes_cli/main.py` | 12,408 | Fire CLI dispatch |
| `hermes_cli/oneshot.py` | 350 | Quiet wrapper, model/provider resolution, AIAgent creation |
| `hermes_cli/config.py` | — | `load_config()` — parse `~/.hermes/config.yaml` |
| `hermes_cli/models.py` | — | `detect_provider_for_model()` |
| `hermes_cli/runtime_provider.py` | — | `resolve_runtime_provider()` — api_key, base_url, api_mode |
| `hermes_cli/tools_config.py` | — | `_get_platform_tools()` — toolset selection |
| `run_agent.py` | **16,408** | `AIAgent.chat()`, `AIAgent.run_conversation()` — core loop |
| `model_tools.py` | 899 | `discover_builtin_tools()`, `handle_function_call()` |
| `tools/registry.py` | — | Tool registration (imported by model_tools) |
| `agent/memory_manager.py` | — | Load MEMORY.md / USER.md |
| `agent/skill_commands.py` | — | Load skills into system prompt |
| `hermes_state.py` | 2,966 | SessionDB — SQLite session persistence |
| `hermes_constants.py` | 345 | Path resolution (`get_hermes_home()`) |

## Data flow diagram

```
User input: "hello"
    │
    ▼
hermes_cli/main.py:main(z="hello")
    │
    ▼
hermes_cli/oneshot.py:run_oneshot("hello")
    │  ├─ stderr → /dev/null (suppress banner/spinner)
    │  ├─ stdout → /dev/null (suppress streaming)
    │  └─ HERMES_YOLO_MODE=1 (skip approvals)
    │
    ▼
hermes_cli/oneshot.py:_run_agent("hello")
    │  ├─ load_config() → config dict
    │  ├─ resolve model + provider + runtime
    │  └─ AIAgent(...)
    │
    ▼
run_agent.py:AIAgent.chat("hello")
    │
    ▼
run_agent.py:AIAgent.run_conversation("hello")
    │
    ├─ [Phase A] Message assembly:
    │   ├─ system = AGENTS.md + memory + skills + rules files
    │   ├─ user = "hello"
    │   └─ tools = discover_builtin_tools() → tool schemas
    │
    ├─ [Phase B] LLM call:
    │   ├─ client.chat.completions.create(
    │   │     model=..., messages=..., tools=...)
    │   │
    │   └─ response.choices[0]:
    │       ├─ finish_reason="stop" → return text
    │       └─ finish_reason="tool_calls" → handle_function_call() → loop
    │
    └─ result = {"final_response": "Hello! How can I help you?", ...}
    │
    ▼
real_stdout.write("Hello! How can I help you?")
```

## Notable design decisions

1. **Oneshot bypasses `cli.py` entirely** — it doesn't import `HermesCLI` or any of its interactive infrastructure. This keeps `-z` startup fast (~200ms vs ~1s+ for interactive).

2. **No streaming in the return path** — streaming is a display optimization. `run_conversation()` always returns the final accumulated text. Streaming callbacks fire during the LLM call for interactive UIs.

3. **Tool discovery by import side-effect** — `model_tools.py` imports `tools/registry.py` which triggers all tool files to `register()` themselves. No config file enumeration.

4. **Session DB is optional** — oneshot creates one best-effort, but if it fails, the agent still works; only `session_search` tool is unavailable.

5. **Model resolution has 3 fallback layers** — arg → env → config. Provider can auto-detect from model name if not explicitly set.
