# Repo Structure — v0.14.0

> **Phase 1** of the repository reading process. File-tree mental map with one-sentence descriptions.

## Dependency chain (critical path)

```
tools/registry.py  (no deps — imported by all tool files)
       ↑
tools/*.py  (each calls registry.register() at import time)
       ↑
model_tools.py  (imports tools/registry + triggers tool discovery via import)
       ↑
run_agent.py, cli.py, batch_runner.py, environments/
```

This chain means: tools are auto-discovered by import side-effect, not by a config file. Adding a new tool means adding a file under `tools/` that calls `registry.register()` at module load time.

## Root-level files (entry points and core infrastructure)

| File | LOC | Role |
|------|-----|------|
| `run_agent.py` | **16,408** | `AIAgent` class — the core conversation loop. Most important file in the repo. |
| `cli.py` | **14,166** | `HermesCLI` — interactive CLI orchestrator. Handles input, dispatching, output. |
| `model_tools.py` | 899 | Bridge between `run_agent.py` and the tool system. `discover_builtin_tools()`, `handle_function_call()`. |
| `hermes_state.py` | 2,966 | `SessionDB` — SQLite session store with FTS5 search. Persists conversations across restarts. |
| `hermes_constants.py` | 345 | Path resolution: `get_hermes_home()`, profile-aware config/log/skill paths. |
| `hermes_logging.py` | 389 | Log setup: `agent.log`, `errors.log`, `gateway.log` (profile-aware). |
| `batch_runner.py` | 1,302 | Parallel batch processing — run agents across many inputs. |
| `toolsets.py` | — | Toolset definitions — `_HERMES_CORE_TOOLS` list, enable/disable logic. |
| `toolset_distributions.py` | — | Distribution profiles for tool sets (CLI vs gateway vs cron). |
| `utils.py` | — | Shared utilities. |
| `hermes_bootstrap.py` | — | Bootstrap/reset logic. |
| `hermes_time.py` | — | Time-related helpers. |

## Directory-by-directory breakdown

### `agent/` — Agent internals (60 files)

The largest functional directory. Core subsystems:

| Subsystem | Files | Role |
|-----------|-------|------|
| **Provider adapters** | `anthropic_adapter.py`, `bedrock_adapter.py`, `codex_responses_adapter.py`, `gemini_native_adapter.py`, `gemini_cloudcode_adapter.py`, `lmstudio_reasoning.py` | Translate each provider's API response shape into the Hermes internal format. Each adapter handles streaming, tool calls, reasoning content differently. |
| **Memory** | `memory_manager.py`, `memory_provider.py` | User preference memory — loads MEMORY.md/USER.md, manages memory CRUD via the `memory` tool. Plugins can provide alternate backends. |
| **Context** | `context_compressor.py`, `context_engine.py`, `context_references.py` | Compress/optimize context windows. Context engine is a plugin interface. |
| **Skill commands** | `skill_commands.py` | Scans `~/.hermes/skills/`, injects matching skills as user messages (not system prompts — cache optimization). |
| **Display** | `display.py` | `KawaiiSpinner` — animated faces during API calls, `┊` activity feed for tool output. Immersive but non-intrusive rendering. |
| **Provider routing** | `auxiliary_client.py`, `plugin_llm.py`, `model_metadata.py`, `models_dev.py`, `credential_pool.py` | Route requests through the right provider, negotiate model availability, manage API keys. |
| **File safety / security** | `file_safety.py` | Protect Hermes control-plane files from prompt injection / malicious writes. |
| **Image generation** | `image_gen_provider.py`, `image_gen_registry.py`, `image_routing.py` | Abstract over multiple image gen backends. |
| **I18n / localization** | `i18n.py` | Internationalization support. |
| **Insights / error** | `insights.py`, `error_classifier.py` | User behavior insights, error classification for diagnostics. |

### `hermes_cli/` — CLI subcommands and tooling (80 files)

The CLI is structured around a **central command registry** (`commands.py`):

```python
COMMAND_REGISTRY = [
    CommandDef(name="model", aliases=["m"], handler=...),
    CommandDef(name="help", aliases=["h", "?"], handler=...),
    ...
]
```

All CLI consumers derive from this registry automatically (CLI, gateway, help text).

Key files:
- `commands.py` — The central command registry (slug commands like `/model`, `/help`, `/reasoning`)
- `main.py` — CLI entry point (after `cli.py` bootstrap)
- `config.py` — `load_cli_config()` — merges defaults + user config YAML
- `skin_engine.py` — Data-driven CLI theming (banner colors, spinner faces, brand text)
- `setup.py` — First-run setup wizard
- `gateway.py` — Gateway subcommand (`hermes gateway`)
- `cron.py` — Cron subcommand (`hermes cron`)
- `kanban.py` — Kanban board subcommand
- `providers.py` — Provider management subcommand
- `profiles.py` — Profile management (`hermes profiles`)
- `plugins.py` / `plugins_cmd.py` — Plugin management
- `skills_config.py` / `skills_hub.py` — Skill management
- `oneshot.py` — One-shot mode (`hermes -z`)
- `doctor.py` — Diagnostics (`hermes doctor`)
- `backup.py` — Backup/restore
- `voice.py` — Voice I/O
- `model_switch.py` / `models.py` / `model_catalog.py` — Model selection

### `gateway/` — Messaging gateway (20 core files + 30 platform adapters)

The gateway runs as a daemon (`hermes run`) and connects to multiple chat platforms simultaneously.

**Architecture:**
```
gateway/run.py  (main entry — starts all platform listeners)
    → gateway/session.py  (session management — one session per chat)
    → gateway/platform_registry.py  (discovers registered platforms)
    → gateway/platforms/{telegram,discord,slack,...}.py  (platform-specific adapters)
    → calls run_agent.py AIAgent (for each incoming message)
```

**Platforms supported in v0.14.0:**

| Platform | File |
|----------|------|
| Telegram | `telegram.py` + `telegram_network.py` |
| Discord | `discord.py` |
| Slack | `slack.py` |
| SMS | `sms.py` |
| Email | `email.py` |
| Signal | `signal.py` + `signal_rate_limit.py` |
| WhatsApp | `whatsapp.py` |
| Matrix | `matrix.py` |
| Mattermost | `mattermost.py` |
| DingTalk | `dingtalk.py` |
| WeCom | `wecom.py` + `wecom_callback.py` + `wecom_crypto.py` |
| Weixin | `weixin.py` |
| Feishu | `feishu.py` + `feishu_comment.py` + `feishu_comment_rules.py` |
| BlueBubbles | `bluebubbles.py` |
| Home Assistant | `homeassistant.py` |
| Webhook | `webhook.py` |
| API Server | `api_server.py` |
| QQ Bot | `qqbot/` (subdirectory) |
| *(plugin-based)* | Google Chat, IRC, Line, SimpleX, Teams |

Key gateway infrastructure:
- `delivery.py` — Message delivery across platforms
- `stream_consumer.py` — Consume streaming LLM responses and forward to platforms
- `slash_access.py` — Slash command handling in gateway context
- `hooks.py` — Hook system for custom gateway behavior
- `restart.py` — Graceful restart handling

### `tools/` — Tool implementations (~40 files)

Every tool is a file that calls `registry.register()` at import time. Tools are auto-discovered:

| Tool file | What it does |
|-----------|-------------|
| `browser_tool.py`, `browser_camofox.py`, `browser_cdp_tool.py` | Browser automation (3 implementations: standard native, Camofox cloud, CDP) |
| `code_execution_tool.py` | Python code execution (`execute_code` tool) |
| `terminal_tool.py` | Shell command execution (`terminal` tool) |
| `file_read_tool.py`, `file_write_tool.py`, `file_search_tool.py` | File I/O (`read_file`, `write_file`, `search_files`) |
| `patch_tool.py` | File patching (`patch` tool — find-and-replace) |
| `web_search_tool.py` | Web search |
| `image_gen_tool.py` | Image generation (`image_generate` tool) |
| `vision_tool.py` | Vision analysis (`vision_analyze` tool) |
| `delegate_tool.py` | Task delegation (`delegate_task` tool) |
| `cronjob_tools.py` | Cron job management (`cronjob` tool) |
| `clarify_tool.py` | User clarification (`clarify` tool) |
| `todo_tool.py` | Task list |
| `memory_tool.py` | Memory management |
| `skill_tool.py` | Skill management |
| `text_to_speech_tool.py` | TTS |
| `computer_use_tool.py` | Computer use (GUI automation) |
| `session_search_tool.py` | Session search |
| `approval.py` | Dangerous command approval |
| `checkpoint_manager.py` | Checkpoint/rollback |
| `budget_config.py` | Iteration budget config |
| `credential_files.py` | Credential file management |

**Environments** (`tools/environments/`): Terminal backends
- `local.py` — Local shell execution
- `ssh.py` — SSH remote execution
- `docker.py` — Docker container execution
- `modal.py` — Modal cloud execution
- `daytona.py` — Daytona workspace
- `singularity.py` — Singularity containers
- `vercel_sandbox.py` — Vercel sandbox
- `managed_modal.py`, `modal_utils.py` — Modal helpers
- `file_sync.py` — File sync across environments

### `plugins/` — Plugin system (120+ files)

**Model providers** (`plugins/model-providers/`, ~30 plugins):
Anthropic, OpenAI (via openrouter/gmi), Gemini, Bedrock, Copilot, Codex, DeepSeek, KiloCode, Kimi Coding, MiniMax, Nous, Novita, NVIDIA, Ollama Cloud, StepFun, xAI, Xiaomi, ZAI, Alibaba, Alibaba Coding Plan, Azure Foundry, HuggingFace, and more. Each is a self-contained plugin that registers a provider adapter.

**Memory providers** (`plugins/memory/`, 8 plugins):
Honcho, Mem0, Supermemory, Hindsight, Holographic, Byterover, RetainDB, OpenViking.

**Other plugins:**
- `kanban/` — Multi-agent board dispatcher + worker
- `image_gen/` — Image generation backends
- `observability/` — Metrics/traces/logs
- `spotify/` — Spotify integration
- `video_gen/` — Video generation
- `teams_pipeline/` — Teams meeting pipeline
- `hermes-achievements/` — Gamified achievement tracking
- `disk-cleanup/` — Disk space management
- `google_meet/` — Google Meet integration
- `example-dashboard/` — Example React dashboard
- `web/` — Web interface plugin
- `platforms/` — Additional gateway platforms (google_chat, irc, line, simplex, teams)

### `cron/` — Scheduler (3 files)

- `scheduler.py` — Main scheduler daemon
- `jobs.py` — Job model and persistence

### `acp_adapter/` — ACP protocol server (10 files)

ACP (Agent Communication Protocol) — the protocol for IDE integration:
- `server.py` — ACP server
- `session.py` — Session management
- `tools.py` — Tool exposure
- `events.py` — Event streaming
- `auth.py` — Authentication
- `permissions.py` — Permission model

### `tui_gateway/` — TUI JSON-RPC backend (8 files)

- `server.py` — JSON-RPC server
- `entry.py`, `event_publisher.py` — Event publishing
- `render.py` — Content rendering for TUI
- `transport.py` — Transport abstraction
- `ws.py` — WebSocket transport
- `slash_worker.py` — Slash command handling in TUI context

### `skills/` — Built-in skills

Skills are organized by category (apple, devops, mlops, software-development, media, etc.). Each skill is a directory with `SKILL.md` and optional supporting files.

### `tests/` — Test suite (~17K tests, ~900 files)

Massive test suite. Tests cover: CLI, gateway, tools, agent loop, memory, provider adapters, security, edge cases. Organized by component.

## "Hot path" candidates

The most important code paths to trace:

1. **CLI → agent loop** (commonest path): `cli.py` → `process_command()` → `AIAgent.run_conversation()` → `model_tools.handle_function_call()` → tool → response
2. **Gateway message → agent**: `gateway/run.py` → platform adapter → `gateway/session.py` → `run_agent.AIAgent()` → same loop
3. **Tool execution end-to-end**: `model_tools.handle_function_call()` → tool function → tool result → inject into messages → next LLM call
4. **Session persistence**: `hermes_state.SessionDB` — when are messages saved? How does FTS5 indexing work?
