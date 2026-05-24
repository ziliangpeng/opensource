# Hermes Agent v0.14.0 — Orientation

> **Phase 0** of the repository reading process. See [`PRINCIPLES.md`](../PRINCIPLES.md) for the methodology.

## Top-level directory survey

| Directory | One-line purpose | Size (files) |
|-----------|-----------------|-------------|
| `run_agent.py` | Entry point — `AIAgent` class, the core agent loop | 1 file, ~12K LOC |
| `cli.py` | CLI orchestrator — `HermesCLI` class with prompt_toolkit input | 1 file, ~11K LOC |
| `model_tools.py` | Tool orchestration — discover tools, call LLM, handle tool responses | 1 file |
| `agent/` | Provider adapters, memory, caching, compression, display, skill_commands | 36 files |
| `hermes_cli/` | CLI subcommands, setup wizard, plugins loader, skin engine, commands registry | 13 files |
| `gateway/` | Messaging gateway — `run.py` + `session.py` + `platforms/` (Telegram, Discord, Slack, SMS, Email, …) | 90 files |
| `tools/` | Tool implementations — code execution, web search, file ops, and `environments/` backends | 30+ files |
| `cron/` | Scheduler — `jobs.py`, `scheduler.py` | 4 files |
| `plugins/` | Plugin system — memory providers, model providers, kanban, image_gen, observability, … | 120+ files |
| `skills/` | Built-in skills bundled with the repo | ~30 files |
| `optional-skills/` | Heavier/niche skills shipped but NOT active by default | — |
| `tests/` | Pytest suite (~17K tests across ~900 files) | 900+ files |
| `ui-tui/` | Ink (React) terminal UI | npm project |
| `tui_gateway/` | Python JSON-RPC backend for the TUI | 4 files |
| `acp_adapter/` | ACP server (VS Code / Zed / JetBrains integration) | — |
| `acp_registry/` | ACP registry | — |
| `providers/` | Inference provider integrations | — |
| `scripts/` | `run_tests.sh`, `release.py`, auxiliary scripts | — |
| `docker/` | Dockerfiles | — |
| `nix/` | Nix flake/package definitions | — |
| `website/` | Docusaurus docs site | npm project |
| `docs/` | Additional documentation | — |
| `web/` | Web interface | — |
| `assets/` | Static assets | — |
| `packaging/` | Packaging (Homebrew, etc.) | — |
| `locales/` | i18n / l10n | — |
| `plans/` | Development plans | — |
| `datagen-config-examples/` | Data generation config examples | — |

**Total:** ~3,452 files. Largest directories: `tests/` (~900), `plugins/` (120+), `gateway/` (90), `agent/` (36), `tools/` (30+).

### Dependency chain (critical path)

From `AGENTS.md`:

```
tools/registry.py  (no deps — imported by all tool files)
       ↑
tools/*.py  (each calls registry.register() at import time)
       ↑
model_tools.py  (imports tools/registry + triggers tool discovery)
       ↑
run_agent.py, cli.py, batch_runner.py, environments/
```

## What problem does this project solve?

Hermes Agent is a **universal AI assistant that runs wherever you need it** — CLI, Telegram, Discord, Slack, SMS, email, VS Code, Kanban board, scheduled cron, and any platform with a messaging API. It:

1. Routes requests through multiple LLM providers (OpenAI, Anthropic, OpenRouter, custom, …) with fallback
2. Provides a large toolset (code execution, file ops, web search, browser automation, image gen, …)
3. Manages conversation state across sessions (SQLite-based FTS5 search)
4. Supports extensions through plugins and skills

## Who is the primary user?

Developers who want a **self-hosted, customizable AI assistant** that works across their workflow — from terminal to chat apps to IDE. The project is designed to be run by the user (not as a hosted service), with config in `~/.hermes/config.yaml`.

## What are the "hard parts"? (technical challenges)

| Challenge | Where it lives | Why it's hard |
|-----------|---------------|--------------|
| **Multi-provider LLM routing** | `agent/`, `providers/` | Each provider has different API shapes (chat vs completions vs codex), auth (API key vs OAuth), model lists, error formats, streaming vs non-streaming |
| **Tool system** | `tools/registry.py`, `model_tools.py` | ~40+ tools with varied signatures; tools produce side-effects; results need to be injected back into the LLM conversation as structured messages |
| **Gateway (multi-platform messaging)** | `gateway/` | Each platform has different API, webhook format, rate limits, media handling, auth flow — all must present a unified session interface |
| **State management** | `hermes_state.py` | SQLite session store with FTS5; needs to handle concurrent writes, session persistence, and cross-session search |
| **Plugin system** | `plugins/` | Plugins can provide: memory backends, model providers, image gen, kanban workers, observability — each with a different interface contract |
| **TUI (React + Python JSON-RPC)** | `ui-tui/`, `tui_gateway/` | Dual-language architecture: Ink (React) frontend + Python backend communicating over JSON-RPC |
| **Cron scheduler** | `cron/` | Schedule LLM-powered tasks that run autonomously; need agent state isolation, delivery to various platforms, script execution |
| **Skills system** | `skills/`, `agent/skill_commands.py` | User-provided instructions loaded at runtime; injection into system prompt without breaking context optimization |
| **Security / prompt injection** | Throughout | CLI/non-LLM interface must handle malicious input; tool orchestration must validate user intent |


## Entry points

| Entry point | How users reach it | File |
|------------|-------------------|------|
| CLI | `hermes` (or `python cli.py`) | `cli.py` |
| TUI | `hermes --tui` | `ui-tui/`, `tui_gateway/` |
| Gateway | `hermes run` | `gateway/run.py` |
| ACP server | `hermes acp` | `acp_adapter/` |
| API server | `hermes serve` | `plugins/platforms/api_server/` |
| Cron scheduler | `hermes cron` | `cron/scheduler.py` |
| One-shot | `hermes -z "message"` | `cli.py` (via `run_agent.py`) |
