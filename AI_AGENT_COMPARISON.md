# AI Agent Architecture Comparison

> A comparison of major open-source AI agent projects, based on reading their source code.
>
> **Last updated:** May 2026
> **Methodology:** Clone the repo at `main`/HEAD, read core architecture, tool system, plugin system, gateway, cron, and key differentiators.

## Overview

| | Hermes Agent | OpenClaw | OpenCode | Claude Code (Anthropic) | Codex CLI (OpenAI) |
|---|---|---|---|---|---|
| **Language** | Python | TypeScript | TypeScript (Bun) | TypeScript | TypeScript |
| **Stars** | 165K | 374K | 165K | ? (proprietary) | ? (proprietary) |
| **License** | MIT | MIT | MIT | Proprietary | Proprietary |
| **Repo** | NousResearch/hermes-agent | openclaw/openclaw | anomalyco/opencode | — | — |
| **Primary identity** | Agent infrastructure platform | Personal assistant + gateway | Coding agent | Coding agent | Coding agent |
| **Open source?** | ✅ Full | ✅ Full | ✅ Full | ❌ (CLI wrapper only) | ❌ (CLI wrapper only) |

## Core Architecture

### Hermes Agent

Built around a **single `AIAgent` class** (`run_agent.py`, ~16K LOC in v0.14.0). All entry points (CLI, gateway daemon, cron scheduler, ACP server, one-shot mode) are thin wrappers around this one loop.

```
tools/registry.py  (no deps — imported by all tool files)
       ↑
tools/*.py  (each calls registry.register() at import time)
       ↑
model_tools.py  (imports tools/registry + triggers tool discovery)
       ↑
run_agent.py, cli.py, batch_runner.py, environments/
```

Key architectural traits:
- **Tools are auto-discovered by import side-effect** — no config file enumeration
- **Entirely synchronous agent loop** — no asyncio in the core loop
- **Streaming is a display optimization** — `run_conversation()` returns the final accumulated text; streaming callbacks fire during the LLM call for interactive UIs
- **Single-file core** — `run_agent.py` at 16K LOC is the biggest file and biggest complexity concentration point
- **Model resolution:** arg → env var → config, with auto-detect for provider

### OpenClaw

TypeScript monorepo with **129 official extension packages**. The architecture is "core stays plugin-agnostic" — everything beyond bare minimum ships as an extension.

- **Channels** (Telegram, WhatsApp, Discord, etc.) are extensions
- **Model providers** are extensions (one package each)
- **Memory** is a special plugin slot with multiple backend options
- **Security boundaries** enforced at code level — plugin code cannot import `src/**` from core
- `src/agents/` has **952 files** — far more modular than Hermes' single `run_agent.py`
- **Commitments** system — extracts action items, tracking, and follow-ups from conversations
- **Talk/voice** — built-in speech recognition, transcription relay, voice call support

### OpenCode

Coding-focused agent built on **Bun** with **Effect-TS** (functional effects system). Monorepo with `packages/` for agent core, CLI, SDK, and extensions.

- **Coding-first** — TUI, tool set, and SDK optimized for software development
- **Effect-TS** — error handling, retry, and resource management built into the type system
- **No gateway/daemon** — pure terminal tool
- **Plugin marketplace** — opencode.ai marketplace for plugins, agents, themes
- **Fast startup** — Bun runtime is significantly faster than Python

### Claude Code (Anthropic) / Codex CLI (OpenAI)

Both are **proprietary CLI wrappers**. The core agent logic, model access, and tool system are closed-source. The open-source component is primarily the CLI interface and terminal UI.

- **Limited extensibility** — no plugin system, no custom model providers, no gateway
- **Single-provider** — locked to Anthropic or OpenAI respectively
- **Terminal-only** — no daemon, no multi-platform messaging

## Feature Comparison

| Feature | Hermes | OpenClaw | OpenCode | Claude Code | Codex CLI |
|---------|--------|----------|----------|-------------|-----------|
| **Multi-platform messaging** | 40 platforms | 20+ channels | ❌ | ❌ | ❌ |
| **Cron scheduler** | ✅ Built-in | ✅ Built-in | Coming (plugin) | ❌ | ❌ |
| **Subagents / delegation** | ✅ delegate_task | ✅ subagent registry | ✅ background sessions | ✅ (limited) | ❌ |
| **Kanban multi-agent** | ✅ | ✅ (plugins) | ❌ | ❌ | ❌ |
| **Memory system** | FTS5 + 8 backends | Commitments + plugins | Plugin-based | Basic (project memory) | Basic |
| **Learning loop** | ✅ Full (auto skill creation, patch, memory extraction) | ⚠️ Partial (commitments only) | ❌ | ❌ | ❌ |
| **Skills system** | `~/.hermes/skills/` | 57 built-in | Plugin-based | ❌ | ❌ |
| **Tool system** | ~75 tools, Python | Extension-based | Plugin SDK | Fixed set | Fixed set |
| **Model providers** | 30+ (plugins) | 129 extensions | Plugin SDK | 1 (Anthropic) | 1 (OpenAI) |
| **Context compression** | ✅ Built-in | ✅ Compaction | ✅ Plugin | ❌ | ❌ |
| **TUI** | Rich + Ink React | Full TUI | Full TUI | Basic | Basic |
| **Batch processing** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **ACP / IDE protocol** | ✅ | ✅ | Native SDK | ❌ | ❌ |
| **Voice / speech** | Basic TTS | ✅ Talk mode | ❌ | ❌ | ❌ |
| **Canvas / interactive output** | ❌ | ✅ Canvas | ❌ | ❌ | ❌ |
| **Security model** | File safety, approval workflow, YOLO mode | Plugin sandboxing, secure defaults | Permission system | Limited | Limited |

## Key Differentiators

### Hermes — unique strengths

| Strength | Details |
|----------|---------|
| **Learning loop** | Background review auto-creates skills, patches them during use, and extracts memory. No other agent (open or proprietary) does this. |
| **Python ecosystem** | Direct access to pandas, numpy, PyTorch, etc. without subprocess. Natural fit for ML/infra work. |
| **Cron with full agent sessions** | `hermes cron` runs complete agent sessions (memory + tools + skills), not just shell scripts, and delivers to any platform. |
| **Research tooling** | `batch_runner.py` for parallel agent runs; `trajectory_compressor.py` for generating training data from agent transcripts. |
| **Multi-platform gateway** | 40 platforms from a single daemon process. All share the same session DB, memory, and skills. |
| **Session search** | FTS5 full-text search across all past conversations. Enables the design principle: one session = one focused task; retrieve context across sessions when needed. |

### OpenClaw — unique strengths

| Strength | Details |
|----------|---------|
| **Largest community** | 374K stars. The biggest ecosystem of extensions, skills, and documentation. |
| **Plugin-first architecture** | 129 official extensions. Core is genuinely lean. Plugin SDK is a first-class citizen with strict code boundaries. |
| **Channel variety** | 20+ channels including niche ones (IRC, Nextcloud Talk, Nostr, Tlon, Synology Chat, Zalo). |
| **Talk / voice** | Built-in speech recognition, transcription relay, and voice call support (via extensions). |
| **Canvas** | Live rendering surface for the assistant to display interactive content. |
| **Security model** | Strong security defaults with plugin sandboxing and operator-controlled risky paths. |

### OpenCode — unique strengths

| Strength | Details |
|----------|---------|
| **Coding-first design** | TUI, tool set, and SDK are all optimized for software development workflows. |
| **Effect-TS** | Strong functional programming foundation. Error handling, retry, and resource management are type-system-level. |
| **Fast runtime** | Bun-based, significantly faster startup than Python agents. |
| **Plugin marketplace** | opencode.ai has a marketplace ecosystem for plugins, agents, and themes. |

### Proprietary CLIs (Claude Code, Codex CLI) — what they lose by being closed

- Cannot add custom model providers or tools
- No multi-platform gateway
- No memory system beyond basic project context
- No learning loop / auto skill extraction
- No cron, no batch processing, no ACP

They gain: zero-config setup, polished UX, direct vendor integration.

## When to use what

| Scenario | Best fit |
|----------|----------|
| Cross-platform assistant (Telegram + Discord + Slack + ...) | Hermes or OpenClaw |
| ML/infra work, Python-native tool access | **Hermes** |
| Learning loop — auto skill/knowledge extraction | **Hermes** (unique) |
| Cron jobs running full agent sessions | **Hermes** (unique) |
| Research / batch trajectory generation | **Hermes** |
| Largest plugin ecosystem and community | **OpenClaw** |
| Voice calls / real-time speech | **OpenClaw** |
| Pure coding agent, terminal-focused | **OpenCode** |
| Zero-config, vendor-backed | Claude Code or Codex CLI |
| Plugin economy / marketplace | OpenCode or OpenClaw |

## Notes

- Stars are approximate as of May 2026.
- Proprietary CLI wrappers (Claude Code, Codex CLI) are included for context but are not open-source agents — their core logic is closed.
- Feature availability for closed-source agents is based on public documentation and may differ from actual capability.
- "Learning loop" refers to the agent's ability to autonomously create, maintain, and improve skills and memory from conversation experience. Only Hermes has a complete implementation.
