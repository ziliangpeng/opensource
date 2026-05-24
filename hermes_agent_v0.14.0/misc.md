# Misc — random questions and observations

## Oneshot mode (`-z`) — tool use and safety

### Can the agent still use tools in oneshot mode?

Yes. Oneshot mode (`hermes -z "..."`) loads the same `enabled_toolsets` as interactive mode. The agent can call all registered tools: web search, file read/write, terminal commands, etc.

The only difference is **no user clarification** — if the agent needs to ask a question (e.g. "which directory?"), the `clarify_callback` returns a synthetic "pick a default and continue" instruction instead of stopping.

### What about YOLO mode?

`HERMES_YOLO_MODE=1` is set in oneshot mode, which bypasses dangerous-command approval prompts. This is necessary because there's no user sitting at a terminal to approve or deny.

### How dangerous is this?

The risk profile:

| Factor | Risk level | Reason |
|--------|-----------|--------|
| User writes the prompt | Low | Oneshot is an explicit, one-off command. You're responsible for what you type. |
| LLM makes unexpected decisions | Moderate | The model might do something you didn't intend, especially with vague prompts. |
| Cron job + oneshot | Higher | A recurring job with a buggy prompt, or a model behavior change, could cause damage unattended. |
| Compared to raw shell | Same | Shell has no approval prompts either. Oneshot just matches shell-level trust. |

The real difference: **shell runs exactly what you type; LLM runs what it interprets from what you type.** Hallucination, misinterpretation, or prompt injection are the real risk vectors — not the YOLO flag itself.

Related: the fork's security patches (`file-safety`, control-plane write-deny) address one aspect of this — protecting Hermes' own configuration files from prompt-injection attacks.

## Session search and cross-session continuity

Hermes uses **FTS5 full-text search** (via `hermes_state.py`) for cross-session recall — not just keyword matching, but LLM-enhanced retrieval. This enables a design where:

- Each session stays **focused on one task** — no need to keep everything in one long context
- Cross-session continuity comes from `session_search` + `agent-kb` + skills, not from cramming all history into the same session
- The user can ask on any platform (Telegram, CLI, etc.) and the agent retrieves context from sessions on other platforms
- The `session_id` printed at the end of each session is a reference handle for resume and recall

This is the architectural opposite of "keep everything in memory" — data is persisted but **not actively loaded** unless queried. It aligns with the memory anti-bloat principle: memory = short universal triggers, agent-kb = declarative facts, session_search = episodic recall.

## Hermes vs OpenClaw vs OpenCode — architecture comparison

### Methodology

I read the source code of all three projects at their current `main`/HEAD (May 2026):

| Project | Language | Stars | Repo |
|---------|----------|-------|------|
| **Hermes Agent** | Python | 165K | NousResearch/hermes-agent |
| **OpenClaw** | TypeScript | 374K | openclaw/openclaw |
| **OpenCode** | TypeScript | 165K | anomalyco/opencode |

This comparison focuses on **architecture and design philosophy**, not feature checklists. All three are capable personal AI assistants; they differ in what they prioritize.

### Core design philosophy

| | Hermes | OpenClaw | OpenCode |
|---|---|---|---|
| **Primary identity** | Agent infrastructure platform | Personal assistant + gateway | Coding agent |
| **Language** | Python | TypeScript | TypeScript (Bun) |
| **User** | Developer who self-hosts | Everyone (channels-first UX) | Developer writing code |
| **CLI** | Rich TUI (prompt_toolkit + Ink React) | Full TUI | TUI (Bun-based) |
| **Gateway/daemon** | ✅ 40 platforms | ✅ 20+ channels (plugin-based) | ❌ (no daemon mode) |
|| **Plugin ecosystem** | 30 model providers, 8 memory backends | 129 official extensions (everything is a plugin) | Plugin SDK + extension marketplace |
| **ACP / IDE integration** | ✅ ACP adapter | ✅ ACP | Native (same codebase powers both CLI and its own IDE integrations via SDK) |

### Architecture comparison

**Hermes Agent** is built around a **single core `AIAgent` class** (`run_agent.py`, 16K LOC). Everything — CLI, gateway, cron, ACP — is a thin wrapper around this one loop. The tools are Python files that register themselves via import side-effect. The plugin system is flexible but feels bolted on: each plugin type (memory, model provider, platform) has its own interface.

**OpenClaw** is a **TypeScript monorepo with 129 official extension packages**. The architecture is "core stays plugin-agnostic" — everything beyond the bare minimum ships as an extension. Channels (WhatsApp, Telegram, Discord, etc.) are extensions. Model providers are extensions. Memory is a special plugin slot. Security boundaries between core and plugins are enforced at the code level (plugin code cannot import core `src/**`). The `src/agents/` directory has **952 files** — the agent logic is far more modular than Hermes' single `run_agent.py`.

**OpenCode** is a **coding-focused agent** (Bun monorepo). Its `packages/` include `opencode` (the main CLI), `core` (agent logic), `sdk` (plugin SDK), and `extensions` (plugins). It has a strong Effect-TS architecture (functional effects system) and is designed primarily as a terminal coding assistant — no gateway/daemon mode, no multi-platform messaging. Its `agent.ts` is short and delegates to the `core` package.

### Feature comparison

| Feature | Hermes | OpenClaw | OpenCode |
|---------|--------|----------|----------|
| **Multi-platform messaging** | 40 platforms (gateway daemon) | 20+ channels (plugins) | ❌ (terminal only) |
| **Cron scheduler** | ✅ Built-in (`cron/`) | ✅ Built-in (`src/cron/`) | Coming (via cron plugin) |
| **Subagents / delegation** | ✅ (`delegate_task` tool) | ✅ (extensive subagent registry + lifecycle management) | ✅ (`background` sessions) |
| **Kanban / multi-agent board** | ✅ (kanban plugin) | ✅ (via plugins) | ❌ (not a focus) |
| **Memory system** | FTS5 + 8 plugin backends | Commitments (extraction from conversation) + multiple memory plugins | Plugin-based |
| **Skills system** | Skill files in `~/.hermes/skills/` | `skills/` directory (57 built-in) | Plugin-based agent skills |
| **Learning loop** | Background review, auto skill creation/patch, memory extraction | Commitments (extracts action items/tracking from conversations) | ❌ (no built-in learning loop) |
| **Tool system** | ~75 tools, Python, import-side-effect registration | Extensions provide tools; core tool infrastructure in `src/tools/` | Plugin SDK provides tools |
| **Model providers** | 30+ (plugins) | 129 extensions (each provider is its own package) | Plugin-based (via SDK) |
| **Context compression** | ✅ Built-in (`context_compressor.py`) | ✅ (compaction system in `src/agents/`) | ✅ (via plugins) |
| **TUI quality** | Rich, Ink React TUI | Full TUI | Full TUI (Bun-based) |
| **Batch processing** | ✅ (`batch_runner.py`) | ❌ (not a focus) | ❌ |
| **ACP / IDE protocol** | ✅ (`acp_adapter/`) | ✅ (`src/acp/`) | Native SDK |
| **Security model** | File safety, approval workflow, YOLO mode | Security-first: secure defaults, operator-controlled risky paths, plugin sandboxing | Permission system via `src/permission.ts` |

### Key differentiators

#### What Hermes does uniquely

1. **Learning loop** — background review that auto-creates and patches skills, extracts memory. Neither OpenClaw nor OpenCode has this. OpenClaw's "commitments" system extracts action items/tracking from conversations but doesn't auto-create reusable knowledge in the same way.

2. **Python ecosystem** — Hermes is the only one in Python. This means direct access to the Python ML/data ecosystem (pandas, numpy, PyTorch, etc.) without subprocess calls. If you work in ML/infra, Hermes fits naturally.

3. **Cron as a first-class concept** — `hermes cron` with LLM-powered job execution and multi-platform delivery. The cron jobs run full agent sessions, not just shell scripts.

4. **Batch processing** — `batch_runner.py` for running many agent sessions in parallel. Research-oriented feature.

5. **Trajectory compression** — `trajectory_compressor.py` compresses full agent transcripts into training data for tool-calling models.

#### What OpenClaw does uniquely

1. **374K stars** — far larger community. More extensions, more skills, more documentation.

2. **Plugin-first architecture** — core is genuinely lean. Almost everything ships as an extension (129 official ones). The plugin SDK is a first-class citizen with strict code boundaries.

3. **Channel count** — 20+ messaging channels including less common ones (IRC, Nextcloud Talk, Nostr, Tlon, Synology Chat, Zalo). Hermes covers 40 but some are in the platform adapter layer rather than plugins.

4. **Talk/voice** — built-in talk mode with speech recognition, transcription relay, and voice call support (via extensions).

5. **Canvas** — live rendering surface for the assistant to show interactive content.

#### What OpenCode does uniquely

1. **Coding-first** — designed from the ground up as a coding agent. The TUI, tool set, and SDK are optimized for software development workflows.

2. **Effect-TS** — strong functional programming foundation. Error handling, retry, and resource management are built into the type system via Effect.

3. **Plugin marketplace** — opencode.ai has a marketplace for plugins, agents, themes. Hermes' `clawhub` (`clawhub.ai`) is similar but newer.

4. **Bun runtime** — very fast startup times compared to Python-based Hermes.

### Summary

| Scenario | Best fit |
|----------|----------|
| You want an assistant that follows you across Telegram/WhatsApp/Discord/Slack | Hermes or OpenClaw (both strong here) |
| You work in ML/infra and want Python-native tool access | **Hermes** |
| You want the richest plugin ecosystem | **OpenClaw** (129 extensions, 374K community) |
| You want a pure coding agent for writing software | **OpenCode** |
| You want automatic skill/knowledge extraction from conversations | **Hermes** (unique learning loop) |
| You want cron jobs that run full agent sessions | **Hermes** |
| You want voice calls / real-time speech with your assistant | **OpenClaw** |
| Community size matters | OpenClaw (374K) > Hermes (165K) ≈ OpenCode (165K) |
