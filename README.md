# opensource

Notes from reading open source project codebases — architecture maps, design decisions, interesting patterns, and "how does this actually work" deep-dives.

## Layout

Two kinds of folders:

**Versioned** (`<project>_<version>/`) — **code deep-dives and walkthroughs.** Version-pinned source analysis: request lifecycle traces, architecture maps, notes on specific files and commits.

```
<project>_<version>/
  README.md         # entry point: what the project is, why it's interesting, reading roadmap
  architecture.md   # high-level component map
  notes/            # focused deep-dives on subsystems / files / commits
```

**Unversioned** (`<project>/`) — **general project reference.** History, contributors, release evolution, study roadmaps, architecture overviews, and anything that spans multiple versions or isn't tied to a specific code snapshot.

```
<project>/
  README.md         # entry point + file index
  history.md        # project history, statistics, contributors
  architecture.md   # codebase structure and component diagrams
  study-roadmap.md  # learning path from beginner to expert
```

## Conventions

- Link to specific commits (`github.com/owner/repo/blob/<sha>/path`) so notes don't rot when upstream changes.
- Quote sparingly; prefer summarizing in your own words and linking to the source.
- Each note starts with **what** is being read (file/module/PR) and **why** it's worth a look.

## Index

- [`AI_AGENT_COMPARISON.md`](./AI_AGENT_COMPARISON.md) — Architecture and feature comparison of major open-source AI agents (Hermes, OpenClaw, OpenCode, Claude Code, Codex CLI).
- [`PRINCIPLES.md`](./PRINCIPLES.md) — Repository reading methodology and conventions.
- [`vllm_0.16.0/`](./vllm_0.16.0/) — vLLM v0.16.0 (released 2026-02-25): engine core, scheduler, worker, attention backends (FlashAttn / FlashInfer / Triton / ROCm AITER), KV transfer, MoE, paged attention, divisibility constraints.
- [`hermes_agent_v0.14.0/`](./hermes_agent_v0.14.0/) — NousResearch/hermes-agent v0.14.0 / v2026.5.16: CLI agent, gateway (Telegram / Discord / etc.), ACP, Kanban, cron, multi-provider routing, TUI. *(in progress)*
- [`prime_rl_v0.5.0/`](./prime_rl_v0.5.0/) — PrimeIntellect-ai/prime-rl v0.5.0 (released 2026-03-30): async RL training (FSDP2 + vLLM), PD-disaggregated inference, MoE EP / context parallelism, verifiers environment integration. *(skeleton)*
- [`triton/`](./triton/) — **Unversioned reference.** Triton language and compiler: project history and contributors, autotune deep-dive, release-by-release feature evolution, codebase architecture map, and structured study roadmap.

## License

MIT — see [LICENSE](./LICENSE). Notes are personal commentary; all referenced code belongs to its respective authors.
