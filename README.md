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
- [`gemma4-rocm-pr-walkthrough.md`](./gemma4-rocm-pr-walkthrough.md) — Gemma4-on-ROCm PR-by-PR walkthrough (aiter#5062/5063/5027/4044 + vllm#53273/53874/53918): layer map, people map (AMD full-stack engineers), tuning-CSV conventions, silent-perf-bug anatomy.
- [`hermes_agent_v0.14.0/`](./hermes_agent_v0.14.0/) — NousResearch/hermes-agent v0.14.0 / v2026.5.16: CLI agent, gateway (Telegram / Discord / etc.), ACP, Kanban, cron, multi-provider routing, TUI. *(in progress)*
- [`prime_rl_v0.5.0/`](./prime_rl_v0.5.0/) — PrimeIntellect-ai/prime-rl v0.5.0 (released 2026-03-30): async RL training (FSDP2 + vLLM), PD-disaggregated inference, MoE EP / context parallelism, verifiers environment integration. *(skeleton)*
- [`triton/`](./triton/) — **Unversioned reference.** Triton language and compiler: project history and contributors, autotune deep-dive, release-by-release feature evolution, codebase architecture map, and structured study roadmap.
- [`hermes-agent/`](./hermes-agent/) — **Unversioned reference.** Hermes Agent fork census (2026-08): what the community adds in forks — platforms, mobile clients, cognitive-science memory experiments, a Rust rewrite, disciplined CN/VN distros.
- [`prime-agent/`](./prime-agent/) — **Unversioned reference.** Prime Agent fork census (2026-08, full 2,046-fork census): 95% fork-and-forget; active forks add model providers and Windows fixes; validated census methodology.
- [`sglang/`](./sglang/) — **Unversioned reference.** SGLang HiCache deep-dives: write_back vs write_through (and why vLLM/LMCache can't do write_back — architectural analysis with vLLM PR #43946 and TensorRT-LLM comparison), SWA device-only design (why 71.4% of KV stays on GPU).

## License

MIT — see [LICENSE](./LICENSE). Notes are personal commentary; all referenced code belongs to its respective authors.
