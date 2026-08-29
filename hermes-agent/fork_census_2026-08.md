# Hermes Agent fork census (2026-08)

**Last Updated**: 2026-08-29
**Method**: `gh` compare API over the 239 most-starred forks of `NousResearch/hermes-agent` (237k★, ~48k forks); read ahead-counts, diffs, commit messages. Over half of top-starred forks have real changes (1 to thousands of commits). Full-census methodology (validated later on Prime Agent): [[../../../ai/labs/prime-intellect/prime_agent_fork_census_2026-08|Prime Agent fork census]].

## What forks add (by theme)

1. **New platforms/channels (most common)**: Telegram deep customization (topic-level model binding, `VKirill`), LINE adapter (`rdnot/medical-research`), Inkbox email+SMS+voice, TrueConf videoconf platform, Slack tweaks.
2. **Mobile clients (hot direction)**: `TheTom/hermes-go` Flutter iOS/Android app (App Store submitted); `adybag14-cyber` (178★, 880 ahead — Android native, on-device Qwen/Gemma via LiteRT, OAuth, release pipeline).
3. **Web API / GUI**: `outsourc-e` api-server (sessions/memory/skills/SSE — CLI agent as web service); `gary-the-ai` web console; `AtomicBot-ai/atomic-hermes` (176★, Electron desktop, rebrand).
4. **Memory enhancement (technically most interesting)**: `zapabob/hermes-agent-windows` (1122 ahead) — Ebbinghaus forgetting curves, semantic graph, sleep-style "dream consolidation", prediction-error reconsolidation; `itsXactlY` Dream Engine; affective-systems experiments.
5. **Performance / ports**: `agent-bob-the-builder/ferris` (373 ahead) — 16 Python modules rewritten as Rust crates (O(1) dispatch HashMaps, precompiled regex); the most hardcore fork.
6. **Localizations**: `Eynzof/Hermes-CN-Core` (73★, disciplined numbered-patch queue + upstream-watch workflow), Vietnamese edition, `xinbenlv/zn-hermes-agent` carried-patch queue.
7. **Vertical specializations**: medical research, crypto KOL analysis (BSC on-chain), Raspberry Pi IoT, threat defense, multi-agent "rooms" (`novique-ai/retinue`).
8. **Architecture experiments**: `waefrebeorn/slermes` (2029 ahead, reverse-engineering-style bootleg), systematic test-hardening, auto-benchmark self-improvement loops.

## Notable forks table

| Fork | ★ | Ahead | One-liner |
|---|---|---|---|
| adybag14-cyber | 178 | 880 | Android native + local models |
| AtomicBot-ai | 176 | 108 | Electron desktop productization |
| Eynzof/Hermes-CN-Core | 73 | 428 | Most disciplined CN distro |
| TheTom/hermes-go | 12 | 124 | Flutter mobile app |
| novique-ai/retinue | 6 | 231 | Multi-agent rooms + web UI |
| agent-bob-the-builder | 5 | 373 | Python→Rust rewrite (16 crates) |
| zapabob/windows | 2 | 1122 | Cognitive-science memory |
| waym0reom3ga/autolycus | 12 | 12,848 | Long-term rewrite branch |

## Takeaways

1. **Forks are distros, not contributions**: top forks are branded derivatives maintained via upstream-sync workflows with a patch layer on top.
2. **Three demand gaps upstream doesn't fill**: mobile clients, web/HTTP API, more messaging platforms.
3. **Memory is the favorite experimental playground** (forgetting curves, dream consolidation, semantic graphs recur constantly).
4. Serious Rust rewrites and 12k-commit branches signal real dissatisfaction with upstream performance/stability.
5. Of ~48k forks, nearly all are sync-only; substantive work concentrates in ~50.

## See also

- kb vault: ai/labs/nous-research · [[../../../ai/labs/prime-intellect/prime_agent_fork_census_2026-08|Prime Agent fork census]]
