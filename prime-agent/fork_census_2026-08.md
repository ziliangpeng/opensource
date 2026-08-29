# Prime Agent fork census (full, 2026-08)

**Last Updated**: 2026-08-29
**Method**: full census of all 2,042 reachable forks of `PrimeIntellect-ai/prime-agent` (19k★, 2,046 forks) — `git ls-remote` HEAD SHA per fork, compared against upstream's reachable refs; compare API per active fork. Raw data: `/tmp/pa_full_compares.json`, `/tmp/pa_own_forks.json`.

## Census pitfalls (validated methodology)

- Stars-sorted sampling of a 0-star-majority fork list is a **random sample**, not a top-list (GitHub ties are arbitrary) — an initial "top 325" scan missed the single largest fork.
- Forks **share the parent's object store**: "commit exists in upstream repo" is meaningless — test reachability from upstream's refs.
- `--filter=blob:none` partial clones lazy-fetch any object, defeating object-existence checks.

## Distribution

| Class | Count | Share |
|---|---|---|
| No own commits (synced or untouched) | 1,942 | 95.0% |
| **Own commits** | **100** | 4.9% |

Only 182 distinct HEAD SHAs across 2,042 forks — most sit on identical upstream commits (one commit has 671 forks parked on it). All 100 active forks have distinct HEADs (independent work, no copying). Scale: 17 forks added exactly 1 commit; ~half under 10; only 6 over 100.

## Largest fork (missed by star-sampling)

**`zhengr/prime-agent` (3,987 ahead, 0★)**: long-term deep customization — internal GLM / Prime Inference model catalog, Docker releases, own version line (v0.2.7), heartbeat/attach_image/subagent-messaging changes. Others: `JeffCarpenter` (322, process governance/adoption guard), `avion23` (128), `junhoyeo/deadhand` (87, refine progress display), `mcclurejt` (77, Bedrock Opus 5 prompt caching), `oneiron-dev` (64, per-child reasoning level).

## Themes of the 100 active forks

- **Model/provider support (largest, ~25)**: GLM, Deep Infra, Venice, MiniMax M3, Grok 4.6, Kimi OAuth compaction, Z.AI reasoning effort, Bedrock caching, OpenRouter web search, Anthropic fast mode — the #1 motivation for forking
- **Windows support (~8)**: daemon sockets, console flicker, native PowerShell — clear pain point
- **Productization/rebrands (~6)**: Electron desktop, codegraph integration — early Hermes-style "distro" pattern
- **RLM/subagent enhancements (~10)**: per-child reasoning level, role-routed subagents, rlm.run cwd/thinking, non-blocking subagents
- **CI/release/governance (~15)** + misc small changes (~35)

## Hermes vs Prime Agent fork ecosystems

| | Hermes Agent | Prime Agent |
|---|---|---|
| Total forks | ~48,000 | 2,046 |
| Own commits | >50% (star-sample) | 100 (4.9%, full census) |
| Top fork | 12,848 commits | 3,987 commits |
| Dominant themes | product distros, messaging platforms, memory experiments | model-provider support, Windows, RLM hacks |
| Overlap | long-run stability, localization, Discord/Telegram | same |

Core conclusion: 95% of Prime Agent forks are "fork and forget" (like most hot GitHub projects); of the active 100, the most common act is **adding the model provider you want**, then Windows fixes — both "can't wait for upstream" needs.

## See also

- kb vault: ai/labs/prime-intellect (prime_agent, release + technical report notes)
- Hermes Agent fork census (this repo, ../hermes-agent/]
