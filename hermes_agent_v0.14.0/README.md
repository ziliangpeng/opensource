# Hermes Agent v0.14.0

Code reading notes for [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) — an open-source, cross-platform AI assistant that runs via CLI, TUI, gateway (Telegram / Discord / Slack / SMS / …), ACP (VS Code / Zed), Kanban board agents, scheduled cron jobs, and more.

## Version

<table>
<tr><th>Date tag</th><th>Semver</th><th>Date</th><th>Commits since previous</th></tr>
<tr><td><code>v2026.3.12</code></td><td><b>v0.2.0</b></td><td>2026-03-12</td><td>—</td></tr>
<tr><td><code>v2026.3.17</code></td><td><b>v0.3.0</b></td><td>2026-03-17</td><td>658</td></tr>
<tr><td><code>v2026.3.23</code></td><td><b>v0.4.0</b></td><td>2026-03-23</td><td>542</td></tr>
<tr><td><code>v2026.3.28</code></td><td><b>v0.5.0</b></td><td>2026-03-28</td><td>188</td></tr>
<tr><td><code>v2026.3.30</code></td><td><b>v0.6.0</b></td><td>2026-03-30</td><td>141</td></tr>
<tr><td><code>v2026.4.3</code></td><td><b>v0.7.0</b></td><td>2026-04-03</td><td>235</td></tr>
<tr><td><code>v2026.4.8</code></td><td><b>v0.8.0</b></td><td>2026-04-08</td><td>337</td></tr>
<tr><td><code>v2026.4.13</code></td><td><b>v0.9.0</b></td><td>2026-04-13</td><td>554</td></tr>
<tr><td><code>v2026.4.16</code></td><td><b>v0.10.0</b></td><td>2026-04-16</td><td>339</td></tr>
<tr><td><code>v2026.4.23</code></td><td><b>v0.11.0</b></td><td>2026-04-23</td><td>1,219</td></tr>
<tr><td><code>v2026.4.30</code></td><td><b>v0.12.0</b></td><td>2026-04-30</td><td>1,150</td></tr>
<tr><td><code>v2026.5.7</code></td><td><b>v0.13.0</b></td><td>2026-05-07</td><td>881</td></tr>
<tr><td><code><b>v2026.5.16</b></code></td><td><b>v0.14.0 🎯</b></td><td><b>2026-05-16</b></td><td><b>847</b></td></tr>
</table>

### Versioning convention

Hermes Agent uses a **dual versioning** scheme:

1. **Date-based tag**: `vYYYY.M.D` — the primary external identifier. Maps to `v0.<minor>.0` semver one-to-one.
2. **Semver**: `v0.<minor>.0` — used internally in release commit messages (e.g. `chore: release v0.14.0 (2026.5.16) (#26862)`).

**There are no patch releases.** Every tag is a minor bump. The project ships continuous delivery: daily commits, cut a release every ~5 days (range 2–9 days). From v0.2.0 (Mar 12) to v0.14.0 (May 16): **13 minor versions in 65 days**, averaging ~600 commits per release.

### Total repo size

- First tag (`v2026.3.12`): ~1,394 commits
- Latest tag (`v2026.5.16`): ~8,485 commits
- Repository: ~17,000+ tests across ~900 test files (as of May 2026)

## Reading roadmap

(To be filled as we dive in.)

## Documents

(To be filled.)

## Source

This directory analyzes version **v0.14.0** (tag `v2026.5.16`). Source code available at `~/.hermes/hermes-agent/` (the fork checkout, which includes backports on top; check the tag directly for pure upstream).

> **Upstream:** https://github.com/NousResearch/hermes-agent
> **Release tag:** `v2026.5.16`
