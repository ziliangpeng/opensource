# Background Review & Curator — Learning Loop

> **Phase 3** deep dive: how Hermes automatically learns from conversations,
> creates memory, builds and maintains skills, and consolidates its knowledge
> over time.
>
> Source: upstream `v2026.5.16` (tag `v2026.5.16`).

## Overview

Hermes has a **two-layer learning loop** that runs entirely in the background,
never blocking the user's conversation:

| Layer | When | What it does | File |
|-------|------|-------------|------|
| **Per-turn background review** | After each `run_conversation()` completes | Reviews the conversation for memory-worthy facts and skill-worthy procedures. Forks a full AIAgent to execute memory/skill tools. | `run_agent.py` (lines 4033–4503) |
| **Curator** | Periodically (~7 days) when agent is idle | Reviews the skill library for consolidation. Archives stale skills, merges overlapping ones into class-level umbrellas. | `agent/curator.py` (1,781 LOC) |

Together they form a closed learning loop:

```
Conversation → Background review → Memory + Skills updated
                                         ↓
                                  Curator (periodic)
                                         ↓
                              Skills consolidated & pruned
                                         ↓
                              Next session starts smarter
```

## Layer 1: Per-turn Background Review

### When does it trigger?

Two independent counters, checked at the end of `run_conversation()`:

| Trigger | Counter | Default interval | Config key |
|---------|---------|-----------------|------------|
| **Memory review** | User turns (`_turns_since_memory`) | Every 10 turns | `memory.nudge_interval` |
| **Skill review** | Tool iterations (`_iters_since_skill`) | Every 10 iters | `skills.creation_nudge_interval` |

Both are initialized during `AIAgent.__init__()`:

```python
self._memory_nudge_interval = 10
self._turns_since_memory = 0
self._iters_since_skill = 0
# ... overridden from config:
mem_config.get("nudge_interval", 10)           # → _memory_nudge_interval
skills_config.get("creation_nudge_interval", 10)  # → _skill_nudge_interval
```

**Memory trigger check** (line 12294, runs at top of each turn):

```python
_should_review_memory = False
if (self._memory_nudge_interval > 0
        and "memory" in self.valid_tool_names
        and self._memory_store):
    self._turns_since_memory += 1
    if self._turns_since_memory >= self._memory_nudge_interval:
        _should_review_memory = True
        self._turns_since_memory = 0
```

Gates: memory tool must be available AND `MemoryStore` must be initialized.

**Skill trigger check** (line 15979, runs AFTER agent loop completes):

```python
_should_review_skills = False
if (self._skill_nudge_interval > 0
        and self._iters_since_skill >= self._skill_nudge_interval
        and "skill_manage" in self.valid_tool_names):
    _should_review_skills = True
    self._iters_since_skill = 0
```

Gates: `skill_manage` tool must be in the valid tool set.

**Session resume hydration** (line 12255): When resuming a session with prior
history (common in gateway mode where each inbound message creates a fresh
AIAgent), the counters are reconstructed from the number of prior user turns
to avoid firing a review immediately on resume.

### The spawn point

At the very end of `run_conversation()`, after the response is delivered
(line 15993):

```python
if final_response and not interrupted and (_should_review_memory or _should_review_skills):
    try:
        self._spawn_background_review(
            messages_snapshot=list(messages),
            review_memory=_should_review_memory,
            review_skills=_should_review_skills,
        )
    except Exception:
        pass  # Background review is best-effort
```

Key design: **review runs AFTER response delivery** so it never competes
with the user's task for model attention.

### _spawn_background_review() — the fork

**File:** `run_agent.py`, lines 4284–4503

**What it does:**

1. Pick the right prompt based on which triggers fired:
   - Both → `_COMBINED_REVIEW_PROMPT`
   - Memory only → `_MEMORY_REVIEW_PROMPT`
   - Skills only → `_SKILL_REVIEW_PROMPT`

2. Create a **daemon thread** (`threading.Thread(target=_run_review, daemon=True)`)

3. Inside the daemon thread (`_run_review()`):
   - **Redirect stdout/stderr to `/dev/null`** — no user-visible output from the fork
   - **Auto-deny dangerous commands** — `_bg_review_auto_deny()` callback prevents deadlocks against the parent's prompt_toolkit TUI
   - **Fork a full AIAgent** with the same model, provider, base_url, api_key, credential_pool as the parent
   - **Inherit the parent's cached system prompt** — byte-identical prefix cache hit (~26% cost reduction on Sonnet 4.5, issue #25322)
   - **Tool whitelist** — only `memory` and `skills` toolsets are allowed. Everything else is denied at runtime:

```python
review_whitelist = {
    t["function"]["name"]
    for t in get_tool_definitions(
        enabled_toolsets=["memory", "skills"],
        quiet_mode=True,
    )
}
set_thread_tool_whitelist(review_whitelist, deny_msg_fmt=(
    "Background review denied non-whitelisted tool: {tool_name}. "
    "Only memory/skill tools are allowed."
))
```

4. Run `review_agent.run_conversation()` with:
   - The review prompt as the user message
   - `conversation_history=messages_snapshot` (the parent's entire message list)
   - Max 16 iterations

5. After completion, scan the review agent's messages for successful tool actions.
   Deduplicate against `messages_snapshot` to avoid re-surfacing stale results
   from the inherited history (issue #14944). Surface a compact summary:

```
💾 Self-improvement review: Memory updated · Skill created
```

### The review prompts

Three prompts live as class-level string constants:

| Prompt | Lines | Purpose |
|--------|-------|---------|
| `_MEMORY_REVIEW_PROMPT` | 4038–4047 | Short (~10 lines). "Review for user persona, desires, preferences, expectations. Save with memory tool or say 'Nothing to save.'" |
| `_SKILL_REVIEW_PROMPT` | 4049–4143 | Long (~95 lines). Detailed decision tree with 4 priority levels, anti-patterns, and format rules. |
| `_COMBINED_REVIEW_PROMPT` | 4145–4218 | Long (~75 lines). Both memory + skill instructions combined. |

**Skill review priority order (from `_SKILL_REVIEW_PROMPT`):**

```
1. UPDATE a currently-loaded skill (patched during this session)
2. UPDATE an existing umbrella skill (found via skills_list)
3. ADD a support file under an existing umbrella (references/ / templates/ / scripts/)
4. CREATE a new class-level umbrella skill
```

**Anti-patterns (things the review agent is told NOT to save):**

- Environment-dependent failures (missing binaries, fresh-install errors)
- Negative claims about tools or features ("browser tools do not work")
- Session-specific transient errors that resolved themselves
- One-off task narratives ("summarize today's market")

## Layer 2: Curator

**File:** `agent/curator.py` (1,781 LOC)

The curator is a **background skill maintenance orchestrator** that runs
periodically when the agent is idle, not per-turn like the background review.

### Configuration

| Parameter | Default | Config key |
|-----------|---------|------------|
| Enabled | True | `curator.enabled` |
| Interval | 7 days | `curator.interval_hours` |
| Min idle before run | 2 hours | `curator.min_idle_hours` |
| Stale threshold | 30 days | `curator.stale_after_days` |
| Archive threshold | 90 days | `curator.archive_after_days` |

Persistent state is stored in `~/.hermes/skills/.curator_state` (JSON).

### Automatic state transitions (pure Python, no LLM)

Before any LLM review, the curator applies deterministic transitions:

```
Agent-created skill
    │
    ├─ No activity for 30 days  →  STATE_STALE
    ├─ No activity for 90 days  →  STATE_ARCHIVED (moved to .archive/)
    └─ Stale but got used again →  STATE_ACTIVE (reactivated)
```

Pinned skills are never touched by any automatic transition.

```python
def apply_automatic_transitions(now=None) -> Dict[str, int]:
    """Returns counts: marked_stale, archived, reactivated, checked"""
```

### LLM-powered curator review

When the curator runs its full pass, it forks an AIAgent with the
`CURATOR_REVIEW_PROMPT` (~115 lines). The prompt instructs the agent to:

1. **Scan the full candidate list** of agent-created skills
2. **Identify prefix clusters** (skills sharing a first word or domain keyword:
   `pr-triage-*`, `hermes-config-*`, `gateway-*`, etc.)
3. **Consolidate** using one of three strategies:
   - **Merge into existing umbrella** — one skill is broad enough; patch it,
     archive the siblings
   - **Create a new umbrella** — no existing member is broad enough;
     write a new class-level SKILL.md, archive the narrow siblings
   - **Demote to support file** — move narrow content into the umbrella's
     `references/ / templates/ / scripts/` directory
4. **Output a structured YAML summary**:

```yaml
consolidations:
  - from: pr-triage-salvage
    into: pr-contribution-workflow
    reason: "PR triage and salvage are both part of PR lifecycle"
prunings:
  - name: debug-2026-03-12-oom
    reason: "session-specific, absorbed into k8s troubleshooting umbrella"
```

The prompt sets an aggressive target: "If you end the pass with fewer than
10 archives, you stopped too early."

### Dry-run mode

`hermes curator run --dry-run` produces the same analysis and structured
output but includes a banner that instructs the fork **not to mutate anything**:

```
═══════════════════════════════════════════════════════════════
DRY-RUN — REPORT ONLY. DO NOT MUTATE THE SKILL LIBRARY.
═══════════════════════════════════════════════════════════════
```

### CLI commands

```
hermes curator run          # Execute a full curator pass
hermes curator run --dry-run  # Preview only, no mutations
hermes curator pause        # Pause the curator schedule
hermes curator resume       # Resume the curator schedule
```

### Report output

Each run writes to `~/.hermes/logs/curator/YYYYMMDD-HHMMSS/`:
- `run.json` — structured run metadata
- `REPORT.md` — human-readable summary

## Design decisions

### Why fork a full AIAgent instead of a lightweight LLM call?

The review agent needs access to the **same tool environment** as the main
agent — `memory` tool reads/writes MEMORY.md/USER.md, `skill_manage` reads/
writes the skills directory. Forking a full AIAgent with a tool whitelist
ensures the review uses the exact same code paths as a user-initiated
memory/skill operation.

### Why not async?

The agent loop is entirely synchronous. The review runs in a **daemon thread**
after the response is delivered, so it never delays the user. The daemon
thread is fire-and-forget — if the main process exits before the review
completes, it's silently dropped.

### Why share the cached system prompt?

Without inheritance, the review fork rebuilds the system prompt from scratch
(fresh timestamp, fresh session_id, narrower toolset → different skills_prompt).
This produces different bytes → misses the Anthropic/OpenRouter prefix cache.
Inheriting the parent's `_cached_system_prompt` byte-identical saves ~26%
end-to-end cost (measured on Sonnet 4.5, PR #17276).

### Why a separate curator from per-turn review?

The per-turn review is optimized for **immediate signal** (user correction,
new preference, emerging technique). The curator is optimized for
**library-level hygiene** (consolidation, de-duplication, archiving).
Different cadences, different scopes, different prompts.
