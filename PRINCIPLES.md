# Repository Reading Principles

A methodology for reading and understanding open-source codebases — developed from reading vLLM, Hermes Agent, prime-rl, and other projects.

## Philosophy

We read code to understand **why** a system is built the way it is, not just **what** it does. The goal is to build a mental model of the architecture that lets us predict where changes live, trace through a request's lifecycle, and recognize which parts are the hardest to get right.

## Reading process

### Phase 0: Orientation (before reading a single source file)

Answer these from the project's README, docs, and file tree:

| Question | Why it matters |
|----------|---------------|
| What problem does this project solve? | Framing determines what code is core vs peripheral. |
| Who is the primary user? | An end-user library and a framework meant to be extended have very different code shapes. |
| What are the "hard parts"? | Inference kernels, distributed communication, stateful session management, async IO — each demands different reading depth. |
| What's the external interface(s)? | CLI, SDK, REST API, library imports — these are the entry points to trace from. |
| What is the repo's size and structure? | How many files? Which directories are the largest? This tells you where complexity concentrates. |

### Phase 1: Skeleton — build a file-tree mental map

1. **Find the entry point(s)** — `main()` or equivalent. Many projects have multiple (CLI, server, library import).
2. **List all top-level directories** and write one sentence for each.
3. **Identify the dependency graph** — what modules import what. ([`tools/registry.py` ← `tools/*.py` ← `model_tools.py` ← `run_agent.py`] is an example of a critical dependency chain.)
4. **Find the "hot path"** — for a request to go from user input to response, which files does it touch?
5. **Place boundary markers** — where does the project depend on external libraries, and what are they for?

Deliverable: `repo_structure.md` — a directory-by-directory breakdown.

### Phase 2: Hot path — trace one complete request

Pick the simplest realistic request and trace it end to end:

1. **Entry** → parsing/deserialization → validation
2. **Core logic** → what's the main loop / state machine?
3. **External call** → model inference / database / API call
4. **Response** → serialization → exit

For each step, note:
- What module/file handles it?
- What data structures are passed along?
- What errors can occur?

Deliverable: `request_lifecycle.md` — the path of one request annotated with files and data structures.

### Phase 3: Deep dives — hardest parts first

Prioritize deep dives by **technical difficulty**, not by your comfort level:

1. **The concurrency/async model** — is it threaded? async event loop? multiprocess? How does it handle contention?
2. **The state management** — what is the persistent state? How is it shared across requests/sessions/users?
3. **The most performance-sensitive path** — what's in the inner loop?
4. **The extension / plugin / customization mechanism** — how does the project let users add behavior?

Each deep dive gets its own file. Start with a clear **what** and **why**.

### Phase 4: Design audit — deferred questions

These come after you've read enough code:

- What design choices surprised you? (These are where the project differs from alternatives.)
- What would you challenge? (Not as criticism — as understanding: "why not X?")
- What patterns are repeated? (Utilities, base classes, decorators — these encode the project's conventions.)

Deliverable: `design_notes.md` — design decisions, patterns, and open questions.

## Writing conventions

- **Link to specific commits/line ranges** (`github.com/owner/repo/blob/<sha>/path#L10-L30`) so notes survive upstream changes. (Clone-relative paths for local reference.)
- **Quote sparingly** — summarize in your own words and link to the source. Inline only when the exact wording matters.
- **Each note starts with what** is being read and **why** it's worth a look.
- **Tables > prose** for comparisons, trade-offs, and structural breakdowns.
- Python files use `module.function` notation (e.g. `run_agent.AIAgent.run_conversation`).
- **Directory names** should be intuitive and topic-based, not numbered.
- When reading a deep-dive, **record the "why"** behind design decisions as you go — don't defer it all to the design audit phase.

## Scope boundaries

- One directory per project (e.g. `vllm_0.16.0/`, `hermes_agent_v0.14.0/`, `triton/`).
- **Versioned folders** (`<project>_<version>/`) are for **code deep-dives and walkthroughs** — tracing request lifecycles, reading source files, understanding the architecture at the code level. These are version-pinned because code changes between releases.
- **Unversioned folders** (`<project>/` without a version suffix) are for **general project reference** — project history, release evolution, contributor analysis, study roadmaps, architecture overviews, and any information that spans multiple versions or isn't tied to a specific code snapshot.
- A project can have both: an unversioned folder for reference materials and a versioned folder for a specific release's code walkthrough.
- Focus on **what the project's authors consider hard** — the code that had the most PRs, the most comments, the most complex test setup.
- Document what you **don't** understand as much as what you do. Open questions are valuable.

## Anti-patterns

- ❌ Reading files alphabetically — you'll drown in utilities before reaching anything interesting.
- ❌ Getting stuck on every detail — accept a hand-wavy understanding on first pass and deepen iteratively.
- ❌ Documenting every file — document the **connections** between files, not the files themselves.
- ❌ Over-focusing on tests for understanding — tests tell you what the code should do, not why it's designed that way. (But tests are great for discovering edge cases and undocumented behavior.)
