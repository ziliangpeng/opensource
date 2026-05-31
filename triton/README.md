# triton

Notes on the Triton language and compiler — kernel authoring, autotuning, compilation pipeline, and GPU architecture-specific tuning considerations.

## Index

- [`history.md`](./history.md) — Project history, statistics (6,284 commits, 590+ contributors, 62M monthly PyPI downloads), top 10 contributors, and the full timeline from ISAAC (2016) → MAPL paper (2019) → OpenAI rewrite → PyTorch 2.0 adoption → community governance.
- [`how-autotune-works.md`](./how-autotune-works.md) — How Triton's `@autotune` works: candidate list, early config prune, perf model, benchmark & cache, and why it doesn't search the space for you.
- [`study-roadmap.md`](./study-roadmap.md) — A structured 4-phase study roadmap from writing Triton kernels to understanding the compiler internals, with reading materials, exercises, and validation checkpoints for each phase.
- [`architecture.md`](./architecture.md) — Complete codebase structure map with annotated directory tree, component architecture diagram (Mermaid), layout lifecycle visualization, key file lookup table, and debugging environment variables.
