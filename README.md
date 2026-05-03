# opensource

Notes from reading open source project codebases — architecture maps, design decisions, interesting patterns, and "how does this actually work" deep-dives.

## Layout

One directory per project:

```
<project-name>/
  README.md         # entry point: what the project is, why it's interesting, reading roadmap
  architecture.md   # high-level component map
  notes/            # focused deep-dives on subsystems / files / commits
```

## Conventions

- Link to specific commits (`github.com/owner/repo/blob/<sha>/path`) so notes don't rot when upstream changes.
- Quote sparingly; prefer summarizing in your own words and linking to the source.
- Each note starts with **what** is being read (file/module/PR) and **why** it's worth a look.

## Index

- [`vllm_0.16.0/`](./vllm_0.16.0/) — vLLM v0.16.0: engine core, scheduler, worker, attention backends (FlashAttn / FlashInfer / Triton / ROCm AITER), KV transfer, MoE, paged attention, divisibility constraints.

## License

MIT — see [LICENSE](./LICENSE). Notes are personal commentary; all referenced code belongs to its respective authors.
