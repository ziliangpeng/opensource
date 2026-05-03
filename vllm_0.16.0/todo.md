# TODO — vLLM v0.16.0 Deep Dives (2‑2‑2)

## 1) Attention backend selection & execution path
- **Why it matters (2):** Core performance path; backend choice drives throughput/latency and KV‑append behavior.
- **Key questions (2):** How is backend selected (FlashAttention/FlashInfer/Triton), and what metadata/code paths differ?
- **Outputs (2):** Flow map runner → backend → kernel + capability/constraint table.

## 2) Distributed execution (TP/PP/EP + NCCL)
- **Why it matters (2):** Multi‑GPU scaling bottlenecks dominate deployments; comm overlap is critical.
- **Key questions (2):** Where are collectives launched, and how do TP/PP/EP boundaries map to modules?
- **Outputs (2):** Stage diagram + list of NCCL call sites and triggers.

## 3) KV cache deep dive (lifecycle + block tables)
- **Why it matters (2):** KV cache dominates memory and limits concurrency.
- **Key questions (2):** How are block tables/slot mappings updated, and where do eviction/preemption hooks live?
- **Outputs (2):** Lifecycle timeline + data‑structure map.
