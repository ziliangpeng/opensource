# vLLM v0.16.0 — Next Deep Dives (2‑2‑2)

## 1) Attention backend selection & execution path
- **Why it matters (2):** This is the core performance path; backend choice directly affects throughput/latency and KV‑append behavior.
- **Key questions (2):** How is the backend selected (FlashAttention/FlashInfer/Triton) and what metadata / code paths differ per backend?
- **Outputs (2):** A flow map from scheduler/runner → attn backend → kernel + a comparison table of backend capabilities/constraints.

## 2) Distributed execution (TP/PP/EP + NCCL)
- **Why it matters (2):** Multi‑GPU scaling bottlenecks dominate real deployments; correctness + comm overlap are critical.
- **Key questions (2):** Where are collectives launched (all‑reduce/all‑gather/send/recv), and how do TP/PP/EP boundaries map to vLLM modules?
- **Outputs (2):** A stage‑by‑stage diagram + a concise list of NCCL calls and their trigger points.

## 3) KV cache deep dive (lifecycle + block tables)
- **Why it matters (2):** KV cache is the dominant memory footprint and defines max concurrency.
- **Key questions (2):** How are block tables/slot mappings updated over time, and where do eviction/preemption policies hook in?
- **Outputs (2):** A lifecycle timeline + a data‑structure map (block tables, slot mappings, cache groups).
