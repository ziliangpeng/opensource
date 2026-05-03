# Triton Attention Backend Flow (v0.16.0)

## Scope
Captures the Triton attention backend’s internal flow and constraints in v0.16.0, based on our discussion and code reading.

## 1) What it is
Backend class: `TritonAttentionBackend` (`vllm/v1/attention/backends/triton_attn.py`).
It is the portable, feature‑complete fallback backend that supports many attention patterns.

## 2) Core constraints (validation)
From `TritonAttentionBackend`:
- **Head size:** `head_size >= 32`
- **Block size:** multiple of **16**
- **Dtypes (compute):** `fp16`, `bf16`, `fp32`
- **KV cache dtype:** supports `fp8` variants (`fp8_e4m3`, `fp8_e5m2`)
- **Sinks:** supported
- **MM prefix:** supported
- **Attention types:** supports decoder + encoder + encoder‑decoder
- **Compute capability:** no explicit restriction (broad support)

## 3) Precision behavior (key point)
- Triton backend supports **FP8 KV cache**, but **attention compute** remains in higher precision (fp16/bf16/fp32).
- True FP8 attention compute (FP8 Q/K/V) is explicitly called out in v0.16.0 docs for **FlashAttention 3**, not Triton.

**Inference (no direct PR/issue found):**
- Triton attention likely avoids true 8‑bit compute due to kernel complexity, portability goals, and hardware/accuracy constraints. FA3/FlashInfer provide the low‑precision compute path instead.

---

## Sources (code/docs)
- `vllm/v1/attention/backends/triton_attn.py`
- `vllm/v1/attention/backend.py`
- https://docs.vllm.ai/en/v0.16.0/design/attention_backends/
