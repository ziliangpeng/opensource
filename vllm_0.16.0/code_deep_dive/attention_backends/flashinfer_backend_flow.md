# FlashInfer Backend Flow (v0.16.0)

## Scope
Captures the FlashInfer backend’s internal decision flow in v0.16.0, based on what we discussed today.

## 1) TRT‑LLM vs native FlashInfer (primary branch)
Decision logic lives in the FlashInfer metadata builder (inside `vllm/v1/attention/backends/flashinfer.py`).

**Key checks:**
- `can_use_trtllm_attention(num_qo_heads, num_kv_heads)` gates TRT‑LLM usage.
- If **KV transfer (P/D disaggregation)** is enabled, TRT‑LLM is **disabled** (requires contiguous KV cache).

**Decode preference:**
- If TRT‑LLM is allowed, **decode prefers TRT‑LLM** (`use_trtllm_decode_attention = True`).

**Sinks:**
- If sinks are required and TRT‑LLM is not available, FlashInfer raises `NotImplementedError` (sinks require TRT‑LLM on Blackwell).

## 2) Q dtype selection
- If TRT‑LLM is allowed and `disable_flashinfer_q_quantization` is **not** set → `q_data_type = kv_cache_dtype` (fp8 when KV cache is fp8).
- Otherwise → `q_data_type = model dtype`.

## 3) Quantization scope (important)
- FlashInfer **does not** implement W4A4 or full weight/activation quantization.
- It supports **FP8 KV cache**, but **compute remains fp16/bf16**.

## 4) Constraints / validation (summary)
- **Compute dtypes:** fp16, bf16
- **KV cache dtypes:** auto, bf16, fp8 (e4m3/e5m2)
- **Head sizes:** 64, 128, 256
- **Block sizes:** 16, 32, 64
- **Compute capability:** SM 7.5–12.1
- **Sinks:** only supported when TRT‑LLM attention is available (SM100); otherwise rejected
- **KV layout:** on SM10, FlashInfer requires **HND** KV layout

## 5) CUDAGraph behavior (summary)
- If decode full‑cudagraphs are enabled, FlashInfer allocates a **decode wrapper per batch size**.
- Capture size is capped by `max_cudagraph_capture_size` when set.
- TRT‑LLM decode is preferred when available to enable better cudagraph behavior.

---

## Sources (code)
- `vllm/v1/attention/backends/flashinfer.py`
