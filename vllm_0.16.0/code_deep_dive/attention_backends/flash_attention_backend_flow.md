# FlashAttention Backend Flow (v0.16.0)

## Scope
Captures the FlashAttention backend’s internal decision flow in v0.16.0, based on our discussion and code reading.

## 1) Kernel version selection (FA2 vs FA3)
Decision logic lives in `vllm/v1/attention/backends/fa_utils.py:get_flash_attn_version()`.

**Defaults:**
- FA3 is used only on **Hopper (SM 9.x)** when supported.
- Otherwise FA2 is used.

**Forced FA2 fallbacks:**
- **Blackwell (SM10)** forces FA2.
- **ALiBi** forces FA2.

**Override:**
- `attention_config.flash_attn_version` can force FA2/FA3 (if supported).

## 2) Backend constraints (validation)
Constraints enforced by `FlashAttentionBackend.validate_configuration(...)`:
- **Head size:** `head_size % 8 == 0` and `head_size <= 256`
- **Block size:** multiple of **16**
- **Dtypes:** `fp16`, `bf16`
- **KV cache dtype:** FP8 KV only if FA3 is available; otherwise `auto/bfloat16`
- **Sinks:** requires FA3
- **Compute capability:** **SM ≥ 8.0**
- **Attention types:** supports decoder + encoder + encoder‑decoder

## 3) Metadata builder + CUDA‑graph support
`FlashAttentionMetadataBuilder` defines CUDA‑graph support:
- **FA3 → `AttentionCGSupport.ALWAYS`**
- **FA2 → `AttentionCGSupport.UNIFORM_BATCH`** (mixed prefill+decode full graphs unsafe)

AOT scheduling is **FA3‑only**:
- `aot_schedule = (get_flash_attn_version() == 3)`

During CUDA‑graph capture, `max_num_splits` is set to pre‑allocate buffers; otherwise FA3 heuristics are used.

## 4) Forward path (high‑level)
- The metadata builder constructs `FlashAttentionMetadata` (seq lengths, block tables, slot mapping, scheduler metadata).
- Forward calls the FlashAttention varlen kernel using that metadata + KV cache.

---

## Sources (code)
- `vllm/v1/attention/backends/fa_utils.py`
- `vllm/v1/attention/backends/flash_attn.py`
