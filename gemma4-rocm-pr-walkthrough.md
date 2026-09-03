# Gemma4 on ROCm: A PR-by-PR Walkthrough of the CAI×AMD Tuning Stack

**Date**: 2026-09-02 · **Source**: Character Weekly Sep 02 deck (p6–p7), Slack `ext-cai-amd-digitalocean` + `tmp-gemma-4-wg` threads, GitHub PRs (read 2026-09-02)
**Related**: `~/code/work-kb/cai/amd-syncs/2026-09-02-vllm-regression-gibberish-timeline.md`

Seven PRs, three layers, four people — what "optimizing one model on one vendor stack" actually looks like when you read the artifacts.

## The PRs

| PR | Repo | Who | Layer | What | Status |
|---|---|---|---|---|---|
| aiter#5062 | AITER | Mustafa (CAI) | tuning | 88-row tuned GEMM CSV for gemma-4-31b FP8-block (85 CK + 3 CKTile shapes, M=1..16384); also fixes a pre-existing multi-arch codegen bug in `candidate_kernels_by_name` | open |
| aiter#5063 | AITER | Mustafa (CAI) | config | Hoists `BLOCK_M=128` out of the `else` branch in `select_2d_config` — wide-head (≥256) prefills were silently getting BLOCK_M=16. −6~8% prefill latency | open |
| aiter#5027 | AITER | Matvei (AMD) | kernel (HIP) | head_dim 512 + weightless V-norm in `fused_qk_norm_rope_cache_pts_quant_shuffle`; runtime `v_norm` toggle | **merged** |
| aiter#4044 | AITER | Alexandra (AMD) | tuning | Unified-attention configs for head 256/512 (Gemma4); bit-comparable, config-only; tuned on CDNA4 | merged |
| vllm#53273 | vLLM | Matvei (AMD) | framework | Fuse post-attention residual add into pre-FFN RMSNorm (2-arg `fused_add_rms_norm`), aligns Gemma4 with gemma3/llama threading | open |
| vllm#53874 | vLLM | Matvei (AMD) | framework | Register GeGLU in `act_mul_and_fp8_group_quant` fusion pass (kernel already had gelu_tanh mode; wrapper hardcoded silu) | open |
| vllm#53918 | vLLM | Matvei (AMD) | framework | Enable `fuse_qk_norm_rope_kvcache` for Gemma4 (60/60 layers fused; +3~7% throughput, −6~9% ITL) | open |

Plus: Luciano Martins (Google DeepMind) wrote the original day-0 Gemma4 vLLM port (PR vllm#38826, 2026-04-02, 5k lines) and the MTP spec-decode support (vllm#41745).

## Layer map (what lives where)

```
HIP/CK kernels        aiter#5027 (head512+V-norm in fused qk/rope/cache op)
tuning artifacts      aiter#5062 CSV (shape → kernelId/splitK + us/tflops/bw/errRatio)
config selection      aiter#5063 (BLOCK_M routing), aiter#4044 (per-head-size configs)
fusion passes         vllm#53273/#53874/#53918 (Matcher + FusionPass registration)
model port            lucianomartins day-0 port (vllm#38826)
```

## Patterns worth stealing

1. **Tuned-CSV convention is per-model but generic-in-content.** `aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_<model>.csv` exists for dsv3/qwen3_235b/glm5_1/kimik3/minimax… Each row is pure (M,N,K)→config; the model name is a discovery key. Auto-picked-up via `get_config_file()` glob — zero code change to deploy.
2. **CSV row = lookup + audit trail.** `us` enables incremental re-tuning (`min_improvement_pct=3`), `errRatio` is the correctness audit column, `tflops/bw` make each row instantly readable as bandwidth- vs compute-bound. See `kernels/docs/kb/aiter-gemm-tuner-errratio.md`.
3. **Silent perf bugs live in config selection, and only E2E measurement catches them.** #5063: BLOCK_M=128 for large prefill was nested under `head_size < 256`, so wide-head prefills ran at 1/8 utilization with zero errors. Found only because Mustafa measured 32k prefill on real hardware. Bitwise-correct ≠ fast.
4. **Coverage of the tuning matrix is ongoing labor.** #4044 tuned wide-head configs on CDNA4/gfx950; the gfx942 wide-head prefill cell stayed untuned until Mustafa hit it. No CI layer catches "this shape's config got worse" (AITER's `kimi-perf-downstream.yaml` is the one existing E2E perf gate — per-model, opt-in, throughput-floor pattern worth copying for Gemma4).
5. **Fusion = shared vendor kernel + matcher registration.** #53874 didn't write a kernel — aiter's `act_mul_and_fp8_group_quant` already had `gelu_tanh`; vLLM's wrapper hardcoded silu. The work is: profile → find standalone passes between GEMMs → check vendor kernel catalog → register a Matcher. `triton_poi_fused_add_*` in a torch-compile trace is the smell.

## People map

- **Matvei Pashkovski** (AMD, `mpashkovskii`) — full-stack: kernel (5027) + fusion passes (53273/53874/53918). Co-author of Gemma4 support effort.
- **Alexandra Sidorova** (AMD) — unified-attention wide-head configs (#4044).
- **Mustafa Yildirim** (CAI) — AITER GEMM tuning (5062/5063), the only CAI-side kernel-repo contributor in this batch.
- **Mike Martin** (CAI) — serving-level tuning: MTP k sweeps, `ROCM_AITER_UNIFIED_ATTN` evals, prompt-replay benchmarks, SGLang-side Gemma4 kernel experiments.
- **Luciano Martins** (Google DeepMind) — day-0 port + MTP spec decode. Role model: framework-side "make the model exist", not perf-tune.
- **Mehmet Cagri** (AMD) — original unified-attention author, reviewed #4044.
