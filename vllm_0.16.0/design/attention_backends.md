# Attention Backends

Based on `docs/design/attention_backends.md` in the vLLM v0.16.0 source.

## Overview

vLLM uses a pluggable attention backend architecture. The memory management layer (paged KV cache, block tables — see [[ai/llm/inference/frameworks/vllm_0.16.0/design/paged_attention]]) is shared across all backends. The backend only handles the attention computation kernel itself.

There are 21 registered backends in `AttentionBackendEnum` (`vllm/v1/attention/backends/registry.py`), covering NVIDIA, AMD/ROCm, CPU, and custom backends.

## Backend Selection

When no backend is specified (`--attention-backend` or `AttentionConfig`), vLLM iterates through a **priority-ordered list** per hardware platform and picks the first compatible one. Compatibility is validated against: model dtype, KV cache dtype, head size, block size, compute capability, attention type, etc. If you manually specify an incompatible backend, vLLM raises an error with the specific reason.

### Standard Attention (MHA/GQA) Priority

| Priority | Blackwell (SM 10.x) | Ampere/Hopper (SM 8.x-9.x) |
|---|---|---|
| 1 | FlashInfer (TRTLLM) | FlashAttention |
| 2 | FlashAttention | FlashInfer |
| 3 | Triton | Triton |
| 4 | Flex Attention | Flex Attention |

The default flips between FlashAttention and FlashInfer depending on GPU generation.

### MLA (DeepSeek-style) Priority

MLA has its own set of backends, with **separate backends for prefill and decode** (unlike standard attention which uses one backend for both).

Decode priority:

| Priority | Blackwell (SM 10.x) | Ampere/Hopper (SM 8.x-9.x) |
|---|---|---|
| 1 | FlashInfer MLA | FlashAttention MLA |
| 2 | CUTLASS MLA | FlashMLA |
| 3 | FlashAttention MLA | FlashInfer MLA |
| 4 | FlashMLA | Triton MLA |

Prefill backends (selected at runtime): TRT-LLM Ragged (default on Blackwell) → FlashInfer → cuDNN → FlashAttention (fallback).

## All Backends

### NVIDIA — Standard Attention

**FlashAttention** (`FLASH_ATTN`) — vLLM's fork of Dao-AILab FlashAttention (see [[ai/llm/inference/frameworks/vllm_0.16.0/design/paged_attention#Kernel Layer: vLLM's FlashAttention Fork and AMD's aiter]]). Outputs two builds: FA2 (SM 8.0+, Ampere/Ada) and FA3 (SM 9.x, Hopper only). Supports all attention types (decoder, encoder, encoder-decoder). FA3 supports attention sinks and FP8 KV cache; FA2 does not.

**FlashInfer** (`FLASHINFER`) — external pip package (`flashinfer-python`). On Blackwell, uses TRT-LLM attention under the hood (supports sinks). On Ampere/Hopper, uses native FlashInfer (no sinks). Optimized for high-concurrency decode. NVIDIA-only.

**Triton Attention** (`TRITON_ATTN`) — Triton-based implementation. Supports attention sinks and multimodal prefix full attention. Works on any compute capability. Less optimized than FA/FlashInfer but most feature-complete.

**Flex Attention** (`FLEX_ATTENTION`) — PyTorch-native API (PyTorch 2.5+) where you define custom attention patterns via a Python `score_mod` function. PyTorch compiles it into a Triton kernel via `torch.compile`. The most flexible backend — only one supporting multimodal prefix full attention alongside Triton. Slowest of the four, used as a fallback when specialized backends can't handle the attention pattern.

**Tree Attention** (`TREE_ATTN`) — specialized for tree-structured decoding (speculative decoding verification).

### NVIDIA — MLA (Multi-head Latent Attention)

MLA backends exist specifically for DeepSeek-style models that project KV into a low-rank compressed latent vector. Each backend has very specific constraints:

| Backend | Block Size | Compute Cap. | Notes |
|---|---|---|---|
| `CUTLASS_MLA` | 128 only | SM 10.x | Blackwell CUTLASS kernel |
| `FLASHINFER_MLA` | 32, 64 | SM 10.x | FlashInfer's MLA path |
| `FLASHMLA` | 64 only | SM 9.x-10.x | Purpose-built for DeepSeek |
| `FLASHMLA_SPARSE` | 64 only | SM 9.x-10.x | Sparse attention variant, bf16 only, head_size=576 only |
| `FLASH_ATTN_MLA` | multiples of 16 | SM 9.x | FlashAttention with MLA head layout |
| `TRITON_MLA` | Any | Any | Triton fallback |

### AMD/ROCm

AMD uses a completely separate ecosystem — no overlap with NVIDIA backends:

| Backend | Library | Notes |
|---|---|---|
| `ROCM_ATTN` | Triton + PagedAttention | Standard attention |
| `ROCM_AITER_FA` | AMD aiter | FlashAttention equivalent |
| `ROCM_AITER_UNIFIED_ATTN` | AMD aiter | Unified attention path |
| `ROCM_AITER_MLA` | AMD aiter | MLA decode, block_size=1 |
| `ROCM_AITER_MLA_SPARSE` | AMD aiter | Sparse MLA, head_size=576 |
| `ROCM_AITER_TRITON_MLA` | aiter + Triton | Hybrid MLA |

### Other

- `CPU_ATTN` — CPU-only attention, supports fp32
- `FLASH_ATTN_DIFFKV` — FlashAttention variant for models with different key and value head dimensions
- `NO_ATTENTION` — placeholder for models without attention (e.g., pure SSM)
- `TORCH_SDPA` — PyTorch's scaled dot-product attention, only used for ViT encoders
- `CUSTOM` — placeholder for third-party backends, must be registered before use

## Feature Matrix — Key Capabilities

Not all backends support the same features:

| Feature | Supported by |
|---|---|
| **Attention sinks** | FA3, FlashInfer (TRTLLM on Blackwell), Triton |
| **FP8 KV cache** | FA3, FlashInfer, Triton, CUTLASS MLA, FlashInfer MLA, FlashMLA |
| **Multimodal prefix full attention** | Flex Attention, Triton |
| **Encoder-decoder** | FlashAttention, Triton, Flex Attention, CPU |
| **Sparse attention** | FlashMLA Sparse, ROCm AITER MLA Sparse |

### Attention Sinks

Attention sinks (from the StreamingLLM paper) are tokens at the very beginning of a sequence that accumulate disproportionately high attention scores regardless of their semantic content. In sliding window attention, if the window slides past these sink tokens, model quality collapses — the attention distribution destabilizes.

The fix: always keep the first K tokens visible to all subsequent tokens, even outside the sliding window. This requires kernel-level support — the attention kernel must handle two non-contiguous KV ranges (sink tokens at position 0 + the sliding window at the current position). Models like GPT-OSS-120B require this.

### Flex Attention

Flex Attention (PyTorch 2.5+) lets you define arbitrary attention patterns via a Python function:

```python
def sliding_window_with_sinks(score, b, h, q_idx, kv_idx):
    is_sink = kv_idx < 4
    is_in_window = q_idx - kv_idx < 4096
    return torch.where(is_sink | is_in_window, score, -float("inf"))
```

PyTorch compiles this into a fused Triton kernel without materializing the full attention matrix. Maximum flexibility (any attention pattern expressible as a score modification), but slower than purpose-built kernels. Used as a fallback when specialized backends don't support the required pattern.

## CUDA Graph Compatibility

Each backend declares its CUDA graph support level (see [[ai/llm/inference/frameworks/vllm_0.16.0/design/cuda_graphs]]):

| Support Level | Backends |
|---|---|
| `ALWAYS` | FA3, Triton |
| `UNIFORM_BATCH` | FA2, FlashMLA, FlashInfer MLA |
| `UNIFORM_SINGLE_TOKEN_DECODE` | FlashInfer (native), Mamba, CUTLASS MLA, AITER MLA |
| `NEVER` | (unlisted backends) |
