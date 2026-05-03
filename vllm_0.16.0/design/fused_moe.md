# Fused MoE Kernels

Based on `docs/design/fused_moe_modular_kernel.md`, `docs/design/moe_kernel_features.md`, and `vllm/model_executor/layers/fused_moe/` in the vLLM v0.16.0 source.

## Overview

Mixture of Experts (MoE) replaces the dense FFN in a transformer layer with N expert FFNs, where only K (top-k) experts are activated per token. The MoE data flow is:

```
input → router → top-k selection → permute tokens by expert
  → matmul W1 (gate+up) → activation (SiLU) → matmul W2 (down)
  → unpermute → apply top-k weights → reduce → output
```

With expert parallelism (EP), tokens must be dispatched to the GPU that owns each expert (All2All dispatch) and results gathered back (All2All combine).

## Why "Fused"

The naive approach runs each expert's FFN separately — N independent small matmuls. "Fused" MoE groups all tokens assigned to the same expert and runs them as one batched matmul per expert, or even fuses multiple experts into a single kernel launch. This is why permute/unpermute exists: tokens must be reordered so each expert's tokens are contiguous in memory for efficient batched GEMM.

## The Combinatorial Problem

There are many independent choices:

- **Activation format**: Contiguous (`[M, K]` flat tensor) vs Batched (`[num_experts, max_tokens, K]`)
- **All2All backend**: DeepEP high-throughput, DeepEP low-latency, Pplx, FlashInfer, naive (no EP)
- **Expert kernel**: Triton, DeepGemm, CUTLASS, FlashInfer, Marlin, TRT-LLM, ROCm aiter
- **Quantization**: unquantized, FP8, INT8, NV FP4, MXFP4, INT4
- **Quantization format**: per-tensor, per-token, grouped (block size 128, 16, 32)

The number of valid combinations is intractable to implement individually.

## Modular Kernel Framework

`FusedMoEModularKernel` (`modular_kernel.py`) solves this with a three-component decomposition:

### 1. FusedMoEPrepareAndFinalize

Handles **All2All dispatch/combine** and **quantization**:

- `prepare()` — quantize activations + dispatch tokens to expert-owning GPUs
- `finalize()` — gather results back + maybe apply top-k weights + reduce

Implementations:

| Backend | Output Format | Quant Types | Async (DBO) |
|---|---|---|---|
| `DeepEPHTPrepareAndFinalize` | standard | FP8 (grouped-128) | Yes |
| `DeepEPLLPrepareAndFinalize` | batched | FP8 | Yes |
| `PplxPrepareAndFinalize` | batched | FP8, INT8 | Yes |
| `FlashInferA2APrepareAndFinalize` | standard | NV FP4, FP8 | No |
| `MoEPrepareAndFinalizeNoEP` | standard | FP8, INT8 | No |
| `BatchedPrepareAndFinalize` | batched | FP8, INT8 | No |

The "naive" backend (no modular kernel) handles all quant types but isn't modular — it's the `FusedMoE.forward_impl` path.

### 2. FusedMoEPermuteExpertsUnpermute

The **core MoE compute**: permute → matmul W1 → activation → matmul W2 → unpermute. This is where the actual GPU kernels run.

| Kernel | Format | Quant Types | Notes |
|---|---|---|---|
| `TritonExperts` / `BatchedTritonExperts` | both | all | Default Triton implementation, most feature-complete |
| `DeepGemmExperts` / `BatchedDeepGemmExperts` | both | FP8 | DeepGemm library, optimized for FP8 grouped-128 |
| `CutlassExpertsFp8` / `CutlassBatchedExpertsFp8` | both | FP8 | CUTLASS-based |
| `CutlassExpertsFp4` | both | NV FP4 | CUTLASS-based |
| `FlashInferExperts` | standard | NV FP4, FP8 | FlashInfer library |
| `MarlinExperts` / `BatchedMarlinExperts` | both | INT4, INT8, FP8, FP4 | Marlin GEMM kernels |
| `TrtLlmGenExperts` | standard | MXFP4, NV FP4 | TensorRT-LLM |
| `OAITritonExperts` | standard | unquantized | GPT-OSS Triton kernels |
| ROCm aiter | standard | FP8 | AMD-only, non-modular |
| CPU | standard | unquantized | CPU-only, non-modular |

### 3. TopKWeightAndReduce

The top-k weight application and reduction can happen in either component:

- If the expert kernel does it internally → returns `TopKWeightAndReduceNoOp`
- If not → returns `TopKWeightAndReduceContiguous` or `TopKWeightAndReduceNaiveBatched`, which runs inside `finalize()`

This is negotiated at init time via `FusedMoEPermuteExpertsUnpermute.finalize_weight_and_reduce_impl()`.

### Composition

The forward pass:

```python
class FusedMoEModularKernel:
    def forward(self, input):
        # 1. Quantize + All2All dispatch
        activations, scales = self.prepare_finalize.prepare(input)

        # 2. Core MoE compute
        output = self.fused_experts.apply(activations, scales, W1, W2)

        # 3. Top-k weights + All2All combine
        war_impl = self.fused_experts.finalize_weight_and_reduce_impl()
        result = self.prepare_finalize.finalize(output, war_impl)

        return result
```

## Compatibility Families

Not all combinations work. The tested families:

| All2All Backend | PrepareAndFinalize | Expert Kernels |
|---|---|---|
| deepep_high_throughput | `DeepEPHTPrepareAndFinalize` | `DeepGemmExperts`, `TritonExperts`, `CutlassExpertsFp8`, `MarlinExperts` |
| deepep_low_latency / pplx | `DeepEPLLPrepareAndFinalize` / `PplxPrepareAndFinalize` | `BatchedDeepGemmExperts`, `BatchedTritonExperts`, `CutlassBatchedExpertsFp8`, `BatchedMarlinExperts` |
| flashinfer | `FlashInferCutlassMoEPrepareAndFinalize` | `FlashInferExperts` |

The key constraint: the activation format must match. High-throughput backends produce standard format → pair with standard expert kernels. Low-latency/Pplx backends produce batched format → pair with batched expert kernels.

## Kernel Selection

`FusedMoEMethodBase` has three methods that build the modular kernel:

1. **`maybe_make_prepare_finalize()`** — picks the All2All backend based on `--all2all-backend` and EP/DP config
2. **`select_gemm_impl()`** — picks the expert kernel based on quantization type and hardware (implemented by each quantization method subclass: `Fp8MoEMethod`, `UnquantizedFusedMoEMethod`, etc.)
3. **`init_prepare_finalize()`** — composes them into a `FusedMoEModularKernel` and overrides `self.fused_experts`, making the calling code agnostic to which implementation is used

## Relationship to Transformers Backend

When the Transformers modeling backend (see [[ai/llm/inference/frameworks/vllm_0.16.0/design/huggingface_integration#Transformers Modeling Backend]]) encounters MoE models, `MoEMixin.recursive_replace()` finds `experts` modules in the HF model and replaces them with `TransformersFusedMoE` — a wrapper that adapts HF's calling convention (pre-routed `topk_ids` + `topk_weights`) to vLLM's `FusedMoE` interface. This ensures all MoE models (native or HF-backed) use the same optimized kernels.
