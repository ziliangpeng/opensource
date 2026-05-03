# torch.compile Integration

Based on `docs/design/torch_compile.md` in the vLLM v0.16.0 source.

## What torch.compile Is (PyTorch Concepts)

`torch.compile` is PyTorch's compiler framework, enabled by default in vLLM V1. It has two main components, both owned by PyTorch:

- **Dynamo** — traces the Python model code into an FX graph (an intermediate representation where every tensor operation is explicit). It intercepts Python bytecode during execution and records what operations happen.
- **Inductor** — the backend compiler that takes the FX graph and generates optimized Triton kernels (or falls back to cuBLAS/ATen when appropriate).

These are general-purpose PyTorch tools, not vLLM-specific.

## What vLLM Designs on Top

vLLM's contribution is the glue that makes `torch.compile` work in an LLM serving system:

1. **Wrapping attention as a custom op** — so Dynamo can trace the full model as one graph without understanding attention internals
2. **Piecewise splitting at attention boundaries** — so each "between-attention" subgraph can be compiled and CUDA-graph-captured independently
3. **Compilation cache management** — so startup isn't painfully slow on repeat launches
4. **Dynamic shapes handling** — so one compiled graph works for all batch sizes without recompilation

## How the Three Optimization Layers Stack

```
torch.compile (Dynamo + Inductor)
  → Fuses ops between attention boundaries into fewer kernels
  → Generates optimized Triton kernels (can beat cuBLAS for small shapes)
  → Result: ~5-7 kernels per layer instead of ~9

CUDA graphs (see cuda_graphs.md)
  → Captures those 5-7 kernels into one graph replay
  → Eliminates per-kernel CPU launch overhead
  → Result: one cudaGraphLaunch() replays an entire layer's non-attention work

vLLM's design
  → Breaks the forward pass at attention boundaries (piecewise compilation)
  → Wraps attention as a custom op so Dynamo sees the full model
  → Dispatcher + nested wrappers for dual-mode CUDA graphs (FULL_AND_PIECEWISE)
  → Cache management so compilation artifacts persist across restarts
```

torch.compile makes each kernel faster and reduces the number of kernels. CUDA graphs eliminates the launch overhead of whatever kernels remain. They are complementary — torch.compile optimizes what runs on the GPU, CUDA graphs optimizes how the CPU dispatches it.

## The Compilation Pipeline

### Stage 1: Dynamo Graph Capture

Dynamo traces the model's `forward()` function, inlining all called functions (linear layers, activations, layernorms, embeddings, communication ops). It produces one FX graph for the entire model.

The key trick: attention is wrapped as a **custom op**:

```python
torch.ops.vllm.unified_attention_with_output
```

Dynamo treats custom ops as opaque — it records the op's inputs and outputs without tracing inside. Since attention's output shares the same shape as its input query, Dynamo can propagate shapes through it. This lets vLLM capture the full model as one graph from Dynamo's perspective, despite attention being extremely complex internally (KV cache, paged memory, variable sequence lengths).

Files traced during compilation (Llama example):
- `vllm/model_executor/models/llama.py` — model forward
- `vllm/model_executor/layers/linear.py` — projections
- `vllm/model_executor/layers/layernorm.py` — RMSNorm
- `vllm/model_executor/layers/activation.py` — SiLU
- `vllm/model_executor/layers/rotary_embedding.py` — RoPE
- `vllm/attention/layer.py` — attention custom op wrapper
- `vllm/distributed/communication_op.py` — tensor parallelism collectives

### Stage 2: Piecewise Graph Splitting

The FX graph is split at `splitting_ops` (the attention custom op). For a 16-layer Llama model, this produces 33 subgraph pieces:

- 1 piece: input embedding → first attention
- 15 pieces: attention → next attention (one per layer boundary, identical structure)
- 16 pieces: the attention ops themselves (treated as opaque, not compiled)
- 1 piece: last attention → output

But only **3 unique subgraphs** need compilation:
1. The "before first attention" piece
2. The "between two attentions" piece (shared across all 15 instances)
3. The "after last attention" piece

### Stage 3: Inductor Compilation

Each unique subgraph piece is passed to Inductor. This is where the actual optimization happens:

**Kernel fusion**: Inductor fuses compatible ops into single Triton kernels. For example, RMSNorm + residual addition becomes one kernel instead of two.

**Kernel replacement**: Inductor can generate Triton kernels that outperform the default implementations. The vLLM doc shows an example where Inductor's auto-tuned Triton matmul is 20% faster than cuBLAS for shape `8×2048×3072` (a small decode-phase shape). For large prefill shapes, cuBLAS typically still wins.

**Two compilation modes**:

- **Symbolic shape** (default): compiled once for any batch size. Uses symbolic dimensions so the same compiled code handles all sizes. General but not maximally tuned.
- **Specific shapes** (`compile_sizes=[1,2,4,8]`): compiled with all shapes known, enabling auto-tuning. Inductor generates multiple Triton config candidates (varying tile sizes, warp counts, pipeline stages) and benchmarks each one. Slow first time but cached for later.

```bash
# Enable auto-tuning for specific batch sizes
vllm serve meta-llama/Llama-3.2-1B \
  --compilation-config '{"compile_sizes": [1, 2, 4, 8]}'
```

### Stage 4: CUDA Graph Capture

Each compiled subgraph piece gets wrapped in a `CUDAGraphWrapper` for piecewise capture. Attention runs eagerly between the pieces. See [[ai/llm/inference/frameworks/vllm_0.16.0/design/cuda_graphs]] for the full CUDA graph design.

## Compilation Cache

All artifacts are stored at `~/.cache/vllm/torch_compile_cache/<hash>/rank_<N>/`:

- `transformed_code.py` — Dynamo's traced Python function
- `computation_graph.py` — the FX graph with all submodules
- `inductor_cache/` — Inductor's compiled Triton kernels and auto-tune results

The `<hash>` is computed from all configs, PyTorch version, and the model's forward function code. Any code change in traced files triggers a cache miss and recompilation.

**Key guarantee**: all compilation finishes before serving any requests. No request will trigger a recompilation, which would cause unpredictable latency spikes.

The cache directory can be copied directly between deployment environments to skip compilation entirely, as long as the hash matches (same model, same config, same PyTorch version).

## Dynamic Shapes

`torch.compile` wants to guard on tensor shapes — if batch_size=8 during tracing, it may specialize for exactly 8 and recompile for other sizes. vLLM needs one compiled graph that works for all batch sizes. This tension is handled by three modes:

```python
class DynamicShapesType(enum.Enum):
    BACKED = ...                  # default
    UNBACKED = ...
    BACKED_SIZE_OBLIVIOUS = ...
```

| Mode | Behavior | Risk |
|---|---|---|
| `BACKED` (default) | Dynamo may add shape guards; vLLM drops them | Dropping guards may be unsound — a guard could be material |
| `UNBACKED` | Guaranteed no guards added | May miss optimizations; picks overly general code paths (e.g., assumes tensors are non-contiguous) |
| `BACKED_SIZE_OBLIVIOUS` | Mostly avoids guards; avoids 0/1 specialization | Experimental middle ground; safer than BACKED |

The only dimension that actually changes between forward passes is the batch size (number of tokens). All other dimensions (hidden size, head dim, num heads, etc.) are static model properties. So the dynamic shapes problem is narrow in practice — it's just one symbolic dimension.
