# CUDA Graphs

Based on `docs/design/cuda_graphs.md` in the vLLM v0.16.0 source.

## Background: What CUDA Graphs Are

CUDA graphs are an NVIDIA CUDA API that records a sequence of kernel launches into a graph object, then replays the entire sequence with a single `cudaGraphLaunch()` call. This eliminates per-kernel CPU-side launch overhead (Python → PyTorch dispatcher → CUDA driver → GPU command queue submission), which is ~5-10μs per kernel.

A well-optimized transformer layer has ~7-9 kernel launches (4 cuBLAS GEMMs + attention + fused activations/norms). For a 32-layer model, that's ~250 kernel launches per forward pass. At 5-10μs each, launch overhead alone is 1-3ms — very noticeable during decode where total forward pass time might be 5-15ms.

The API:

```c
cudaStreamBeginCapture(stream);
// kernels are recorded, NOT executed
run_model_forward();
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&exec, graph);

// later, replay everything in one call:
cudaGraphLaunch(exec, stream);
```

### Key Constraint: Everything Is Baked In

CUDA graphs capture raw GPU state — memory addresses, kernel grid dimensions, integer arguments. The graph has no concept of tensor shapes or PyTorch abstractions. This means:

- **All memory addresses are fixed** — input/output tensors must be pre-allocated at fixed addresses. Before replay, `memcpy` new data into the same buffer, replay, read output from the same buffer.
- **Shapes can't change** — a graph captured for batch_size=8 has that size baked into every kernel's grid dimensions and arguments. It cannot run batch_size=5.
- **No dynamic control flow** — no `if` statements or variable-length loops inside the graph.

vLLM handles the shape constraint by **padding batches** to pre-determined sizes (1, 2, 4, 8, ...) and capturing one graph per size at warmup. At runtime, the actual batch is padded up to the next captured size, the matching graph is replayed, and padded outputs are discarded.

## The Design Problem

CUDA graphs themselves are simple. The design challenge in vLLM is that **the model forward pass is not a single static graph**:

1. **Prefill vs decode run different code paths** — different attention kernel routines, different `max_query_len`. A graph captured for prefill cannot be replayed for decode.

2. **Not all ops are graph-capturable** — attention is the main problem. Many backends (FlashInfer, Mamba) don't support CUDA graph capture at all, or only for pure decode. So the model must be broken into pieces: capture the non-attention parts, leave attention eager.

3. **Batch size varies** — each size needs its own pre-captured graph.

4. **These decisions interact** — "full graph for decode, piecewise for prefill, no graph for cascade attention" is a runtime dispatch problem that the old code had scattered across multiple components.

The old design tightly coupled CUDA graph capture with `torch.compile` piecewise compilation inside `PiecewiseBackend`, resulting in all-or-nothing behavior, inconsistent backend support, and increasingly complex code.

## `CUDAGraphMode`: The Single Config Knob
**File:** `vllm/config/compilation.py`

Set via `--compilation-config '{"cudagraph_mode": "..."}'`:

```python
class CUDAGraphMode(enum.Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_DECODE_ONLY = (FULL, NONE)          # (2, 0)
    FULL_AND_PIECEWISE = (FULL, PIECEWISE)   # (2, 1)
```

The first three are **single-mode** — they use the same strategy for both decode and prefill. The last two are **dual-mode tuples** of `(decode_strategy, prefill_strategy)`, dispatched at runtime depending on the batch type.

| Mode | Enum value | Decode | Prefill/Mixed | Notes |
|---|---|---|---|---|
| `NONE` | `0` | Eager | Eager | Debugging |
| `PIECEWISE` | `1` | Piecewise | Piecewise | Old default. Requires piecewise compilation. |
| `FULL` | `2` | Full | Full | Only works when backend supports it for both batch types. |
| `FULL_DECODE_ONLY` | `(2, 0)` | Full | Eager | Good for P/D disaggregated decode instances — saves memory by not storing piecewise graphs. |
| `FULL_AND_PIECEWISE` | `(2, 1)` | Full | Piecewise | **New default.** Most performant, most memory. |

The dual-mode values are accessed via `decode_mode()` and `mixed_mode()` methods, which return the first and second tuple elements respectively. Single-mode values return `self` for both.

Missing combinations like `(PIECEWISE, NONE)` or `(PIECEWISE, FULL)` don't exist because: piecewise for decode when you could do full is always suboptimal, and full for prefill when you can't even do full for decode makes no sense.

`PIECEWISE` means: break the model around attention (attention stays eager, everything else is captured in graphs). This works with any backend since attention is excluded from the graph.

`FULL` means: capture the entire model including attention as one graph. This only works if the attention backend supports CUDA graph capture.

`FULL_AND_PIECEWISE` is the key innovation — decode gets full graphs (maximum performance for the latency-sensitive path), prefill/mixed gets piecewise graphs (compatible with any backend). Both use the same compiled model, just dispatched differently.

## Core Components

### `BatchDescriptor` — the dispatch key

```python
class BatchDescriptor(NamedTuple):
    num_tokens: int
    num_reqs: int
    uniform: bool = False   # all requests have the same query length
    has_lora: bool = False
```

`uniform=True` for pure decode batches (or speculative decode where all queries are `1 + num_spec_tokens`). Many attention backends only support full CUDA graphs for uniform batches.

### `CudagraphDispatcher` — central controller
**File:** `vllm/v1/cudagraph_dispatcher.py`

The single source of truth for CUDA graph selection. Before each forward pass:

1. Takes the incoming `BatchDescriptor`
2. Searches its sets of valid dispatch keys with priority: `FULL` > `PIECEWISE` > `NONE`
3. Returns the selected runtime mode and final batch descriptor

```python
batch_descriptor = BatchDescriptor(num_tokens=num_input_tokens, uniform=...)
runtime_mode, batch_descriptor = cudagraph_dispatcher.dispatch(batch_descriptor)

with set_forward_context(..., cudagraph_runtime_mode=runtime_mode,
                         batch_descriptor=batch_descriptor):
    output = self.model(...)
```

The dispatcher is initialized after all attention backends are set up, so it knows which graph sizes and modes are valid.

### `CUDAGraphWrapper` — capture and replay
**File:** `vllm/compilation/cuda_graph.py`

Wraps a callable with CUDA graph capture/replay ability. Each wrapper is bound to a specific runtime mode (`FULL` or `PIECEWISE`). At runtime it reads the `ForwardContext` and:

- If runtime mode matches its own mode → capture (first time) or replay (subsequent)
- If runtime mode doesn't match → pass through to the underlying callable directly

### The Nested Wrapper Design

This is the structural trick that makes `FULL_AND_PIECEWISE` work with a single compiled model:

```
FULL wrapper (around entire model)
  └── Model forward
        ├── Linear layers, norms, etc.
        │     └── PIECEWISE wrapper (inside compiled subgraph)
        ├── Attention (eager, not wrapped)
        ├── Linear layers, norms, etc.
        │     └── PIECEWISE wrapper (inside compiled subgraph)
        └── ...
```

- **`FULL` runtime mode**: outer wrapper captures/replays the entire model. Inner PIECEWISE wrappers see a mismatched mode and pass through — they're invisible.
- **`PIECEWISE` runtime mode**: outer FULL wrapper sees a mismatch and passes through. Inner PIECEWISE wrappers each capture/replay their subgraph. Attention runs eagerly between them.
- **`NONE` runtime mode**: both wrappers pass through. Pure eager execution.

No duplicate compilation needed — the same compiled model supports all three modes, selected per-batch by the dispatcher.

## Attention Backend Compatibility

Each attention backend declares its CUDA graph support level:

```python
class AttentionCGSupport(enum.Enum):
    ALWAYS = 3                       # Supports all batch types
    UNIFORM_BATCH = 2                # Only uniform batches (all same query length)
    UNIFORM_SINGLE_TOKEN_DECODE = 1  # Only pure decode (query_len=1)
    NEVER = 0                        # No CUDA graph support
```

| Attention Backend | Support Level |
|---|---|
| FlashAttention v3 | `ALWAYS` |
| Triton Attention | `ALWAYS` |
| FlashAttention v2 | `UNIFORM_BATCH` |
| FlashMLA, FlashInfer MLA | `UNIFORM_BATCH` |
| FlashInfer | `UNIFORM_SINGLE_TOKEN_DECODE` |
| Mamba | `UNIFORM_SINGLE_TOKEN_DECODE` |
| AITER MLA, CUTLASS MLA | `UNIFORM_SINGLE_TOKEN_DECODE` |

For hybrid models (e.g., Mamba + attention layers), vLLM takes the **minimum** capability across all backends. If the requested mode isn't compatible, vLLM auto-downgrades — e.g., `FULL` with a `UNIFORM_BATCH` backend becomes `FULL_AND_PIECEWISE` (full graph only for decode, piecewise for prefill).

## Warmup and Capture

CUDA graph capture happens during the model runner's first dummy forward passes (`_dummy_run`). The warmup sequence:

1. Run with `NONE` mode first — eager execution to warm up kernels, allocate memory, etc.
2. For each captured batch size, run with the target mode — this triggers `CUDAGraphWrapper` to capture
3. For full graphs, attention metadata is explicitly set (e.g., `max_query_len=1` for decode) to ensure the correct kernel routine is recorded

The dispatcher initializes its valid key sets after warmup via `initialize_cudagraph_keys`, which is called after all attention backends are initialized.
