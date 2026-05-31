# How Triton's `@autotune` Works

Triton's `@autotune` is **not an automatic search engine** — it does not explore a search space or try to "figure out" good configurations from scratch. It is a **launch-time benchmark selector**: you provide a list of candidate configurations, and Triton picks the fastest one by actually running them on your GPU.

This note explains the full pipeline, what each moving part does, and why the design makes sense.

---

## 1. The API: you provide the candidates

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=5, num_warps=4),
    ],
    key=["M", "N", "K"],        # ← input shape → cache key
    prune_configs_by={
        "early_config_prune": my_prune_fn,   # filter out bad configs
        "perf_model": my_model_fn,           # analytical ranking (optional)
        "top_k": TOPK,                       # keep top K for benchmarking
    },
)
```

**No DSL for search spaces.** No `BLOCK_M ∈ [16, 32, 64, 128]`. Each `triton.Config` is a hand-written point in parameter space.

---

## 2. The pipeline: 4 stages, called once per unique input shape

For each new set of `key` values (e.g., the first time the kernel is called with M=1024, N=4096, K=4096):

### Stage 1: `early_config_prune` — heuristic filter

Before any GPU activity, this optional function receives the full candidate list and returns a filtered subset. Typical heuristics:

- Shared memory budget: `(BLOCK_M + BLOCK_N) × BLOCK_K × num_stages × dtype_size ≤ max_shared_mem`
- Register budget: estimated register pressure ≤ device limit
- Warp/wavefront ceiling: on AMD, `num_warps × 64 ≤ 1024` (so `num_warps ≤ 16`)
- Tile size sanity: any dimension that exceeds the matrix size is wasteful

This is your **hook for domain knowledge**. No GPU launch, just Python arithmetic.

```python
def prune_configs(configs, named_args, **kwargs):
    dtsize = named_args["dtype"].itemsize
    max_shared = driver.active.utils.get_device_properties(0)["max_shared_mem"]
    filtered = []
    for cfg in configs:
        smem = (cfg.kwargs["BLOCK_M"] + cfg.kwargs["BLOCK_N"]) * \
               cfg.kwargs["BLOCK_K"] * cfg.num_stages * dtsize
        if smem <= max_shared:
            filtered.append(cfg)
    return filtered
```

### Stage 2: Perf model ranking (NVIDIA only)

Triton ships with a built-in analytical performance model for matmul: `triton.ops.matmul_perf_model.estimate_matmul_time`. This is a **C extension** that calls into `triton._C.libtriton.triton.runtime` and estimates each config's runtime without launching the kernel.

The output is a scalar score. Configs are sorted by score and only the **top K** (from `top_k`) advance to real benchmarking.

On AMD, this C extension does not exist. You must either:

- Provide your own `perf_model` function (returning `0.0` effectively disables ranking and benchmarks all survived configs), or
- Omit `perf_model` entirely, in which case the autotuner falls back to benchmarking all survived configs.

### Stage 3: Real GPU benchmark

Each surviving config is launched *once* (or a few times) with the actual input tensor shapes. Triton uses `triton.testing.do_bench` internally, which:

1. Warms up the GPU (a few dummy launches)
2. Runs the kernel repeatedly over a short duration (~100ms or a fixed number of iterations)
3. Returns median latency in milliseconds

The config with the lowest latency wins.

### Stage 4: Cache

The winning config is cached in a dictionary keyed by the tuple `(key_values, device_properties)`. Future calls with the same shape skip stages 1-3 entirely. The cache is per-process, per-device.

```python
cache = {}  # (M, N, K, device) -> best Config
```

After the first call with shape `(1024, 4096, 4096)`, subsequent calls are a dictionary lookup — zero overhead.

---

## 3. What Triton's autotuner does NOT do

- ❌ No grid search over parameter ranges
- ❌ No Bayesian optimization or Gaussian processes
- ❌ No genetic / evolutionary search
- ❌ No inter-config interpolation or "try between 32 and 64"
- ❌ No learning across different input shapes

It strictly does: **enumerate → filter → rank → benchmark top-K → cache**.

---

## 4. Why this design: the economics of kernel autotuning

You might think "this is just brute-force over my candidate list." Yes — and that's fine.

### Kernel launches are cheap

A single config benchmark takes ~0.1–5 ms (one launch × a few iterations). Even with 40 candidates, the total overhead is ~50–100 ms. For a matmul kernel that runs millions of times, this amortizes instantly.

### Per-shape caches make the overhead one-time

The cache key includes the actual matrix dimensions. If your model uses fixed shapes (common in transformer inference), each distinct shape pays the cost exactly once. After that, `@autotune` adds zero overhead.

### Search space exploration would be slower than brute force

Suppose you wanted the autotuner to "find" good parameters from `BLOCK_M ∈ {16, 32, 64, 128}`, `BLOCK_K ∈ {16, 32, 64, 128}`, `num_stages ∈ {2, 3, 4, 5}`, `num_warps ∈ {2, 4, 8}` — that's 4×4×4×3 = 192 combos. A Bayesian optimizer might need 30–50 launches to converge. But a naive sweep of all 192 takes only ~200 ms. There's nothing to optimize.

### The real problem is picking the right candidates, not searching

The hard part is knowing *which* combinations of parameters are worth trying on a given GPU architecture. That requires understanding:

- Tensor core instruction formats (wgmma vs MFMA)
- TMA vs manual load pipelines
- Wavefront vs warp sizes
- Shared memory vs register budgets

No amount of autotune search magic replaces this knowledge. The autotuner is just the final selector — it doesn't generate the options.

---

## 5. The AMD vs NVIDIA trap: same autotune API, radically different parameters

Because the candidates are hand-written, and because H100 and AMD CDNA3 have opposite tuning preferences, **a candidate pool tuned for H100 will perform catastrophically on AMD**, and vice versa.

| Parameter | H100 typical | AMD MI325X typical | Root cause |
|-----------|-------------|-------------------|------------|
| BLOCK_K | 32–64 | 128–512 | wgmma efficient at small K; MFMA needs large K to amortize instruction overhead |
| num_stages | 3–5 | 2–3 | TMA handles prefetch on H100; AMD has no TMA, manual loads |
| num_warps | 4–8 | 2–4 | AMD wavefront = 64 threads, half the warp count per block ceiling |
| SPLIT_K | sometimes | rarely needed | Large K already keeps MFMA busy |

This means you need **architecture-specific candidate pools**, often derived from offline sweeps.

### Example: AMD-tuned matmul pool (from 4480-measurement sweep)

```python
AMD_MATMUL_TUNED_POOL = [
    # Decode (small M)
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 256}, num_stages=3, num_warps=2),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 512}, num_stages=2, num_warps=2),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 512}, num_stages=2, num_warps=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 16, "BLOCK_K": 512}, num_stages=2, num_warps=2),
    # Prefill (large M)
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 256}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 128}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128}, num_stages=2, num_warps=4),
]
```

Key pattern: **large BLOCK_K (128–512), shallow pipeline (stages=2–3), conservative num_warps (2–4)**.

---

## 6. What if you really need to explore a large space?

You write an external sweep script. Triton provides `triton.testing.do_bench` for this:

```python
import triton
from triton.testing import do_bench
import itertools

param_space = itertools.product(
    [16, 32, 64, 128],     # BLOCK_M
    [16, 32, 64, 128],     # BLOCK_N
    [32, 64, 128, 256],    # BLOCK_K
    [2, 3, 4],             # num_stages
    [2, 4, 8],             # num_warps
)

results = []
for BM, BN, BK, stages, warps in param_space:
    if BM * BN * BK * stages * 4 > max_shared_mem:  # Heuristic filter
        continue
    kernel = matmul_kernel[ (BM, BN, BK), stages, warps ]
    ms, min_ms, max_ms = do_bench(lambda: kernel(ptr_a, ptr_b, ptr_c, M, N, K))
    results.append((BM, BN, BK, stages, warps, ms))
```

Then analyze `results` to pick the stable, top-performing configs that go into production.

Triton's autotuner is intentionally **not** this script — because once you know what configs work, you don't want to benchmark all 256+ combos every time you run the kernel.

---

## 7. Summary: analogy

```
Triton @autotune is NOT a chef who invents recipes.
It's a taste-tester who picks the best from a menu you wrote.
```

You bring the architectural knowledge; the autotuner brings reliable, GPU-measured selection.
