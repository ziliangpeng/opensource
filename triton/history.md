# Triton Project History

> **Summary**: 6,284 commits, 590+ contributors, 19K GitHub stars, 62M monthly PyPI downloads. From a Harvard PhD side project to the GPU codegen backend powering PyTorch 2.0+.

---

## Timeline

### 2016 — ISAAC: the predecessor

Philippe Tillet, during his PhD at Harvard (advisor: H.T. Kung), started a project called **ISAAC** — an ML-driven library for input-aware kernel selection (essentially a smarter cuBLAS autotune wrapper). ISAAC's approach of "let the computer figure out the best tiling for your hardware" was the conceptual seed for Triton.

### 2019 — MAPL/PLDI paper

Philippe Tillet, H.T. Kung, and David Cox published **"Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations"** at MAPL 2019 (a PLDI-affiliated workshop).

**Core idea**: a programming model centered around **tiles** — statically-shaped multi-dimensional sub-arrays — with an LLVM-based IR. Developers write GPU kernels without manually managing memory coalescing, shared memory allocation, or warp-level synchronization.

At this point, Triton was an **academic prototype**: C-based syntax (not Python), LLVM backend (not MLIR), no `@triton.jit` decorator.

### 2020-02-06 — First commit on GitHub

The earliest commit in the repository. The commit author email is `ptillet@g.harvard.edu`. The codebase at this point inherited history from ISAAC.

### 2021 — Philippe Tillet joins OpenAI → ground-up rewrite

Philippe joined OpenAI Research, and Triton underwent a radical transformation:

- **Python DSL** replaced C-based syntax (`@triton.jit` decorator model)
- **MLIR-based compiler pipeline** replaced LLVM IR — MLIR's dialect architecture made GPU optimizations (tiling, coalescing, shared memory promotion) implementable as composable MLIR passes rather than LLVM pattern-matching hacks
- **2021-07-27**: old ISAAC history was stripped from the repo. The commit message reads: *"History prior to this date belonged to the now deprecated ISAAC project, and was deleted to save space"*. The 6,284 commits counted today are all from this point onward.

### 2022 — PyTorch 2.0 adopts Triton

`torch.compile` announced Triton as its **default GPU codegen backend**. The `inductor` backend generates Triton kernels for every PyTorch operation. This was the inflection point: Triton went from a niche tool for GPU kernel hackers to a core component of the PyTorch ecosystem, running billions of kernel launches daily.

### 2023 — Project moves to community governance

The repository moved from `openai/triton` to `triton-lang/triton`. A hierarchical governance structure was established:

```
Community contributors (issues, PRs)
    ↓
Module maintainers (own subsystems, drive development)
    ↓
Project lead (Philippe Tillet)
```

### 2023–2024 — AMD ROCm backend

A multi-company effort (AMD engineers, open-source contributors, OpenAI) added HIP backend support. This was a major engineering challenge, not a trivial port:

- Wavefront size 64 (vs NVIDIA's warp size 32)
- No TMA unit (threads execute manual loads)
- LLVM AMDGPU target differences
- Different tensor core instruction formats (MFMA vs wgmma)

Key AMD contributors: Zahi Moudallal, Anatoly Myachev, Kyle Wang, yanxuer-999.

### 2024–2025 — Hopper (H100) features land

With H100 reaching wide deployment, Triton added:

- **TMA (Tensor Memory Accelerator)** descriptor support — dedicated DMA for global→shared memory
- **`num_ctas`** — cluster kernel support, multiple CTA groups sharing L1 data path
- **wgmma instruction lowering** — Hopper's new tensor core instruction format

### 2025 — 3rd Triton Developer Conference

Hosted at the Microsoft Silicon Valley Campus in Mountain View, CA (October 21, 2025). The conference demonstrated Triton's cross-company adoption: OpenAI, NVIDIA, AMD, Microsoft, and others all contributing.

### 2026-05 — Triton 3.7.0

Latest release. 62.4M monthly PyPI downloads — on par with PyTorch in ecosystem infrastructure penetration.

---

## Statistics (as of 2026-05-30)

| Metric | Value |
|--------|-------|
| Total commits | 6,284 |
| Contributors (unique names) | ~590 |
| Contributors (unique emails) | ~698 |
| GitHub stars | ~19K |
| PyPI monthly downloads | 62.4M |
| First commit | 2020-02-06 (Harvard era) |
| Repository moved to `triton-lang` | 2023 |
| Latest release | Triton 3.7.0 (May 2026) |
| Activity status | **Extremely active** — commits within the last 24 hours |

---

## Top 10 Contributors (by commit count)

### 1. Philippe Tillet — 715 commits

**The creator.** Started Triton as a PhD student at Harvard (advisor: H.T. Kung), where he first built ISAAC (2016), an ML-driven kernel selection library. Published the MAPL 2019 paper "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations" (17K+ citations). After graduating in 2020, joined OpenAI Research and led the ground-up rewrite of Triton from a C+LLVM prototype into the Python+MLIR compiler we know today. Still leads the Triton project at OpenAI as Member of Technical Staff. Also co-authored NVIDIA's blog on Triton on Blackwell. GitHub: [@ptillet](https://github.com/ptillet).

### 2. Keren Zhou (周可人) — 566 commits

Core compiler engineer on the Triton team. Previously at OpenAI; now **Assistant Professor at George Mason University**, leading the CAT Lab (Compiler, Architecture, and Tools). PhD from Rice University. Built **Proton**, a multi-level adaptive profiler for Triton kernels, and **Triton-Sanitizer**, a device-agnostic memory sanitizer. Co-author of the **Linear Layouts** paper (arXiv 2505.23819) that redesigned Triton's tensor layout system. GitHub: [@Jokeren](https://github.com/Jokeren).

### 3. Thomas Raoux — 438 commits

Triton compiler team at OpenAI. Deep expertise in MLIR-based code generation — works on the Triton→MLIR→LLVM pipeline, GPU dialect lowering, and Hopper-specific features (TMA, cluster kernels). Previously at NVIDIA on MLIR infrastructure. His GitHub also shows contributions to IREE (Google's MLIR compiler). GitHub: [@ThomasRaoux](https://github.com/ThomasRaoux).

### 4. peterbell10 (Peter Bell) — 309 commits

Member of Technical Staff at OpenAI on the Triton compiler team. Major contributor to the CUDA backend — kernel launch infrastructure, memory model, code generation. Prior to OpenAI was a staff software engineer working on GPU-accelerated scientific computing (radiative transfer, spatial data structures). Author on NVIDIA Technical Blog. GitHub: [@peterbell10](https://github.com/peterbell10).

### 5. Jeff Niu (Mogball) — 272 commits

Member of Technical Staff at OpenAI. Self-describes as a "TensorFlow refugee married to MLIR." Built **triton_lite**, a Triton clone in Mojo, at Modular. Works on Triton's MLIR dialect architecture — the modular compiler design that allows Triton to target multiple backends (CUDA, HIP, CPU). University of Waterloo alumni. GitHub: [@Mogball](https://github.com/Mogball).

### 6. Mario Lezcano Casado (Lezcano) — 253 commits

Member of Technical Staff at OpenAI. Previously **tech lead of the PyTorch core dev team at Quansight** (the company that maintains PyTorch core alongside Meta). Core PyTorch developer working on `torch.compile`, TorchDynamo, TorchInductor, `torch.linalg`, and autodiff. His Triton contributions are at the PyTorch–Triton boundary: making sure `torch.compile` generates correct and efficient Triton kernels. PhD from University of Oxford. GitHub: [@Lezcano](https://github.com/Lezcano).

### 7. Lei Zhang (antiagainst) — 199 commits

**Senior Manager at AMD**, leading an AI compiler/runtime team. GitHub handle `antiagainst`. Works on Triton, IREE, MLIR, and LLVM. Before AMD, worked on SPIR-V, Vulkan, and Metal graphics standards. Author on the ROCm Blogs (e.g., "Unleash Full GPU Potential: Overlap Communication and Computation with Triton-Distributed"). Also involved in ByteDance-Seed's **Triton-distributed** project. His presence in the top 10 reflects AMD's deep investment in Triton as a compiler platform. GitHub: [@antiagainst](https://github.com/antiagainst).

### 8. pawelszczerbuk (Paweł Szczerbuk) — 167 commits

Member of Technical Staff at OpenAI on the Triton compiler team. Career dedicated to GPU compiler development. Core owner of Triton's **software pipelining pass** (pipeline scheduler) — the pass that overlaps global memory loads with computation (`num_stages`). Frequently cc'd on pipeline-related issues and bugs. Author on NVIDIA Technical Blog. GitHub: [@pawelszczerbuk](https://github.com/pawelszczerbuk).

### 9. Justin Lebar (jlebar) — 131 commits

Decade-long compiler engineer across **Google, Waymo, and OpenAI**. Worked on CUDA support in clang, XLA:GPU, Triton, and OpenAI's custom hardware. Recently made waves with a blog post on *Semianalysis* about spending $10K+ running AI agents (Codex CLI) over Triton's C++ codebase to find miscompilations — finding hundreds of plausible bugs, several of which were confirmed and fixed. Known for a meticulous, correctness-obsessed engineering style. GitHub: [@jlebar](https://github.com/jlebar).

### 10. Alexander Weinrauch — 130 commits

**Senior Software Development Engineer at AMD.** PhD in Computer Science focused on High Performance Computing and Computer Graphics. The top AMD engineer contributing to Triton's **HIP/ROCm backend** — making Triton work on AMD GPUs. His work covers the HIP runtime integration, AMDGPU LLVM target lowering, and GPU-specific optimizations for CDNA architectures. GitHub: [@alexweinrauch](https://github.com/alexweinrauch).

---

### Other notable contributors

| Name | Commits | Notes |
|------|---------|-------|
| Anatoly Myachev | 108 | Triton compiler, backend infrastructure |
| Zahi Moudallal | 96 | AMD, HIP/ROCm backend |
| Kyle Wang | 63 | AMD, Triton-distributed, ROCm |
| neildhar | 57 | Kernel optimizations, PyTorch-Triton integration |
| Hongtao Yu (htyu) | 56 | AMD, compiler passes, automatic warp specialization |
| apgoucher (Adam Goucher) | 53 | Linear layouts co-author, mathematical foundations |

### Ranks 11–20 in detail

#### 11. Anatoly Myachev — 108 commits
OpenAI Triton compiler team. Works on build system, CI, and backend infrastructure — the unglamorous but essential work that makes the compiler compilable and testable.

#### 12. Zahi Moudallal — 96 commits
**AMD engineer.** Core developer of the Triton HIP/ROCm backend. Making Triton actually work on AMD GPUs.

#### 13. Kyle Wang — 63 commits
**AMD software engineer.** Primary contributor to **Triton-distributed**, the extension that enables computation-communication overlap for distributed Triton kernels. Co-author with Lei Zhang on ROCm Blogs.

#### 14. Lixun Zhang — 60 commits
**AMD compiler engineer.** Tsinghua BS, UT Austin PhD (automatic code generation + GPU optimization). Works with Lei Zhang on Triton for AMD GPUs, focusing on chiplet architecture optimization and instruction scheduling.

#### 15. neildhar — 57 commits
Kernel optimization and PyTorch-Triton integration. Visible in PyTorch issues fixing `torch.compile` + Triton bugs. Specific affiliation unclear.

#### 16. Ilya V (joviliast) — 56 commits
Likely Intel-related. Forks ROCm/Triton, possibly working on Intel GPU Triton backend. Identity unclear.

#### 17. Hongtao Yu (htyu) — 56 commits
**AMD compiler engineer**, highly active. Designer and implementer of **Automatic Warp Specialization (autoWS)** — the compiler pass that automatically splits warps into roles (some load data, others compute) without developer-written synchronization. Also proposed **TLX**, a minimally invasive extension to Triton that adds explicit multi-warp orchestration only where needed. Author of the PyTorch blog post on autoWS design and roadmap.

#### 18. Alexander Efimov (binarman) — 54 commits
Backend and build infrastructure contributor.

#### 19. apgoucher (Adam Goucher) — 53 commits
**Linear Layouts co-author** (ASPLOS 2026 paper). Provided the F₂ vector space algebraic framework that unified Triton's tensor layout description. Role is more mathematician/researcher than compiler engineer. (Not the Olympic runner of the same name.)

#### 20. Phil Tillet — 46 commits
Same person as **Philippe Tillet** (#1). Git author string inconsistency ("Philippe Tillet" vs "Phil Tillet") split the count. True total: 715 + 46 = **761 commits**.

### Ranks 21–30 summary

| # | Name | Commits | Notes |
|---|------|---------|-------|
| 21 | Yi Qian | 45 | AMD, ROCm-related |
| 22 | Pengzhan Zhao | 45 | Academia (UIUC?), ML compiler research |
| 23 | Maksim Levental | 44 | Independent MLIR/compiler researcher; built nelli (MLIR frontend) |
| 24 | masahi | 40 | Intel GPU backend |
| 25 | Christian Sigg | 40 | NVIDIA, PhD ETH Zurich, MLIR |
| 26 | Thomas | 38 | Possibly Thomas Raoux alias |
| 27 | Jungwook Park | 37 | AMD, Senior AI GPU Compiler Engineer |
| 28 | Tori Baker | 31 | Triton runtime/infra |
| 29 | Nick Riasanovsky | 31 | — |
| 30 | Corbin Robeck | 31 | — |

### Affiliations summary

| Organization | Key contributors |
|-------------|-----------------|
| **OpenAI** (Triton team) | Philippe Tillet, Thomas Raoux, Peter Bell, Jeff Niu, Mario Lezcano, Paweł Szczerbuk, Justin Lebar, Anatoly Myachev, Keren Zhou (formerly) |
| **AMD** | Lei Zhang, Alexander Weinrauch, Zahi Moudallal, Kyle Wang, Lixun Zhang, Hongtao Yu, Yi Qian, Jungwook Park |
| **Academia** | Keren Zhou (George Mason University, current), Philippe Tillet (Harvard, former PhD) |
| **Intel** | masahi, Ilya V (possible) |

---

## Release History & Feature Evolution

Git tags (sorted chronologically):

```
v0.1 → v0.2.3 → v0.4  (pre-1.0 prototype era)
v1.0 → v1.1 → v1.1.2  (1.x: initial public release)
v2.0.0 → v2.1.0       (2.x: MLIR rewrite)
v3.0.0 → ... → v3.7.0 (3.x: multi-GPU maturity)
```

Plus special tags: `gfx950-tutorial-v0.1` (AMD tutorial), `isaac` (predecessor reference), `legacy-backend` (pre-MLIR snapshot).

### v0.1 / v0.2.3 / v0.4 — 2021 prototype era

Before the July 2021 OpenAI blog announcing Triton 1.0. `@triton.jit` just born, Python DSL taking shape. Supported basic element-wise, reduction, and matmul kernels. The thesis: a Python DSL for GPU kernels can match hand-written CUDA performance.

### v1.0 — 2021-07-28 🎯 First Public Release

OpenAI blog: "Introducing Triton: Open-source GPU programming for neural networks." Core positioning:
- **Python DSL + JIT compiler** (`@triton.jit`)
- **Tile-based programming model** — developer writes block-level ops, compiler handles memory coalescing and shared memory
- **Autotuner** — select best tile config via GPU benchmark
- GPU support: Volta (SM70) through Ampere (SM80)

This is the Triton that PyTorch 2.0's inductor would later adopt.

### v1.1 / v1.1.1 / v1.1.2 — 2021–2022

Bug fixes, performance improvements, expanded GPU support. Flash Attention 1/2 ports appeared in the community. Triton started gaining adoption as the go-to tool for custom PyTorch GPU kernels.

### v2.0.0 — 2023-02 🏗️ MLIR Ground-Up Rewrite

**The most significant architectural change in Triton's history.** The backend was completely rewritten from PTX emission to an **MLIR pipeline**:

```
@triton.jit Python DSL
    ↓
Triton IR (tt dialect)
    ↓
TritonGPU dialect (ttg)   ← GPU-specific layout/coalescing analysis
    ↓
LLVM IR (nvvm/gpu dialect)
    ↓
PTX / cubin
```

**Why a rewrite?** The old backend (direct PTX emission) couldn't handle complex kernel patterns like **back-to-back matmuls** (e.g., Flash Attention pattern: `tl.dot` result fed directly into the next `tl.dot` without going through global memory). MLIR's dialect architecture made each optimization a composable pass.

**What it enabled:**
- Native support for Flash Attention pattern
- Better register allocation
- **Multi-backend foundation** — same pipeline could target CUDA or HIP

This shipped in the same month as PyTorch 2.0 (March 2023), which made Triton the default GPU codegen backend for `torch.compile`. Not a coincidence — the MLIR rewrite was partly driven by inductor's requirements.

### v2.1.0 — 2023

Continued MLIR pipeline improvements. More operator coverage, better Turing/Ampere support.

### v3.0.0 — 2024 🚀 AMD Backend + Hopper Preparation

Major version jump:

- **AMD/HIP backend** officially joins (`third_party/amd/`). Triton is no longer NVIDIA-only. Supports MI250/MI300 series. This is the result of heavy AMD investment.
- **NVIDIA Hopper (SM90)** foundation — `wgmma` instruction lowering begins. TMA and cluster kernels come later.
- **TritonGPU dialect layout system** refactored (paving the way for linear layouts).

### v3.1.0 — 2024

Performance improvements, bug fixes, AMD backend maturation.

### v3.2.0 — 2025-01-22

- **NVIDIA Blackwell (SM100)** preliminary support
- Hopper TMA support continues maturing
- **AMD MI350 (CDNA3)** support
- `num_ctas` cluster kernel support
- Autotune improvements (better caching, warmup logic)

### v3.3.0 — 2025-04-09

- **TLX** (Hongtao Yu's multi-warp orchestration extension) merged
- **Proton profiler** initial version — Keren Zhou's multi-level adaptive profiler integrated into Triton
- **Warp specialization** foundation — autoWS pipeline pass
- AMD gfx942/gfx950 support

### v3.3.1 — 2025-05-29

Bug fix release.

### v3.4.0 — 2025-07-30

- **Gluon architecture** introduced — new frontend IR layer for tensor-level (not block-level) operations. Product of the **Linear Layouts** paper
- **Linear layouts** — F₂ vector space algebra framework redesigning the tensor layout system, fixing many legacy layout bugs
- FP8 support improvements
- AMD ROCm 6.2 support

### v3.5.0 — 2025-10-13

- **TMA (Tensor Memory Accelerator)** mature — H100 TMA descriptors correct, pipeline pass handles `tma.load`
- **CUDA Tile backend** — NVIDIA-contributed backend compiling Triton to CUDA Tile IR instead of PTX
- Warp specialization continues strengthening
- `tl.cat` (tensor concatenation) added

### v3.5.1 — 2025-11-11

Bug fix release.

### v3.6.0 — 2026-01-20

- Scaled BMM (batched matmul with scaling) in frontend
- `tl.squeeze` / `tl.unsqueeze` — tensor shape manipulation
- FP8 constant creation from Python DSL
- AMD gfx1200/gfx1201 (RDNA4) support begins
- `constexpr` return from JIT functions
- Build infrastructure overhaul (pre-built wheels for more platforms)

### v3.7.0 — 2026-05-07 (Latest)

- **gfx1250 (RDNA4) maturation** — major AMD GPU backend update
- **Warp specialization on AMD** — autoWS now works on AMD GPUs
- **Tensor Data Movement (TDM)** — new data movement abstraction
- **Warp-pipeline path rewrite** — pipeline scheduler overhaul
- **Proton profiler production-ready** — standard tool in Triton
- Linear layouts adoption expanded to more kernels
- `tl.cat(can_reorder=False)` — non-reordering concatenation
- `get_int_attr` for out-of-tree IR walks (plugin developer API)
- Breaking changes section in release notes

### Release cadence

| Version | Date | Interval |
|---------|------|----------|
| v3.0.0 | 2024 mid | — |
| v3.1.0 | 2024 | ~3-4 months |
| v3.2.0 | 2025-01-22 | ~3 months |
| v3.3.0 | 2025-04-09 | ~3 months |
| v3.4.0 | 2025-07-30 | ~4 months |
| v3.5.0 | 2025-10-13 | ~3 months |
| v3.6.0 | 2026-01-20 | ~3 months |
| v3.7.0 | 2026-05-07 | ~4 months |

A new minor release every 3-4 months, with daily commits in between.

---

### Evolution diagram

```
2021        2022       2023          2024         2025              2026
v0.4 → v1.0    v1.1    v2.0           v3.0         v3.2  v3.4       v3.7
│              │       │              │            │     │          │
Python DSL     PyTorch MLIR          AMD           TMA   Gluon      RDNA4
prototype      2.0     rewrite       backend       Hopper+Linear    maturation
              inductor               加入           成熟  Layouts     warp-spec
              採用                                               on AMD
```

Three inflection points:
1. **v1.0** — Python DSL born, tile-based abstraction proven
2. **v2.0** — MLIR backend rewrite; Triton becomes a real compiler, not a script→PTX translator. Also what made PyTorch 2.0 inductor trust it
3. **v3.0** — AMD backend joins; NVIDIA-only → multi-GPU platform

---

## Current State: Maturity, Not Stagnation

A reasonable observer might look at the release history and think: "Triton's API barely changed since v1.0. The evolution seems slow."

This is correct as an observation but misses what's actually happening. Here's the distinction:

### What's stable (user-facing API)

The core programming model hasn't changed since v1.0:

```python
@triton.jit
def kernel(X, Y, N, BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK), mask=...)
```

A kernel written in 2021 compiles on Triton 3.7 in 2026 with zero changes. New additions (`tl.squeeze`, `tl.cat(can_reorder=False)`) are marginal. **This is by design** — stability is what allows the ecosystem (PyTorch inductor, Flash Attention, vLLM) to build on Triton without constant breakage.

### What's evolving rapidly (compiler internals)

| Feature | User-visible? |
|---------|--------------|
| Linear layouts | No — kernels are just faster and less buggy |
| Warp specialization (autoWS) | No — compiler does it automatically |
| TMA descriptor support | No — unless you write advanced Hopper kernels |
| Proton profiler | Yes — but it's a dev tool, not runtime |
| AMD backend | Yes — but only if you use AMD GPUs |
| CUDA Tile backend | No — output format changed under the hood |
| Pre-built wheels | No — `pip install` is just faster |

### Compare to compilers like LLVM or GCC

LLVM's API has been relatively stable since ~LLVM 10 (2019). But the codegen improvements — new scheduling passes, better register allocation, new target support — are massive. Triton is in the same phase: **platform stability + deep compiler innovation**. Innovation at this depth doesn't surface as flashy API changes.

### What should still concern the project

1. **JIT compile time** — large kernels can take seconds to tens of seconds on first compilation. This is the #1 community complaint.
2. **Error messages** — cryptic failures like `OutOfResources` or `AssertionError` wrapped in `CompilationError`. Debugging is painful, especially for newcomers.
3. **Autotune cache not persistent** — every process restart re-learns all shape→config mappings. Production serving pays this cost repeatedly.
4. **Codebase complexity** — 6,284 commits, 590 contributors, MLIR + Python + C++ hybrid. New contributor onboarding is hard.
5. **Documentation** — the official docs cover the basics but deep compiler internals are under-documented. Most knowledge lives in GitHub issues and PR descriptions.

### One-phrase summary

> Triton's user-facing API has reached stability, and its innovation has shifted *downward* into the compiler stack — linear layouts, warp specialization, multi-backend support, and profiling infrastructure. This is the natural trajectory of a compiler that has become infrastructure.

---

## Why Triton Matters

```
PyTorch model
    ↓
torch.compile / inductor
    ↓
Triton kernels   ←───────────────   ←──  You are here
    ↓
CUDA / HIP driver
    ↓
GPU
```

Before Triton, writing a custom GPU kernel meant either:

- **cuBLAS / cuDNN** — fast but inflexible (you can only call what NVIDIA shipped)
- **CUDA C++** — fully flexible but painful (memory coalescing, shared memory, warp synchronization, 100+ lines for a simple matmul)

Triton's bet: **a tile-level abstraction is the sweet spot**. The programmer specifies *what* tile computation to perform; the compiler handles *how* to map it to GPU hardware. The intuition is that most neural network primitives can be expressed as tiled programs, and that the tile abstraction is high enough to be productive while low enough to give the compiler room to optimize.

The bet paid off. Today, `torch.compile` generates Triton kernels for PyTorch ops on the fly, custom attention kernels are written in Triton (Flash Attention 2/3), and the ROCm backend means AMD GPU users get the same programming model.

---

## References

- [2019 MAPL paper](http://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf) — "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations"
- [GitHub repository](https://github.com/triton-lang/triton)
- [Official documentation](https://triton-lang.org)
- [Naming ambiguity with NVIDIA Triton Inference Server](https://github.com/triton-lang/triton/issues/156)
