# Triton Study Roadmap

A structured path from zero to compiler-deep understanding of Triton.

---

## 概覽：四階段計畫

```
Phase 1 (1-2 週) — Triton 用戶：寫 kernel
    ↓  熟悉 DSL、autotune、benchmark
Phase 2 (2-3 週) — Compiler 前端：看懂編譯流程
    ↓  建立 MLIR 基礎、trace 實際 IR
Phase 3 (3-4 週) — Compiler 核心：Pass Pipeline
    ↓  理解 layout 系統、pipeline scheduler
Phase 4 (選修) — Backend 專家
    ↓  NVIDIA/AMD GPU-specific lowering
```

每個階段都有：學習目標、閱讀材料、實作練習、驗收標準。

---

## Phase 0: 前置知識（必要，不要跳過）

### GPU 基礎

| 主題 | 資源 |
|------|------|
| CUDA thread hierarchy | grid → block → warp → thread |
| Memory hierarchy | global → L2 → shared → register |
| Memory coalescing | 相鄰 threads 存取相鄰 addresses |
| Occupancy | warp 數 / SM 上限，受 register/SMEM 限制 |

**推薦資源：**
- [CUDA Mode](https://github.com/cuda-mode) lectures（Lecture 1-5）
- NVIDIA's "CUDA C++ Programming Guide" — 至少讀 memory hierarchy 和 execution model 章節
- "Programming Massively Parallel Processors"（書）— 前三章

> 💡 如果你已經熟悉 CUDA，Phase 0 可以跳過。

---

## Phase 1 — Triton 用戶（1–2 週）

### 目標

能獨立用 Triton 寫常見 kernel（element-wise、reduction、matmul、softmax、layer norm），能用 autotune 選最佳參數，能用 `do_bench` 做效能評估。

### 學習路徑

#### Step 1.1: 基礎概念（1-2 天）

讀 [Triton 官方文檔 Programming Guide](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html)：

- Chapter 1: Introduction — 為什麼 tile-level abstraction？
- Chapter 2: `@triton.jit`、`tl.constexpr`、`tl.arange`、`tl.load`/`tl.store`
- Chapter 3: Reduction、mask、`tl.sum`、`tl.max`

**同時做**：跑 [Triton 官方 Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)：
- 01-vector-add
- 02-fused-softmax
- 03-matrix-multiplication

> 在 `~/code` 下開一個 `triton-study/` 目錄，所有練習寫在裡面。

#### Step 1.2: 自己動手寫（3-4 天）

用 Triton 重新實作這些 kernel：

1. **Layer Normalization** — 了解 reduction + element-wise 組合
2. **Flash Attention forward** — 了解 back-to-back matmul（用 `tl.dot` 兩次）
3. **RMS Norm**
4. **Simple element-wise fusion**（例如 `silu(x) * x` 的 fused kernel）

**驗收標準：** 每個 kernel 跟 PyTorch baseline 做 correctness + benchmark，寫入筆記。

#### Step 1.3: Autotune 深度理解（1-2 天）

- 讀 `python/triton/runtime/autotuner.py`（~500 lines）
- 實作一個自定義 `early_config_prune` — 比如限制 SMEM 消耗在 48KB 以下
- 實驗：同樣的 matmul，不同的 config pool，看效能差距多少
- 理解 `key` 參數對 cache 行為的影響

#### Step 1.4: GPU MODE Lecture 14（1 天）

看 [GPU MODE Lecture 14: Practitioners Guide to Triton](https://christianjmills.com/posts/cuda-mode-notes/lecture-014/)，跟著做範例。

#### Step 1.5: Triton Puzzles（選修，推薦）

做 [Triton Puzzles](https://github.com/gpu-mode/Triton-Puzzles) — Sasha Rush 設計的一系列 puzzle。不需要 GPU（用 interpreter mode）。從簡單的 vector add 到 flash attention。

### Phase 1 驗收

✅ 能在 30 分鐘內從零寫出一個 fused softmax kernel
✅ 理解 `tl.load` 的 `mask` / `other` 語義
✅ 能用 `@triton.autotune` 並解讀 benchmark 結果
✅ 知道什麼時候用 `tl.dot` vs 手動乘加

---

## 過渡：MLIR 基礎（必要，1 週）

在進入 compiler 內部之前，你需要 MLIR 的最低存活知識。

### 學習目標

能讀懂 Triton 的 MLIR IR dump，知道 operation、type、region、block、pass 是什麼。

### 資源

1. [MLIR 官方文檔 — 第一章](https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/) — 只讀 "Understanding the IR Structure"，理解：
   - `Operation`（`%0 = arith.addi %a, %b : i32`）
   - `Value` 和 `SSA`
   - `Block` 和 `Region`
   - `OpTrait` 和 `Interface`

2. Kapil Sharma 的 [Deep Dive into Triton Internals Part 1](http://www.kapilsharma.dev/posts/deep-dive-into-triton-internals/) — 最好的一篇 Triton compiler 入門文。他 trace 了 `tl.dot` 從 Python DSL → TTIR → TTGIR → LLVM IR 的完整路徑。

> **不要**試著讀完整個 MLIR 文檔。只需要夠讀懂 Triton IR dump 即可。

### 實作

編譯以下程式，dump 每一層 IR：

```python
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    # ... 標準 matmul 實作
```

用 `TRITON_DEBUG=1` 或自定義 compile options 印出：
- **TTIR** — 最初的 Triton IR，包含高階 `tt.dot` op
- **TTGIR** — 轉成 TritonGPU dialect 後，tensor layout 被賦予
- **LLVM IR** — 最終轉成 LLVM
- **PTX** — 最終 GPU assembly

使用 Triton 的 interpreter mode（`TRITON_INTERPRET=1`）可以不依賴 GPU 執行並 debug。

### 驗收

✅ 能讀懂 TTGIR dump，指出哪些 op 在 shared memory、哪些在 register
✅ 理解 `tt.load` → `ttg.load` 的 lowering 做了 layout conversion

---

## Phase 2 — Compiler 前端（2–3 週）

### 目標

理解 `@triton.jit` → `triton.compile()` → TTIR 的完整前端路徑。能 trace 一段 `tl.load` 怎麼變成 `tt.load` op。

### 閱讀材料

#### 2.1 Python DSL 層

| 檔案 | 行數 | 內容 |
|------|------|------|
| `python/triton/compiler/compiler.py` | ~600 | 編譯入口：`compile()` 的 orchestration |
| `python/triton/language/semantic.py` | ~1300 | `tl.load`/`tl.store`/`tl.dot` 的語義 — 這是前端核心 |
| `python/triton/language/core.py` | ~800 | `triton.language` 的 AST 構建 |
| `python/triton/runtime/jit.py` | ~400 | `@triton.jit` decorator 的實作 |

**關鍵閱讀順序：**

1. 讀 `jit.py` — 理解 `@triton.jit` 怎麼攔截 function 呼叫、收集 AST
2. 讀 `compiler.py` — 理解 `compile()` 怎麼把 AST 餵給 MLIR pipeline
3. 讀 `semantic.py` 中 `tl.load` 和 `tl.store` 的實作 — 理解 Python AST 節點怎麼 map 到 MLIR operation

#### 2.2 Triton IR（tt dialect）

| 檔案 | 行數 | 內容 |
|------|------|------|
| `include/triton/Dialect/Triton/IR/` | — | `TritonOps.td` — op 定義（TableGen） |
| `lib/Dialect/Triton/IR/Ops.cpp` | — | op 的 C++ 實作 |

**不需要讀所有 op。** 專注於：`tt.load`、`tt.store`、`tt.dot`、`tt.reduce`、`tt.reshape`。

#### 2.3 實作練習

用 `triton.compile` 的 Python API，手動構建一個簡單的 kernel（不用 `@triton.jit`），直接輸出 TTIR。

```python
# 概念性範例（非完整實作）
module = triton.ir.Module()
ir = """
#ttir = #triton.ir<dialect = ttir>
module {
    tt.func @add_kernel(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>, %n: i32) {
        // ...
    }
}
"""
asm = triton.ir.parse_mlir_module(ir)
```

> 這個練習的目的是理解 `@triton.jit` 幫你省了什麼事。

### Phase 2 驗收

✅ 能解釋 `@triton.jit` 從 Python function 到 TTIR 的路徑
✅ 知道 `tl.load` 的 `mask` 和 `other` 參數怎麼影響產生的 IR
✅ 能人工讀懂一份 TTIR dump

---

## Phase 3 — Compiler 核心：Pass Pipeline（3–4 週）

### 目標

理解 TritonGPU dialect 的 layout 系統、核心 optimization passes（coalescing、shared memory allocation、pipelining、warp specialization），以及 autotune 怎麼跟 compiler 互動。

### 前置閱讀

**Kapil Sharma 的系列文章是必讀：**
- [Part 2: Frontend](http://www.kapilsharma.dev/posts/deep-dive-into-triton-internals-2/)
- [Part 3: MLIR Internals](http://www.kapilsharma.dev/posts/deep-dive-into-triton-internals-3/)
- [GPU Mode Talk](http://www.kapilsharma.dev/posts/gpu-mode-triton-internals-talk/)

### 核心主題

#### 3.1 Layout 系統（最重要）

TritonGPU 的核心創新是同一個 logical tensor 在 GPU 生命周期中有多種 physical layout：

| Layout | 位置 | 描述 |
|--------|------|------|
| `BlockedLayout` | Global memory | Tensor 被切成 block，block→thread mapping |
| `SharedLayout` | Shared memory | swizzled / row-major data layout |
| `MmaLayout` | Register (warp) | NVIDIA `wgmma` / AMD `mfma` 指令的 register layout |
| `LinearLayout` | Any | F₂ 向量空間代數框架（3.4+）統一描述以上布局 |
| `SliceLayout` | Meta | 子 tensor 的視圖 |
| `DotOperandLayout` | Register (warp) | `tl.dot` 的 operand layout |

**如何讀（循序漸進）：**

1. 讀 `include/triton/Dialect/TritonGPU/IR/TritonGPUAttrs.td` — layout attribute 的定義
2. 讀 Linear Layouts 論文 ([arXiv 2505.23819](https://arxiv.org/abs/2505.23819)) — 這是近年 Triton 最重要的架構改動
3. 讀 `lib/Tools/LinearLayout.cpp` — 實作

**實作：** 用 `TRITON_DEBUG=1` 看同一個 kernel 在不同 layout assignment 下的表現。

#### 3.2 Coalescing Pass

| 檔案 | 內容 |
|------|------|
| `lib/Dialect/TritonGPU/Transforms/Coalesce.cpp` | Memory coalescing pass |

把 `BlockedLayout` 的 thread-block→data mapping 最佳化，確保相鄰 threads 存取相鄰 addresses。

#### 3.3 Shared Memory Allocation

| 檔案 | 內容 |
|------|------|
| `lib/Dialect/TritonGPU/Transforms/AllocateSharedMemory.cpp` | SMEM 分配 pass |

決定哪些 tensor 需要放 shared memory，分配 offset。

#### 3.4 Pipeline Scheduler（num_stages）

| 檔案 | 內容 |
|------|------|
| `lib/Dialect/TritonGPU/Transforms/Pipeline.cpp` | 軟體 pipelining pass（Paweł Szczerbuk） |

就是 autotune 中 `num_stages` 參數背後的 pass — 把 global memory load 跟 computation overlap。

**關鍵理解：** 這個 pass 對 TMA vs manual load 的行為完全不同。AMD 上沒有 TMA，pipeline 層數受限。

#### 3.5 Warp Specialization（autoWS）

| 檔案 | 內容 |
|------|------|
| `lib/Dialect/TritonGPU/Transforms/WarpSpecialize.cpp` | 自動 warp specialization（htyu） |

把 warp 分成 producer/consumer role — 一些專門做 load，另一些做 compute。

#### 3.6 Triton → TritonGPU Conversion

| 檔案 | 內容 |
|------|------|
| `lib/Conversion/TritonToTritonGPU/` | TTIR → TTGIR |

把高階 `tt.dot` op 降低到 `ttg.dot` + layout constraints。決定每個 tensor 用什麼 layout。

### Phase 3 驗收

✅ 能解釋 `BlockedLayout` 跟 `MmaLayout` 的差別
✅ 能畫出一個 matmul kernel 的 pipeline pass 示意圖（load A/B → compute C → store C 的指令排程）
✅ 理解為什麼 AMD 上 `num_stages=2` 而 H100 上 `num_stages=4`
✅ 讀懂一份 TTGIR dump 中每個 tensor 的 layout attribute

---

## Phase 4 — Backend（選修，需數月）

### 目標

選擇一個 backend（NVIDIA 或 AMD），理解從 TTGIR → device binary 的最終 lowering。

#### NVIDIA Backend

| 檔案/目錄 | 內容 |
|-----------|------|
| `third_party/nvidia/` | CUDA backend orchestration (Python) |
| `lib/Conversion/TritonGPUToLLVM/` | TTGIR → NVVM 的核心 lowering |
| `include/triton/Dialect/TritonNvidiaGPU/` | `ttng` dialect — TMA、cluster、wgmma |

關鍵 lowering 路徑：

```
tt.dot → ttg.dot (with MmaLayout)
       → nvvm.wgmma (Hopper) / nvvm.mma (Ampere)
       → PTX instruction
```

**需要理解的 NVIDIA 專有概念：**
- `wgmma` instruction format
- TMA descriptor setup + `tma.load`
- Tensor Core operations（`tcgen` dialect）
- `num_ctas` cluster kernel
- CUDA Tile IR backend

#### AMD Backend

| 檔案/目錄 | 內容 |
|-----------|------|
| `third_party/amd/` | AMD backend orchestration (Python), 120K lines |
| `lib/Conversion/TritonGPUToLLVM/` | 也包含 AMD 的 lowering path（有 `#ifdef USE_ROCM`） |

**需要理解的 AMD 專有概念：**
- Wavefront size 64 的影響
- MFMA instruction format
- 無 TMA → 手動 double buffering
- gfx942/CDNA3 vs gfx950/CDNA4 vs gfx1200/RDNA4 差異
- `ds_swizzle` bit操作

### Phase 4 驗收

✅ 能讀懂一份 PTX/SASS dump，指出哪個 instruction 對應 kernel 中的哪個操作
✅ 能解釋為什麼同一個 Triton kernel 在 NVIDIA 和 AMD 上產生的 assembly 完全不同

---

## Resources 彙總

### 必讀（依次序）

| 資源 | 類型 | 對應階段 |
|------|------|---------|
| [Triton Official Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) | 互動 | Phase 1 |
| [Triton Puzzles](https://github.com/gpu-mode/Triton-Puzzles) | 互動 | Phase 1 |
| Kapil Sharma — [Triton Internals Part 1](http://www.kapilsharma.dev/posts/deep-dive-into-triton-internals/) | Blog | MLIR 過渡 |
| Kapil Sharma — [Part 2](http://www.kapilsharma.dev/posts/deep-dive-into-triton-internals-2/) | Blog | Phase 2 |
| Kapil Sharma — [Part 3](http://www.kapilsharma.dev/posts/deep-dive-into-triton-internals-3/) | Blog | Phase 3 |
| [Linear Layouts 論文](https://arxiv.org/abs/2505.23819) | 論文 | Phase 3 |
| [TritonGPU Dialect 文檔](https://triton-lang.org/main/dialects/TritonGPUDialect.html) | 參考 | Phase 3 |
| [Triton Meetup Videos](https://triton-lang.org/main/meetups/) | 影片 | Phase 3-4 |
| [AMD Triton Compilation 深入](https://medium.com/@nzhangnju/a-deep-dive-into-amd-triton-compilation-912d96e68e45) | Blog | Phase 4 (AMD) |

### 選修參考

| 資源 | 類型 | 適合 |
|------|------|------|
| [GPU MODE Lecture 14](https://christianjmills.com/posts/cuda-mode-notes/lecture-014/) | 影片+筆記 | Phase 1 |
| [ML-Triton 論文](https://arxiv.org/pdf/2503.14985) | 論文 | Phase 3 |
| [MLIR 官方文檔 — Understanding IR](https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/) | 參考 | MLIR 過渡 |
| [Triton 2019 MAPL 論文](http://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf) | 論文 | 背景知識 |
| [OpenAI Triton 1.0 發表 Blog](https://openai.com/index/triton/) | Blog | 背景知識 |
| [Proton Profiler 論文](https://www.semanticscholar.org/paper/Proton%3A-Towards-Multi-level%2C-Adaptive-Profiling-for-Zhou-Zhong/05ff8cb22c2070c230666db047e450ca04c9fd3c) | 論文 | Phase 3 |

---

## 學習節奏建議

對正常全職工作的節奏：

| 時間 | 階段 | 每週投入 |
|------|------|---------|
| 第 1-2 週 | Phase 1: 寫 kernel | 3-4 小時 |
| 第 3 週 | MLIR 基礎 | 2-3 小時 |
| 第 4-6 週 | Phase 2: 前端 | 3-4 小時 |
| 第 7-10 週 | Phase 3: Pass Pipeline | 4-5 小時 |
| 第 11+ 週 | Phase 4: Backend | 選修 |

> 「讀 compiler code」不同於讀應用程式 code。不要預期每行都看懂。目標是建立 mental model — 知道每個 component 存在、負責什麼、跟其他 component 怎麼互動。

---

## Codebase 中最重要的 10 個檔案（按閱讀優先級）

| # | 檔案 | 行數 | 為什麼重要 |
|---|------|------|-----------|
| 1 | `python/triton/runtime/autotuner.py` | ~500 | Autotune 完整實作 |
| 2 | `python/triton/compiler/compiler.py` | ~600 | 編譯入口，pipeline orchestration |
| 3 | `python/triton/language/semantic.py` | ~1300 | `tl.load`/`tl.store`/`tl.dot` 語義 |
| 4 | `python/triton/runtime/jit.py` | ~400 | `@triton.jit` decorator |
| 5 | `include/triton/Dialect/TritonGPU/IR/TritonGPUAttrs.td` | — | Layout attributes 定義 |
| 6 | `lib/Dialect/TritonGPU/Transforms/Coalesce.cpp` | — | Coalescing pass |
| 7 | `lib/Dialect/TritonGPU/Transforms/Pipeline.cpp` | — | Software pipeline |
| 8 | `lib/Dialect/TritonGPU/Transforms/AllocateSharedMemory.cpp` | — | SMEM 分配 |
| 9 | `lib/Conversion/TritonToTritonGPU/TritonGPUConversion.cpp` | — | TTIR → TTGIR |
| 10 | `third_party/nvidia/backend/compiler.py` 或 `third_party/amd/backend/compiler.py` | — | Backend compilation流程 |
