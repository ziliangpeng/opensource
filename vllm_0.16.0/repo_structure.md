# vLLM Repo Structure

Analyzing version **v0.16.0** (source available in `code/` submodule).

## Top-Level Layout

```
vllm/                  # Main Python package — the inference engine (see code-structure.md)
csrc/                  # C++/CUDA custom kernels (attention, quantization, sampling)
tests/                 # Test suite (models, kernels, engine, distributed, API)
benchmarks/            # Performance benchmarks (throughput, latency)
examples/              # Usage examples (offline inference, API server, chat templates, tool calling)
docs/                  # Documentation site (MkDocs)
docker/                # Dockerfiles per platform (CUDA, ROCm, TPU, CPU, XPU, s390x, ppc64le)
requirements/          # Pip requirements split by backend (CUDA, ROCm, TPU, etc.)
cmake/                 # CMake modules for C++/CUDA build
tools/                 # Dev/CI utility scripts (profiling, formatting, release)
.buildkite/            # Buildkite CI pipelines (primary CI system)
.github/               # GitHub Actions, issue templates, CODEOWNERS
```

## File Types Per Directory

```
vllm/                  1308 py, 514 json, 7 jinja, 4 md, 1 proto, 1 js, 1 css
csrc/                  75 cu, 69 cuh, 34 hpp, 32 h, 24 cpp, 5 py, 1 inl
tests/                 931 py, 61 yaml, 14 txt, 9 sh, 7 json, 3 png, 3 md
benchmarks/            78 py, 5 yaml, 5 sh, 5 md, 3 txt, 2 json
examples/              129 py, 42 jinja, 22 yaml, 17 md, 16 sh, 4 json
docs/                  178 md, 72 png, 5 yml, 5 py, 5 js, 2 jpg, 2 html
docker/                10 Dockerfiles (no ext), 1 json, 1 hcl
requirements/          17 txt, 1 in
cmake/                 6 cmake, 1 py
tools/                 15 py, 13 sh, 3 png, 2 md, 1 patch, 1 json
.buildkite/            65 yaml, 44 sh, 14 json, 7 txt, 6 py, 2 md
.github/               23 yml, 5 sh, 3 json, 1 md, 1 js
```

## Top-Level Files

- `CMakeLists.txt` — build system for C++/CUDA extensions (46KB)
- `setup.py` — Python package build with custom CUDA compilation logic (39KB)
- `pyproject.toml` — project metadata and tool configuration
- `mkdocs.yaml` — documentation site configuration
- `README.md`, `RELEASE.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`
- `LICENSE` — Apache 2.0
