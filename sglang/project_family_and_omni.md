# SGLang project family and SGLang-Omni (2026-08)

**Last Updated**: 2026-08-29
**Sources**: GitHub API survey of the SGLang org (2026-08-29), SGLang-Omni README, LMSYS blog on Higgs Audio v3 serving, Red Hat vLLM-Omni architecture post.

## Note Role

How SGLang grew from one repo into a full inference-infrastructure family, and why SGLang-Omni exists. Companion to [[inference-ecosystem-vllm-llamacpp-gguf-timeline]].

## Family map (from GitHub API data)

```
├─ Education: mini-sglang (4.9k★ — dissected teaching engine), sgl-learning-materials, sgl-cookbook, sgl-docs
├─ Tooling: genai-bench (load testing), sgl-eval, sgl-whl
├─ Orchestration: sgl-model-gateway (Rust, control+data plane), rbg (K8s operator)
├─ Core engine: sglang (32.6k★, LMSYS/Berkeley, RadixAttention; multi-backend: cpu/npu/xpu)
├─ Modality: sglang-omni (audio/multimodal), rust/sglang-mm, SpecForge (spec-dec training→serving)
└─ Operators: sgl-kernel + DeepSeek kernel mirrors (DeepGEMM, FlashMLA, DeepEP, sgl-flash-attn)
   + hardware: sgl-kernel-npu (Ascend), sgl-kernel-xpu (Intel), sglang-jax
```

Timeline: 2024-01 sglang born (RadixAttention = radix-tree KV prefix reuse); 2024-H2 router+kernel split (since re-merged); 2025 SpecForge, hardware diversification, DeepSeek kernel adoption (made SGLang the de-facto DeepSeek-architecture serving standard), mini-sglang; 2026 sglang-omni (Jan), Rust gateway/workspace.

Three strategic intents: **vertical integration** (kernel → K8s, no vLLM/Triton dependency at any layer), **hardware neutrality** (Ascend/Intel/JAX — China + non-CUDA play), **engine → platform** (gateway + rbg sell "LLM inference infrastructure", not "a fast engine"). Also: main repo uses AI-agent skills (`.claude/skills/`: babysit-pr-to-pass-ci, add-jit-kernel) for CI and maintenance.

## SGLang-Omni: why it must exist

> Text-LLM serving assumes generation = one autoregressive decode loop. Speech/omni models generate via a **multi-stage pipeline** — existing engines (including SGLang core) can't hold it.

Higgs Audio v3 pipeline: text stream (starts mid-sentence) → Thinker (AR, Qwen3-4B) → interleaved text+audio tokens → tokenizer encodes 8 codebooks @25fps (delayed-interleave) → multi-codebook fusion → AR audio chunks → vocoder → 24kHz waveform. Three compute regimes on one pipeline: AR decoding (KV cache, batch sched), streaming chunk generation (low latency), DSP-like vocoder (constant-rate streaming). A single-AR scheduler mismanages every stage; GPU memory budgets can't be separated; pipeline can't be split across machines.

**Killer app: real-time voice agents.** Human turn-taking budget ~500ms; TTS must start speaking from the first half-sentence while text keeps flowing. Traditional LLM→TTS adds seconds.

SGLang-Omni = "multi-stage serving runtime": pipeline topology, per-stage schedulers, transport-aware execution (control plane + relay data plane over shared memory/NCCL/NIXL/Mooncake — cross-GPU and cross-machine stage links), model-family adapters (Qwen3-Omni, Higgs Audio v3, MOSS-TTS, Fish Speech S2-Pro, MiniMax Music 3, ZONOS2...), OpenAI-compatible `/v1/audio/speech` (streaming) + `/v1/audio/transcriptions` (diarization).

**Ecosystem design**: single-stage models stay in SGLang core (AR) or SGLang-Diffusion; Omni owns the third regime (multi-stage heterogeneous). AR stages *compose* SGLang core, not rewrite it. Separate repo because audio communities (Boson, MOSS, Fish Audio, MiniMax) have their own cadence.

## Correct classification axis

Common misconception: "Omni = inference for non-Transformer models". Actually the thinkers/talkers **are Transformers** (Higgs = Qwen3-4B AR; Qwen3-Omni Thinker–Talker both transformers). Only pipeline heads/tails (vocoders, encoders) are non-transformer. The real axis is **single-stage vs multi-stage** — three workloads need three schedulers. Note: "non-AR ≠ non-Transformer" (LLaDA 2.0 is a 100B discrete-diffusion LM with a transformer-ish backbone; SGLang has day-0 support). Cross-validation: Red Hat built vLLM-Omni (2026-07) for the same problem — two rivals independently building the same runtime is the best evidence the problem is real.

One-liner: **Omni exists because speech/omni models upgraded serving from "queueing decode tokens" to "orchestrating a heterogeneous pipeline" — a new systems problem, not a plugin.**

## See also

- see vLLM/llama.cpp/gguf ecosystem timeline in the `kb` vault (ai/llm/inference/frameworks)
- [[../../multimodal|multimodal]] subtree
