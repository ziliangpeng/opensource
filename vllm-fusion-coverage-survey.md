# vLLM Fusion Coverage Survey — 294 model files, one afternoon

**Method**: static scan of `vllm/model_executor/models/*.py` on main (2026-09-02, 294 of 299 files fetched; commit `HEAD@2026-09-02`). Classified each decoder's residual/norm call pattern:
- **fused** — ≥1 two-arg `rms_norm(x, residual)` call, no explicit `+ residual`
- **naive** — explicit `+ residual`, no two-arg calls
- **mixed** — both patterns in one file

## Results

| Pattern | Count | Who |
|---|---|---|
| fused | **53** | llama, qwen2/qwen3/×moe, mistral, mixtral, gemma/gemma2/gemma3, glm4(+moe), internlm2, olmoe, jamba, minimax_m2, gpt_oss, lfm2, bailing_moe, commandr, … |
| mixed | 7 | deepseek_v2, nemotron, molmo(+2), solar |
| naive | **20** | **gemma4, gemma4_mtp**, dbrx, bloom, gpt2, gpt_j, falcon(+h1), phi, chatglm, mimo, jais2, step3p5, iquest_loopcoder, minicpmv4_6, nemotron_h_mtp, llama_eagle, phimoe, interfaces |
| no-residual / other | 214 | vision encoders, embedding-only, pre-RMSNorm legacy archs |

## Reading the numbers

1. **The sweep already happened, informally.** 53 fused files = every mainstream serving model. The mechanism is template-following: new models copy `llama.py`'s pattern, so post-2024 additions (qwen3, gpt_oss, lfm2, minimax_m2) are fused by construction.
2. **Half the naive list is structurally exempt** — bloom/gpt2/falcon/phi are pre-RMSNorm architectures where residual threading doesn't apply. The real stragglers: **gemma4 (+ its MTP draft, which also runs the unfused path — relevant since CAI's production uses MTP spec decode)**, dbrx, mimo, jais2.
3. **Gemma4 is the cautionary tale**: a 2026-04 day-0 port (5,051 lines, Luciano Martins/Google) shipped the pre-fusion pattern. AMD filed #53273 to bring it to llama-parity — and gemma4_mtp.py still isn't fused (pending in #53273's MTP commit).
4. **A static sweep costs one afternoon.** The scan is ~40 lines of regex over raw files (no GPU needed). The expensive part is not detection — it's the per-model × per-hardware *verification* each fix needs.

## Survey script (reusable)

```python
# two-arg fused call: layernorm(..., residual) — pattern from llama.py
two_arg = re.findall(r'(?:input_layernorm|post_attention_layernorm|_norm)\([^()\n]+,\s*residual\)', src)
explicit = re.findall(r'\+\s*residual\b', src)
```

Caveat: heuristic, not AST. `interfaces.py`/abstract base files will misclassify; treat counts as a triage list, not ground truth.
