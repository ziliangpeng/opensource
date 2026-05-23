# Future Roadmap — Complete Reasoning Coverage

> All items target vLLM upstream main (0.16+).

## Legend

| Icon | Meaning |
|---|---|
| ✅ | Done |
| 🔜 | Next priority |
| 📋 | Planned |
| 💡 | Future / stretch |

---

## Phase 1: Parser internals (covered ✅)

| # | Topic | Status | File |
|---|---|---|---|
| 1 | Reasoning on/off control (3 layers) | ✅ | `01_reasoning_on_off_control.md` |
| 2 | DeepSeek R1 parser walkthrough | ✅ | `02_deepseek_r1_parser.md` |
| 3 | Gemma4 parser (nested thinking) | ✅ | `03_gemma4_parser.md` |
| 4 | MiniMax M2 parser (no start token) | ✅ | `04_minimax_m2_parser.md` |

---

## Phase 2: Remaining parsers 🔜

Each follows the same pattern (start/end tokens + streaming logic), but some have unique behaviors worth noting.

| # | Parser | Why interesting | Status |
|---|---|---|---|
| 5 | **IdentityReasoningParser** | What happens when reasoning is disabled — trivial but essential | 🔜 |
| 6 | **Qwen3ReasoningParser** | Used by Qwen3, but also aliased as Mimo — check if different | 📋 |
| 7 | **Step3 / Step3p5** | Step 3.5 has different streaming behavior vs Step 3 | 📋 |
| 8 | **Cohere Command** | Uses `<results>` XML-style tags instead of thinking tokens | 📋 |
| 9 | **MistralReasoningParser** | Integrated with Mistral grammar-based tool parser | 📋 |
| 10 | **Kimi K2** | Aliased to Qwen3 parser, but may have implicit end-by-tool-call | 📋 |
| 11 | **Ernie 4.5 / GPT-OSS / Seed-OSS / Hunyuan / HY-V3 / Olmo3 / Granite / Nemotron V3** | Mostly alias to existing parsers — verify and document mappings | 📋 |

---

## Phase 3: ReasoningConfig — The engine-side config 📋

| # | Topic | What to cover |
|---|---|---|
| 12 | `vllm/config/reasoning.py` | How `ReasoningConfig` stores start/end strings and token IDs |
| 13 | `initialize_token_ids()` | How strings → token IDs, and why it can fail silently |
| 14 | `ReasoningConfig.enabled` vs `parser` | The difference between "parser exists" and "reasoning is enabled" |
| 15 | Where `ReasoningConfig` feeds into engine | `VllmConfig.__post_init__` → `initialize_token_ids()` |

---

## Phase 4: Per-request reasoning parameters 📋

| # | Topic | What to cover |
|---|---|---|
| 16 | `thinking_token_budget` in `SamplingParams` | How it limits max reasoning tokens |
| 17 | `reasoning_effort` (low/medium/high/xhigh/max) | How it's converted to chat_template_kwargs or parameters |
| 18 | `include_reasoning` — the output filter | When `false`, everything becomes content at the API layer |

---

## Phase 5: Chat template — The prompt side 📋

| # | Topic | What to cover |
|---|---|---|
| 19 | How `enable_thinking` is used in Jinja2 templates | Where the ` thinking` token is inserted |
| 20 | DeepSeek V3/V4 tokenizer_config.json chat template | The actual template logic |
| 21 | Qwen3 chat template | Different template structure |
| 22 | Gemma4 chat template | `<|think|>` injection mechanism |
| 23 | Per-request chat_template_kwargs | How `{"thinking": true}` propagates through rendering |

---

## Phase 6: Architecture & flow 📋

| # | Topic | What to cover |
|---|---|---|
| 24 | Request lifecycle with reasoning | From API → chat template → tokenizer → engine → detokenizer → parser → API |
| 25 | How `reasoning_ended` flows through the engine | The flag passed to `engine_client.generate()`, how it affects sampling |
| 26 | `StructuredOutputsConfig` vs `ReasoningConfig` | Two configs, different roles — when reasoning and structured output interact |
| 27 | `enable_in_reasoning` — structured output during reasoning | Newer feature for JSON mode during thinking phase |

---

## Phase 7: Rust reasoning parser 📋

| # | Topic | What to cover |
|---|---|---|
| 28 | `rust/src/reasoning-parser/` architecture | How Rust parser works alongside Python parser |
| 29 | Prompt-based initialization | Why prompt token IDs > model-family convention |
| 30 | `Qwen3ReasoningParser` in Rust | Shared by 7+ model families |
| 31 | `DelimitedReasoningParser` | Generic parser for any start/end delimiter model |
| 32 | How Python ↔ Rust parsers interact | Which path takes priority |

---

## Phase 8: Edge cases & debugging 💡

| # | Topic | What to cover |
|---|---|---|
| 33 | What happens when reasoning never terminates | Model never emits end token → infinite reasoning? |
| 34 | What happens when start/end tokens are multi-token | e.g. `  response` is 2 tokens, not 1 |
| 35 | Nested thinking in streaming | Multiple `<|channel>`...`<channel|>` pairs in one generation |
| 36 | Reasoning + tool calling interaction | Mistral combined parser, Gemma4 tool call interleave |
| 37 | What happens with `skip_special_tokens=True` | How invisible boundary tokens affect parser behavior |
| 38 | Logprob during reasoning phase | Do reasoning tokens show up in `logprobs`? |

---

## Current priority order

1. IdentityReasoningParser (quick, fills the gap on "what does disabled look like")
2. Request lifecycle from API to parser output (the big picture)
3. Chat template — where the ` thinking` token actually comes from
4. ReasoningConfig — the other side of the config coin
5. Per-request parameters (thinking_token_budget, reasoning_effort)
6. Rust reasoning parser
7. Edge cases and debugging

**Phase 2-8 items will be opened as separate `notebook/` sessions** — each gets a dedicated deep dive.