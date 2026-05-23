# Roadmap — Complete Reasoning Coverage

> All items target vLLM upstream main (0.16+).

## Legend

| Icon | Meaning |
|---|---|
| ✅ | Done |
| 🔜 | Next priority |
| 📋 | Planned |
| 💡 | Future / stretch |

---

## All topics in recommended learning order

| # | Topic | Type | Why interesting | Status |
|---|---|---|---|---|
| 1 | **Reasoning on/off control (3 layers)** | Overview | `enable_thinking`、`--reasoning-parser`、`include_reasoning`—the three independent controls | ✅ |
| 2 | **IdentityReasoningParser** | Parser walkthrough | Simplest parser—when reasoning is disabled, everything is content. 10 lines of code. | 🔜 |
| 3 | **MiniMax M2** | Parser walkthrough | Model never emits start token, so only needs to check end token. Minimal streaming state machine. | ✅ |
| 4 | **DeepSeek R1** | Parser walkthrough | Standard two-phase parser with start+end tokens. Most models (Qwen3, Kimi K2, Step3...) share this pattern. | ✅ |
| 5 | **Gemma4** | Parser walkthrough | Nested/multi-turn reasoning, `<\|channel>` boundaries, `thought\n` prefix stripping with buffering. | ✅ |
| 6 | **Cohere Command** | Parser walkthrough | Uses `<results>` XML-style tags instead of thinking tokens—different delimiter paradigm. | 📋 |
| 7 | **Mistral** | Parser walkthrough | Reasoning + tool calling combined in one parser, grammar-based. The most complex one. | 📋 |
| 8 | **Remaining parsers (Qwen3, Step3/3p5, Kimi K2, Ernie 4.5, GPT-OSS, Seed-OSS, Hunyuan, HY-V3, Olmo3, Granite, Nemotron V3)** | Quick audit | Mostly aliased to existing parsers—verify mappings and note any unique behavior. | 📋 |
| 9 | **`vllm/config/reasoning.py`** | Engine config | How `ReasoningConfig` stores start/end strings and token IDs for the engine layer | 📋 |
| 10 | **`initialize_token_ids()`** | Engine config | How strings → token IDs, and why it can fail silently | 📋 |
| 11 | **`ReasoningConfig.enabled` vs parser** | Engine config | The difference between "a parser is registered" and "reasoning is actually enabled" | 📋 |
| 12 | **Where `ReasoningConfig` feeds into engine** | Engine config | `VllmConfig.__post_init__` → `initialize_token_ids()` at startup | 📋 |
| 13 | **`thinking_token_budget` in `SamplingParams`** | Per-request param | How it limits max reasoning tokens | 📋 |
| 14 | **`reasoning_effort` (low/medium/high/xhigh/max)** | Per-request param | How it's converted to chat_template_kwargs or parameters | 📋 |
| 15 | **`include_reasoning` — the output filter** | Per-request param | When `false`, everything becomes content at the API layer | 📋 |
| 16 | **How `enable_thinking` is used in Jinja2 templates** | Chat template | Where the ` thinking` token is inserted in the prompt | 📋 |
| 17 | **DeepSeek V3/V4 tokenizer_config.json chat template** | Chat template | The actual template logic for the most popular model | 📋 |
| 18 | **Qwen3 chat template** | Chat template | Different template structure | 📋 |
| 19 | **Gemma4 chat template** | Chat template | `<\|think\|>` injection mechanism | 📋 |
| 20 | **Per-request chat_template_kwargs propagation** | Chat template | How `{"thinking": true}` flows through rendering | 📋 |
| 21 | **Request lifecycle with reasoning** | Architecture | From API → chat template → tokenizer → engine → detokenizer → parser → API | 📋 |
| 22 | **How `reasoning_ended` flows through the engine** | Architecture | The flag passed to `engine_client.generate()`, how it affects sampling | 📋 |
| 23 | **`StructuredOutputsConfig` vs `ReasoningConfig`** | Architecture | Two configs, different roles—when reasoning and structured output interact | 📋 |
| 24 | **`enable_in_reasoning` — structured output during reasoning** | Architecture | Newer feature for JSON mode during thinking phase | 📋 |
| 25 | **Rust reasoning parser architecture** | Rust | How Rust parser works alongside Python parser | 📋 |
| 26 | **Prompt-based initialization** | Rust | Why prompt token IDs > model-family convention | 📋 |
| 27 | **`Qwen3ReasoningParser` in Rust** | Rust | Shared by 7+ model families | 📋 |
| 28 | **`DelimitedReasoningParser`** | Rust | Generic parser for any start/end delimiter model | 📋 |
| 29 | **Python ↔ Rust parser interaction** | Rust | Which path takes priority | 📋 |
| 30 | **What happens when reasoning never terminates** | Edge case | Model never emits end token → infinite reasoning? | 💡 |
| 31 | **Multi-token start/end tokens** | Edge case | e.g. `  response` is 2 tokens, not 1 | 💡 |
| 32 | **Nested thinking in streaming** | Edge case | Multiple `<\|channel\|>...<channel\|>` pairs in one generation | 💡 |
| 33 | **Reasoning + tool calling interaction** | Edge case | Mistral combined parser, Gemma4 tool call interleave | 💡 |
| 34 | **`skip_special_tokens=True` impact** | Edge case | How invisible boundary tokens affect parser behavior | 💡 |
| 35 | **Logprob during reasoning phase** | Edge case | Do reasoning tokens show up in `logprobs`? | 💡 |