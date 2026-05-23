# Reasoning On/Off Control

> Date: 2026-05-21
> Source: vLLM upstream main, Python layer

## Core Discovery

**vLLM does NOT have a single `--enable-reasoning` / `--disable-reasoning` CLI flag.** Reasoning enable/disable is controlled through **three independent layers** that work together:

1. **Reasoning parser selection** (server-level, CLI) — which parser class gets instantiated
2. **`enable_thinking` / `thinking`** (per-request-level, chat_template_kwargs) — tells the parser whether to extract reasoning content
3. **`include_reasoning`** (per-request-level, API parameter) — output filter

---

## Layer 1: Server-level — Reasoning parser selection

### CLI parameter

```python
# vllm/config/structured_outputs.py
class StructuredOutputsConfig:
    reasoning_parser: str = ""          # empty = no reasoning parser
    reasoning_parser_plugin: str = ""   # external plugin path
    enable_in_reasoning: bool = False   # structured output during reasoning phase
```

Passed via `--reasoning-parser deepseek_v3`.

### Registration mechanism

All parsers are lazily registered in `vllm/reasoning/__init__.py`:

```python
_REASONING_PARSERS_TO_REGISTER = {
    "deepseek_r1": ("deepseek_r1_reasoning_parser", "DeepSeekR1ReasoningParser"),
    "deepseek_v3": ("deepseek_v3_reasoning_parser", "DeepSeekV3ReasoningParser"),
    "deepseek_v4": ("deepseek_v3_reasoning_parser", "DeepSeekV3ReasoningParser"),  # same as v3
    "qwen3":       ("qwen3_reasoning_parser", "Qwen3ReasoningParser"),
    "gemma4":      ("gemma4_reasoning_parser", "Gemma4ReasoningParser"),
    "step3":       ("step3_reasoning_parser", "Step3ReasoningParser"),
    "step3p5":     ("step3p5_reasoning_parser", "Step3p5ReasoningParser"),
    "minimax_m2":  ("minimax_m2_reasoning_parser", "MiniMaxM2ReasoningParser"),
    # ... full list in `vllm/reasoning/__init__.py`
}
```

### Flow

```
CLI --reasoning-parser=deepseek_v3
  → StructuredOutputsConfig.reasoning_parser = "deepseek_v3"
    → OpenAIServingChat.__init__() calls:
        ParserManager.get_reasoning_parser("deepseek_v3")
          → lazy-imports vllm.reasoning.deepseek_v3_reasoning_parser
          → returns DeepSeekV3ReasoningParser CLASS (not instance)
```

The `ReasoningConfig` at `vllm/config/reasoning.py` stores the start/end token strings and their tokenized IDs, but does **not** control parser selection. It is called in `VllmConfig.__post_init__` via `initialize_token_ids()` to convert start/end strings into token IDs for the engine layer.

---

## Layer 2: Per-request control — `enable_thinking`

### chat_template_kwargs propagation path

```
Request body (OpenAI API)
  → ChatCompletionRequest (protocol.py)
    → build_chat_params() → ChatParams
      → .with_defaults(self.default_chat_template_kwargs)
        → chat_template_kwargs dict
```

Three sources (priority: request > CLI default > chat template default):

1. **CLI default** `--default-chat-template-kwargs '{"enable_thinking": false}'`
2. **Request body** `chat_template_kwargs: {"enable_thinking": false}`
3. **Chat template Jinja2** `{% if enable_thinking %}...{% endif %}` in tokenizer_config.json

### DeepSeekV3ReasoningParser dispatch logic

```python
# vllm/reasoning/deepseek_v3_reasoning_parser.py
class DeepSeekV3ReasoningParser(ReasoningParser):
    def __init__(self, tokenizer, *args, **kwargs):
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        thinking = bool(chat_kwargs.get("thinking", False))
        enable_thinking = bool(chat_kwargs.get("enable_thinking", False))
        thinking = thinking or enable_thinking

        if thinking:
            self._parser = DeepSeekR1ReasoningParser(tokenizer, ...)
            # → parses think...endthink → splits into reasoning vs content
        else:
            self._parser = IdentityReasoningParser(tokenizer, ...)
            # → entire output is content, no reasoning extraction
```

### What IdentityReasoningParser does

```python
class IdentityReasoningParser(ReasoningParser):
    def extract_reasoning(self, model_output, request):
        return None, model_output        # (reasoning=None, content=full_output)

    def extract_reasoning_streaming(self, ...):
        if delta_text:
            return DeltaMessage(content=delta_text)  # everything is content
        return None

    def is_reasoning_end(self, input_ids):
        return True                      # skip reasoning state tracking

    reasoning_start_str = None           # no boundary tokens at all
    reasoning_end_str   = None
```

### DeepSeekV3ReasoningWithThinkingParser

```python
# Used by glm45, holo2 etc. — models that default to thinking mode
class DeepSeekV3ReasoningWithThinkingParser(DeepSeekV3ReasoningParser):
    def __init__(self, tokenizer, *args, **kwargs):
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        thinking = chat_kwargs.get("thinking", None)
        enable_thinking = chat_kwargs.get("enable_thinking", None)
        if thinking is None and enable_thinking is None:
            chat_kwargs["thinking"] = True     # default ON
            chat_kwargs["enable_thinking"] = True
```

---

## Layer 3: `include_reasoning` — Output filter

API request parameter `include_reasoning: bool = True`.

```python
# vllm/entrypoints/openai/chat_completion/serving.py
if not request.include_reasoning:
    reasoning_ended = True   # tell engine: don't track reasoning state
```

This flag does NOT affect whether the model generates reasoning content. It only controls whether the server separates `reasoning_content` from the delta. When `False`, the server still receives model output with ` thinking...` text, but no `reasoning` field appears in the API response — everything goes into `content`.

---

## Complete data flow

```
                        Server-level                              Request-level
                        ============                              ==============

CLI --reasoning-parser=deepseek_v3           CLI --default-chat-template-kwargs
         |                                                '{"enable_thinking": false}'
         v                                                |
  StructuredOutputsConfig                          ChatCompletionRequest
    .reasoning_parser = "deepseek_v3"                  .chat_template_kwargs
         |                                                |
         v                                                |
  ParserManager.get_reasoning_parser()              _effective_chat_template_kwargs()
         |                                                |
         |    +-----------+                                |
         +--->|   class   |<--- chat_template_kwargs ------+
              |   DeepSeek |
              |   V3       |
              +-----+------+
                    |
       ┌────────────┼──────────────┐
       v            v              v
   enable_thinking=True   enable_thinking=False
       |                       |
       v                       v
  DeepSeekR1              IdentityReasoningParser
  ReasoningParser            |
       |                     v
       v                 (reasoning=None,
  (reasoning, content)      content=full_output)
```

---

## Key insight

**`enable_thinking=false` does NOT prevent the model from thinking. It only prevents the server from separating reasoning from the response.**

The real thinking on/off switch is in the chat template layer — if the Jinja2 template skips inserting the ` thinking` prefix when `enable_thinking=false`, the model will likely not enter reasoning mode. But the parser layer (IdentityReasoningParser) would work regardless — even if the model still outputs ` thinking... response`, everything would be treated as content.