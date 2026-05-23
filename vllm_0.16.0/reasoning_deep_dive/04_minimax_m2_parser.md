# MiniMax M2 Reasoning Parser

> Date: 2026-05-21
> Source: vLLM upstream main, Python layer

## Key difference from DeepSeek R1

**MiniMax M2 models never emit the ` thinking` start token.** The chat template already inserts it into the prompt, so the model's raw output starts directly with reasoning content and only produces `  response` when moving to the final answer.

### Raw model output (MiniMax M2)

```
This is the reasoning content without any start token...
It just goes directly...
  responseThis is the final answer.
```

### Raw model output (DeepSeek R1 for comparison)

```
 thinkingThis is reasoning...  responseThis is the answer.
```

---

## MiniMaxM2ReasoningParser

### Token definition (same as DeepSeek R1!)

```python
class MiniMaxM2ReasoningParser(BaseThinkingReasoningParser):
    start_token = " thinking"    # defined but never emitted by model
    end_token = "  response"
```

### Streaming — much simpler than DeepSeek R1

```python
def extract_reasoning_streaming(self, ...):
    # Skip single end token itself
    if len(delta_token_ids) == 1 and delta_token_ids[0] == self.end_token_id:
        return None

    # End token already appeared → we're in content phase
    if self.end_token_id in previous_token_ids:
        return DeltaMessage(content=delta_text)

    # End token is in this delta → split
    if self.end_token_id in delta_token_ids:
        end_index = delta_text.find(self.end_token)
        reasoning = delta_text[:end_index]
        content = delta_text[end_index + len(self.end_token):]
        return DeltaMessage(
            reasoning=reasoning if reasoning else None,
            content=content if content else None,
        )

    # No end token anywhere → all reasoning
    return DeltaMessage(reasoning=delta_text)
```

**Why this works without checking start_token_id**: Since the model never outputs ` thinking`, any text before `  response` IS reasoning. There's no need to check `self.start_token_id in previous/delta` — the only delimiter is the end token.

### Comparison with DeepSeek R1

| Scenario | DeepSeek R1 | MiniMax M2 |
|---|---|---|
| Start token in prev/delta | Must check both | Doesn't need to check (never present) |
| Fallback when no start token | Custom override (20+ lines) | Not needed (default behavior) |
| Logic branches | 4 branches (start in prev, start in delta, end in prev/delta, neither) | 3 branches (end in prev, end in delta, neither) |

---

## MiniMaxM2AppendThinkReasoningParser

An even more extreme variant — this parser doesn't reason extraction at all. It treats the entire output as content and manually prepends ` thinking` to the first delta.

```python
class MiniMaxM2AppendThinkReasoningParser(ReasoningParser):
    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.end_token_id = self.vocab.get(" response")
        self.start_token_id = self.vocab.get(" thinking")

    def extract_reasoning_streaming(self, ...):
        # If this is the very first step, prepend " thinking"
        if len(previous_token_ids) == 0:
            delta_text = " thinking" + delta_text
        return DeltaMessage(content=delta_text)   # everything is content!

    def extract_reasoning(self, model_output, request):
        return None, " thinking" + model_output    # also non-streaming
```

This parser exists for cases where the downstream system expects a ` thinking` prefix in the content (e.g. chat template rendering logic), but the model itself doesn't produce one.

---

## Summary: three models side by side

| | DeepSeek R1 | MiniMax M2 | MiniMax M2 (AppendThink) |
|---|---|---|---|
| **Model emits start token** | Yes | No | No |
| **Model emits end token** | Yes | Yes | Yes (unused) |
| **Reasoning extracted?** | Yes | Yes | **No** (everything is content) |
| **Extra text added?** | No | No | `" thinking"` prepended to first delta |
| **Code complexity** | ~30 lines + fallback | ~20 lines | ~15 lines |

This is a great example of how the same fundamental parser pattern (token delimited split) adapts to different model behaviors — from full extraction (R1) to minimal detection (M2) to no extraction at all (AppendThink).