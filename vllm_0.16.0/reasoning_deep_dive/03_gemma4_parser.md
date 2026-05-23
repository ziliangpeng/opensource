# Gemma4 Reasoning Parser

> Date: 2026-05-21
> Source: vLLM upstream main, Python layer

## Key difference from DeepSeek R1

Gemma4 supports **nested / multi-turn reasoning** — it can enter reasoning, emit a tool call, receive a tool response, and enter reasoning again, all within a single generation session. DeepSeek R1 is a simple 2-phase (reasoning → answer).

Also, unlike DeepSeek R1's simple ` thinking` / `  response` tokens, Gemma4 uses structured XML-like boundary tokens:

| Token | Value |
|---|---|
| Start | `<|channel>` |
| End | `<channel|>` |
| Additional | `<|turn>`, `<|tool_call>`, `<|tool_response>` |

### Raw model output

```
<|channel>thought
...chain of thought reasoning...<channel|>
Final answer text here.
<|tool_call>
{...}
```

Note: `thought\n` is a **role label** that must be stripped.

---

## `is_reasoning_end()` — Multi-turn aware

```python
def is_reasoning_end(self, input_ids):
    start_token_id = self.start_token_id        # <|channel>
    end_token_id = self.end_token_id            # <channel|>
    new_turn_token_id = self.new_turn_token_id    # <|turn>
    tool_call_token_id = self.tool_call_token_id  # <|tool_call>
    tool_response_token_id = self.tool_response_token_id  # <|tool_response>

    # Scan from the end backwards
    for i in range(len(input_ids) - 1, -1, -1):
        if input_ids[i] == start_token_id:
            return False          # <|channel> → entering reasoning
        if input_ids[i] == tool_call_token_id:
            return True           # <|tool_call> → reasoning is over (tool call mode)
        if input_ids[i] in (new_turn_token_id, tool_response_token_id):
            return False          # <|turn> or <|tool_response> → new reasoning round
        if input_ids[i] == end_token_id:
            return True           # <channel|> → reasoning ended
    return False
```

**Key difference from DeepSeek R1**: `new_turn_token_id` and `tool_response_token_id` cause reasoning to start **again**. This enables multi-round reasoning within a single generation.

---

## Streaming: `thought\n` prefix stripping

### The problem

Gemma4's output format includes a role label `thought\n` right after the `<|channel>` token:

```
<|channel>thought\n
```

When `skip_special_tokens=True` (vLLM default), the `<|channel>` and `<channel|>` tokens are invisible in the text. So the streaming parser receives text like `"thought\n..."` with no way to know where the actual reasoning starts.

### The solution: buffering

```python
class Gemma4ReasoningParser(BaseThinkingReasoningParser):
    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        # Instance state for streaming prefix stripping
        self._reasoning_text: str = ""   # tracks ONLY reasoning text from base parser
        self._prefix_stripped: bool = False

    def adjust_request(self, request):
        """Disable special-token stripping to preserve boundary tokens."""
        request.skip_special_tokens = False
        return request
```

Note: `adjust_request()` overrides `skip_special_tokens` to False. This ensures `<|channel>` and `<channel|>` appear in the text, making boundary detection possible.

### Streaming extraction with prefix buffering

```python
def extract_reasoning_streaming(self, ...):
    result = super().extract_reasoning_streaming(...)  # base class processing

    if result is None:
        return None

    if result.reasoning is None:
        return result      # no reasoning content in this delta → pass through

    # Accumulate reasoning text ONLY (not current_text)
    self._reasoning_text += result.reasoning

    if self._prefix_stripped:
        return result      # prefix already handled

    # Case 1: We've accumulated enough to confirm the prefix is present
    if self._reasoning_text.startswith("thought\n"):
        # Calculate how much of prefix was in this delta vs previous
        prefix_len = len("thought\n")          # = 7
        prev_reasoning_len = len(self._reasoning_text) - len(result.reasoning)

        if prev_reasoning_len >= prefix_len:
            # Prefix was entirely in prior deltas → pass through
            self._prefix_stripped = True
            return result
        else:
            # This delta contains part (or all) of the prefix
            chars_of_prefix_in_delta = prefix_len - prev_reasoning_len
            stripped = result.reasoning[chars_of_prefix_in_delta:]

            if stripped:
                self._prefix_stripped = True
                result.reasoning = stripped
                return result
            else:
                # This delta was ENTIRELY the remaining prefix
                if len(self._reasoning_text) >= prefix_len:
                    self._prefix_stripped = True
                    result.reasoning = ""          # emit empty reasoning
                    return result
                return None                        # suppress this delta

    # Case 2: accumulated text is a prefix of "thought\n"
    # e.g. we've only seen "th" or "thou"
    if "thought\n".startswith(self._reasoning_text):
        return None    # buffer — don't emit anything yet

    # Case 3: accumulated text diverged from prefix
    # e.g. first delta was "therefore" not "thought"
    self._prefix_stripped = True
    result.reasoning = self._reasoning_text  # flush buffered content
    return result
```

### Why buffering matters

Imagine the streaming deltas arrive like this:

| Step | Delta | `_reasoning_text` | Output | Why |
|---|---|---|---|---|
| 1 | `"tho"` | `"tho"` | `None` | Case 2 — prefix match, buffer |
| 2 | `"ught\n"` | `"thought\n"` | `""` (empty reasoning) | Case 1 — prefix complete, strip |
| 3 | `"L"` | `"thought\nL"` | `DeltaMessage(reasoning="L")` | Case 1 — prefix done, pass through |
| 4 | `"et me..."` | `"thought\nLet me..."` | `DeltaMessage(reasoning="et me...")` | Pass through |

If instead the first delta was `"therefore"`:

| Step | Delta | `_reasoning_text` | Output | Why |
|---|---|---|---|---|
| 1 | `"therefore"` | `"therefore"` | `DeltaMessage(reasoning="therefore")` | Case 3 — diverged from prefix, flush |
| 2 | ... | ... | ... | Normal |

---

## Summary: DeepSeek R1 vs Gemma4

| | DeepSeek R1 | Gemma4 |
|---|---|---|
| **Start token** | ` thinking` | `<|channel>` |
| **End token** | `  response` | `<channel|>` |
| **Multi-turn reasoning** | No | Yes (tool call interleave) |
| **Extra structure** | None | `thought\n` role label to strip |
| **Streaming special** | Fallback for no-start-token | Prefix buffering mechanism |
| **skip_special_tokens** | True (default) | **False** (overrides in adjust_request) |
| **Code complexity** | ~30 lines | ~200 lines (mostly prefix logic) |
| **State requirement** | Stateless (infer from tokens) | Instance field `_reasoning_text` + `_prefix_stripped` |