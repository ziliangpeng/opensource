# DeepSeek R1 Parser Walkthrough

> Date: 2026-05-21
> Source: vLLM upstream main, Python layer

## Class hierarchy

```
DeepSeekR1ReasoningParser
  → BaseThinkingReasoningParser
    → ReasoningParser (abstract base)
```

---

## What the inputs/outputs look like

### Model raw output

```
 thinkingLet me break down this problem step by step...
First, I need to...  responseThe answer is 42.
```

### API responses

```json
// Streaming deltas:
{ "reasoning": "Let me break down this problem..." }
{ "reasoning": " step by step...\nFirst, I need to..." }
{ "reasoning": "\nThe answer is" }
{ "content": " 42." }

// If a delta contains BOTH the end of reasoning and the start of content:
{ "reasoning": " 42.", "content": null }

// Non-streaming (one shot):
{ "reasoning": "Let me break down...\nFirst...\nThe answer is",
  "content": " 42." }
```

---

## Abstact base: `abs_reasoning_parsers.py`

```python
class ReasoningParser:
    @property
    def reasoning_start_str(self) -> str | None:
        return None  # subclasses override

    @property
    def reasoning_end_str(self) -> str | None:
        return None

    @abstractmethod
    def extract_reasoning(model_output, request) -> tuple[str | None, str | None]:
        """Non-streaming: split the complete string in one shot."""

    @abstractmethod
    def extract_reasoning_streaming(
        previous_text, current_text, delta_text,
        previous_token_ids, current_token_ids, delta_token_ids
    ) -> DeltaMessage | None:
        """Streaming: called each decode step, returns a single delta."""
```

---

## Core: `BaseThinkingReasoningParser` in `basic_parsers.py`

### Constructor — resolve token IDs at init time

```python
class BaseThinkingReasoningParser(ReasoningParser):
    # Subclasses must provide these
    @property
    @abstractmethod
    def start_token(self) -> str: ...
    @property
    @abstractmethod
    def end_token(self) -> str: ...

    def __init__(self, tokenizer):
        # Resolve string → token ID at construction time.
        # This is KEY: streaming boundary detection uses integer comparison,
        # not string matching, which is much faster.
        self.start_token_id = self.vocab.get(self.start_token)
        self.end_token_id = self.vocab.get(self.end_token)

        if self.start_token_id is None or self.end_token_id is None:
            raise RuntimeError(
                f"{self.__class__.__name__} could not locate "
                "think start/end tokens in the tokenizer!"
            )
```

### `is_reasoning_end()` — Check if reasoning phase is over

```python
def is_reasoning_end(self, input_ids):
    """
    Scan from the end of input_ids backwards:
    - Last token is start_token → still in reasoning (return False)
    - Last token is end_token   → reasoning is done (return True)
    - Neither found             → default to still reasoning (return False)
    """
    for i in range(len(input_ids) - 1, -1, -1):
        if input_ids[i] == self.start_token_id:
            return False
        if input_ids[i] == self.end_token_id:
            return True
    return False
```

Note: the input is the **entire output token list so far**, not a single token. Called each decode step with the complete accumulated tokens.

### `is_reasoning_end_streaming()` — Optimized for streaming

```python
def is_reasoning_end_streaming(self, input_ids, delta_ids):
    """Only check delta_ids for the end token."""
    end_token_id = self.end_token_id
    return end_token_id in delta_ids
```

### `extract_reasoning()` — Non-streaming

```python
def extract_reasoning(self, model_output, request):
    # Partition on start_token:
    #   "prefix" + start_token + "rest"
    model_output_parts = model_output.partition(self.start_token)
    model_output = (
        model_output_parts[2]    # keep the part after start_token
        if model_output_parts[1] # if start_token was found
        else model_output_parts[0]  # otherwise keep everything
    )

    # Partition on end_token:
    if self.end_token not in model_output:
        return model_output, None    # all reasoning, no content
    else:
        reasoning, _, content = model_output.partition(self.end_token)
        return reasoning, content or None
```

### `extract_reasoning_streaming()` — The streaming state machine

**This is the most important method.** Called once per decode step. The state (am I in reasoning or content phase) is NOT stored in a mutable field — it's inferred from the token sequences each time.

```python
def extract_reasoning_streaming(
    self,
    previous_text,       # complete text from previous step
    current_text,        # previous_text + delta_text
    delta_text,          # new text in this decode step
    previous_token_ids,  # complete token IDs from previous step
    current_token_ids,   # previous_token_ids + delta_token_ids
    delta_token_ids,     # new token IDs in this decode step
):
```

**Step 1: Skip single special tokens**

```python
    # If the delta is literally just the start or end token itself,
    # skip it — these tokens don't appear in the API response.
    if len(delta_token_ids) == 1 and \
       delta_token_ids[0] in [self.start_token_id, self.end_token_id]:
        return None
```

**Step 2: Determine where the start token is**

```python
    if self.start_token_id in previous_token_ids:
        # start token was found in a PREVIOUS step
        # → we already entered reasoning phase

        if self.end_token_id in delta_token_ids:
            # This delta contains the end token.
            # Split: before end = reasoning, after = content
            end_index = delta_text.find(self.end_token)
            reasoning = delta_text[:end_index]
            content = delta_text[end_index + len(self.end_token):]
            return DeltaMessage(
                reasoning=reasoning,
                content=content if content else None
            )

        elif self.end_token_id in previous_token_ids:
            # End token appeared BEFORE this delta → we're in content phase
            return DeltaMessage(content=delta_text)

        else:
            # No end token anywhere → still in reasoning
            return DeltaMessage(reasoning=delta_text)

    elif self.start_token_id in delta_token_ids:
        # start token appears in THIS delta for the first time
        # Same logic as above, but anchored in delta_text
        ...

    else:
        # Haven't seen start token yet → default to content
        return DeltaMessage(content=delta_text)
```

---

## DeepSeek R1: `deepseek_r1_reasoning_parser.py`

### Token definition

```python
class DeepSeekR1ReasoningParser(BaseThinkingReasoningParser):
    @property
    def start_token(self) -> str:
        return " thinking"    # note the leading space!

    @property
    def end_token(self) -> str:
        return "  response"   # note TWO leading spaces!
```

### Override: handling models that never emit start_token

```python
    def extract_reasoning_streaming(self, ...):
        ret = super().extract_reasoning_streaming(...)   # call base class

        # If the base class returned something AND start token is
        # NOT in either previous or delta tokens:
        if ret is not None and \
           self.start_token_id not in previous_token_ids and \
           self.start_token_id not in delta_token_ids:
            # Model never output " thinking" — chat template may have
            # already inserted it. Determine phase by checking end_token.

            if self.end_token_id in delta_token_ids:
                # Found end token in this delta → split
                end_index = delta_text.find(self.end_token)
                reasoning = delta_text[:end_index]
                content = delta_text[end_index + len(self.end_token):]
                return DeltaMessage(
                    reasoning=reasoning,
                    content=content if content else None
                )

            elif self.end_token_id in previous_token_ids:
                # End token appeared earlier → content phase
                return DeltaMessage(content=delta_text)

            else:
                # No end token anywhere → still reasoning
                return DeltaMessage(reasoning=delta_text)

        return ret   # fall through to base class result
```

**Why this override exists**: Some chat templates insert the ` thinking` token into the prompt itself. In that case, the model doesn't output ` thinking` — it just starts generating reasoning tokens. The base class would see `start_token_id not in previous/delta` and default to `DeltaMessage(content=...)`, treating everything as content. This override detects that case and correctly splits on `  response`.

---

## Summary: streaming state machine design

| Property | Value |
|---|---|
| **State tracking** | No mutable `self.in_reasoning` field. State is inferred from token content each time. |
| **Input granularity** | Called once per decode step. Gets ALL accumulated tokens, not just one. |
| **Boundary detection** | Integer comparison on token IDs (not strings) — fast path. |
| **Special token skip** | Start/end tokens themselves are suppressed from output. |
| **Fallback case** | When model never outputs start token (chat template inserted it). |

The parser is essentially a **stateless function with access to full history** — pure enough to be deterministic but context-aware enough to handle streaming.