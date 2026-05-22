# Reasoning 開關控制：完整路徑追蹤

> vLLM upstream main, 2026-05-21。Python 層 + Rust 層 reasoning parser。

## 重點結論

**vLLM 沒有單一的 `--enable-reasoning` / `--disable-reasoning` CLI flag。** Reasoning 的啟用/停用是透過兩層機制協作達成的：

1. **Reasoning parser 選擇**（server-level）— 指定哪個 parser class 被實例化
2. **`enable_thinking` / `thinking`**（per-request-level）— 告訴 parser 要不要解析 reasoning content

兩者都設對，reasoning 才真正作用。

---

## 第一層：Server-level — reasoning parser 選擇

### CLI 參數

參數名：`--reasoning-parser`（存在 `StructuredOutputsConfig` 中）

```python
# vllm/config/structured_outputs.py
class StructuredOutputsConfig:
    reasoning_parser: str = ""          # 空字串 = 無 reasoning parser
    reasoning_parser_plugin: str = ""   # 外部 plugin 路徑
    enable_in_reasoning: bool = False   # 是否在 reasoning 階段啟用 structured output
```

註冊在 CLI：
```python
# vllm/entrypoints/openai/cli_args.py
parser.add_argument('--reasoning-parser', ...)
```

### 註冊機制

所有 parser 在 `vllm/reasoning/__init__.py` 中以 lazy module 方式註冊：

| 名稱 | Module | Class |
|---|---|---|
| `deepseek_r1` | `deepseek_r1_reasoning_parser` | `DeepSeekR1ReasoningParser` |
| `deepseek_v3` | `deepseek_v3_reasoning_parser` | `DeepSeekV3ReasoningParser` |
| `deepseek_v4` | → 同 deepseek_v3 | → 同 DeepSeekV3ReasoningParser |
| `qwen3` | `qwen3_reasoning_parser` | `Qwen3ReasoningParser` |
| `step3` | `step3_reasoning_parser` | `Step3ReasoningParser` |
| `step3p5` | `step3p5_reasoning_parser` | `Step3p5ReasoningParser` |
| `minimax_m2` | `minimax_m2_reasoning_parser` | `MiniMaxM2ReasoningParser` |
| `identity` | 無（直接實例化） | `IdentityReasoningParser` |
| ...全列表在 `vllm/reasoning/__init__.py` `_REASONING_PARSERS_TO_REGISTER` | | |

### 選擇到哪個 parser

```
CLI --reasoning-parser=deepseek_v3
  → StructuredOutputsConfig.reasoning_parser = "deepseek_v3"
    → OpenAIServingChat.__init__() 中呼叫:
        ParserManager.get_reasoning_parser("deepseek_v3")
          → 動態 import vllm.reasoning.deepseek_v3_reasoning_parser
          → 回傳 DeepSeekV3ReasoningParser class (不是實例)
```

注意：`ReasoningConfig`（`vllm/config/reasoning.py`）是另一個 config class，它存的是 `reasoning_parser` 字串和 tokenize 後的 start/end token IDs，但**不直接控制 parser 選擇**。它在 `VllmConfig.__post_init__` 中被調用 `initialize_token_ids()` 來把 start/end strings 轉成 token IDs，供 engine 層的 structured output 等機制使用。

---

## 第二層：Request-level — enable_thinking / thinking

### chat_template_kwargs 傳遞路徑

```
Request body (OpenAI API)
  → ChatCompletionRequest (protocol.py)
    → build_chat_params() → ChatParams
      → .with_defaults(self.default_chat_template_kwargs)
        → chat_template_kwargs dict
```

三個來源：

1. **CLI default** `--default-chat-template-kwargs '{"enable_thinking": false}'`
2. **Request body** `chat_template_kwargs: {"enable_thinking": false}`
3. **Chat template 本身**（Jinja2 tokenizer_config.json 中的 `{% if enable_thinking %}...`）

優先級：request > CLI default > chat template default。

### DeepSeekV3ReasoningParser 的分派邏輯

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
            # → 解析 think...endthink → split reasoning vs content
        else:
            self._parser = IdentityReasoningParser(tokenizer, ...)
            # → 整個 output 當 content，不解析 reasoning
```

關鍵：`enable_thinking=False` 時，delegate 到 `IdentityReasoningParser`，它：
- `extract_reasoning()` → 回傳 `(None, full_output)` — 全部當 content
- `extract_reasoning_streaming()` → 每個 delta 直接當 content
- `is_reasoning_end()` → 永遠 True（跳過 reasoning state tracking）
- `reasoning_start_str` / `reasoning_end_str` → 都 None

### DeepSeekV3ReasoningWithThinkingParser

```python
# 用於 glm45、holo2 等預設 thinking 要開啟的模型
class DeepSeekV3ReasoningWithThinkingParser(DeepSeekV3ReasoningParser):
    def __init__(self, tokenizer, *args, **kwargs):
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        thinking = chat_kwargs.get("thinking", None)
        enable_thinking = chat_kwargs.get("enable_thinking", None)
        if thinking is None and enable_thinking is None:
            chat_kwargs["thinking"] = True    # 預設開
            chat_kwargs["enable_thinking"] = True
```

---

## 第三層：include_reasoning — 輸出層的 filter

API request 參數 `include_reasoning: bool = True`（OpenAI protocol 原生欄位）。

```python
# vllm/entrypoints/openai/chat_completion/serving.py
if not request.include_reasoning:
    reasoning_ended = True   # 告訴 engine：不用追蹤 reasoning state
```

這個 flag 不影響模型**產生** reasoning content，只影響 server 在 stream 時是否把 `reasoning_content` 從 delta 中分離出來。設 `False` 時，server 仍然收到含 ` thinking...` 的模型輸出，但 delta 中不產生 `reasoning` field，全部塞進 `content`。

---

## 第四層（新增）：Rust reasoning parser

最新 vLLM main 已將 reasoning parser 移植到 Rust，放在 `rust/src/reasoning-parser/`。

```
rust/src/reasoning-parser/src/
├── lib.rs          # ReasoningParser trait + delta type
├── deepseek_r1.rs  #  thinking... response
├── qwen3.rs        # (shared by deepseek_v3/v4, glm45, kimi_k2, minimax_m2, nemotron_v3, step3)
├── gemma4.rs       # special gemma4 format
├── kimi.rs         # original kimi format (not k2)
├── cohere_cmd.rs   # cohere command
├── delimited.rs    # generic delimited (think...endthink）
```

Rust parser 使用**基於 prompt token IDs 的初始化**，不依賴 model-family convention，同一 parser 可服務多個 model family。

---

## 完整資料流圖

```
                     Server-level                          Request-level
                     ============                          ==============

CLI --reasoning-parser=deepseek_v3        CLI --default-chat-template-kwargs
         |                                              '{"enable_thinking": false}'
         v                                              |
  StructuredOutputsConfig                       ChatCompletionRequest
    .reasoning_parser = "deepseek_v3"              .chat_template_kwargs
         |                                              |
         v                                              |
  ParserManager.get_reasoning_parser()            _effective_chat_template_kwargs()
         |                                              |
         |    +-----------+                             |
         +--->|   class   |<--- chat_template_kwargs ---+
              |   DeepSeek |  (combined request + default)
              |   V3       |
              +-----+-----+
                    |
       ┌────────────┼────────────┐
       v            v            v
   enable_thinking=True    enable_thinking=False
       |                        |
       v                        v
  DeepSeekR1              IdentityReasoningParser
  ReasoningParser            |
       |                     |
       v                     v
  extract_reasoning()    extract_reasoning()
  → (reasoning, content) → (None, full_output)
  → DeltaMessage {       → DeltaMessage {
      reasoning, content    content only
    }                     }
```

## 與 reasoning parser（第二題）的關係

Reasoning on/off 控制的是**輸出端解析** — 模型本身產生的 token 序列不受影響（模型總是輸出 ` thinking... response...`），差別只在 server 端是否把 ` thinking...` 部分從 API response 中分離出來。

真正的 reasoning content **產生**是由 chat template 控制（`enable_thinking` 也是 chat template kwarg！），如果 chat template 中 `{% if enable_thinking %}` 部分不插入 thinking prompt prefix，模型可能根本不會開始 reasoning。

---

**待確認**（需要看具體 chat template Jinja2 內容）：
- DeepSeek V3/V4 的 tokenizer_config.json 中 chat_template 如何處理 `enable_thinking`
- 當 `enable_thinking=false` 時，chat template 是否真的不插入 think prefix token
- 如果 chat template 仍然插了 think prefix token，那 model output 仍有 reasoning content，只是 server 層用 `IdentityReasoningParser` 全部當 content 輸出