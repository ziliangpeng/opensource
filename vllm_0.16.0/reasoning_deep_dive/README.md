# Reasoning Deep Dive — vLLM 0.16

> vLLM upstream main, May 2026. Notes from code reading.

## Contents

1. [Reasoning On/Off Control](01_reasoning_on_off_control.md) — `enable_thinking`、`--reasoning-parser`、`include_reasoning` 三層控制
2. [DeepSeek R1 Parser Walkthrough](02_deepseek_r1_parser.md) — `BaseThinkingReasoningParser` streaming 狀態機
3. [Gemma4 Parser](03_gemma4_parser.md) — Nested thinking、`<|channel>` boundary、`thought\n` prefix stripping
4. [MiniMax M2 Parser](04_minimax_m2_parser.md) — 無 start token 的特殊情況
5. [Future Roadmap](05_roadmap.md) — 待涵蓋內容總覽