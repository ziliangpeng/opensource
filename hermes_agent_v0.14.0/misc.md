# Misc — random questions and observations

## Oneshot mode (`-z`) — tool use and safety

### Can the agent still use tools in oneshot mode?

Yes. Oneshot mode (`hermes -z "..."`) loads the same `enabled_toolsets` as interactive mode. The agent can call all registered tools: web search, file read/write, terminal commands, etc.

The only difference is **no user clarification** — if the agent needs to ask a question (e.g. "which directory?"), the `clarify_callback` returns a synthetic "pick a default and continue" instruction instead of stopping.

### What about YOLO mode?

`HERMES_YOLO_MODE=1` is set in oneshot mode, which bypasses dangerous-command approval prompts. This is necessary because there's no user sitting at a terminal to approve or deny.

### How dangerous is this?

The risk profile:

| Factor | Risk level | Reason |
|--------|-----------|--------|
| User writes the prompt | Low | Oneshot is an explicit, one-off command. You're responsible for what you type. |
| LLM makes unexpected decisions | Moderate | The model might do something you didn't intend, especially with vague prompts. |
| Cron job + oneshot | Higher | A recurring job with a buggy prompt, or a model behavior change, could cause damage unattended. |
| Compared to raw shell | Same | Shell has no approval prompts either. Oneshot just matches shell-level trust. |

The real difference: **shell runs exactly what you type; LLM runs what it interprets from what you type.** Hallucination, misinterpretation, or prompt injection are the real risk vectors — not the YOLO flag itself.

Related: the fork's security patches (`file-safety`, control-plane write-deny) address one aspect of this — protecting Hermes' own configuration files from prompt-injection attacks.
