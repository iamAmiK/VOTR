# Router Policy 

## Request Format
Use this payload:

```json
{
  "server_intent": "string",
  "tool_intent": "string",
  "session_id": "string",
  "skip_session_filter": false,
  "record_session": true
}
```

## Input Rules
- `server_intent`: name app/domain if known (e.g., `GitHub repository operations`).
- `tool_intent`: `verb + object + key constraint` (e.g., `list open pull requests`).
- Keep short: `server_intent` 4-10 words, `tool_intent` 6-16 words.
- Multi-step user asks: split into separate `/route` calls (one verb+object per call).
- No extra LLM rewrite step.
- Create a random constant session_id for this conversation/thread

## Allowed Cleanup
- typos: `githb` -> `github`
- shorthand: `gh` -> `github`, `prs` -> `pull requests`
Do not invent missing facts (IDs, dates, repo names, server names).
