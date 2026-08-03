# Agent History Adapter compatibility

Storybook reads only local files available to the current OS user. Adapters do
not upload transcripts. External session IDs and file identities are converted
to Profile-local HMAC values before persistence; absolute paths are normalized
into ContextEnvelope aliases/fingerprints.

`supported` means the repository contains a versioned parser, a redacted schema
fixture, incremental/idempotency tests, and a stable failure mode for unknown
schemas. `experimental` means detection evidence exists but the local contract
is not stable enough to import automatically.

| Agent | Platforms / default local evidence | Schema / workspace / cursor | Privacy-sensitive fields | Fixture/version evidence | Status |
|---|---|---|---|---|---|
| Claude Code | macOS/Linux `~/.claude/projects/*/*.jsonl` | JSONL `user`/`assistant`; cwd metadata; complete-record fingerprint | cwd, session ID, tool payloads | Existing collector fixtures and regression suite | supported |
| Codex CLI/app | macOS/Linux `$CODEX_HOME/sessions/**/*.jsonl` (`CODEX_HOME` defaults to `~/.codex`) | `session_meta`, `response_item`, `event_msg`; cwd; complete-line cursor + SHA-256 | auth/tool arguments, cwd, external session ID | Codex CLI 0.145 local schema probe; `tests/test_history_adapters.py` | supported |
| Cursor | macOS workspaceStorage `*/state.vscdb` | read-only `ItemTable`; workspace metadata; DB fingerprint | workspace URI, opaque cache IDs | Existing Cursor fixtures migrated to adapter contract | supported |
| Gemini CLI | macOS/Linux `~/.gemini/tmp/*/chats/session-*.json` | 0.38+ `sessionId` + `messages`; project-scoped; file fingerprint | session ID, project hash, tool content | [official session management docs](https://github.com/google-gemini/gemini-cli/blob/main/docs/reference/commands.md); redacted fixture test | supported |
| Cline | VS Code/VSCodium globalStorage `saoudrizwan.claude-dev/tasks/*/api_conversation_history.json` | MessageParam array; task ID; file fingerprint | environment details, tool blocks, provider secrets | [official Cline repository](https://github.com/cline/cline); redacted fixture test | supported |
| Roo Code | VS Code globalStorage `rooveterinaryinc.roo-cline/tasks` | Cline-derived UI/API history, but release-specific drift remains | environment details, tool/provider payloads | Path and family evidence evaluated; no frozen current fixture | experimental |
| OpenCode | OS data directory, current releases use SQLite | normalized session/message/part tables; migrations vary by release | provider/auth state adjacent to sessions | Stable cross-version export contract not verified | experimental |
| GitHub Copilot CLI/Agent | CLI and IDE surfaces differ | no single verified cross-surface local transcript contract | account/repo telemetry may be adjacent | Public local schema insufficient | unsupported |
| Windsurf | editor-owned local state | undocumented and release-sensitive | workspace/account state | No stable, legal, versioned local fixture | unsupported |
| Continue | editor/global state has changed across releases | no current cross-editor stable transcript contract | model/provider config adjacent | Current schema not frozen | experimental |

## Adapter contract

Every adapter exposes `detect`, `discover`, `parse`, `cursor`, `fingerprint`,
`context`, and `diagnostics`. The ingestion manager owns source enablement,
checkpoint persistence, HMAC file identity, upsert, structured summaries, and
failure isolation.

For JSONL, only newline-terminated records are consumed. A truncated tail stays
eligible for the next scan. Corrupt complete records increment `invalid` without
blocking valid records or other sources. Unknown schemas return stable
`SB_SOURCE_*` codes and are never guessed into Sessions.
