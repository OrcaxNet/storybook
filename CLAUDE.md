# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An offline "coding memory" system (project name: **Storybook**) that ingests AI-coding session logs (Claude Code session logs, JSON files, or a built-in simulator), then runs a **"dream cycle"** that consolidates each session into a structured memory unit (a *Story*) and links Stories into a weighted association graph. Retrieval is vector-similarity plus edge-graph activation. All LLM/embedding work runs through a **local Ollama** instance — the system is fully offline.

Source comments, docstrings, and LLM prompts are bilingual Chinese/English.

## Environment & running

- Python **3.11+** (venv at `.venv/`). Dependencies: `click`, `requests`, `numpy`, `sqlite-vec`.
- **Ollama must be running** at `http://localhost:11434` (override with `OLLAMA_HOST`) with two models pulled:
  - LLM: `qwythos-hermes:latest` (override `STORYBOOK_LLM_MODEL`)
  - Embedding: `qwen3-embedding:0.6b`, **1024-dim** (override `STORYBOOK_EMBED_MODEL`). `EMBED_DIM` in config must match.
- The venv has no `pip` (created with `uv`). Install editable to get the `storybook` command: `VIRTUAL_ENV=$(pwd)/.venv uv pip install -e .` (re-run if the project dir moves and the `storybook` shebang goes stale). Without installing, run via `PYTHONPATH=src .venv/bin/python -m storybook.cli <command>`.

## Commands

```
storybook init                          # create SQLite schema + vec0 virtual table (also auto-run by other commands)
storybook import-data                   # default: collect Claude Code sessions from ~/.claude/projects (incremental, dedup by sessionId)
storybook import-data --claude          # same as above (explicit)
storybook import-data --sample [--n 100]   # generate & import simulated Claude Code sessions (no real sessions needed)
storybook import-data --cursor              # scan ~/Library/Application Support/Cursor/User/workspaceStorage (backup source)
storybook import-data <file|dir>            # import JSON (list, {sessions:[...]}, or {messages:[...]} chat-log shape)
storybook process [--session ID]        # "dream cycle": process all pending sessions (or one)
storybook search "<query>" [--top 3]    # vector search + related-story activation
storybook stats | storybook list | storybook show <id>
```

Note the command is **`import-data`**, not `import` (click auto-hyphenates the `import_data` function). With no flags/path it defaults to `--claude`. The `--claude`, `--sample`, `--cursor`, and `<path>` forms are mutually exclusive.

There is **no test suite**. `test_logs/*.json` and `hermes_sessions.json` are sample data sources for `import-data`.

## Architecture

Module flow (all under `src/storybook/`): `collector` → `store` → `processor` (uses `llm` + `embeddings`) → `search`. `cli.py` wires commands; `config.py` holds all paths, model names, and thresholds.

### Storage layer (`store.py`) — SQLite + sqlite-vec
Single file at `data/memory.db`. Three tables (`sessions`, `stories`, `edges`) plus the **`story_vectors` vec0 virtual table**. Each `get_db()` call opens a fresh connection (WAL mode, foreign keys on) and loads the sqlite-vec extension.

**Embeddings are stored in two places and must stay in sync:** as a float32 BLOB in `stories.embedding` *and* as a row in the `story_vectors` vec0 table. `add_story`/`update_story` write both; if you add any other path that mutates an embedding, update both. `update_story` deletes-then-reinserts the vec0 row (vec0 rows are immutable).

`search_by_vector` queries the vec0 table by L2 distance and converts to cosine similarity as `1 - distance²/2` (exact for L2-normalized vectors; `embeddings.embed` L2-normalizes on write). A numpy brute-force cosine fallback (`search_by_vector_numpy`) exists but is **not wired into the active code paths** — `processor` and `search` call `search_by_vector` directly.

### The dream cycle (`processor.process_session`)
For each pending session: LLM extracts keywords → embed `keywords + problem_desc` (not raw_content, for focus) → vector-search top-K existing stories → branch on best similarity:

| Branch | Trigger | Action |
|--------|---------|--------|
| **create** | best sim < 0.85 | LLM summarizes to ≤400-char "问题/步骤/结果" story; link to low-match (0.75–0.85) stories with `weight = sim` |
| **merge** | 0.85 ≤ sim < 0.92 | LLM merges old+new content; if result >400 chars or LLM says `SPLIT:YES`, split into child stories (`parent_id`, parent-child edge weight 1.0, sibling edges 0.5). On split the parent's vector is dropped from the index (`delete_story_vector`) so it no longer matches search; the `stories` row is kept for lineage. |
| **update** | sim ≥ 0.92 | merge keywords only, re-embed, strengthen existing edge weights (+0.1, capped 1.0) |

Thresholds live in `config.py`, including `SIM_THRESHOLD_UPDATE_ONLY` (0.92, the merge-vs-update-only boundary). Story text is capped at `STORY_MAX_CHARS` (400).

### Retrieval (`search.search`)
Embed query (keywords + query) → vec0 top-K (fetches `top_k*2`, filters by `SIM_THRESHOLD_SEARCH=0.50`) → for each hit, surface related stories via `edges` (weight desc) and bump edge weights between co-retrieved stories. Hits also increment `stories.access_count`.

### Edge graph
`edges` table, `UNIQUE(source_id, target_id)`. Types: `semantic`, `parent_child` (1.0, fixed), `sibling` (0.5). Edges are **undirected**: `add_or_update_edge`/`increment_edge_weight` normalize endpoint order to `(min_id, max_id)` via `_edge_pair`, so call direction doesn't matter and `(A,B)`/`(B,A)` can't create duplicate rows. `add_or_update_edge` takes `max(existing, new)`; `increment_edge_weight` adds a delta capped at `WEIGHT_MAX` (1.0). Queries (`get_related_stories`) match either endpoint.

## Configuration knobs (`config.py`)

- Paths: `DB_PATH` (`data/memory.db`), `LOG_DIR`, `CLAUDE_PROJECTS_PATH` (`~/.claude/projects`, primary source), `CURSOR_STORAGE_PATH` (backup).
- Thresholds: `SIM_THRESHOLD_HIGH` (0.85), `SIM_THRESHOLD_UPDATE_ONLY` (0.92), `SIM_THRESHOLD_LOW` (0.75), `SIM_THRESHOLD_SEARCH` (0.50), `TOP_K_RETRIEVAL` (5), `TOP_K_SEARCH` (3), `STORY_MAX_CHARS` (400).
- Weight rules: `WEIGHT_INCREMENT` (0.1), `WEIGHT_MAX` (1.0), `WEIGHT_PARENT_CHILD` (1.0).
- LLM call options (temp 0.3, `num_ctx` 8192, 120s timeout) are hardcoded in `llm._chat`/`_generate`, except `think` which follows `config.LLM_THINK` (`STORYBOOK_LLM_THINK` env, default **off**). `qwythos-hermes` is Qwen3-arch with a thinking mode that makes extraction calls ~9× slower; thinking is unnecessary for keyword/summary/split tasks, so it's off by default. Set `STORYBOOK_LLM_THINK=1` only if retrieval accuracy drops.

## Notes

- Not a git repository.
- `docs/TECH_DESIGN.md` is the original design doc; its directory layout and `storybook import` examples predate the implementation (command is `import-data`, no `tests/` or `scripts/` dirs exist, no launchd plist is set up).
- LLM output parsing is tolerant: it slices between `[`/`]` for keyword JSON and splits on `TITLE:`/`CONTENT:` markers, with string-split fallbacks when the model doesn't follow the format.
