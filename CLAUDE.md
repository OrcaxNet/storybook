# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An offline "coding memory" system (project name: **Storybook**) that ingests AI-coding session logs (Claude Code session logs, JSON files, or a built-in simulator), then runs a **"dream cycle"** that consolidates each session into a structured memory unit (a *Story*) and links Stories into a weighted association graph. Retrieval is vector-similarity plus edge-graph activation. All LLM/embedding work runs through a **local Ollama** instance — the system is fully offline.

Source comments, docstrings, and LLM prompts are bilingual Chinese/English.

## Environment & running

- Python **3.11+** (venv at `.venv/`). Dependencies: `click`, `requests`, `numpy`, `sqlite-vec`, `mcp`.
- **Ollama must be running** at `http://localhost:11434` (override with `OLLAMA_HOST`) with two models pulled:
  - LLM: `qwythos-hermes:latest` (override `STORYBOOK_LLM_MODEL`)
  - Embedding: `qwen3-embedding:0.6b`, **1024-dim** (override `STORYBOOK_EMBED_MODEL`). `EMBED_DIM` in config must match.
- `config.py` auto-loads a project-root `.env` at import (no error if absent; copy `.env.example`). Pre-existing env vars / command-line `VAR=val` take priority over `.env` (`.env` never overwrites them).
- The venv has no `pip` (created with `uv`). Install editable to get the `storybook` command: `VIRTUAL_ENV=$(pwd)/.venv uv pip install -e .` (re-run if the project dir moves and the `storybook` shebang goes stale). Without installing, run via `PYTHONPATH=src .venv/bin/python -m storybook.cli <command>`.

## Commands

```
storybook init                          # create SQLite schema + vec0 virtual table (also auto-run by other commands)
storybook setup [--yes|--dry-run|--json]  # user Profile + detected Claude/Cursor/Codex adapters + smoke tests
storybook uninstall [--purge-data]      # restore managed config nodes; keep memory by default
storybook profile show|list             # inspect the user-level Profile registry
storybook profile create NAME           # create an isolated Profile
storybook profile switch ID_OR_NAME     # switch the active Profile
storybook migration discover            # find project-level v1 databases read-only
storybook migration run PATH --dry-run  # zero-write migration plan
storybook migration run PATH            # backup, convert, verify, atomic cut-over
storybook migration rollback ID         # atomically point back to a v1 rollback copy
storybook sync status                   # v0.2: explicit local_only status
storybook doctor [--fix]                # env health check (Ollama/models/dim/sqlite-vec/vector double-write); --fix repairs double-write inconsistency
storybook import-data                   # default: collect Claude Code sessions from ~/.claude/projects (incremental, dedup by sessionId)
storybook import-data --claude          # same as above (explicit)
storybook import-data --sample [--n 100]   # generate & import simulated Claude Code sessions (no real sessions needed)
storybook import-data --cursor              # scan ~/Library/Application Support/Cursor/User/workspaceStorage (backup source)
storybook import-data <file|dir>            # import JSON (list, {sessions:[...]}, or {messages:[...]} chat-log shape)
storybook process [--session ID]        # "dream cycle": process all pending sessions (or one)
storybook search "<query>" [--top 3]    # vector search + related-story activation
storybook status --performance          # recent query p50/p95 + cache/fallback ratios
storybook benchmark --model-state warm  # isolated 10k Story performance/quality baseline
storybook stats | storybook list | storybook show <id>
```

Note the command is **`import-data`**, not `import` (click auto-hyphenates the `import_data` function). With no flags/path it defaults to `--claude`. The `--claude`, `--sample`, `--cursor`, and `<path>` forms are mutually exclusive.

The pytest suite lives in `tests/` and mocks Ollama by default. `test_logs/*.json` and `hermes_sessions.json` are sample data sources for `import-data`.

## Architecture

Module flow (all under `src/storybook/`): `collector` → `store` → `processor` (uses `llm` + `embeddings`) → `search`. `context.py` owns ContextEnvelope capture, privacy normalization and environment-fit scoring; `cli.py` wires commands; `config.py` holds all paths, model names, and thresholds; `health.py` powers `storybook doctor` (env + vector double-write consistency self-check, reads via `store`). `setup_manager.py` orchestrates one-click setup/uninstall; `setup_adapters/` owns plugin-registered, node-scoped Claude Code/Cursor/Codex config merges and rollback.

### Storage layer (`store.py`) — SQLite + sqlite-vec
Each random-UUID user Profile owns a database generation under `profiles/<profile_id>/`; the default is `db/memory.db`, while safe migrations use `migrations/<migration_id>/v2.db`. The registry stores only that Profile-relative `database_ref`, and switches it atomically after conversion validation. `profiles.py` is the sole registry/path resolver used by CLI, collectors, hooks and MCP; repository paths are not memory boundaries. Three tables (`sessions`, `stories`, `edges`) plus the **`story_vectors` vec0 virtual table** carry path-independent `global_id`, `profile_id`, and `sync_state=local_only`. Each `get_db()` call opens a fresh connection (WAL mode, foreign keys on) and loads the sqlite-vec extension.

`migration.py` inspects and backs up v1 sources read-only, keeps the consistent v1 backup for at least 30 days, upgrades a private copy, compares all Session/Story/edge counts and relationships, verifies `legacy_raw`, integrity/FKs and a real sqlite-vec lookup, then changes the registry pointer. Immediately before cut-over it holds a SQLite `BEGIN IMMEDIATE` guard (or a stable read transaction for an OS-read-only source), recomputes the logical source hash, and stages durable deny-write triggers on every writable retiring generation. It commits every retiring fence and target unfence before the registry compare-and-swap, so no fallible SQLite commit remains after the pointer is durable; ambiguous prepare commits are compensated to their pre-switch states while the registry still names the old authority. Stale connections fail with `SB_MIGRATION_GENERATION_FENCED`. A post-backup commit or active writer aborts before pointer change. Migration IDs are deterministic from Profile ID + logical source hash. Rollback fences the active v2 generation with the same recoverable protocol, creates an independent v1 copy, records its authority baseline hash, and refuses to reuse a stale v2 generation after that copy changes.

**Embeddings are stored in two places and must stay in sync:** as a float32 BLOB in `stories.embedding` *and* as a row in the `story_vectors` vec0 table. `add_story`/`update_story` write both; if you add any other path that mutates an embedding, update both. `update_story` deletes-then-reinserts the vec0 row (vec0 rows are immutable).

`search_by_vector` queries the vec0 table by L2 distance and converts to cosine similarity as `1 - distance²/2` (exact for L2-normalized vectors; `embeddings.embed` L2-normalizes on write). A numpy brute-force cosine fallback (`search_by_vector_numpy`) exists but is **not wired into the active code paths** — `processor` and `search` call `search_by_vector` directly.

### The dream cycle (`processor.process_session`)
For each pending session: LLM extracts keywords → embed `keywords + problem_desc` (not raw_content, for focus) → vector-search top-K existing stories → branch on best similarity:

| Branch | Trigger | Action |
|--------|---------|--------|
| **create** | best sim < 0.85 | Persist Story v2 title/abstract/structured detail/sources without hard-truncating detail; link low matches with `weight = sim` |
| **merge** | 0.85 ≤ sim < 0.92 | Merge structured evidence; split only for independent reusable conclusions/applicability boundaries, never character count. Parent revision remains auditable. |
| **update** | sim ≥ 0.92 | merge keywords only, re-embed, strengthen existing edge weights (+0.1, capped 1.0) |

Thresholds live in `config.py`, including `SIM_THRESHOLD_UPDATE_ONLY` (0.92). Only the abstract/recall presentation has a budget; persisted detail and sources are lossless.

### Retrieval (`search.search`)
Embed the raw query → vec0 top-K (expanded candidate lane, filtered by `SIM_THRESHOLD_SEARCH=0.50`) → use direct hits as seeds for bounded typed-graph expansion → deduplicate, suppress superseded Stories, apply environment policy, and rank. `graph_enabled=False` preserves a direct-only fallback. Graph candidates expose `seed_story_id`, full `graph_path` (including provenance/version), and `score_components`; independent hop/path/fan-out/time/token budgets return partial results with top-level `truncated=true`.

Every query returns and locally records a privacy-safe latency breakdown:
`cache/embed/vector/graph/rerank/serialize/total`. Diagnostics are written to
`query_performance.jsonl` in the active Profile's log directory, with a strict
metadata-only schema (never raw query,
Story content, absolute paths, hostnames, or repository URLs). `storybook benchmark`
uses an isolated fixed-seed dataset and reports performance together with recall/MRR.

When a caller supplies a ContextEnvelope, semantic similarity remains the primary
signal and environment fit contributes a bounded secondary score
(`ENVIRONMENT_SCORE_WEIGHT`). Default `scope="profile"` keeps cross-environment
results and adds explainable warnings; only explicit `scope="strict"` filters
environment conflicts. Story environment history is a collection derived from all
source Sessions, never a last-write-wins field.

### Memory Graph
`edges` uses `UNIQUE(source_id, target_id, edge_type)` and supports `semantic`, `temporal`, `causal`, `same_environment`, `parent_child`, `co_recall`, and `supersedes`. Temporal (old→new), causal (cause→effect), parent-child (parent→child), and supersedes (new→old) are directed; the rest canonicalize undirected endpoints. Every edge carries JSON provenance, version, observations, reinforcement timestamps, and a soft-delete timestamp. Legacy `sibling` rows remain readable, while new split siblings are standard semantic edges with `relationship=sibling` provenance. Recall feedback is queued and non-blocking, creates/reinforces capped `co_recall` edges, and supports half-life decay.

## Configuration knobs (`config.py`)

- Paths: `DB_PATH`/`INDEX_DIR`/`CACHE_DIR`/`LOG_DIR` resolve from the active user Profile; `PERFORMANCE_LOG_PATH` follows that Profile's `LOG_DIR`; `CLAUDE_PROJECTS_PATH` (`~/.claude/projects`, primary source), `CURSOR_STORAGE_PATH` (backup).
- Thresholds/budgets: `SIM_THRESHOLD_HIGH` (0.85), `SIM_THRESHOLD_UPDATE_ONLY` (0.92), `SIM_THRESHOLD_LOW` (0.75), `SIM_THRESHOLD_SEARCH` (0.50), `TOP_K_RETRIEVAL` (5), `TOP_K_SEARCH` (3), `STORY_ABSTRACT_MAX_CHARS` (600), plus Graph RAG hop/path/fan-out/time/token budgets in `GRAPH_*`.
- Weight rules: `WEIGHT_INCREMENT` (0.1), `WEIGHT_MAX` (1.0), `WEIGHT_PARENT_CHILD` (1.0).
- LLM call options (temp 0.3, `num_ctx` 8192, 120s timeout) are hardcoded in `llm._chat`/`_generate`, except `think` which follows `config.LLM_THINK` (`STORYBOOK_LLM_THINK` env, default **off**). `qwythos-hermes` is Qwen3-arch with a thinking mode that makes extraction calls ~9× slower; thinking is unnecessary for keyword/summary/split tasks, so it's off by default. Set `STORYBOOK_LLM_THINK=1` only if retrieval accuracy drops.

## Notes

- This is a git repository; preserve unrelated worktree changes.
- `docs/TECH_DESIGN.md` is the original design doc; some directory layout and `storybook import` examples predate the implementation (the command is now `import-data`).
- LLM output parsing is tolerant: it slices between `[`/`]` for keyword JSON and splits on `TITLE:`/`CONTENT:` markers, with string-split fallbacks when the model doesn't follow the format.
