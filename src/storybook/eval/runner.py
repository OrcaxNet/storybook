"""评测编排：建语料 / 跑查询 / 跑加工分支 / 跑分裂，产出可序列化报告。

设计要点：
  * **隔离 DB** -- 每次评测把 ``config.DB_PATH`` 重定向到临时文件，绝不污染用户 Profile 数据库。
  * **embedding 经 ``embeddings.embed`` 模块属性** -- 真实运行走 Ollama；测试用 fake_embedder 夹具
    monkeypatch 该属性即可注入确定性桩（processor 内部也走同一属性，故加工/分裂评测同样受控）。
  * **加工/分裂评测用确定性 CuratedLLM** -- 用 benchmark 里人工标注的 keywords/summary 替代 LLM 输出，
    使分支决策只由「真实 embedding 相似度 vs 阈值」决定，从而隔离度量 0.85/0.92 阈值是否合理
    （这正是 issue 的核心疑问）。检索评测不用 LLM（story 内容为人工标注）。
  * **阈值敏感性曲线** -- 复用同一批 embedding，仅改阈值重过滤/重分类，零额外 Ollama 调用。
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import re
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .. import adaptive, config, store, embeddings, llm as llm_mod, processor
from .. import search as search_mod, story_v2
from . import benchmark as bm
from . import metrics as M

logger = logging.getLogger(__name__)

KS = (1, 3, 5)
# 阈值敏感性扫描点
SEARCH_THRESHOLD_SWEEP = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
HIGH_THRESHOLD_SWEEP = [0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.95]
ABLATION_MODES = ("legacy", "default", "full", "multi_vector")
RETRIEVAL_STRATEGIES = (
    "direct_vector",
    "hybrid",
    "hybrid_graph",
    "hybrid_graph_reranker",
    "hybrid_graph_rewrite",
    "hybrid_graph_hyde",
    "hybrid_graph_hyde_reranker",
)
TRANSFORM_EVIDENCE_SOURCES = frozenset({
    "live_generated",
    "query_only_pre_generated",
    "oracle_upper_bound",
})
EXACT_TERM_CASES = (
    "SQLITE_BUSY",
    "ERR_MODULE_NOT_FOUND",
    "ECONNRESET",
    "ORA-00060",
    "E11000",
    "SIGSEGV",
    "HTTP_429",
    "OOMKILLED",
)
EXACT_TERM_CORPUS_CONTRACT = (
    "shared compact semantic vector; exact token only in FTS-indexed "
    "content/keywords"
)


# ═══════════════════════════════════════════════
#  隔离 DB + LLM/embed 注入
# ═══════════════════════════════════════════════

@contextlib.contextmanager
def _isolated_db(db_path: Optional[Path] = None):
    """把 config.DB_PATH 重定向到临时库，结束后恢复。db_path=None 则用 tempfile。"""
    saved = config.DB_PATH
    keep = db_path is not None
    tmp = None
    if db_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        db_path = Path(tmp.name)
    config.DB_PATH = Path(db_path)
    try:
        # 若使用既有文件路径，先清掉旧库内容，保证干净
        if keep and Path(db_path).exists():
            Path(db_path).unlink()
        store.init_db()
        yield config.DB_PATH
    finally:
        config.DB_PATH = saved
        if tmp is not None:
            with contextlib.suppress(FileNotFoundError):
                Path(tmp.name).unlink(missing_ok=True)


@contextlib.contextmanager
def _patch_llm(extract=None, summarize=None, merge=None, judge=None, split=None):
    """临时把 storybook.llm 模块的对外函数替换为给定桩，结束后恢复。"""
    saved = {
        "extract_keywords": llm_mod.extract_keywords,
        "summarize_session": llm_mod.summarize_session,
        "form_stories": llm_mod.form_stories,
        "merge_stories": llm_mod.merge_stories,
        "judge_split": llm_mod.judge_split,
        "split_story": llm_mod.split_story,
    }
    if extract is not None:
        llm_mod.extract_keywords = extract
    if summarize is not None:
        llm_mod.summarize_session = summarize
        llm_mod.form_stories = lambda content: [summarize(content)]
    if merge is not None:
        llm_mod.merge_stories = merge
    if judge is not None:
        llm_mod.judge_split = judge
    if split is not None:
        llm_mod.split_story = split
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(llm_mod, k, v)


class CuratedLLM:
    """确定性 LLM 桩：按 session 逐条注入人工 keywords/summary/split 决策。

    processor 调用顺序：extract_keywords(raw) -> summarize(raw) -> (merge/judge/split 仅 merge 分支)。
    评测前用 ``configure(spec, ...)`` 设好当前会话的期望输出。
    """

    def __init__(self):
        self.keywords: list[str] = []
        self.summary: dict = {"title": "", "content": ""}
        self.merged: dict = {"title": "", "content": ""}
        self.should_split: bool = False
        self.sub_stories: list[dict] = []
        self.calls: dict[str, int] = {}

    def _tick(self, name: str):
        self.calls[name] = self.calls.get(name, 0) + 1

    def configure(self, spec: bm.SessionSpec, *, merged: dict = None,
                  should_split: bool = False, sub_stories: list[dict] = None):
        self.keywords = list(spec.keywords)
        self.summary = dict(spec.summary)
        self.merged = dict(merged or spec.summary)
        self.should_split = should_split
        self.sub_stories = list(sub_stories or [])

    def extract_keywords(self, text):
        self._tick("extract_keywords")
        return list(self.keywords)

    def summarize_session(self, content):
        self._tick("summarize_session")
        return dict(self.summary)

    def merge_stories(self, old, new):
        self._tick("merge_stories")
        return dict(self.merged)

    def judge_split(self, merged_text):
        self._tick("judge_split")
        return bool(self.should_split)

    def split_story(self, merged_text):
        self._tick("split_story")
        return [dict(s) for s in self.sub_stories]


# ═══════════════════════════════════════════════
#  检索评测
# ═══════════════════════════════════════════════

def run_retrieval_eval(
    db_path: Optional[Path] = None,
    ks: tuple = KS,
    sweep_thresholds: list[float] = None,
    benchmark_path: Path | str = None,
) -> dict:
    """构建人工语料（真实 embedding 索引），跑全部查询变体，算 recall@k/precision@k/MRR。

    返回 ``{summary, per_variant, per_query, negatives, threshold_sweep, corpus_size, ...}``。
    """
    sweep_thresholds = sweep_thresholds if sweep_thresholds is not None else SEARCH_THRESHOLD_SWEEP
    bench = bm.load_benchmark(benchmark_path)
    with _isolated_db(db_path):
        # ── 建语料：每条 topic 一条 story，索引向量 = embed(keywords + problem_desc) ──
        topic_to_sid: dict[str, int] = {}
        embed_failures = 0
        for t in bench.topics:
            vec = embeddings.embed(t.index_text())
            if not vec:
                embed_failures += 1
                logger.warning("topic %s embedding 失败，跳过", t.id)
                continue
            sid = store.add_story(
                title=t.title, content=t.content, keywords=t.keywords,
                embedding=vec, source_session_ids=[],
            )
            topic_to_sid[t.id] = sid
        corpus_size = len(topic_to_sid)

        # ── 跑查询：取完整排序（top=语料规模），阈值过滤在指标侧做 ──
        per_query: list[dict] = []
        # 缓存 (query -> ranked [(sid, sim)]) 供阈值扫描复用，避免重复 embed
        query_rankings: list[dict] = []
        for pair in bench.query_pairs:
            target_sid = topic_to_sid.get(pair["topic_id"])
            qvec = embeddings.embed(pair["query"])
            if not qvec:
                embed_failures += 1
                continue
            ranked = store.search_by_vector(qvec, top_k=corpus_size)
            ranked_ids = [r["story_id"] for r in ranked]
            ranked_sims = {r["story_id"]: r["similarity"] for r in ranked}
            target_sim = ranked_sims.get(target_sid)
            # 基线阈值 = config.SIM_THRESHOLD_SEARCH（与 search.search 一致）
            th = config.SIM_THRESHOLD_SEARCH
            filtered_ids = [r["story_id"] for r in ranked if r["similarity"] >= th]
            row = {
                "query": pair["query"], "variant": pair["variant"],
                "topic_id": pair["topic_id"], "target_story_id": target_sid,
                "target_similarity": round(target_sim, 4) if target_sim is not None else None,
                "ranked_ids": ranked_ids,
                "filtered_ids": filtered_ids,
            }
            for k in ks:
                row[f"recall@{k}"] = M.recall_at_k(filtered_ids, [target_sid], k)
                row[f"precision@{k}"] = M.precision_at_k(filtered_ids, [target_sid], k)
            row["mrr"] = M.mrr(filtered_ids, [target_sid])
            per_query.append(row)
            query_rankings.append({"target_sid": target_sid, "ranked": ranked})

        # ── 负例：应无强匹配（sim < SEARCH 阈值） ──
        neg_rows = []
        for q in bench.negatives:
            qvec = embeddings.embed(q)
            if not qvec:
                continue
            ranked = store.search_by_vector(qvec, top_k=corpus_size)
            top_sim = ranked[0]["similarity"] if ranked else 0.0
            neg_rows.append({
                "query": q, "top_similarity": round(top_sim, 4),
                "rejected": top_sim < config.SIM_THRESHOLD_SEARCH,
            })
        specificity = (
            sum(1 for r in neg_rows if r["rejected"]) / len(neg_rows) if neg_rows else 0.0
        )

    # ── 汇总 ──
    def agg(rows, key):
        return round(sum(r[key] for r in rows) / len(rows), 4) if rows else 0.0

    summary = {}
    for k in ks:
        summary[f"recall@{k}"] = agg(per_query, f"recall@{k}")
        summary[f"precision@{k}"] = agg(per_query, f"precision@{k}")
    summary["mrr"] = agg(per_query, "mrr")
    summary["query_count"] = len(per_query)
    summary["corpus_size"] = corpus_size
    summary["embed_failures"] = embed_failures
    summary["specificity"] = round(specificity, 4)
    summary["target_threshold"] = config.SIM_THRESHOLD_SEARCH

    # 按变体分组
    per_variant: dict[str, dict] = {}
    for v in ("exact", "synonym", "cross_lang"):
        vrows = [r for r in per_query if r["variant"] == v]
        vd = {"count": len(vrows)}
        for k in ks:
            vd[f"recall@{k}"] = agg(vrows, f"recall@{k}")
            vd[f"precision@{k}"] = agg(vrows, f"precision@{k}")
        vd["mrr"] = agg(vrows, "mrr")
        per_variant[v] = vd

    # ── 阈值敏感性曲线：复用排名，仅改阈值重过滤 ──
    def sweep_fn(th):
        recalls = {f"recall@{k}": 0.0 for k in ks}
        n = 0
        for qr in query_rankings:
            target = qr["target_sid"]
            if target is None:
                continue
            filt = [r["story_id"] for r in qr["ranked"] if r["similarity"] >= th]
            for k in ks:
                recalls[f"recall@{k}"] += M.recall_at_k(filt, [target], k)
            n += 1
        return {kk: round(vv / n, 4) for kk, vv in recalls.items()} if n else recalls

    curve = M.threshold_sweep(sweep_fn, sweep_thresholds, metric_key="recall@3")

    # 达标判断：recall@3 >= 0.70（对应 PRD「重复 bug 检索准确率≥70%」+ MVP 验收 Top3）
    passes_70 = summary["recall@3"] >= 0.70

    return {
        "summary": summary,
        "per_variant": per_variant,
        "per_query": per_query,
        "negatives": neg_rows,
        "threshold_sweep": curve,
        "passes_70_percent_recall_at_3": passes_70,
    }


def run_exact_term_hybrid_ablation(
    db_path: Optional[Path] = None,
    *,
    top_k: int = 3,
) -> dict:
    """Measure exact-code-token recall for vector-only versus hybrid search.

    Every Story deliberately shares the same semantic title/abstract vector;
    its distinguishing error token lives in ``content`` and ``keywords``.
    This models a real Story v2 boundary: the default compact embedding omits
    full detail, while FTS indexes it.  Ground truth is the Story containing
    the queried token, and the isolated database is discarded after the run.
    """

    top_k = max(1, int(top_k))
    with _isolated_db(db_path):
        shared_vector = embeddings.embed(
            "generic incident recovery with a verified remediation"
        )
        if not shared_vector:
            return {
                "query_count": 0,
                "embed_failures": 1,
                "top_k": top_k,
                "vector_recall_at_k": 0.0,
                "hybrid_recall_at_k": 0.0,
                "absolute_gain": 0.0,
                "passes_improvement_gate": False,
                "per_query": [],
                "corpus_contract": EXACT_TERM_CORPUS_CONTRACT,
            }
        targets = {}
        for token in EXACT_TERM_CASES:
            targets[token] = store.add_story(
                title="Incident recovery note",
                abstract="A verified remediation for a production incident.",
                content=f"Observed exact diagnostic token {token}.",
                keywords=[token],
                embedding=shared_vector,
                source_session_ids=[],
            )

        rows = []
        embed_failures = 0
        candidate_limit = len(targets)
        for token, target_story_id in targets.items():
            query_vector = embeddings.embed(token)
            if not query_vector:
                embed_failures += 1
                continue
            vector_rows = [
                item for item in store.search_by_vector(
                    query_vector, top_k=candidate_limit
                )
                if item["similarity"] >= config.SIM_THRESHOLD_SEARCH
            ]
            lexical_rows = store.search_by_lexical(
                token,
                top_k=candidate_limit,
                timeout_seconds=max(0.5, config.QUERY_FALLBACK_TIMEOUT_SECONDS),
            )
            hybrid_rows = adaptive.fuse_rankings(
                vector_rows, lexical_rows, limit=candidate_limit
            )
            vector_ids = [item["story_id"] for item in vector_rows]
            hybrid_ids = [item["story_id"] for item in hybrid_rows]
            rows.append({
                "query": token,
                "target_story_id": target_story_id,
                "vector_recall": M.recall_at_k(
                    vector_ids, [target_story_id], top_k
                ),
                "hybrid_recall": M.recall_at_k(
                    hybrid_ids, [target_story_id], top_k
                ),
            })

    count = len(rows)
    vector_recall = (
        sum(row["vector_recall"] for row in rows) / count if count else 0.0
    )
    hybrid_recall = (
        sum(row["hybrid_recall"] for row in rows) / count if count else 0.0
    )
    return {
        "query_count": count,
        "embed_failures": embed_failures,
        "top_k": top_k,
        "vector_recall_at_k": round(vector_recall, 4),
        "hybrid_recall_at_k": round(hybrid_recall, 4),
        "absolute_gain": round(hybrid_recall - vector_recall, 4),
        "passes_improvement_gate": hybrid_recall > vector_recall,
        "per_query": rows,
        "corpus_contract": EXACT_TERM_CORPUS_CONTRACT,
    }


# ═══════════════════════════════════════════════
#  Story v2 embedding representation ablation
# ═══════════════════════════════════════════════

def _topic_v2_payload(topic: bm.Topic) -> dict:
    """Map the existing human-labelled topic into the Story v2 contract."""

    return story_v2.normalize_story_payload({
        "title": topic.title,
        "abstract": topic.problem_desc,
        "detail": {
            "problem": topic.problem_desc,
            "actions": [topic.content],
            "outcome": topic.content,
            "pitfalls": [],
            "evidence": [topic.content],
            "applicability": {
                "applies_when": [{"domain": topic.domain}],
                "excludes_when": [],
            },
        },
        "sources": [{"evidence": ["retrieval benchmark ground truth"]}],
        "keywords": topic.keywords,
    })


def _ablation_query_pairs(bench: bm.Benchmark) -> list[dict]:
    pairs = list(bench.query_pairs)
    # v1 of the benchmark predates ContextEnvelope. Add a deterministic
    # cross-tool paraphrase from its human synonym label so reports always
    # expose the required group without changing ground-truth topic identity.
    for topic in bench.topics:
        if "cross_tool" not in topic.queries:
            pairs.append({
                "query": (
                    "在另一个 Agent 工具（Cursor/Codex/Claude Code）中遇到："
                    + topic.queries.get("synonym", topic.problem_desc)
                ),
                "variant": "cross_tool",
                "topic_id": topic.id,
            })
    return pairs


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * percentile))
    return round(ordered[index], 3)


def run_embedding_ablation(
    benchmark_path: Path | str = None,
    *,
    baseline_mode: str = "legacy",
) -> dict:
    """Compare default/full/per-field multi-vector quality and local latency.

    Query vectors and story vectors use the real configured embedding provider;
    unit tests monkeypatch the module function. Multi-vector ranks a Story by
    its best title/abstract/applicability field similarity and therefore records
    three index vectors per Story instead of one.
    """

    bench = bm.load_benchmark(benchmark_path)
    topics = {topic.id: topic for topic in bench.topics}
    payloads = {topic.id: _topic_v2_payload(topic) for topic in bench.topics}
    query_pairs = _ablation_query_pairs(bench)
    results: dict[str, dict] = {}

    for mode in ABLATION_MODES:
        index_vectors: dict[str, list[list[float]]] = {}
        index_latencies = []
        embed_failures = 0
        for topic_id, topic in topics.items():
            payload = payloads[topic_id]
            if mode == "legacy":
                texts = [topic.index_text()]
            elif mode == "multi_vector":
                fields = story_v2.embedding_fields(payload)
                texts = [
                    fields[name] for name in ("title", "abstract", "applicability")
                    if fields[name]
                ]
            else:
                texts = [story_v2.embedding_input(payload, mode)]
            vectors = []
            for text in texts:
                started = time.perf_counter()
                vector = embeddings.embed(text)
                index_latencies.append((time.perf_counter() - started) * 1000)
                if vector:
                    vectors.append(vector)
                else:
                    embed_failures += 1
            if vectors:
                index_vectors[topic_id] = vectors

        rows = []
        query_latencies = []
        retrieval_latencies = []
        for pair in query_pairs:
            started = time.perf_counter()
            query_vector = embeddings.embed(pair["query"])
            query_latencies.append((time.perf_counter() - started) * 1000)
            if not query_vector:
                embed_failures += 1
                continue
            query_array = np.asarray(query_vector, dtype=np.float32)
            ranking_started = time.perf_counter()
            ranking = []
            for topic_id, vectors in index_vectors.items():
                similarities = [
                    float(np.dot(query_array, np.asarray(vector, dtype=np.float32)))
                    for vector in vectors
                ]
                ranking.append((topic_id, max(similarities)))
            ranking.sort(key=lambda item: item[1], reverse=True)
            retrieval_latencies.append(
                (time.perf_counter() - ranking_started) * 1000
            )
            filtered = [
                topic_id for topic_id, similarity in ranking
                if similarity >= config.SIM_THRESHOLD_SEARCH
            ]
            rows.append({
                "variant": pair["variant"],
                "topic_id": pair["topic_id"],
                "recall@3": M.recall_at_k(
                    filtered, [pair["topic_id"]], 3
                ),
                "mrr": M.mrr(filtered, [pair["topic_id"]]),
            })

        def aggregate(subset, key):
            return round(
                sum(row[key] for row in subset) / len(subset), 4
            ) if subset else 0.0

        groups = {}
        for variant in ("exact", "synonym", "cross_tool", "cross_lang"):
            subset = [row for row in rows if row["variant"] == variant]
            groups[variant] = {
                "count": len(subset),
                "recall@3": aggregate(subset, "recall@3"),
                "mrr": aggregate(subset, "mrr"),
            }
        results[mode] = {
            "recall@3": aggregate(rows, "recall@3"),
            "mrr": aggregate(rows, "mrr"),
            "query_count": len(rows),
            "story_count": len(index_vectors),
            "index_vectors_per_story": (
                3 if mode == "multi_vector" else 1
            ),
            "embed_failures": embed_failures,
            "groups": groups,
            "latency_ms": {
                "index_mean_per_vector": round(
                    sum(index_latencies) / len(index_latencies), 3
                ) if index_latencies else 0.0,
                "index_mean_per_story": round(
                    sum(index_latencies) / len(index_vectors), 3
                ) if index_vectors else 0.0,
                "query_p50": _percentile(query_latencies, 0.50),
                "query_p95": _percentile(query_latencies, 0.95),
                "retrieval_p50": _percentile(retrieval_latencies, 0.50),
                "retrieval_p95": _percentile(retrieval_latencies, 0.95),
            },
        }

    baseline = results[baseline_mode]["recall@3"]
    default = results["default"]["recall@3"]
    delta = round(default - baseline, 4)
    passes = default >= baseline - 0.02
    return {
        "benchmark_version": bench.version,
        "embedding_model": config.EMBED_MODEL,
        "embedding_dimension": config.EMBED_DIM,
        "similarity_threshold": config.SIM_THRESHOLD_SEARCH,
        "topic_count": len(bench.topics),
        "query_count": len(query_pairs),
        "baseline_mode": baseline_mode,
        "selected_mode": "default" if passes else baseline_mode,
        "selection_gate": "recall@3 >= baseline - 0.02",
        "default_delta_recall_at_3": delta,
        "passes_two_point_non_regression": passes,
        "modes": results,
        "notes": [
            "default = title + abstract + applicability",
            "full = title + abstract + structured detail + applicability",
            "multi_vector = max(title, abstract, applicability) field similarity",
            "cross_tool is a deterministic paraphrase of each human synonym label",
        ],
    }


# ═══════════════════════════════════════════════
#  Retrieval strategy ablation (FLO-152)
# ═══════════════════════════════════════════════

def _strategy_query_pairs(bench: bm.Benchmark) -> list[dict]:
    """Expose the five acceptance groups with explicit ground truth."""

    pairs = []
    for topic in bench.topics:
        outcome = topic.content.rsplit("结果：", 1)[-1].strip()
        outcome_hint = outcome
        # The user remembers an outcome/evidence fragment, not the technology
        # name already present in title/abstract.  Mask labelled keywords while
        # retaining exact measurements and action/result phrasing for FTS.
        masks = [*topic.keywords, *re.findall(r"[A-Za-z0-9_.+-]{3,}", topic.title)]
        for mask in sorted(set(masks), key=len, reverse=True):
            outcome_hint = re.sub(
                re.escape(mask), " ", outcome_hint, flags=re.IGNORECASE
            )
        outcome_hint = " ".join(outcome_hint.split()).strip("，。；; ")
        if len(outcome_hint) < 8:
            outcome_hint = outcome
        pairs.extend([
            {
                "query": topic.queries["exact"],
                "variant": "exact",
                "topic_id": topic.id,
            },
            {
                "query": topic.queries["synonym"],
                "variant": "synonym",
                "topic_id": topic.id,
            },
            {
                "query": topic.queries["cross_lang"],
                "variant": "cross_language",
                "topic_id": topic.id,
            },
            {
                "query": (
                    "在另一个 Agent 工具（Cursor/Codex/Claude Code）中遇到："
                    + topic.queries["synonym"]
                ),
                "variant": "cross_tool",
                "topic_id": topic.id,
            },
            {
                "query": (
                    "问题名称和工具已经记不清，只记得最后的结果或指标："
                    + outcome_hint
                ),
                "variant": "ambiguous",
                "topic_id": topic.id,
            },
        ])
    return pairs


def pre_generated_transform_provider(path: Path | str):
    """Load a query-keyed transform artifact that cannot address ground truth."""

    raw_bytes = Path(path).read_bytes()
    artifact = json.loads(raw_bytes.decode("utf-8"))
    if artifact.get("source") != "query_only_pre_generated":
        raise ValueError("transform artifact source must be query_only_pre_generated")
    expected_inputs = [
        "raw_query", "requested_transformations", "timeout_seconds"
    ]
    if artifact.get("generation_inputs") != expected_inputs:
        raise ValueError("transform artifact must declare the query-only input contract")
    entries = artifact.get("entries")
    if not isinstance(entries, list):
        raise ValueError("transform artifact entries must be a list")

    cache = {}
    forbidden = {"topic_id", "target_story", "title", "problem_desc", "content"}
    for entry in entries:
        if not isinstance(entry, dict) or forbidden & set(entry):
            raise ValueError("transform artifact entry contains ground-truth fields")
        query = adaptive.normalize_query(entry.get("query"))
        digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
        if not query or entry.get("query_sha256") != digest:
            raise ValueError("transform artifact query hash mismatch")
        cache[digest] = entry.get("output")

    stats = {"cache_hits": 0, "cache_misses": 0}

    def provider(query, transformations, *, timeout_seconds):
        del transformations, timeout_seconds
        digest = hashlib.sha256(
            adaptive.normalize_query(query).encode("utf-8")
        ).hexdigest()
        if digest not in cache:
            stats["cache_misses"] += 1
            return None
        stats["cache_hits"] += 1
        output = cache[digest]
        return dict(output) if isinstance(output, dict) else None

    provider.evidence_metadata = {
        "generator": str(artifact.get("generator") or "unknown"),
        "prompt_version": str(artifact.get("prompt_version") or "unknown"),
        "artifact_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "entry_count": len(cache),
        "stats": stats,
    }
    return provider


def run_retrieval_strategy_ablation(
    db_path: Optional[Path] = None,
    benchmark_path: Path | str = None,
    *,
    transform_provider=None,
    transform_source: str = "live_generated",
) -> dict:
    """Compare cumulative direct/hybrid/graph/rewrite/HyDE/rerank stacks.

    Ground truth is deliberately unavailable to the transformation provider:
    ``topic_id`` is used only after ranking, to score the returned story ids.
    The default provider is the production query transformer and receives only
    the raw query, requested transformation names, and its timeout budget.

    ``query_only_pre_generated`` is supported for reproducible offline output,
    while ``oracle_upper_bound`` is explicitly excluded from default selection.
    Callers supplying either source must also inject the corresponding provider.
    """

    if transform_source not in TRANSFORM_EVIDENCE_SOURCES:
        raise ValueError(
            "transform_source must be live_generated, "
            "query_only_pre_generated, or oracle_upper_bound"
        )
    if transform_provider is None:
        if transform_source != "live_generated":
            raise ValueError(
                f"{transform_source} requires an explicit transform_provider"
            )

        def transform_provider(query, transformations, *, timeout_seconds):
            return llm_mod.transform_search_query(
                query,
                transformations,
                timeout_seconds=timeout_seconds,
            )

    bench = bm.load_benchmark(benchmark_path)
    pairs = _strategy_query_pairs(bench)
    rows_by_strategy: dict[str, list[dict]] = {
        strategy: [] for strategy in RETRIEVAL_STRATEGIES
    }
    latencies: dict[str, list[float]] = {
        strategy: [] for strategy in RETRIEVAL_STRATEGIES
    }
    embed_failures = 0
    transform_attempts = 0
    transform_successes = 0
    transform_failures = 0
    generated_counts = {"rewrite": 0, "multi_query": 0, "hyde": 0}

    with _isolated_db(db_path):
        topic_to_sid: dict[str, int] = {}
        for topic in bench.topics:
            payload = _topic_v2_payload(topic)
            vector = embeddings.embed(story_v2.embedding_input(payload, "default"))
            if not vector:
                embed_failures += 1
                continue
            topic_to_sid[topic.id] = store.add_story(
                title=topic.title,
                abstract=topic.problem_desc,
                content=topic.content,
                keywords=topic.keywords,
                embedding=vector,
                applicability=payload["applicability"],
            )
        corpus_size = len(topic_to_sid)
        candidate_limit = max(12, corpus_size)

        def retrieve(text: str) -> tuple[list[dict], list[dict]]:
            nonlocal embed_failures
            vector = embeddings.embed(text)
            vector_rows = []
            if vector:
                vector_rows = [
                    item for item in store.search_by_vector(
                        vector, top_k=candidate_limit
                    )
                    if item["similarity"] >= config.SIM_THRESHOLD_SEARCH
                ]
            else:
                embed_failures += 1
            lexical_rows = store.search_by_lexical(
                text,
                top_k=candidate_limit,
                timeout_seconds=max(0.5, config.QUERY_FALLBACK_TIMEOUT_SECONDS),
            )
            return vector_rows, lexical_rows

        for pair in pairs:
            target_sid = topic_to_sid.get(pair["topic_id"])
            if target_sid is None:
                continue
            started = time.perf_counter()
            vector_rows, lexical_rows = retrieve(pair["query"])
            direct_rows = vector_rows
            direct_ms = (time.perf_counter() - started) * 1000.0

            hybrid_rows = adaptive.fuse_rankings(
                vector_rows, lexical_rows, limit=candidate_limit
            )
            hybrid_ms = (time.perf_counter() - started) * 1000.0

            graph_rows, _, _ = search_mod._graph_rerank(
                hybrid_rows,
                top_k=candidate_limit,
                retrieval_source="hybrid",
                graph_enabled=True,
                current_context=None,
                scope="profile",
            )
            graph_ms = (time.perf_counter() - started) * 1000.0

            graph_rerank_input, _ = search_mod._hydrate_rerank_details(
                graph_rows, enabled=True
            )
            graph_reranked_rows, _ = adaptive.rerank(
                pair["query"], graph_rerank_input, enabled=True
            )
            graph_reranked_rows = search_mod._strip_rerank_details(
                graph_reranked_rows
            )
            graph_rerank_ms = (time.perf_counter() - started) * 1000.0

            transform_attempts += 1
            try:
                generated = transform_provider(
                    pair["query"],
                    list(adaptive.VALID_TRANSFORMS),
                    timeout_seconds=config.QUERY_DEEP_TRANSFORM_TIMEOUT_SECONDS,
                )
            except Exception as exc:
                logger.warning("strategy transformation failed: %s", exc)
                generated = None
            generated = _normalize_strategy_transformations(
                pair["query"], generated
            )
            if any((
                generated["rewrite"],
                generated["queries"],
                generated["hypothetical_document"],
            )):
                transform_successes += 1
            else:
                transform_failures += 1

            rewrite_rankings = []
            rewrite_inputs = []
            if generated["rewrite"]:
                rewrite_inputs.append(("rewrite", generated["rewrite"]))
                generated_counts["rewrite"] += 1
            for transformed_query in generated["queries"]:
                rewrite_inputs.append(("multi_query", transformed_query))
                generated_counts["multi_query"] += 1
            for query_index, (kind, transformed_query) in enumerate(rewrite_inputs):
                transformed_vector, transformed_lexical = retrieve(transformed_query)
                rewrite_rankings.append({
                    "transform": kind,
                    "query_index": query_index,
                    "matches": adaptive.fuse_rankings(
                        transformed_vector,
                        transformed_lexical,
                        limit=candidate_limit,
                    ),
                })
            rewrite_rows = adaptive.merge_transformed_rankings(
                graph_rows,
                rewrite_rankings,
                mode="deep",
                limit=candidate_limit,
            )
            rewrite_ms = (time.perf_counter() - started) * 1000.0

            hyde_rankings = []
            if generated["hypothetical_document"]:
                generated_counts["hyde"] += 1
                hyde_vector, hyde_lexical = retrieve(
                    generated["hypothetical_document"]
                )
                hyde_rankings.append({
                    "transform": "hyde",
                    "query_index": len(rewrite_inputs),
                    "matches": adaptive.fuse_rankings(
                        hyde_vector, hyde_lexical, limit=candidate_limit
                    ),
                })
            hyde_rows = adaptive.merge_transformed_rankings(
                rewrite_rows,
                hyde_rankings,
                mode="deep",
                limit=candidate_limit,
            )
            hyde_ms = (time.perf_counter() - started) * 1000.0

            rerank_input, _ = search_mod._hydrate_rerank_details(
                hyde_rows, enabled=True
            )
            reranked_rows, _ = adaptive.rerank(
                pair["query"], rerank_input, enabled=True
            )
            reranked_rows = search_mod._strip_rerank_details(reranked_rows)
            rerank_ms = (time.perf_counter() - started) * 1000.0
            outputs = {
                "direct_vector": direct_rows,
                "hybrid": hybrid_rows,
                "hybrid_graph": graph_rows,
                "hybrid_graph_reranker": graph_reranked_rows,
                "hybrid_graph_rewrite": rewrite_rows,
                "hybrid_graph_hyde": hyde_rows,
                "hybrid_graph_hyde_reranker": reranked_rows,
            }
            timings = {
                "direct_vector": direct_ms,
                "hybrid": hybrid_ms,
                "hybrid_graph": graph_ms,
                "hybrid_graph_reranker": graph_rerank_ms,
                "hybrid_graph_rewrite": rewrite_ms,
                "hybrid_graph_hyde": hyde_ms,
                "hybrid_graph_hyde_reranker": rerank_ms,
            }
            for strategy, ranked in outputs.items():
                ranked_ids = [item["story_id"] for item in ranked]
                rows_by_strategy[strategy].append({
                    "variant": pair["variant"],
                    "topic_id": pair["topic_id"],
                    "recall@3": M.recall_at_k(ranked_ids, [target_sid], 3),
                    "mrr": M.mrr(ranked_ids, [target_sid]),
                })
                latencies[strategy].append(timings[strategy])

    def aggregate(rows: list[dict], key: str) -> float:
        return round(sum(row[key] for row in rows) / len(rows), 4) if rows else 0.0

    results = {}
    for strategy in RETRIEVAL_STRATEGIES:
        rows = rows_by_strategy[strategy]
        groups = {}
        for variant in (
            "exact", "synonym", "cross_language", "cross_tool", "ambiguous"
        ):
            subset = [row for row in rows if row["variant"] == variant]
            groups[variant] = {
                "count": len(subset),
                "recall@3": aggregate(subset, "recall@3"),
                "mrr": aggregate(subset, "mrr"),
            }
        hard = [row for row in rows if row["variant"] != "exact"]
        results[strategy] = {
            "recall@3": aggregate(rows, "recall@3"),
            "mrr": aggregate(rows, "mrr"),
            "hard_recall@3": aggregate(hard, "recall@3"),
            "hard_mrr": aggregate(hard, "mrr"),
            "query_count": len(rows),
            "groups": groups,
            "latency_ms": {
                "p50": _percentile(latencies[strategy], 0.50),
                "p95": _percentile(latencies[strategy], 0.95),
            },
        }

    baseline = results["direct_vector"]
    eligible = []
    for strategy, row in results.items():
        quality_gain = (
            row["hard_mrr"] - baseline["hard_mrr"] >= 0.05
            if baseline["hard_recall@3"] >= 0.90
            else row["hard_recall@3"] - baseline["hard_recall@3"] >= 0.10
        )
        overall_non_regression = row["recall@3"] >= baseline["recall@3"] - 0.02
        latency_limit = (
            5_000.0 if strategy in {
                "hybrid_graph_rewrite", "hybrid_graph_hyde",
                "hybrid_graph_hyde_reranker",
            } else 1_000.0
        )
        latency_pass = row["latency_ms"]["p95"] <= latency_limit
        uses_transform = strategy in {
            "hybrid_graph_rewrite", "hybrid_graph_hyde",
            "hybrid_graph_hyde_reranker",
        }
        ground_truth_isolation = not (
            uses_transform and transform_source == "oracle_upper_bound"
        )
        online_latency_evidence = not uses_transform or (
            transform_source == "live_generated"
        )
        row["evidence_source"] = (
            transform_source if uses_transform else "raw_query_only"
        )
        # The baseline is reference-only; every new default must demonstrate
        # the issue's hard-query quality gain as well as non-regression/latency.
        row["gates"] = {
            "hard_quality_gain": quality_gain,
            "overall_non_regression": overall_non_regression,
            "latency": latency_pass,
            "ground_truth_isolation": ground_truth_isolation,
            "online_latency_evidence": online_latency_evidence,
            "eligible_for_default": (
                strategy != "direct_vector"
                and quality_gain and overall_non_regression and latency_pass
                and ground_truth_isolation and online_latency_evidence
            ),
        }
        if row["gates"]["eligible_for_default"]:
            eligible.append(strategy)

    selected = max(
        eligible,
        key=lambda name: (
            results[name]["hard_recall@3"], results[name]["hard_mrr"],
            results[name]["recall@3"], -results[name]["latency_ms"]["p95"],
        ),
        default="direct_vector",
    )
    provider_metadata = getattr(transform_provider, "evidence_metadata", {})
    provider_stats = provider_metadata.get("stats", {})
    return {
        "benchmark_version": bench.version,
        "embedding_model": config.EMBED_MODEL,
        "topic_count": len(topic_to_sid),
        "query_count": len(pairs),
        "groups": [
            "exact", "synonym", "cross_language", "cross_tool", "ambiguous"
        ],
        "baseline": "direct_vector",
        "strategies": results,
        "eligible_strategies": eligible,
        "selected_default": selected,
        "passes_default_gate": selected != "direct_vector",
        "embed_failures": embed_failures,
        "transformation_provenance": {
            "source": transform_source,
            "generator": (
                provider_metadata.get("generator")
                or (
                    config.LLM_MODEL
                    if transform_source == "live_generated" else "injected_provider"
                )
            ),
            "generation_inputs": [
                "raw_query", "requested_transformations", "timeout_seconds"
            ],
            "ground_truth_fields_used_for_generation": [],
            "ground_truth_usage": "topic_id is used only for post-ranking scoring",
            "pre_generated_outputs": (
                transform_source == "query_only_pre_generated"
            ),
            "oracle_upper_bound": transform_source == "oracle_upper_bound",
            "oracle_eligible_for_default": False,
            "online_latency_evidence": transform_source == "live_generated",
            "prompt_version": provider_metadata.get("prompt_version"),
            "artifact_sha256": provider_metadata.get("artifact_sha256"),
            "artifact_entry_count": provider_metadata.get("entry_count"),
            "artifact_cache_hits": provider_stats.get("cache_hits"),
            "artifact_cache_misses": provider_stats.get("cache_misses"),
            "attempts": transform_attempts,
            "successes": transform_successes,
            "failures": transform_failures,
            "generated_counts": generated_counts,
        },
        "gate": {
            "hard_query": (
                "recall@3 +10pp; when baseline >=90%, MRR +5pp"
            ),
            "overall": "recall@3 no worse than direct-vector by more than 2pp",
            "latency": "fast p95 <=1s; deep p95 <=5s",
        },
        "notes": [
            "transform generation receives the raw query only; labels and target "
            "Story fields are unavailable until post-ranking scoring",
            "live generation latency is included in rewrite/HyDE cumulative latency",
            "pre-generated query-only evidence is labelled; oracle evidence can "
            "never select a default",
            "graph quality has a dedicated typed-path benchmark in graph_eval",
        ],
    }


def _normalize_strategy_transformations(query: str, generated) -> dict:
    """Bound provider output without consulting benchmark labels or target data."""

    if not isinstance(generated, dict):
        generated = {}

    def bounded(value, limit=1200):
        if not isinstance(value, str):
            return ""
        return adaptive.normalize_query(value)[:limit]

    raw_query = adaptive.normalize_query(query)
    rewrite = bounded(generated.get("rewrite"), 600)
    if rewrite == raw_query:
        rewrite = ""
    queries = []
    values = generated.get("queries", [])
    if isinstance(values, list):
        for value in values:
            candidate = bounded(value, 600)
            if candidate and candidate != raw_query and candidate not in queries:
                queries.append(candidate)
            if len(queries) >= max(1, config.QUERY_MULTI_QUERY_LIMIT):
                break
    return {
        "rewrite": rewrite,
        "queries": queries,
        "hypothetical_document": bounded(
            generated.get("hypothetical_document"), 1200
        ),
    }


# ═══════════════════════════════════════════════
#  加工分支评测（merge/update vs create）
# ═══════════════════════════════════════════════

def _branch_from_sim(sim: float) -> str:
    """复刻 processor 的分支判定：sim>=0.92 update / [0.85,0.92) merge / <0.85 create。"""
    if sim >= config.SIM_THRESHOLD_UPDATE_ONLY:
        return "update"
    if sim >= config.SIM_THRESHOLD_HIGH:
        return "merge"
    return "create"


def run_processing_eval(
    db_path: Optional[Path] = None,
    sweep_thresholds: list[float] = None,
    benchmark_path: Path | str = None,
) -> dict:
    """逐对评测 merge/update 分支是否选对（确定性 CuratedLLM + 真实 embedding）。

    每对一个干净 DB：先加工 a（create），再加工 b，观测实际分支（由 b-vs-a 相似度决定，
    并与 processor 实际产物 story id 交叉验证）。返回分支正确率 + 阈值敏感性曲线。
    """
    sweep_thresholds = sweep_thresholds if sweep_thresholds is not None else HIGH_THRESHOLD_SWEEP
    bench = bm.load_benchmark(benchmark_path)
    curated = CuratedLLM()

    pair_results: list[dict] = []
    sim_samples: list[dict] = []   # 供阈值扫描复用：{expected, sim}
    for pair in bench.merge_pairs:
        with _isolated_db(db_path):
            with _patch_llm(
                extract=curated.extract_keywords,
                summarize=curated.summarize_session,
                merge=curated.merge_stories,
                judge=curated.judge_split,
                split=curated.split_story,
            ):
                # 加工 a -> create
                curated.configure(pair.a, should_split=False)
                sid_a = store.add_session("eval", pair.a.problem_desc + " raw",
                                          pair.a.problem_desc, "[]", "")
                ret_a = processor.process_session(sid_a)
                if ret_a is None:
                    pair_results.append({
                        "id": pair.id, "expected_branch": pair.expected_branch,
                        "actual_branch": "error", "sim": None,
                        "same_story": False, "note": "a 加工失败",
                    })
                    continue
                story_a = ret_a

                # b-vs-a 相似度（直接向量比较，与 processor 检索一致）
                bvec = embeddings.embed(pair.b.index_text())
                if not bvec:
                    pair_results.append({
                        "id": pair.id, "expected_branch": pair.expected_branch,
                        "actual_branch": "error", "sim": None,
                        "same_story": False, "note": "b embedding 失败",
                    })
                    continue
                hits = store.search_by_vector(bvec, top_k=5)
                sim = next((h["similarity"] for h in hits if h["story_id"] == story_a), 0.0)

                # 加工 b
                curated.configure(pair.b, should_split=False)
                sid_b = store.add_session("eval", pair.b.problem_desc + " raw",
                                          pair.b.problem_desc, "[]", "")
                ret_b = processor.process_session(sid_b)

                actual = _branch_from_sim(sim)
                same_story = (ret_b == story_a)
                # 交叉验证：sim>=0.85 时 processor 应并入 story_a
                processor_agrees = (same_story if sim >= config.SIM_THRESHOLD_HIGH
                                    else not same_story)
                pair_results.append({
                    "id": pair.id,
                    "relation": pair.relation,
                    "expected_branch": pair.expected_branch,
                    "actual_branch": actual,
                    "sim": round(sim, 4),
                    "same_story": same_story,
                    "processor_agrees_with_sim": processor_agrees,
                    "note": pair.note,
                })
                sim_samples.append({"expected": pair.expected_branch, "sim": sim,
                                    "relation": pair.relation})

    accuracy = M.merge_branch_accuracy(pair_results)

    # 阈值敏感性：在不同 SIM_THRESHOLD_HIGH 下重新分类，看正确率如何变化
    def sweep_fn(th):
        correct = 0
        for s in sim_samples:
            sim = s["sim"]
            if sim >= config.SIM_THRESHOLD_UPDATE_ONLY:
                actual = "update"
            elif sim >= th:
                actual = "merge"
            else:
                actual = "create"
            exp = s["expected"]
            if exp == "merge_or_update":
                ok = actual in ("merge", "update")
            elif exp == "update":
                ok = actual == "update"
            elif exp == "create":
                ok = actual == "create"
            else:
                ok = actual == exp
            if ok:
                correct += 1
        return {"merge_branch_accuracy": round(correct / len(sim_samples), 4),
                "correct": correct, "total": len(sim_samples)}

    curve = M.threshold_sweep(sweep_fn, sweep_thresholds, metric_key="merge_branch_accuracy")

    return {
        "accuracy": accuracy,
        "pairs": pair_results,
        "threshold_sweep": curve,
        "note": ("分支由真实 embedding 相似度决定；keywords/summary 用 benchmark 人工标注"
                 "（CuratedLLM），隔离度量 0.85/0.92 阈值，排除 LLM 关键词质量波动。"),
    }


# ═══════════════════════════════════════════════
#  分裂评测
# ═══════════════════════════════════════════════

def run_split_eval(
    db_path: Optional[Path] = None,
    benchmark_path: Path | str = None,
) -> dict:
    """对每个 split_case：加工 existing（create）-> 加工 incoming（应 merge->split），
    校验分裂结构正确性（父向量移除、父子/兄弟边、子 story 可检索）。
    """
    bench = bm.load_benchmark(benchmark_path)
    curated = CuratedLLM()

    results: list[dict] = []
    for case in bench.split_cases:
        with _isolated_db(db_path):
            # 构造包含两段完整结论的 merged 文本，并用人工 split 标注触发。
            long_merged = case.incoming.summary["content"] + " " + (
                "补充细节：" + case.existing.summary["content"]
            ) * 4
            sub_stories = [
                {"title": f"{case.id}-子记忆1", "content": case.existing.summary["content"]},
                {"title": f"{case.id}-子记忆2", "content": case.incoming.summary["content"]},
            ]
            with _patch_llm(
                extract=curated.extract_keywords,
                summarize=curated.summarize_session,
                merge=curated.merge_stories,
                judge=curated.judge_split,
                split=curated.split_story,
            ):
                curated.configure(case.existing, should_split=False)
                sid_ex = store.add_session("eval", case.existing.problem_desc + " raw",
                                           case.existing.problem_desc, "[]", "")
                parent = processor.process_session(sid_ex)

                # incoming-vs-existing 相似度：必须在加工 incoming 之前度量
                # （split 会把 parent 向量从索引移除，事后度量会得到 0）。仅 merge 区间
                # [0.85,0.92) 才进入分裂路径，故 sim 同时解释了分裂是否本应触发。
                in_vec = embeddings.embed(case.incoming.index_text())
                sim = 0.0
                if in_vec:
                    hits = store.search_by_vector(in_vec, top_k=5)
                    sim = next((h["similarity"] for h in hits if h["story_id"] == parent), 0.0)
                actual_branch_before = _branch_from_sim(sim)

                curated.configure(
                    case.incoming,
                    merged={"title": case.incoming.summary["title"], "content": long_merged},
                    should_split=True,
                    sub_stories=sub_stories,
                )
                sid_in = store.add_session("eval", case.incoming.problem_desc + " raw",
                                           case.incoming.problem_desc, "[]", "")
                ret = processor.process_session(sid_in)

            # ── 校验分裂结构（直接查 vec0 / stories / edges） ──
            db = store.get_db()
            try:
                parent_in_index = db.execute(
                    "SELECT 1 FROM story_vectors WHERE story_id = ?", (parent,)
                ).fetchone() is not None
                child_rows = db.execute(
                    "SELECT id FROM stories WHERE parent_id = ?", (parent,)
                ).fetchall()
                child_ids = [c["id"] for c in child_rows]
                # 父子边
                pc_edges = []
                sibling_edges = []
                if child_ids:
                    placeholders = ",".join("?" * len(child_ids))
                    all_edges = db.execute(
                        f"""SELECT source_id, target_id, weight, edge_type FROM edges
                            WHERE (source_id=? AND target_id IN ({placeholders}))
                               OR (target_id=? AND source_id IN ({placeholders}))""",
                        (parent, *child_ids, parent, *child_ids),
                    ).fetchall()
                    for e in all_edges:
                        if e["edge_type"] == "parent_child":
                            pc_edges.append(e)
                        elif e["edge_type"] in {"semantic", "sibling"}:
                            sibling_edges.append(e)
                # 子 story 向量是否在索引中
                children_in_index = []
                for cid in child_ids:
                    r = db.execute(
                        "SELECT 1 FROM story_vectors WHERE story_id = ?", (cid,)
                    ).fetchone()
                    children_in_index.append(r is not None)
            finally:
                db.close()

            split_happened = len(child_ids) >= 2
            pc_edges_ok = all(
                abs(e["weight"] - config.WEIGHT_PARENT_CHILD) < 1e-6
                and e["edge_type"] == "parent_child" for e in pc_edges
            ) and len(pc_edges) == len(child_ids)
            # 子 story 可检索：用 incoming 查询向量检索，top-k 应包含至少一个子 story
            children_retrievable = False
            if child_ids:
                qvec = embeddings.embed(case.incoming.index_text())
                if qvec:
                    hits = store.search_by_vector(qvec, top_k=max(5, len(child_ids) + 2))
                    children_retrievable = any(h["story_id"] in child_ids for h in hits)

            checks = {
                "split_triggered": split_happened,
                "parent_vector_removed": not parent_in_index,
                "child_count": len(child_ids),
                "parent_child_edges_weight_1": pc_edges_ok,
                "children_vectors_in_index": all(children_in_index) if children_in_index else False,
                "children_retrievable": children_retrievable,
            }

            results.append({
                "id": case.id,
                "expected_split": case.expected_split,
                "actual_split": split_happened,
                "sim": round(sim, 4),
                "actual_branch": actual_branch_before,
                "returned_story_id": ret,
                "parent_story_id": parent,
                "checks": checks,
                "note": case.note,
            })

    passed = sum(1 for r in results
                 if r["actual_split"] == r["expected_split"]
                 and all(v for k, v in r["checks"].items()
                         if k != "child_count"))
    return {
        "cases": results,
        "passed": passed,
        "total": len(results),
        "accuracy": round(passed / len(results), 4) if results else 0.0,
        "note": ("分裂触发由独立结论的人工 SPLIT 标注决定（不使用字符硬规则）；"
                 "结构校验：父向量移除、父子边 1.0、子向量入索引、子 story 可检索。"),
    }


# ═══════════════════════════════════════════════
#  汇总 + 文本报告
# ═══════════════════════════════════════════════

class EvalReport:
    """质量、加工、分裂与表示消融的汇总容器。"""
    def __init__(self, retrieval=None, processing=None, split=None,
                 ablation=None, strategy=None, exact_term=None, meta=None):
        self.retrieval = retrieval
        self.processing = processing
        self.split = split
        self.ablation = ablation
        self.strategy = strategy
        self.exact_term = exact_term
        self.meta = meta or {}

    def to_dict(self) -> dict:
        return {
            "meta": self.meta,
            "retrieval": self.retrieval,
            "processing": self.processing,
            "split": self.split,
            "ablation": self.ablation,
            "strategy": self.strategy,
            "exact_term": self.exact_term,
        }


def run_all(
    db_path: Optional[Path] = None,
    parts: tuple = (
        "retrieval", "processing", "split", "ablation", "strategy",
        "exact_term",
    ),
    benchmark_path: Path | str = None,
    *,
    transform_provider=None,
    transform_source: str = "live_generated",
) -> EvalReport:
    """跑指定子评测集合，返回 EvalReport。"""
    report = EvalReport(meta={"parts": list(parts)})
    if "retrieval" in parts:
        report.retrieval = run_retrieval_eval(db_path=db_path,
                                              benchmark_path=benchmark_path)
    if "processing" in parts:
        report.processing = run_processing_eval(db_path=db_path,
                                                benchmark_path=benchmark_path)
    if "split" in parts:
        report.split = run_split_eval(db_path=db_path,
                                      benchmark_path=benchmark_path)
    if "ablation" in parts:
        report.ablation = run_embedding_ablation(
            benchmark_path=benchmark_path
        )
    if "strategy" in parts:
        report.strategy = run_retrieval_strategy_ablation(
            db_path=db_path,
            benchmark_path=benchmark_path,
            transform_provider=transform_provider,
            transform_source=transform_source,
        )
    if "exact_term" in parts:
        report.exact_term = run_exact_term_hybrid_ablation(db_path=db_path)
    return report


def format_report(report: EvalReport) -> str:
    """把 EvalReport 渲染为人类可读文本（CLI / 评论用）。"""
    lines = ["=" * 64, "📊 Storybook 检索质量评测报告", "=" * 64]
    meta = report.meta or {}
    if meta.get("embed_mode"):
        lines.append(f"embedding 模式: {meta['embed_mode']}")
    lines.append("")

    if report.retrieval:
        s = report.retrieval["summary"]
        lines.append("─" * 64)
        lines.append("① 检索评测 (retrieval)")
        lines.append("─" * 64)
        lines.append(f"  语料规模: {s['corpus_size']} stories | 查询数: {s['query_count']} "
                     f"(embed 失败 {s['embed_failures']})")
        lines.append(f"  recall@1 = {s['recall@1']:.2%}  | recall@3 = {s['recall@3']:.2%} "
                     f"| recall@5 = {s['recall@5']:.2%}")
        lines.append(f"  precision@1 = {s['precision@1']:.2%} | precision@3 = {s['precision@3']:.2%} "
                     f"| precision@5 = {s['precision@5']:.2%}")
        lines.append(f"  MRR = {s['mrr']:.4f} | 负例特异性 = {s['specificity']:.2%} "
                     f"(阈值 {s['target_threshold']})")
        lines.append(f"  达标 (recall@3 ≥ 70%): {'✅ 是' if report.retrieval['passes_70_percent_recall_at_3'] else '❌ 否'}")
        lines.append("  按查询变体:")
        for v in ("exact", "synonym", "cross_lang"):
            vd = report.retrieval["per_variant"][v]
            lines.append(f"    {v:10s}: recall@1/3/5 = {vd['recall@1']:.2%}/{vd['recall@3']:.2%}/{vd['recall@5']:.2%}"
                         f"  (n={vd['count']})")
        # 未达标的查询明细
        misses = [r for r in report.retrieval["per_query"] if r["recall@3"] == 0.0]
        if misses:
            lines.append(f"  recall@3=0 的查询 ({len(misses)}):")
            for r in misses[:12]:
                lines.append(f"    [{r['variant']}] {r['query']!r} -> 目标 topic={r['topic_id']} "
                             f"最高相似度={r['target_similarity']}")
        lines.append("  阈值敏感性 (SIM_THRESHOLD_SEARCH -> recall@3):")
        for pt in report.retrieval["threshold_sweep"]:
            lines.append(f"    {pt['threshold']:.2f}: recall@1={pt['recall@1']:.2%} "
                         f"recall@3={pt['recall@3']:.2%} recall@5={pt['recall@5']:.2%}")
        lines.append("")

    if report.processing:
        a = report.processing["accuracy"]
        lines.append("─" * 64)
        lines.append("② 加工分支评测 (merge/update vs create)")
        lines.append("─" * 64)
        lines.append(f"  合并正确率: {a['accuracy']:.2%} ({a['correct']}/{a['total']})")
        for exp, b in a["by_expected"].items():
            lines.append(f"    expected={exp}: {b['correct']}/{b['total']} = {b['accuracy']:.2%}")
        lines.append("  逐对明细:")
        for p in report.processing["pairs"]:
            mark = "✅" if p.get("actual_branch") != "error" and _pair_ok(p) else "❌"
            lines.append(f"    {mark} {p['id']:30s} rel={p.get('relation','?'):14s} "
                         f"exp={p['expected_branch']:16s} actual={p['actual_branch']:8s} "
                         f"sim={p['sim']}")
        lines.append("  阈值敏感性 (SIM_THRESHOLD_HIGH -> 合并正确率):")
        for pt in report.processing["threshold_sweep"]:
            lines.append(f"    {pt['threshold']:.2f}: accuracy={pt['merge_branch_accuracy']:.2%} "
                         f"({pt['correct']}/{pt['total']})")
        lines.append(f"  说明: {report.processing['note']}")
        lines.append("")

    if report.split:
        sp = report.split
        lines.append("─" * 64)
        lines.append("③ 分裂质量评测 (split)")
        lines.append("─" * 64)
        lines.append(f"  结构正确率: {sp['accuracy']:.2%} ({sp['passed']}/{sp['total']})")
        for c in sp["cases"]:
            ok = c["actual_split"] == c["expected_split"] and all(
                v for k, v in c["checks"].items() if k != "child_count")
            mark = "✅" if ok else "❌"
            lines.append(f"    {mark} {c['id']}: sim={c.get('sim')} branch={c.get('actual_branch')} "
                         f"split={c['actual_split']} checks={c['checks']}")
        lines.append(f"  说明: {sp['note']}")
        lines.append("")

    if report.ablation:
        ab = report.ablation
        lines.append("─" * 64)
        lines.append("④ Story v2 embedding 表示消融")
        lines.append("─" * 64)
        for mode in ABLATION_MODES:
            row = ab["modes"][mode]
            latency = row["latency_ms"]
            lines.append(
                f"  {mode:12s}: recall@3={row['recall@3']:.2%} "
                f"MRR={row['mrr']:.4f} vectors/story={row['index_vectors_per_story']} "
                f"index/story={latency['index_mean_per_story']:.1f}ms "
                f"query p95={latency['query_p95']:.1f}ms "
                f"retrieval p95={latency['retrieval_p95']:.1f}ms"
            )
            groups = row["groups"]
            lines.append(
                " " * 16
                + " | ".join(
                    f"{name}={groups[name]['recall@3']:.2%}"
                    for name in ("exact", "synonym", "cross_tool", "cross_lang")
                )
            )
        lines.append(
            f"  选型: {ab['selected_mode']} | default-baseline "
            f"Δrecall@3={ab['default_delta_recall_at_3']:+.2%} | "
            f"2pp 非劣门槛: {'✅' if ab['passes_two_point_non_regression'] else '❌'}"
        )
        lines.append("")

    if report.strategy:
        strategy_report = report.strategy
        lines.append("─" * 64)
        lines.append("⑤ 自适应检索策略消融")
        lines.append("─" * 64)
        provenance = strategy_report["transformation_provenance"]
        lines.append(
            f"  transformation evidence: {provenance['source']} | "
            f"ground truth: scoring-only | "
            f"online latency={'yes' if provenance['online_latency_evidence'] else 'no'} | "
            f"oracle={'yes' if provenance['oracle_upper_bound'] else 'no'} | "
            f"success={provenance['successes']}/{provenance['attempts']}"
        )
        for name in RETRIEVAL_STRATEGIES:
            row = strategy_report["strategies"][name]
            gates = row["gates"]
            lines.append(
                f"  {name:30s}: recall@3={row['recall@3']:.2%} "
                f"hard={row['hard_recall@3']:.2%} MRR={row['mrr']:.4f} "
                f"p95={row['latency_ms']['p95']:.1f}ms "
                f"default={'✅' if gates['eligible_for_default'] else '❌'}"
            )
            lines.append(
                " " * 34
                + " | ".join(
                    f"{group}={row['groups'][group]['recall@3']:.2%}"
                    for group in strategy_report["groups"]
                )
            )
        lines.append(
            f"  默认选型: {strategy_report['selected_default']} | "
            f"质量+时延门禁: {'✅' if strategy_report['passes_default_gate'] else '❌'}"
        )
        lines.append("")

    if report.exact_term:
        exact = report.exact_term
        lines.append("─" * 64)
        lines.append("⑥ 精确术语 Hybrid 消融")
        lines.append("─" * 64)
        if exact["query_count"] == 0:
            lines.append(
                "  ⚠️ 评测不可用：无法生成语料 embedding "
                f"(embed failures={exact['embed_failures']})"
            )
            lines.append("  改善门禁: 未评估（无有效查询）")
        else:
            lines.append(
                f"  queries={exact['query_count']} | recall@{exact['top_k']}: "
                f"vector={exact['vector_recall_at_k']:.2%} -> "
                f"hybrid={exact['hybrid_recall_at_k']:.2%} "
                f"(Δ={exact['absolute_gain']:+.2%})"
            )
            lines.append(
                "  改善门禁: "
                + ("✅" if exact["passes_improvement_gate"] else "❌")
            )
        lines.append("")

    lines.append("=" * 64)
    return "\n".join(lines)


def _pair_ok(p: dict) -> bool:
    """单对是否判对（与 metrics.merge_branch_accuracy 一致）。"""
    exp, act = p["expected_branch"], p["actual_branch"]
    if exp == "merge_or_update":
        return act in ("merge", "update")
    if exp == "update":
        return act == "update"
    if exp == "create":
        return act == "create"
    return act == exp
