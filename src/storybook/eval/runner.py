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
import logging
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Optional

from .. import config, store, embeddings, llm as llm_mod, processor
from . import benchmark as bm
from . import metrics as M

logger = logging.getLogger(__name__)

KS = (1, 3, 5)
# 阈值敏感性扫描点
SEARCH_THRESHOLD_SWEEP = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
HIGH_THRESHOLD_SWEEP = [0.80, 0.82, 0.85, 0.88, 0.90, 0.92, 0.95]


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
        "merge_stories": llm_mod.merge_stories,
        "judge_split": llm_mod.judge_split,
        "split_story": llm_mod.split_story,
    }
    if extract is not None:
        llm_mod.extract_keywords = extract
    if summarize is not None:
        llm_mod.summarize_session = summarize
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
        # 复刻真实 llm.judge_split 的硬规则：合并文本 > STORY_MAX_CHARS 必分裂
        if len(merged_text) > config.STORY_MAX_CHARS:
            return True
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
    max_k = max(ks)

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
            # 构造 >400 字的 merged 文本以强制触发分裂（judge_split 硬规则）
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
                        elif e["edge_type"] == "sibling":
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
        "note": ("分裂触发由 judge_split 决定（>400 字硬规则 + 人工 SPLIT 标注）；"
                 "结构校验：父向量移除、父子边 1.0、子向量入索引、子 story 可检索。"),
    }


# ═══════════════════════════════════════════════
#  汇总 + 文本报告
# ═══════════════════════════════════════════════

class EvalReport:
    """三轮评测的汇总容器（dict-like，可 JSON 序列化）。"""
    def __init__(self, retrieval=None, processing=None, split=None, meta=None):
        self.retrieval = retrieval
        self.processing = processing
        self.split = split
        self.meta = meta or {}

    def to_dict(self) -> dict:
        return {
            "meta": self.meta,
            "retrieval": self.retrieval,
            "processing": self.processing,
            "split": self.split,
        }


def run_all(
    db_path: Optional[Path] = None,
    parts: tuple = ("retrieval", "processing", "split"),
    benchmark_path: Path | str = None,
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
