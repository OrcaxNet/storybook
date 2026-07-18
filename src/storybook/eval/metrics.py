"""纯函数指标：recall@k / precision@k / MRR / 合并正确率 / 阈值敏感性曲线。

刻意与 store/embeddings 解耦——只吃「排序后的 id 列表 + relevant 集合」，
便于用合成数据单测，且不依赖 Ollama。
"""
from __future__ import annotations

from typing import Callable, Iterable


def recall_at_k(ranked_ids: list, relevant_ids: Iterable, k: int) -> float:
    """recall@k = |relevant ∩ top-k| / |relevant|。relevant 为空时返回 0.0。"""
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    topk = set(ranked_ids[:k])
    return len(relevant & topk) / len(relevant)


def precision_at_k(ranked_ids: list, relevant_ids: Iterable, k: int) -> float:
    """precision@k = |relevant ∩ top-k| / k。top-k 不足 k 条时按实际条数分母。"""
    relevant = set(relevant_ids)
    topk = ranked_ids[:k]
    if not topk:
        return 0.0
    return len(relevant & set(topk)) / len(topk)


def mrr(ranked_ids: list, relevant_ids: Iterable) -> float:
    """Mean Reciprocal Rank：首个命中 relevant 的倒数秩；无命中返回 0。"""
    relevant = set(relevant_ids)
    for rank, rid in enumerate(ranked_ids, start=1):
        if rid in relevant:
            return 1.0 / rank
    return 0.0


def _is_merge_or_update(actual_branch: str) -> bool:
    """merge 与 update 都属于「并入既有 story」分支（区别仅在 sim≥0.92 与否）。"""
    return actual_branch in ("merge", "update")


def merge_branch_accuracy(results: list[dict]) -> dict:
    """合并/更新分支正确率。

    ``results`` 每项形如 ``{id, expected_branch, actual_branch, sim, same_story}``。
    判定规则（与 benchmark 标注对齐）：
      * expected=merge_or_update -> actual 应为 merge/update（而非 create）
      * expected=update          -> actual 应为 update（sim≥0.92）
      * expected=create          -> actual 应为 create（不应误并入）

    返回总体正确率 + 按 expected 分组的命中矩阵 + 各 pair 明细是否正确。
    """
    if not results:
        return {"accuracy": 0.0, "correct": 0, "total": 0, "by_expected": {}, "details": []}

    correct = 0
    by_expected: dict[str, dict] = {}
    details = []
    for r in results:
        exp = r["expected_branch"]
        act = r["actual_branch"]
        if exp == "merge_or_update":
            ok = _is_merge_or_update(act)
        elif exp == "update":
            ok = act == "update"
        elif exp == "create":
            ok = act == "create"
        else:
            ok = act == exp
        r = {**r, "correct": ok}
        details.append(r)
        if ok:
            correct += 1
        bucket = by_expected.setdefault(exp, {"correct": 0, "total": 0})
        bucket["total"] += 1
        if ok:
            bucket["correct"] += 1
    for bucket in by_expected.values():
        bucket["accuracy"] = round(bucket["correct"] / bucket["total"], 4) if bucket["total"] else 0.0
    return {
        "accuracy": round(correct / len(results), 4),
        "correct": correct,
        "total": len(results),
        "by_expected": by_expected,
        "details": details,
    }


def threshold_sweep(
    sweep_fn: Callable[[float], dict],
    thresholds: list[float],
    metric_key: str = "recall_at_3",
) -> list[dict]:
    """对一组阈值逐点调用 ``sweep_fn(threshold) -> metrics_dict``，
    收集 ``{threshold, metric_key, ...full_metrics}``。用于阈值敏感性曲线。

    ``sweep_fn`` 负责在给定阈值下重算指标（如重新过滤检索结果）。
    """
    curve = []
    for th in thresholds:
        m = sweep_fn(th)
        curve.append({"threshold": round(th, 3), metric_key: m.get(metric_key), **m})
    return curve
