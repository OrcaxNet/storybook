"""Adaptive query planning, hybrid fusion, and bounded local reranking.

The module is deliberately independent from storage and model clients.  Search
supplies ranked candidates and invokes a transformation callback only after the
deterministic gate has opted into the auto/deep second stage.
"""
from __future__ import annotations

import math
import queue
import re
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

from . import config

VALID_RETRIEVAL_MODES = frozenset({"fast", "auto", "deep"})
VALID_TRANSFORMS = ("rewrite", "multi_query", "hyde")

T = TypeVar("T")


def normalize_query(query: str) -> str:
    """Collapse transport whitespace while preserving the user's language."""

    return " ".join(str(query or "").split()).strip()


def resolve_mode(mode: str | None) -> str:
    resolved = (mode or config.QUERY_DEFAULT_MODE or "fast").strip().lower()
    if resolved not in VALID_RETRIEVAL_MODES:
        raise ValueError("retrieval_mode 必须是 fast、auto 或 deep")
    return resolved


def plan_query(
    query: str,
    matches: list[dict],
    *,
    requested_mode: str,
    current_context: dict | None,
    strict_filtered: int = 0,
    transform_enabled: bool | None = None,
) -> dict:
    """Return an explainable, deterministic second-stage decision."""

    mode = resolve_mode(requested_mode)
    enabled = (
        config.QUERY_TRANSFORM_ENABLED
        if transform_enabled is None else bool(transform_enabled)
    )
    text = normalize_query(query)
    has_cjk = bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", text))
    has_latin = bool(re.search(r"[A-Za-z]", text))
    conjunctions = re.findall(
        r"(?:\band\b|\bthen\b|\balso\b|以及|并且|同时|然后|另外|而且|；|;)",
        text,
        flags=re.IGNORECASE,
    )
    compound = (
        len(text) >= max(1, config.QUERY_AUTO_COMPLEX_CHARS)
        or len(conjunctions) >= 2
    )

    confidences = [_candidate_confidence(item) for item in matches]
    top_confidence = confidences[0] if confidences else 0.0
    scores = [float(item.get("score", item.get("similarity", 0.0))) for item in matches]
    score_gap = scores[0] - scores[1] if len(scores) > 1 else 1.0
    warning_count = sum(bool(item.get("warnings")) for item in matches[:3])
    environment_ambiguity = bool(
        current_context is not None
        and matches
        and (
            warning_count >= max(1, math.ceil(min(3, len(matches)) / 2))
            or strict_filtered > 0
        )
    )

    reasons: list[str] = []
    if not matches:
        reasons.append("zero_results")
    elif top_confidence < config.QUERY_AUTO_CONFIDENCE_THRESHOLD:
        reasons.append("low_confidence")
    if (
        len(matches) > 1
        and score_gap <= config.QUERY_AUTO_SCORE_GAP_THRESHOLD
    ):
        reasons.append("ambiguous_ranking")
    if compound:
        reasons.append("long_compound_query")
    # Mixed CJK + technical identifiers is common in exact coding queries.
    # Treat it as cross-language ambiguity only when the fast evidence is also
    # weak; this is what keeps a simple exact query off the generative path.
    if has_cjk and has_latin and (
        not matches or top_confidence < config.QUERY_AUTO_CONFIDENCE_THRESHOLD
    ):
        reasons.append("cross_language")
    if environment_ambiguity:
        reasons.append("environment_ambiguity")

    if mode == "deep":
        trigger_reasons = ["explicit_deep", *reasons]
        selected = list(VALID_TRANSFORMS)
    elif mode == "auto":
        trigger_reasons = reasons
        selected = _select_auto_transforms(reasons)
    else:
        trigger_reasons = []
        selected = []

    should_transform = bool(
        enabled and selected and mode in {"auto", "deep"}
    )
    if mode == "fast":
        skip_reason = "fast_mode"
    elif not enabled:
        skip_reason = "transformation_disabled"
    elif not selected:
        skip_reason = "simple_high_confidence_query"
    else:
        skip_reason = None

    return {
        "requested_mode": mode,
        "effective_mode": "second_stage" if should_transform else "fast_path",
        "transform_enabled": enabled,
        "should_transform": should_transform,
        "trigger_reasons": trigger_reasons,
        "selected_transforms": selected,
        "skip_reason": skip_reason,
        "features": {
            "query_chars": len(text),
            "compound_markers": len(conjunctions),
            "mixed_language": has_cjk and has_latin,
            "result_count": len(matches),
            "top_confidence": round(top_confidence, 6),
            "score_gap": round(score_gap, 6),
            "environment_ambiguity": environment_ambiguity,
            "strict_filtered": max(0, int(strict_filtered)),
        },
        "thresholds": {
            "confidence": config.QUERY_AUTO_CONFIDENCE_THRESHOLD,
            "score_gap": config.QUERY_AUTO_SCORE_GAP_THRESHOLD,
            "complex_chars": config.QUERY_AUTO_COMPLEX_CHARS,
        },
    }


def _select_auto_transforms(reasons: list[str]) -> list[str]:
    selected: list[str] = []
    if any(reason in reasons for reason in ("cross_language", "environment_ambiguity")):
        selected.append("rewrite")
    if any(reason in reasons for reason in ("long_compound_query", "ambiguous_ranking")):
        selected.append("multi_query")
    if any(reason in reasons for reason in ("zero_results", "low_confidence")):
        selected.append("hyde")
    limit = max(0, min(3, config.QUERY_AUTO_MAX_TRANSFORMS))
    return selected[:limit]


def fuse_rankings(
    vector_matches: list[dict],
    lexical_matches: list[dict],
    *,
    limit: int,
) -> list[dict]:
    """Fuse vector and lexical ranks with normalized weighted RRF."""

    lanes = (
        ("vector", vector_matches, max(0.0, config.HYBRID_VECTOR_WEIGHT)),
        ("lexical", lexical_matches, max(0.0, config.HYBRID_LEXICAL_WEIGHT)),
    )
    active_weight = sum(weight for _, rows, weight in lanes if rows and weight > 0)
    rrf_k = max(1, config.HYBRID_RRF_K)
    normalizer = active_weight / (rrf_k + 1) if active_weight else 1.0
    merged: dict[int, dict] = {}

    for lane, rows, weight in lanes:
        if weight <= 0:
            continue
        previous_value: float | None = None
        tie_rank = 0
        for ordinal, source in enumerate(rows, start=1):
            rank_value = float(
                source.get("similarity", 0.0)
                if lane == "vector"
                else source.get("lexical_score", source.get("similarity", 0.0))
            )
            if previous_value is None or abs(rank_value - previous_value) > 1e-12:
                tie_rank = ordinal
                previous_value = rank_value
            rank = tie_rank
            story_id = int(source["story_id"])
            item = merged.setdefault(story_id, dict(source))
            item.setdefault("source_paths", [])
            item["source_paths"].append({
                "source": lane,
                "rank": rank,
                "score": round(float(source.get("similarity", 0.0)), 8),
            })
            components = item.setdefault("score_components", {})
            components[f"{lane}_rank"] = rank
            if lane == "vector":
                components["vector_similarity"] = float(source.get("similarity", 0.0))
            else:
                components["lexical_score"] = float(source.get("lexical_score", 0.0))
                components["lexical_similarity"] = float(source.get("similarity", 0.0))
            components["rrf_raw"] = components.get("rrf_raw", 0.0) + weight / (
                rrf_k + rank
            )

    fused = []
    for item in merged.values():
        components = item["score_components"]
        rrf_score = min(1.0, components.pop("rrf_raw", 0.0) / normalizer)
        vector_similarity = float(components.get("vector_similarity", 0.0))
        lexical_similarity = float(components.get("lexical_similarity", 0.0))
        semantic_confidence = max(vector_similarity, lexical_similarity * 0.9)
        fusion_score = 0.68 * semantic_confidence + 0.32 * rrf_score
        sources = {path["source"] for path in item["source_paths"]}
        item["retrieval_source"] = (
            "hybrid" if len(sources) > 1 else next(iter(sources))
        )
        item["similarity"] = round(
            vector_similarity if vector_similarity > 0 else lexical_similarity, 4
        )
        item["fusion_score"] = round(min(1.0, fusion_score), 8)
        item["score"] = item["fusion_score"]
        components.update({
            "rrf_score": round(rrf_score, 8),
            "semantic_confidence": round(semantic_confidence, 8),
            "fusion_score": item["fusion_score"],
            "final_score": item["fusion_score"],
        })
        fused.append(item)
    fused.sort(
        key=lambda item: (
            item["fusion_score"],
            item.get("similarity", 0.0),
            -int(item["story_id"]),
        ),
        reverse=True,
    )
    return fused[:max(0, limit)]


def merge_transformed_rankings(
    base_matches: list[dict],
    transformed_rankings: list[dict],
    *,
    mode: str,
    limit: int,
) -> list[dict]:
    """Fuse second-stage ranks into the already usable fast fallback."""

    best: dict[int, dict] = {
        int(item["story_id"]): dict(item) for item in base_matches
    }
    rrf_k = max(1, config.HYBRID_RRF_K)
    contributions: dict[int, float] = {}
    max_contribution = 0.0
    for ranking in transformed_rankings:
        rows = ranking.get("matches", [])
        if rows:
            max_contribution += 1.0 / (rrf_k + 1)
        for rank, source in enumerate(rows, start=1):
            story_id = int(source["story_id"])
            contributions[story_id] = contributions.get(story_id, 0.0) + 1.0 / (
                rrf_k + rank
            )
            if story_id not in best:
                best[story_id] = dict(source)
            item = best[story_id]
            item.setdefault("source_paths", []).append({
                "source": "transformed_query",
                "transform": ranking.get("transform"),
                "query_index": ranking.get("query_index", 0),
                "rank": rank,
                "score": round(float(source.get("score", 0.0)), 8),
            })

    transform_weight = (
        config.HYBRID_TRANSFORM_WEIGHT_DEEP
        if mode == "deep" else config.HYBRID_TRANSFORM_WEIGHT_AUTO
    )
    transform_weight = max(0.0, min(1.0, transform_weight))
    for story_id, item in best.items():
        base_score = float(item.get("fusion_score", item.get("score", 0.0)))
        transform_rrf = (
            contributions.get(story_id, 0.0) / max_contribution
            if max_contribution else 0.0
        )
        transformed_score = max(
            (
                float(path.get("score", 0.0))
                for path in item.get("source_paths", [])
                if path.get("source") == "transformed_query"
            ),
            default=0.0,
        )
        transform_signal = 0.6 * transform_rrf + 0.4 * transformed_score
        combined = 1.0 - (1.0 - base_score) * (
            1.0 - transform_weight * transform_signal
        )
        item["fusion_score"] = round(max(0.0, min(1.0, combined)), 8)
        item["score"] = item["fusion_score"]
        if contributions.get(story_id):
            item["retrieval_source"] = (
                "hybrid_transform"
                if story_id in {int(row["story_id"]) for row in base_matches}
                else "transformed_query"
            )
        components = item.setdefault("score_components", {})
        components.update({
            "transform_rrf_score": round(transform_rrf, 8),
            "transform_signal": round(transform_signal, 8),
            "fusion_score": item["fusion_score"],
            "final_score": item["fusion_score"],
        })

    ranked = sorted(
        best.values(),
        key=lambda item: (
            item["fusion_score"], item.get("similarity", 0.0),
            -int(item["story_id"]),
        ),
        reverse=True,
    )
    return ranked[:max(0, limit)]


def call_with_deadline(
    callback: Callable[[], T], timeout_seconds: float
) -> tuple[str, T | None]:
    """Run an uncontrollable local/model dependency behind a hard deadline."""

    output: queue.Queue[tuple[str, T | None]] = queue.Queue(maxsize=1)

    def run() -> None:
        try:
            output.put_nowait(("ok", callback()))
        except Exception:  # noqa: BLE001 -- callers expose only stable reasons
            output.put_nowait(("error", None))

    worker = threading.Thread(
        target=run, name="storybook-adaptive-deadline", daemon=True
    )
    worker.start()
    worker.join(max(0.001, timeout_seconds))
    if worker.is_alive():
        return "timeout", None
    try:
        return output.get_nowait()
    except queue.Empty:
        return "error", None


_RERANK_LOCK = threading.Lock()
_RERANK_FAILURES = 0
_RERANK_OPEN_UNTIL = 0.0


def reset_reranker_circuit() -> None:
    """Reset process-local breaker state (also useful for isolated tests)."""

    global _RERANK_FAILURES, _RERANK_OPEN_UNTIL
    with _RERANK_LOCK:
        _RERANK_FAILURES = 0
        _RERANK_OPEN_UNTIL = 0.0


def rerank(
    query: str,
    matches: list[dict],
    *,
    enabled: bool | None = None,
    top_n: int | None = None,
    timeout_seconds: float | None = None,
) -> tuple[list[dict], dict]:
    """Rerank a bounded prefix; failures always return the original ranking."""

    use_reranker = config.RERANK_ENABLED if enabled is None else bool(enabled)
    bounded_top_n = max(0, min(
        len(matches), config.RERANK_TOP_N if top_n is None else int(top_n)
    ))
    timeout = max(
        0.001,
        config.RERANK_TIMEOUT_SECONDS
        if timeout_seconds is None else float(timeout_seconds),
    )
    info = {
        "enabled": use_reranker,
        "status": "disabled" if not use_reranker else "not_run",
        "top_n": bounded_top_n,
        "timeout_ms": round(timeout * 1000.0, 3),
        "degraded_reason": None,
    }
    if not use_reranker or bounded_top_n == 0:
        return matches, info

    now = time.monotonic()
    with _RERANK_LOCK:
        circuit_open = now < _RERANK_OPEN_UNTIL
    if circuit_open:
        info.update({
            "status": "circuit_open",
            "degraded_reason": "reranker_circuit_open",
        })
        return matches, info

    started = time.perf_counter()
    status, reranked_head = call_with_deadline(
        lambda: _local_rerank(query, matches[:bounded_top_n]), timeout
    )
    info["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
    if status != "ok" or reranked_head is None:
        reason = "reranker_timeout" if status == "timeout" else "reranker_unavailable"
        _record_rerank_failure()
        info.update({"status": status, "degraded_reason": reason})
        return matches, info

    global _RERANK_FAILURES
    with _RERANK_LOCK:
        _RERANK_FAILURES = 0
    info["status"] = "ok"
    return [*reranked_head, *matches[bounded_top_n:]], info


def _record_rerank_failure() -> None:
    global _RERANK_FAILURES, _RERANK_OPEN_UNTIL
    with _RERANK_LOCK:
        _RERANK_FAILURES += 1
        if _RERANK_FAILURES >= max(1, config.RERANK_FAILURE_THRESHOLD):
            _RERANK_OPEN_UNTIL = (
                time.monotonic() + max(0.0, config.RERANK_CIRCUIT_COOLDOWN_SECONDS)
            )


def _local_rerank(query: str, matches: list[dict]) -> list[dict]:
    """Deterministic local relevance pass with no model or network calls."""

    query_terms = _terms(_rerank_query_focus(query))
    prepared = []
    document_frequency: dict[str, int] = {}
    for original in matches:
        document = " ".join((
            str(original.get("title") or ""),
            str(original.get("abstract") or ""),
            str(original.get("_rerank_text") or original.get("content") or ""),
            " ".join(str(keyword) for keyword in original.get("keywords", [])),
        ))
        document_terms = _terms(document)
        prepared.append((original, document_terms))
        for term in query_terms & document_terms:
            document_frequency[term] = document_frequency.get(term, 0) + 1
    population = max(1, len(matches))
    term_weights = {
        term: 1.0 + math.log((population + 1) / (document_frequency.get(term, 0) + 1))
        for term in query_terms
    }
    total_query_weight = sum(term_weights.values())

    ranked = []
    for position, (original, document_terms) in enumerate(prepared):
        item = dict(original)
        item.pop("_rerank_text", None)
        overlap = (
            sum(term_weights[term] for term in query_terms & document_terms)
            / total_query_weight
            if total_query_weight else 0.0
        )
        base_score = float(item.get("score", item.get("similarity", 0.0)))
        rerank_score = 0.5 * base_score + 0.5 * overlap
        item["score"] = round(max(0.0, min(1.0, rerank_score)), 8)
        components = item.setdefault("score_components", {})
        components.update({
            "pre_rerank_score": round(base_score, 8),
            "rerank_overlap": round(overlap, 8),
            "rerank_score": item["score"],
            "final_score": item["score"],
        })
        ranked.append((item, position))
    ranked.sort(
        key=lambda pair: (pair[0]["score"], -pair[1]), reverse=True
    )
    return [item for item, _ in ranked]


def _rerank_query_focus(query: str) -> str:
    """Drop explicit recall boilerplate while retaining the remembered clue."""

    text = normalize_query(query)
    prefix, separator, suffix = text.rpartition("：")
    if not separator:
        prefix, separator, suffix = text.rpartition(":")
    recall_markers = ("只记得", "记不清", "结果或指标", "remember only")
    if separator and suffix and any(
        marker in prefix.casefold() for marker in recall_markers
    ):
        return suffix.strip()
    return text


def _terms(text: str) -> set[str]:
    lowered = str(text or "").casefold()
    terms = set(re.findall(r"[a-z0-9_]+", lowered))
    for token in re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]+", lowered):
        if len(token) <= 2:
            terms.add(token)
        else:
            terms.update(token[index:index + 2] for index in range(len(token) - 1))
    return terms


def _candidate_confidence(item: dict[str, Any]) -> float:
    components = item.get("score_components") or {}
    if "vector_similarity" in components:
        return float(components["vector_similarity"])
    if "lexical_similarity" in components:
        return float(components["lexical_similarity"])
    return float(item.get("similarity", 0.0))
