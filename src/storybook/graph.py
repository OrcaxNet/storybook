"""Budgeted, explainable graph expansion for experiential memories.

Vector/lexical retrieval supplies trustworthy seed Stories.  This module walks
the typed Memory Graph without invoking a generative model, and returns only
bounded candidates plus the complete edge path that produced each score.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from . import config, store


@dataclass(frozen=True)
class _State:
    node_id: int
    seed_id: int
    seed_similarity: float
    score: float
    edge_product: float
    hub_penalty: float
    path: tuple[dict, ...]
    visited: tuple[int, ...]


def expand(
    seeds: list[dict],
    *,
    max_hops: int | None = None,
    max_paths: int | None = None,
    fan_out: int | None = None,
    time_budget_ms: float | None = None,
    token_budget: int | None = None,
) -> dict:
    """Expand direct retrieval seeds under independent graph budgets.

    ``matches`` contains graph-only candidates.  Callers merge these with the
    direct lane and apply environment-aware final ranking.  A candidate appears
    once, using its best-scoring path across all seeds.
    """

    max_hops = max(0, min(int(
        config.GRAPH_MAX_HOPS if max_hops is None else max_hops
    ), 8))
    max_paths = max(0, min(int(
        config.GRAPH_MAX_PATHS if max_paths is None else max_paths
    ), 10_000))
    fan_out = max(0, min(int(
        config.GRAPH_FAN_OUT if fan_out is None else fan_out
    ), 256))
    time_budget_ms = max(0.0, float(
        config.GRAPH_TIME_BUDGET_MS
        if time_budget_ms is None else time_budget_ms
    ))
    token_budget = max(0, int(
        config.GRAPH_TOKEN_BUDGET if token_budget is None else token_budget
    ))
    started = time.perf_counter()
    deadline = started + time_budget_ms / 1000.0
    seed_ids = {int(seed["story_id"]) for seed in seeds}
    frontier = [
        _State(
            node_id=int(seed["story_id"]),
            seed_id=int(seed["story_id"]),
            seed_similarity=_clamp(seed.get("score", seed.get("similarity", 0.0))),
            score=_clamp(seed.get("score", seed.get("similarity", 0.0))),
            edge_product=1.0,
            hub_penalty=1.0,
            path=(),
            visited=(int(seed["story_id"]),),
        )
        for seed in seeds
    ]
    candidates: dict[int, dict] = {}
    best_state: dict[tuple[int, int], float] = {
        (state.seed_id, state.node_id): state.score for state in frontier
    }
    reasons: set[str] = set()
    path_count = 0
    token_used = 0
    cycles_suppressed = 0
    path_policy_suppressed = 0

    if not seeds or not max_hops or not max_paths or not fan_out:
        return _result(
            candidates, seed_ids, started, reasons,
            path_count=0, token_used=0, cycles_suppressed=0,
            path_policy_suppressed=0, budgets={
                "max_hops": max_hops,
                "max_paths": max_paths,
                "fan_out": fan_out,
                "time_ms": time_budget_ms,
                "tokens": token_budget,
            },
        )

    stop = False
    for hop in range(1, max_hops + 1):
        if time.perf_counter() >= deadline:
            reasons.add("time_budget")
            break
        adjacency = store.get_graph_neighbors_batch(
            [state.node_id for state in frontier], fan_out=fan_out
        )
        if any(len(items) >= fan_out for items in adjacency.values()):
            reasons.add("fan_out")
        if time.perf_counter() >= deadline:
            reasons.add("time_budget")
            break
        next_frontier: list[_State] = []
        for state in frontier:
            previous_type = state.path[-1]["edge_type"] if state.path else None
            for neighbor in adjacency.get(state.node_id, []):
                if time.perf_counter() >= deadline:
                    reasons.add("time_budget")
                    stop = True
                    break
                if path_count >= max_paths:
                    reasons.add("path_budget")
                    stop = True
                    break
                edge = neighbor["edge"]
                if not _can_traverse(edge):
                    path_policy_suppressed += 1
                    continue
                if not _allowed_transition(previous_type, edge["edge_type"], hop):
                    path_policy_suppressed += 1
                    continue
                neighbor_id = int(neighbor["story_id"])
                if neighbor_id in state.visited:
                    cycles_suppressed += 1
                    continue

                type_factor = _clamp(
                    config.GRAPH_EDGE_TYPE_FACTORS.get(edge["edge_type"], 0.5)
                )
                direction_factor = _direction_factor(edge)
                effective_weight = _clamp(edge["weight"]) * type_factor * direction_factor
                edge_product = state.edge_product * effective_weight
                degree = max(1, int(neighbor.get("degree") or 1))
                node_hub_penalty = 1.0 / math.sqrt(max(1.0, degree / max(1, fan_out)))
                hub_penalty = state.hub_penalty * node_hub_penalty
                hop_decay = _clamp(config.GRAPH_HOP_DECAY) ** max(0, hop - 1)
                score = state.seed_similarity * edge_product * hub_penalty * hop_decay
                path_count += 1
                if score < max(0.0, float(config.GRAPH_MIN_SCORE)):
                    continue
                state_key = (state.seed_id, neighbor_id)
                if score <= best_state.get(state_key, -1.0):
                    continue
                best_state[state_key] = score

                step = {
                    "from_story_id": state.node_id,
                    "to_story_id": neighbor_id,
                    "edge_id": edge["id"],
                    "edge_global_id": edge["global_id"],
                    "edge_type": edge["edge_type"],
                    "source_id": edge["source_id"],
                    "target_id": edge["target_id"],
                    "directed": edge["directed"],
                    "traversal": edge["traversal"],
                    "weight": round(float(edge["weight"]), 8),
                    "type_factor": round(type_factor, 8),
                    "direction_factor": round(direction_factor, 8),
                    "effective_weight": round(effective_weight, 8),
                    "hub_degree": degree,
                    "hub_penalty": round(node_hub_penalty, 8),
                    "version": edge["version"],
                    "provenance": edge["provenance"],
                }
                path = (*state.path, step)
                next_state = _State(
                    node_id=neighbor_id,
                    seed_id=state.seed_id,
                    seed_similarity=state.seed_similarity,
                    score=score,
                    edge_product=edge_product,
                    hub_penalty=hub_penalty,
                    path=path,
                    visited=(*state.visited, neighbor_id),
                )
                next_frontier.append(next_state)

                if neighbor_id in seed_ids:
                    continue
                candidate = {
                    key: value for key, value in neighbor.items()
                    if key not in {"edge", "degree"}
                }
                candidate.update({
                    "similarity": round(score, 4),
                    "score": round(score, 8),
                    "graph_score": round(score, 8),
                    "retrieval_source": "graph",
                    "seed_story_id": state.seed_id,
                    "graph_path": list(path),
                    "score_components": {
                        "seed_similarity": round(state.seed_similarity, 8),
                        "edge_product": round(edge_product, 8),
                        "hop_decay": round(hop_decay, 8),
                        "hub_penalty": round(hub_penalty, 8),
                        "graph_score": round(score, 8),
                    },
                })
                existing = candidates.get(neighbor_id)
                if existing is not None and existing["graph_score"] >= score:
                    continue
                if existing is None:
                    candidate_tokens = _estimate_candidate_tokens(candidate)
                    if token_used + candidate_tokens > token_budget:
                        reasons.add("token_budget")
                        stop = True
                        break
                    token_used += candidate_tokens
                candidates[neighbor_id] = candidate
            if stop:
                break
        if hop == max_hops and next_frontier:
            reasons.add("hop_budget")
        if stop or not next_frontier:
            break
        # Strong paths are expanded first on the next hop.  The global path
        # budget still bounds duplicate seed/node states.
        frontier = sorted(next_frontier, key=lambda state: state.score, reverse=True)

    return _result(
        candidates, seed_ids, started, reasons,
        path_count=path_count,
        token_used=token_used,
        cycles_suppressed=cycles_suppressed,
        path_policy_suppressed=path_policy_suppressed,
        budgets={
            "max_hops": max_hops,
            "max_paths": max_paths,
            "fan_out": fan_out,
            "time_ms": time_budget_ms,
            "tokens": token_budget,
        },
    )


def _result(
    candidates: dict[int, dict],
    seed_ids: set[int],
    started: float,
    reasons: set[str],
    *,
    path_count: int,
    token_used: int,
    cycles_suppressed: int,
    path_policy_suppressed: int,
    budgets: dict,
) -> dict:
    all_ids = [*seed_ids, *candidates]
    superseded = store.get_superseded_story_ids(all_ids)
    matches = [
        candidate for story_id, candidate in candidates.items()
        if story_id not in superseded
    ]
    matches.sort(key=lambda item: item["graph_score"], reverse=True)
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
    return {
        "matches": matches,
        "suppressed_story_ids": sorted(superseded),
        "truncated": bool(reasons),
        "truncated_reasons": sorted(reasons),
        "trace": {
            "seed_story_ids": sorted(seed_ids),
            "expanded_candidates": len(matches),
            "paths_considered": path_count,
            "tokens_used": token_used,
            "cycles_suppressed": cycles_suppressed,
            "path_policy_suppressed": path_policy_suppressed,
            "superseded_suppressed": len(superseded),
            "elapsed_ms": elapsed_ms,
            "budgets": budgets,
        },
    }


def _can_traverse(edge: dict) -> bool:
    """Apply direction semantics without erasing the recorded direction.

    A supersedes edge is stored new→old but recall traverses old→new so an old
    seed can be replaced.  Parent/child is useful in both hierarchy directions;
    causal and temporal paths only follow their declared direction.
    """

    if not edge.get("directed"):
        return True
    edge_type = edge.get("edge_type")
    traversal = edge.get("traversal")
    if edge_type == "supersedes":
        return traversal == "inbound"
    if edge_type == "parent_child":
        return traversal in {"outbound", "inbound"}
    return traversal == "outbound"


def _allowed_transition(
    previous_type: str | None, edge_type: str, hop: int
) -> bool:
    # Learned co-occurrence and legacy sibling edges are deliberately terminal:
    # chaining them turns popular hubs into ranking shortcuts.
    if previous_type in {"co_recall", "sibling"}:
        return False
    if hop > 1 and edge_type in {"co_recall", "sibling"}:
        return False
    if previous_type == edge_type == "same_environment":
        return False
    if previous_type == edge_type == "supersedes":
        return False
    return True


def _direction_factor(edge: dict) -> float:
    if not edge.get("directed") or edge.get("traversal") == "outbound":
        return 1.0
    if edge.get("edge_type") == "supersedes":
        return 1.0
    # Reverse hierarchy traversal is useful but slightly weaker than the
    # declared parent→child direction.
    return 0.9


def _estimate_candidate_tokens(candidate: dict) -> int:
    text = " ".join((candidate.get("title") or "", candidate.get("content") or ""))
    cjk = sum("\u3400" <= char <= "\u9fff" for char in text)
    other = len(text) - cjk
    # Include a conservative fixed allowance for path/provenance JSON.
    return max(1, cjk + (other + 3) // 4 + 80 * len(candidate["graph_path"]))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
