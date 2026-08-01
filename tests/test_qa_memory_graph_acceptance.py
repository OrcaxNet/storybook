"""Independent QA acceptance probes for FLO-153."""
from __future__ import annotations

import pytest

from storybook import config, search as search_module, store
from ._helpers import basis


def _story(title: str, dimension: int) -> int:
    return store.add_story(title, f"{title} detail", [title], basis(dimension))


def test_inbound_causal_edges_do_not_expand_against_declared_direction(
    fake_embedder,
):
    cause = _story("cause", 5)
    outcome = _story("outcome", 0)
    store.add_or_update_edge(cause, outcome, 1.0, "causal")
    fake_embedder.register("q", basis(0))

    result = search_module.search("q", top_k=3)

    assert [item["story_id"] for item in result["top_matches"]] == [outcome]
    assert result["graph_trace"]["path_policy_suppressed"] >= 1


def test_fanout_is_applied_after_direction_policy_so_valid_path_is_not_starved(
    fake_embedder, monkeypatch
):
    seed = _story("seed", 0)
    relevant = _story("relevant", 40)
    # Invalid inbound causal edges are intentionally stronger. They must not
    # consume the entire traversal fan-out before direction policy is applied.
    for index in range(8):
        source = _story(f"inbound-{index}", index + 1)
        store.add_or_update_edge(source, seed, 1.0, "causal")
    store.add_or_update_edge(seed, relevant, 0.9, "causal")
    fake_embedder.register("q", basis(0))
    monkeypatch.setattr(config, "GRAPH_FAN_OUT", 8)

    result = search_module.search("q", top_k=3)

    assert relevant in [item["story_id"] for item in result["top_matches"]]


def test_token_budget_exhaustion_is_truncated_and_keeps_direct_result(
    fake_embedder, monkeypatch
):
    seed = _story("seed", 0)
    related = _story("related", 5)
    store.add_or_update_edge(seed, related, 1.0, "causal")
    fake_embedder.register("q", basis(0))
    monkeypatch.setattr(config, "GRAPH_TOKEN_BUDGET", 0)

    result = search_module.search("q", top_k=3)

    assert [item["story_id"] for item in result["top_matches"]] == [seed]
    assert result["truncated"] is True
    assert "token_budget" in result["truncated_reasons"]


@pytest.mark.parametrize(
    ("setting", "reason"),
    [
        ("GRAPH_MAX_HOPS", "hop_budget"),
        ("GRAPH_MAX_PATHS", "path_budget"),
        ("GRAPH_FAN_OUT", "fan_out"),
    ],
)
def test_zero_graph_budget_is_reported_as_truncated(
    fake_embedder, monkeypatch, setting, reason
):
    seed = _story("seed", 0)
    related = _story("related", 5)
    store.add_or_update_edge(seed, related, 1.0, "causal")
    fake_embedder.register("q", basis(0))
    monkeypatch.setattr(config, setting, 0)

    result = search_module.search("q", top_k=3)

    assert [item["story_id"] for item in result["top_matches"]] == [seed]
    assert result["truncated"] is True
    assert reason in result["truncated_reasons"]


def test_parent_child_reverse_path_retains_full_explanation(fake_embedder):
    parent = _story("parent", 6)
    child = _story("child", 0)
    store.add_or_update_edge(
        parent,
        child,
        1.0,
        "parent_child",
        provenance={"source": "qa", "evidence": "parent-id"},
    )
    fake_embedder.register("q", basis(0))

    result = search_module.search("q", top_k=3)
    match = next(item for item in result["top_matches"] if item["story_id"] == parent)
    step = match["graph_path"][0]

    assert step["traversal"] == "inbound"
    assert step["provenance"]["evidence"] == "parent-id"
    assert match["seed_story_id"] == child
    assert match["score_components"]["final_score"] == match["score"]
