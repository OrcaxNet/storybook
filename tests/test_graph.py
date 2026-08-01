"""Memory Graph schema, traversal, budgets and concurrent feedback tests."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest

from storybook import config, feedback, search as search_module, store
from ._helpers import basis


def _seed(title: str, vector: list[float]) -> int:
    return store.add_story(title, f"{title} detail", [title], vector)


class TestTypedEdges:
    def test_same_pair_supports_multiple_types_with_direction_and_provenance(self):
        first = _seed("first", basis(0))
        second = _seed("second", basis(1))

        semantic_id = store.add_or_update_edge(
            second, first, 0.6, "semantic",
            provenance={"source": "test", "evidence": "semantic-match"},
        )
        causal_id = store.add_or_update_edge(
            first, second, 0.8, "causal",
            provenance={"source": "test", "evidence": "observed-outcome"},
        )

        edges = {edge["edge_type"]: edge for edge in store.get_edges(first)}
        assert set(edges) == {"semantic", "causal"}
        assert edges["semantic"]["id"] == semantic_id
        assert edges["semantic"]["directed"] is False
        assert (edges["semantic"]["source_id"], edges["semantic"]["target_id"]) == (
            first, second,
        )
        assert edges["causal"]["id"] == causal_id
        assert edges["causal"]["directed"] is True
        assert edges["causal"]["direction"] == "outbound"
        assert edges["causal"]["provenance"]["evidence"] == "observed-outcome"

    def test_edge_update_versions_clamps_and_soft_delete_is_auditable(self):
        first = _seed("first", basis(0))
        second = _seed("second", basis(1))
        store.add_or_update_edge(first, second, 0.4, "temporal")

        before = store.get_edges(first)[0]
        store.add_or_update_edge(
            first, second, 2.0, "temporal",
            provenance={"source": "timeline", "evidence": "session-order"},
        )
        updated = store.get_edges(first)[0]
        assert updated["weight"] == config.WEIGHT_MAX
        assert updated["version"] == before["version"] + 1
        assert updated["provenance"]["evidence"] == "session-order"

        assert store.delete_edge(first, second, "temporal") == 1
        assert store.get_edges(first) == []
        deleted = store.get_edges(first, include_deleted=True)[0]
        assert deleted["deleted_at"] is not None
        assert deleted["version"] == updated["version"] + 1

    @pytest.mark.parametrize(
        ("edge_type", "directed"),
        [
            ("semantic", False),
            ("temporal", True),
            ("causal", True),
            ("same_environment", False),
            ("parent_child", True),
            ("co_recall", False),
            ("supersedes", True),
        ],
    )
    def test_standard_edge_direction_contract(self, edge_type, directed):
        first = _seed("first", basis(0))
        second = _seed("second", basis(1))

        store.add_or_update_edge(first, second, 0.5, edge_type)

        assert store.get_edges(first)[0]["directed"] is directed


class TestGraphRecall:
    def test_one_hop_causal_candidate_has_complete_explanation(self, fake_embedder):
        seed = _seed("seed", basis(0))
        outcome = _seed("outcome", basis(5))
        store.add_or_update_edge(
            seed, outcome, 0.9, "causal",
            provenance={"source": "episode", "evidence": "command-result"},
        )
        fake_embedder.register("q", basis(0))

        result = search_module.search("q", top_k=3)

        expanded = next(
            item for item in result["top_matches"] if item["story_id"] == outcome
        )
        assert expanded["retrieval_source"] == "graph"
        assert expanded["seed_story_id"] == seed
        assert expanded["graph_path"][0]["edge_type"] == "causal"
        assert expanded["graph_path"][0]["provenance"]["evidence"] == "command-result"
        assert expanded["score_components"]["final_score"] == expanded["score"]

    def test_multi_hop_cycle_is_bounded_and_results_are_deduplicated(
        self, fake_embedder, monkeypatch
    ):
        first = _seed("first", basis(0))
        second = _seed("second", basis(5))
        third = _seed("third", basis(6))
        store.add_or_update_edge(first, second, 0.95, "causal")
        store.add_or_update_edge(second, third, 0.95, "causal")
        store.add_or_update_edge(third, first, 0.95, "causal")
        fake_embedder.register("q", basis(0))
        monkeypatch.setattr(config, "GRAPH_MAX_HOPS", 3)

        result = search_module.search("q", top_k=5)

        ids = [item["story_id"] for item in result["top_matches"]]
        assert ids.count(first) == ids.count(second) == ids.count(third) == 1
        third_match = next(item for item in result["top_matches"] if item["story_id"] == third)
        assert len(third_match["graph_path"]) == 2
        assert result["graph_trace"]["cycles_suppressed"] >= 1
        assert result["graph_trace"]["paths_considered"] <= config.GRAPH_MAX_PATHS

    def test_supersedes_replaces_old_seed_by_default(self, fake_embedder):
        old = _seed("old", basis(0))
        replacement = _seed("replacement", basis(7))
        store.add_or_update_edge(
            replacement, old, 1.0, "supersedes",
            provenance={"source": "story_update", "version": 2},
        )
        fake_embedder.register("q", basis(0))

        result = search_module.search("q", top_k=3)

        assert [item["story_id"] for item in result["top_matches"]] == [replacement]
        path = result["top_matches"][0]["graph_path"][0]
        assert path["traversal"] == "inbound"
        assert result["graph_trace"]["superseded_suppressed"] == 1

    def test_graph_can_be_disabled_for_direct_retrieval_fallback(self, fake_embedder):
        seed = _seed("seed", basis(0))
        related = _seed("related", basis(5))
        store.add_or_update_edge(seed, related, 1.0, "causal")
        fake_embedder.register("q", basis(0))

        result = search_module.search("q", top_k=3, graph_enabled=False)

        assert [item["story_id"] for item in result["top_matches"]] == [seed]
        assert result["graph_enabled"] is False
        assert result["graph_trace"]["status"] == "disabled"

    def test_zero_time_budget_safely_truncates_to_direct_results(
        self, fake_embedder, monkeypatch
    ):
        seed = _seed("seed", basis(0))
        related = _seed("related", basis(5))
        store.add_or_update_edge(seed, related, 1.0, "causal")
        fake_embedder.register("q", basis(0))
        monkeypatch.setattr(config, "GRAPH_TIME_BUDGET_MS", 0)

        result = search_module.search("q", top_k=3)

        assert [item["story_id"] for item in result["top_matches"]] == [seed]
        assert result["truncated"] is True
        assert "time_budget" in result["truncated_reasons"]

    def test_hub_penalty_prevents_popularity_from_beating_causal_path(
        self, fake_embedder
    ):
        seed = _seed("seed", basis(0))
        causal = _seed("causal", basis(5))
        hub = _seed("hub", basis(6))
        store.add_or_update_edge(seed, causal, 0.9, "causal")
        store.add_or_update_edge(seed, hub, 0.99, "semantic")
        for index in range(20):
            leaf = _seed(f"leaf-{index}", basis(20 + index))
            store.add_or_update_edge(hub, leaf, 0.9, "semantic")
        fake_embedder.register("q", basis(0))

        result = search_module.search("q", top_k=3)

        ids = [item["story_id"] for item in result["top_matches"]]
        assert causal in ids
        assert hub not in ids

    def test_co_recall_is_terminal_and_does_not_create_noisy_multihop_path(
        self, fake_embedder
    ):
        seed = _seed("seed", basis(0))
        co_recalled = _seed("co-recalled", basis(5))
        remote = _seed("remote", basis(6))
        store.add_or_update_edge(seed, co_recalled, 1.0, "co_recall")
        store.add_or_update_edge(co_recalled, remote, 1.0, "causal")
        fake_embedder.register("q", basis(0))

        result = search_module.search("q", top_k=5)

        ids = [item["story_id"] for item in result["top_matches"]]
        assert co_recalled in ids
        assert remote not in ids
        assert result["graph_trace"]["path_policy_suppressed"] >= 1


class TestCoRecallLifecycle:
    def test_feedback_creates_reinforces_caps_and_decays_co_recall_edge(self):
        first = _seed("first", basis(0))
        second = _seed("second", basis(1))

        for _ in range(20):
            store.apply_recall_feedback([first, second])
        co_recall = next(
            edge for edge in store.get_edges(first)
            if edge["edge_type"] == "co_recall"
        )
        assert co_recall["weight"] == config.WEIGHT_MAX
        assert co_recall["observations"] == 20

        old = datetime.now(UTC) - timedelta(days=2)
        db = store.get_db(load_vector_extension=False)
        try:
            db.execute(
                "UPDATE edges SET last_reinforced_at = ? WHERE id = ?",
                (old.isoformat(), co_recall["id"]),
            )
            db.commit()
        finally:
            db.close()
        report = store.decay_co_recall_edges(
            now=datetime.now(UTC), half_life_days=1, min_weight=0.3
        )

        assert report["deleted"] == 1
        assert not any(
            edge["edge_type"] == "co_recall" for edge in store.get_edges(first)
        )

    def test_concurrent_graph_reads_and_feedback_writes_remain_available(
        self, fake_embedder
    ):
        seed = _seed("seed", basis(0))
        related = _seed("related", basis(5))
        store.add_or_update_edge(seed, related, 0.9, "causal")
        fake_embedder.register("q", basis(0))

        with ThreadPoolExecutor(max_workers=5) as pool:
            results = list(pool.map(
                lambda _: search_module.search("q", top_k=3), range(20)
            ))

        assert all(
            related in [item["story_id"] for item in result["top_matches"]]
            for result in results
        )
        assert feedback.flush_feedback(timeout=5.0)
        assert store.get_story(seed)["access_count"] >= 1
