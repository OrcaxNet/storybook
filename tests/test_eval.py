"""评测体系（storybook.eval）自测：全程 mock ``embeddings.embed``，无需 Ollama。

覆盖：
  * 指标纯函数（recall@k / precision@k / MRR / merge_branch_accuracy / threshold_sweep）
  * 检索评测：确定性向量构造已知排序，验证 recall@k 计算、阈值过滤、达标判定、阈值曲线
  * 加工分支评测：duplicate/near_identical 走 merge/update、distinct 走 create
  * 分裂评测：合并区间相似度触发分裂，结构校验全通过

这是 eval 作为「慢测试/独立命令」与 Stage 1 测试套件的集成入口。
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from storybook import config
from storybook import eval as eval_module
from storybook.eval import runner as runner_module
from storybook.eval import metrics as M
from ._helpers import basis

DIM = config.EMBED_DIM


def _cos_vec(target_idx: int, cos: float, residual_idx: int = 100) -> list[float]:
    """单位向量：与 ``basis(target_idx)`` 余弦相似度 == cos，残差放在无 story 占用的 dim。"""
    cos = float(np.clip(cos, -1.0, 1.0))
    v = np.zeros(DIM, dtype=np.float32)
    v[target_idx] = cos
    v[residual_idx] = float(np.sqrt(max(0.0, 1.0 - cos * cos)))
    return v.tolist()


# ═══════════════════════════════════════════════
#  指标纯函数
# ═══════════════════════════════════════════════

class TestMetrics:
    def test_recall_at_k(self):
        ranked = [10, 20, 30, 40, 50]
        assert M.recall_at_k(ranked, [20], 1) == 0.0
        assert M.recall_at_k(ranked, [20], 3) == 1.0
        assert M.recall_at_k(ranked, [20], 5) == 1.0
        # 多 relevant
        assert M.recall_at_k(ranked, [20, 40], 3) == 0.5
        assert M.recall_at_k(ranked, [99], 5) == 0.0
        assert M.recall_at_k(ranked, [], 3) == 0.0

    def test_precision_at_k(self):
        ranked = [10, 20, 30, 40, 50]
        assert M.precision_at_k(ranked, [20], 1) == 0.0
        assert M.precision_at_k(ranked, [20], 3) == pytest.approx(1 / 3)
        assert M.precision_at_k(ranked, [20, 40], 5) == pytest.approx(2 / 5)

    def test_mrr(self):
        assert M.mrr([10, 20, 30], [20]) == 0.5
        assert M.mrr([10, 20, 30], [10]) == 1.0
        assert M.mrr([10, 20, 30], [99]) == 0.0

    def test_merge_branch_accuracy(self):
        results = [
            {"id": "a", "expected_branch": "merge_or_update", "actual_branch": "update"},
            {"id": "b", "expected_branch": "merge_or_update", "actual_branch": "merge"},
            {"id": "c", "expected_branch": "merge_or_update", "actual_branch": "create"},  # ❌
            {"id": "d", "expected_branch": "create", "actual_branch": "create"},
            {"id": "e", "expected_branch": "update", "actual_branch": "update"},
        ]
        acc = M.merge_branch_accuracy(results)
        assert acc["correct"] == 4
        assert acc["total"] == 5
        assert acc["accuracy"] == 0.8
        assert acc["by_expected"]["merge_or_update"]["correct"] == 2
        assert acc["by_expected"]["create"]["correct"] == 1
        assert acc["by_expected"]["update"]["correct"] == 1

    def test_threshold_sweep(self):
        def fn(th):
            return {"recall@3": 1.0 if th <= 0.5 else 0.0}
        curve = M.threshold_sweep(fn, [0.3, 0.5, 0.7], metric_key="recall@3")
        assert [c["threshold"] for c in curve] == [0.3, 0.5, 0.7]
        assert [c["recall@3"] for c in curve] == [1.0, 1.0, 0.0]


# ═══════════════════════════════════════════════
#  检索评测（确定性向量驱动）
# ═══════════════════════════════════════════════

def _register_retrieval_corpus(fake_embedder, cross_lang_sim: float):
    """注册 24 topic 索引向量 + 72 查询向量，构造已知排序。

    exact -> sim 1.0；synonym -> sim 0.80；cross_lang -> sim ``cross_lang_sim``。
    残差放在 dim 100（无 story 占用），保证 target 是唯一非零相似命中。
    """
    bench = eval_module.load_benchmark()
    for i, t in enumerate(bench.topics):
        fake_embedder.register(t.index_text(), basis(i))
        fake_embedder.register(t.queries["exact"], basis(i))
        fake_embedder.register(t.queries["synonym"], _cos_vec(i, 0.80))
        fake_embedder.register(t.queries["cross_lang"], _cos_vec(i, cross_lang_sim))


class TestRetrievalEval:
    def test_all_variants_above_threshold_passes_70(self, fake_embedder):
        _register_retrieval_corpus(fake_embedder, cross_lang_sim=0.60)
        res = eval_module.run_retrieval_eval()
        s = res["summary"]
        assert s["corpus_size"] == 24
        assert s["query_count"] == 72
        assert s["recall@1"] == 1.0
        assert s["recall@3"] == 1.0
        assert s["recall@5"] == 1.0
        assert res["passes_70_percent_recall_at_3"] is True
        # 三变体都命中
        for v in ("exact", "synonym", "cross_lang"):
            assert res["per_variant"][v]["recall@3"] == 1.0
        # 负例特异性：默认 hash 向量与语料正交 -> top_sim < 0.5 -> 全部拒绝
        assert s["specificity"] == 1.0

    def test_cross_lang_below_threshold_filtered_fails_70(self, fake_embedder):
        """cross_lang sim=0.45 < 0.50 阈值 -> 被过滤；整体 recall@3=48/72≈0.667 < 0.70。"""
        _register_retrieval_corpus(fake_embedder, cross_lang_sim=0.45)
        res = eval_module.run_retrieval_eval()
        s = res["summary"]
        assert res["per_variant"]["exact"]["recall@3"] == 1.0
        assert res["per_variant"]["synonym"]["recall@3"] == 1.0
        assert res["per_variant"]["cross_lang"]["recall@3"] == 0.0   # 被阈值过滤
        assert s["recall@3"] == pytest.approx(48 / 72, abs=1e-4)
        assert res["passes_70_percent_recall_at_3"] is False

    def test_threshold_sweep_recovers_filtered(self, fake_embedder):
        """阈值降到 0.40 后，被 0.50 过滤的 cross_lang 重新召回。"""
        _register_retrieval_corpus(fake_embedder, cross_lang_sim=0.45)
        res = eval_module.run_retrieval_eval()
        sweep = {pt["threshold"]: pt for pt in res["threshold_sweep"]}
        assert sweep[0.50]["recall@3"] == pytest.approx(48 / 72, abs=1e-4)   # cross_lang 被滤
        assert sweep[0.40]["recall@3"] == 1.0                                # cross_lang 恢复
        # 阈值越高召回越低（单调非增）
        recall3 = [sweep[th]["recall@3"] for th in sorted(sweep)]
        assert recall3 == sorted(recall3, reverse=True)

    def test_report_format_contains_headline(self, fake_embedder):
        _register_retrieval_corpus(fake_embedder, cross_lang_sim=0.60)
        rep = eval_module.run_all(parts=("retrieval",))
        text = eval_module.format_report(rep)
        assert "recall@3" in text
        assert "达标" in text


class TestEmbeddingAblation:
    def test_reports_all_modes_groups_latency_and_selection_gate(self, fake_embedder):
        result = eval_module.run_embedding_ablation()
        assert set(result["modes"]) == {
            "legacy", "default", "full", "multi_vector"
        }
        assert result["selection_gate"] == "recall@3 >= baseline - 0.02"
        for mode, row in result["modes"].items():
            assert set(row["groups"]) == {
                "exact", "synonym", "cross_tool", "cross_lang"
            }
            assert row["query_count"] == 96
            assert row["latency_ms"]["query_p95"] >= 0
        assert result["modes"]["multi_vector"]["index_vectors_per_story"] == 3

        report = eval_module.EvalReport(ablation=result)
        text = eval_module.format_report(report)
        assert "embedding 表示消融" in text
        assert "cross_tool" in text


class TestRetrievalStrategyAblation:
    def test_pre_generated_artifact_enforces_query_only_contract(self, tmp_path):
        query = "remembered clue"
        artifact = {
            "source": "query_only_pre_generated",
            "generator": "local-test-model",
            "prompt_version": "test-v1",
            "generation_inputs": [
                "raw_query", "requested_transformations", "timeout_seconds"
            ],
            "entries": [{
                "query": query,
                "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                "output": {
                    "rewrite": "safe rewrite",
                    "queries": [],
                    "hypothetical_document": "",
                },
            }],
        }
        path = tmp_path / "transforms.json"
        path.write_text(json.dumps(artifact), encoding="utf-8")

        provider = eval_module.pre_generated_transform_provider(path)

        assert provider(query, [], timeout_seconds=1) == artifact["entries"][0][
            "output"
        ]
        assert provider("missing", [], timeout_seconds=1) is None
        assert provider.evidence_metadata["stats"] == {
            "cache_hits": 1, "cache_misses": 1
        }

        artifact["entries"][0]["topic_id"] = "forbidden-label"
        path.write_text(json.dumps(artifact), encoding="utf-8")
        with pytest.raises(ValueError, match="ground-truth"):
            eval_module.pre_generated_transform_provider(path)

    def test_reports_cumulative_strategies_groups_latency_and_gates(
        self, fake_embedder, fake_llm
    ):
        result = eval_module.run_retrieval_strategy_ablation()

        assert list(result["strategies"]) == [
            "direct_vector",
            "hybrid",
            "hybrid_graph",
            "hybrid_graph_reranker",
            "hybrid_graph_rewrite",
            "hybrid_graph_hyde",
            "hybrid_graph_hyde_reranker",
        ]
        assert result["groups"] == [
            "exact", "synonym", "cross_language", "cross_tool", "ambiguous"
        ]
        assert result["query_count"] == 120
        for row in result["strategies"].values():
            assert set(row["groups"]) == set(result["groups"])
            assert row["latency_ms"]["p95"] >= 0
            assert set(row["gates"]) == {
                "hard_quality_gain", "overall_non_regression", "latency",
                "ground_truth_isolation", "online_latency_evidence",
                "eligible_for_default",
            }
        assert result["selected_default"] in result["strategies"]
        assert result["transformation_provenance"] == {
            "source": "live_generated",
            "generator": config.LLM_MODEL,
            "generation_inputs": [
                "raw_query", "requested_transformations", "timeout_seconds"
            ],
            "ground_truth_fields_used_for_generation": [],
            "ground_truth_usage": "topic_id is used only for post-ranking scoring",
            "pre_generated_outputs": False,
            "oracle_upper_bound": False,
            "oracle_eligible_for_default": False,
            "online_latency_evidence": True,
            "prompt_version": None,
            "artifact_sha256": None,
            "artifact_entry_count": None,
            "artifact_cache_hits": None,
            "artifact_cache_misses": None,
            "attempts": 120,
            "successes": 0,
            "failures": 120,
            "generated_counts": {"rewrite": 0, "multi_query": 0, "hyde": 0},
        }

        report = eval_module.EvalReport(strategy=result)
        text = eval_module.format_report(report)
        assert "自适应检索策略消融" in text
        assert "hybrid_graph_hyde_reranker" in text
        assert "ambiguous" in text
        assert "live_generated" in text

    def test_transform_retrieval_never_reads_target_story_fields(
        self, fake_embedder, monkeypatch
    ):
        bench = eval_module.load_benchmark()
        target = bench.topics[0]
        raw_query = "raw query without target labels"
        monkeypatch.setattr(
            runner_module,
            "_strategy_query_pairs",
            lambda _bench: [{
                "query": raw_query,
                "variant": "ambiguous",
                "topic_id": target.id,
            }],
        )
        provider_calls = []

        def provider(query, transformations, **kwargs):
            provider_calls.append((query, transformations, kwargs))
            return {
                "rewrite": "query-only rewrite",
                "queries": ["query-only alternate"],
                "hypothetical_document": "query-only hypothetical memory",
            }

        result = eval_module.run_retrieval_strategy_ablation(
            transform_provider=provider,
            transform_source="query_only_pre_generated",
        )

        assert provider_calls == [(
            raw_query,
            ["rewrite", "multi_query", "hyde"],
            {"timeout_seconds": config.QUERY_DEEP_TRANSFORM_TIMEOUT_SECONDS},
        )]
        # One embed call per indexed Story precedes all retrieval calls.
        retrieval_inputs = fake_embedder.calls[len(bench.topics):]
        assert set(retrieval_inputs) == {
            raw_query,
            "query-only rewrite",
            "query-only alternate",
            "query-only hypothetical memory",
        }
        assert target.title not in retrieval_inputs
        assert target.problem_desc not in retrieval_inputs
        assert target.content not in retrieval_inputs
        provenance = result["transformation_provenance"]
        assert provenance["source"] == "query_only_pre_generated"
        assert provenance["pre_generated_outputs"] is True
        assert provenance["ground_truth_fields_used_for_generation"] == []
        assert provenance["online_latency_evidence"] is False

    def test_oracle_upper_bound_cannot_select_default(
        self, fake_embedder, monkeypatch
    ):
        target = eval_module.load_benchmark().topics[0]
        monkeypatch.setattr(
            runner_module,
            "_strategy_query_pairs",
            lambda _bench: [{
                "query": "opaque query",
                "variant": "ambiguous",
                "topic_id": target.id,
            }],
        )

        result = eval_module.run_retrieval_strategy_ablation(
            transform_provider=lambda *_args, **_kwargs: {
                "rewrite": target.problem_desc,
                "queries": [],
                "hypothetical_document": target.content,
            },
            transform_source="oracle_upper_bound",
        )

        assert result["transformation_provenance"]["oracle_upper_bound"] is True
        for name in (
            "hybrid_graph_rewrite", "hybrid_graph_hyde",
            "hybrid_graph_hyde_reranker",
        ):
            gates = result["strategies"][name]["gates"]
            assert gates["ground_truth_isolation"] is False
            assert gates["online_latency_evidence"] is False
            assert gates["eligible_for_default"] is False


# ═══════════════════════════════════════════════
#  加工分支评测
# ═══════════════════════════════════════════════

class TestProcessingEval:
    def test_branches_classified_correctly(self, fake_embedder):
        """duplicate/near_identical -> 同一 topic 向量 -> sim 1.0 -> update（并入同一 story）；
        distinct -> 不同 topic 正交向量 -> sim 0 -> create（各自新建）。合并正确率 100%。

        用 topic-based 注册：同一 topic 的会话（即便跨 pair 复用）映射到同一向量，
        忠实反映「相同 bug 的会话应有相同向量」的真实语义，避免共享 embedder 的注册冲突。
        """
        bench = eval_module.load_benchmark()
        topics = bench.topics

        def topic_basis(spec):
            """按关键词重合度把会话归到一个 topic，返回该 topic 的基向量。"""
            kw = set(spec.keywords)
            best_i, best_ov = 0, -1
            for i, t in enumerate(topics):
                ov = len(kw & set(t.keywords))
                if ov > best_ov:
                    best_ov, best_i = ov, i
            return basis(best_i)

        for pair in bench.merge_pairs:
            fake_embedder.register(pair.a.index_text(), topic_basis(pair.a))
            fake_embedder.register(pair.b.index_text(), topic_basis(pair.b))

        res = eval_module.run_processing_eval()
        acc = res["accuracy"]
        assert acc["total"] == len(bench.merge_pairs)
        # 每对都判对
        for p in res["pairs"]:
            assert p["actual_branch"] != "error", p
            if p["expected_branch"] == "create":
                assert p["actual_branch"] == "create"
                assert p["same_story"] is False
            else:  # merge_or_update / update
                assert p["actual_branch"] in ("merge", "update")
                assert p["same_story"] is True
        assert acc["accuracy"] == 1.0
        # processor 实际产物与相似度判定一致
        assert all(p["processor_agrees_with_sim"] for p in res["pairs"])


# ═══════════════════════════════════════════════
#  分裂评测
# ═══════════════════════════════════════════════

class TestSplitEval:
    def test_split_triggered_and_structure_ok(self, fake_embedder):
        """incoming 与 existing 相似度 0.88（merge 区间）-> 合并 -> 触发分裂；
        结构校验：父向量移除、父子边 1.0、子向量入索引、子 story 可检索。"""
        bench = eval_module.load_benchmark()
        case = bench.split_cases[0]
        merge_sim = 0.88
        qvec = _cos_vec(0, merge_sim)
        fake_embedder.register(case.existing.index_text(), basis(0))
        fake_embedder.register(case.incoming.index_text(), qvec)
        # 预注册子 story 的 embed 输入文本（processor: " ".join(merged_kw) + " " + sub_content）
        # 使子向量与检索查询同向 -> 可检索
        parent_kw = case.existing.keywords
        incoming_kw = case.incoming.keywords
        merged_kw = list(dict.fromkeys(parent_kw + incoming_kw))
        for sub_content in (case.existing.summary["content"], case.incoming.summary["content"]):
            fake_embedder.register(" ".join(merged_kw) + " " + sub_content, qvec)

        res = eval_module.run_split_eval()
        assert res["total"] == 1
        c = res["cases"][0]
        # sim 落在 merge 区间 [0.85, 0.92) -> 走 merge 分支 -> 触发分裂
        assert c["sim"] == pytest.approx(0.88, abs=1e-3)
        assert c["actual_branch"] == "merge"
        assert c["actual_split"] is True
        checks = c["checks"]
        assert checks["split_triggered"] is True
        assert checks["parent_vector_removed"] is True
        assert checks["child_count"] >= 2
        assert checks["parent_child_edges_weight_1"] is True
        assert checks["children_vectors_in_index"] is True
        assert checks["children_retrievable"] is True
        assert res["accuracy"] == 1.0
