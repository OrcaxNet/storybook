"""FLO-212: ``book --verbose search`` 标签化关键日志。

验收点（PRD §5）：
1. verbose 时 stderr 出现 ``[SB] [req=<request_id>] [stage=...]`` 各阶段行，
   且同一查询所有行共享同一 request_id；
2. 非 verbose 无任何 ``[stage=`` trace；
3. verbose 与不加的 stdout（人读与 ``--json``）完全一致；
4. 标签稳定；``grep 'stage=graph'`` 可过滤；
5. 向量日志为摘要（维度 + 前 N 维预览），不 dump 全量；
6. caplog 可断言各阶段标签与 request_id，非 verbose 无 trace。
"""
from __future__ import annotations

import json
import logging

import pytest
from click.testing import CliRunner

from storybook import search as search_module, store
from storybook.cli import cli, setup_logging
from tests._helpers import basis, with_cos


def _seed(fake_embedder) -> int:
    """写入一条可命中 story 并注册查询向量，保证 fast 路径各阶段真实执行。"""
    story_id = store.add_story(
        "语音机器人开发经验",
        "问题：如何做一个语音机器人 步骤：1.搭建 2.调试 结果：完成",
        ["语音", "机器人"],
        with_cos(0, 0.9),
    )
    fake_embedder.register("语音机器人", basis(0))
    return story_id


def _sb_stages(caplog) -> dict[str, list[logging.LogRecord]]:
    """按 stage 聚合 [SB] trace 记录。"""
    grouped: dict[str, list[logging.LogRecord]] = {}
    for record in caplog.records:
        if getattr(record, "sb_trace", False):
            grouped.setdefault(record.sb_stage, []).append(record)
    return grouped


@pytest.fixture
def cli_stderr_handler():
    """模拟新进程：给 root 挂一个 stderr StreamHandler，测试后移除。

    pytest 的日志插件会让 root 已有 handler，导致 ``logging.basicConfig`` 不再
    自动添加 StreamHandler（与真实 ``book`` 进程行为不一致）。本 fixture 补上
    该 handler，使 CLI 的日志（含 [SB] trace）能进入 ``CliRunner`` 的 stderr。
    """
    root = logging.getLogger()
    handler = logging.StreamHandler()
    root.addHandler(handler)
    try:
        yield handler
    finally:
        root.removeHandler(handler)
        handler.close()


class TestVerboseCli:
    def test_verbose_search_emits_labeled_stages(
        self, fake_embedder, fake_llm, cli_stderr_handler
    ):
        """端到端：CLI verbose 时 stderr 出现全部实际执行阶段的标签行。"""
        _seed(fake_embedder)
        result = CliRunner().invoke(
            cli,
            ["--verbose", "search", "语音机器人", "--context", "none", "--json"],
        )
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        rid = payload["request_id"]
        sb_lines = [
            line for line in result.stderr.splitlines()
            if "[SB] [req=" in line
        ]
        assert sb_lines, "stderr 缺少 [SB] trace 行"
        stages = set()
        for line in sb_lines:
            assert f"[SB] [req={rid}] [stage=" in line, line
            stage = line.split("[stage=", 1)[1].split("]", 1)[0]
            stages.add(stage)
        # 其余 stderr 行（如 store 的 INFO）不得携带 [stage= trace
        for line in result.stderr.splitlines():
            if "[SB]" not in line:
                assert "[stage=" not in line, line
        # 该路径实际执行到的阶段必须全部出现（transform 在 fast 模式记为 skipped）
        assert {
            "query", "embed", "recall-vector", "recall-lexical", "fusion",
            "transform", "graph", "rerank", "final",
        } <= stages

    def test_non_verbose_has_no_stage_traces(
        self, fake_embedder, fake_llm, caplog
    ):
        """非 verbose：stderr 与 caplog 均不出现 [SB]/[stage= trace。"""
        _seed(fake_embedder)
        caplog.set_level(logging.DEBUG)
        result = CliRunner().invoke(
            cli,
            ["search", "语音机器人", "--context", "none", "--json"],
        )
        assert result.exit_code == 0
        assert "[stage=" not in result.stderr
        assert "[SB]" not in result.stderr
        assert not [
            record for record in caplog.records
            if getattr(record, "sb_trace", False)
        ]

    def test_verbose_does_not_change_stdout(
        self, monkeypatch, cli_stderr_handler
    ):
        """相同结果下 verbose 与不加的 --json stdout 逐字节一致，trace 只进 stderr。"""
        expected = {
            "query": "语音机器人",
            "retrieval_mode": "fast",
            "mode": "vector",
            "degraded": False,
            "degraded_reasons": [],
            "latency_ms": {"total": 12.5},
            "keywords": ["语音机器人"],
            "top_matches": [
                {
                    "story_id": 42,
                    "title": "语音机器人",
                    "content": "搭建语音机器人",
                    "keywords": ["voice"],
                    "similarity": 0.91,
                    "warnings": [],
                    "related": [],
                }
            ],
        }

        def fake_search(query, **kwargs):
            search_module._sb_trace("fixed-req", "query", f"original={query!r}")
            search_module._sb_trace("fixed-req", "final", "top_k=1 ids=[42]")
            return expected

        monkeypatch.setattr("storybook.cli.store.init_db", lambda: None)
        monkeypatch.setattr("storybook.cli.search_module.search", fake_search)
        runner = CliRunner()
        base = runner.invoke(
            cli, ["search", "语音机器人", "--context", "none", "--json"]
        )
        verbose = runner.invoke(
            cli,
            ["--verbose", "search", "语音机器人", "--context", "none", "--json"],
        )
        assert base.exit_code == 0
        assert verbose.exit_code == 0
        assert verbose.stdout == base.stdout
        assert "[SB]" not in base.stderr
        assert "[SB] [req=fixed-req] [stage=query]" in verbose.stderr
        assert "[SB] [req=fixed-req] [stage=final]" in verbose.stderr


class TestVerboseCaplog:
    def test_caplog_stage_labels_and_shared_request_id(
        self, fake_embedder, fake_llm, caplog
    ):
        """caplog 可捕获各阶段标签；同一查询所有 trace 共享一个 request_id。"""
        _seed(fake_embedder)
        caplog.set_level(logging.DEBUG)
        setup_logging(verbose=True)
        result = search_module.search("语音机器人", top_k=3, retrieval_mode="fast")
        grouped = _sb_stages(caplog)
        assert {
            "query", "embed", "recall-vector", "recall-lexical", "fusion",
            "transform", "graph", "rerank", "final",
        } <= set(grouped)
        rids = {
            record.sb_request_id
            for records in grouped.values()
            for record in records
        }
        assert len(rids) == 1
        assert rids == {result["request_id"]}

    def test_embed_trace_is_summary_not_full_dump(
        self, fake_embedder, fake_llm, caplog
    ):
        """向量日志只输出维度 + 前 N 维预览（带截断标记），不 dump 全量。"""
        _seed(fake_embedder)
        caplog.set_level(logging.DEBUG)
        setup_logging(verbose=True)
        search_module.search("语音机器人", top_k=3, retrieval_mode="fast")
        embed_records = _sb_stages(caplog).get("embed", [])
        assert len(embed_records) == 1
        message = embed_records[0].getMessage()
        assert "dim=" in message
        assert "preview=" in message
        assert "source=" in message
        assert "…" in message  # 截断标记，证明是摘要而非全量
        assert message.count(",") < 20

    def test_deep_transform_emits_generated_queries(
        self, fake_embedder, fake_llm, caplog
    ):
        """deep 模式：transform 阶段记录生成的 rewrite/multi_query/hyde 查询列表。"""
        _seed(fake_embedder)
        fake_llm.transformation = {
            "rewrite": "语音机器人 实现步骤",
            "queries": ["语音机器人 架构", "机器人 调试"],
            "hypothetical_document": "搭建语音机器人并调试的完整流程",
        }
        caplog.set_level(logging.DEBUG)
        setup_logging(verbose=True)
        result = search_module.search("语音机器人", top_k=3, retrieval_mode="deep")
        transform_records = _sb_stages(caplog).get("transform", [])
        assert transform_records
        text = " | ".join(record.getMessage() for record in transform_records)
        assert "status=ok" in text
        assert "rewrite=" in text or "multi_query=" in text or "hyde=" in text

    def test_lexical_fallback_emits_stages(
        self, fake_embedder, fake_llm, caplog
    ):
        """embedding 不可用时降级路径仍输出实际执行的阶段标签。"""
        story_id = store.add_story(
            "数据库连接经验",
            "问题：连不上数据库 步骤：1.查端口 结果：解决",
            ["数据库"],
            with_cos(3, 0.9),
        )
        fake_embedder.register("数据库连接", None)  # 强制 embedding 不可用
        caplog.set_level(logging.DEBUG)
        setup_logging(verbose=True)
        result = search_module.search("数据库连接", top_k=3, retrieval_mode="fast")
        assert result["mode"] == "lexical_fallback"
        grouped = _sb_stages(caplog)
        assert "embed" in grouped
        assert "recall-lexical" in grouped
        assert "fusion" in grouped
        assert "graph" in grouped
        assert "rerank" in grouped
        assert "final" in grouped
