"""会话启动主动注入（晨间简报）测试：query 构造、相关度门槛、token 预算、静默不注入。

全程 mock ``embeddings.embed``（返回精确构造的查询向量），``search.search`` /
``store`` 走真实 SQLite，复用 conftest 的隔离 DB。与 ``test_search`` / ``test_mcp_server``
同源，验证 prime 在其之上的"门槛 + 预算 + 静默"行为。
"""
from __future__ import annotations

from click.testing import CliRunner
import pytest

from storybook import config
from storybook import prime as prime_module
from storybook import store
from storybook.cli import cli
from ._helpers import basis, with_cos


def _seed(title: str, vec: list[float], content: str = "c",
          keywords: list[str] | None = None) -> int:
    return store.add_story(title, content, keywords or ["k"], vec)


# ═══════════════════════════════════════════════
#  token 估算
# ═══════════════════════════════════════════════

class TestEstimateTokens:
    def test_empty(self):
        assert prime_module.estimate_tokens("") == 0

    def test_pure_ascii(self):
        # "hello world" 11 字符 -> (11+3)//4 = 3
        assert prime_module.estimate_tokens("hello world") == 3

    def test_cjk(self):
        # 4 个中文字符 -> 4 token
        assert prime_module.estimate_tokens("排查订单") == 4

    def test_mixed(self):
        # 2 CJK (2) + "ab" 2 字符 -> (2+3)//4 = 1；合计 3
        assert prime_module.estimate_tokens("排查ab") == 3


# ═══════════════════════════════════════════════
#  query 构造
# ═══════════════════════════════════════════════

class TestBuildQuery:
    def test_first_prompt_wins(self):
        assert prime_module.build_query("/x/y", "如何调试 db") == "如何调试 db"

    def test_first_prompt_whitespace_falls_back_to_cwd(self):
        assert prime_module.build_query("/x/payment-service", "   ") == "payment service"

    def test_cwd_only_basename(self):
        assert prime_module.build_query("/Users/orca/work/payment-service") == "payment service"

    def test_cwd_underscore_to_space(self):
        assert prime_module.build_query("/x/order_service") == "order service"

    def test_cwd_skips_generic_segments(self):
        # basename "src" 过于通用，向前取 "billing"
        assert prime_module.build_query("/a/billing/src") == "billing"

    def test_cwd_all_generic_falls_back_to_last(self):
        assert prime_module.build_query("/usr/src/app") == "app"

    def test_both_empty_returns_empty(self):
        assert prime_module.build_query("", "") == ""
        assert prime_module.build_query("", "   ") == ""


# ═══════════════════════════════════════════════
#  prime_context 核心行为
# ═══════════════════════════════════════════════

class TestPrimeSources:
    def test_core_uses_explicit_source(self):
        out = prime_module.prime_context(
            cwd="",
            first_prompt="",
            tool_type="claude_code",
            integration_mode="hook",
        )

        assert out["context"]["tool"]["type"] == "claude_code"
        assert out["context"]["tool"]["integration_mode"] == "hook"

    def test_cli_prime_marks_claude_session_start_hook(self, monkeypatch):
        captured = {}

        def fake_prime_context(**kwargs):
            captured.update(kwargs)
            return {
                "cwd": kwargs["cwd"],
                "query": "",
                "count": 0,
                "injected": False,
                "briefing": "",
                "matches": [],
                "truncated": False,
                "note": None,
            }

        monkeypatch.setattr(prime_module, "prime_context", fake_prime_context)
        monkeypatch.setattr(store, "init_db", lambda: None)

        result = CliRunner().invoke(
            cli, ["prime", "--cwd", "/work/repo", "--format", "json"]
        )

        assert result.exit_code == 0
        assert captured["tool_type"] == "claude_code"
        assert captured["integration_mode"] == "hook"


class TestPrimeContextSilence:
    def test_empty_db_silent_no_inject(self, fake_embedder):
        fake_embedder.register("q", basis(0))
        out = prime_module.prime_context(cwd="/x/proj", first_prompt="q")
        assert out["injected"] is False
        assert out["briefing"] == ""
        assert out["count"] == 0
        assert out["matches"] == []
        # 无匹配是正常情况，不产生 note 噪声
        assert out["note"] is None

    def test_below_prime_threshold_silent(self, fake_embedder):
        """sim 高于检索门槛 0.50 但低于主动注入门槛 0.60 -> 静默不注入。"""
        _seed("low", with_cos(0, 0.55), content="弱相关记忆内容")
        fake_embedder.register("q", basis(0))
        out = prime_module.prime_context(cwd="/x/proj", first_prompt="q")
        assert out["injected"] is False
        assert out["count"] == 0
        assert out["briefing"] == ""

    def test_embedding_failure_silent_with_note(self, fake_embedder):
        """Ollama 不可用等：静默不注入，不抛错，note 给排查提示。"""
        fake_embedder.register("q", None)
        out = prime_module.prime_context(cwd="/x/proj", first_prompt="q")
        assert out["injected"] is False
        assert out["briefing"] == ""
        assert out["note"] is not None
        assert "storybook doctor" in out["note"]

    def test_empty_query_silent(self, fake_embedder):
        """cwd 与 first_prompt 均空 -> 无 query，静默不注入。"""
        # 不注册任何 embed，也不应触发 embed（query 为空时直接返回）
        out = prime_module.prime_context(cwd="", first_prompt="")
        assert out["injected"] is False
        assert out["briefing"] == ""
        assert out["note"] is not None
        # query 为空时不该调用 embed
        assert fake_embedder.calls == []

    def test_schema_missing_silent_not_raise(self, fake_embedder, tmp_db, monkeypatch):
        """DB schema 未初始化（story_vectors 表缺失）-> 静默不注入，不抛错。

        模拟全新环境未跑 ``storybook init``：清空 DB 文件后不 init，prime_context
        不应抛异常，而是返回 injected=False + 可排查 note。
        """
        import sqlite3 as _sqlite3
        # 用一个空 DB 文件（无 schema）覆盖隔离 DB
        empty = tmp_db.parent / "empty.db"
        _sqlite3.connect(str(empty)).close()
        monkeypatch.setattr(config, "DB_PATH", empty)
        fake_embedder.register("q", basis(0))
        out = prime_module.prime_context(cwd="/x/proj", first_prompt="q")
        assert out["injected"] is False
        assert out["briefing"] == ""
        assert out["note"] is not None
        assert "init" in out["note"] or "doctor" in out["note"]


class TestPrimeContextInjection:
    def test_match_above_threshold_injects(self, fake_embedder):
        _seed("排查用户下单失败", basis(0),
              content="问题：下单接口超时 步骤：1.查日志 2.定位回调 结果：调整超时配置",
              keywords=["下单", "超时", "日志"])
        fake_embedder.register("q", basis(0))
        out = prime_module.prime_context(cwd="/x/proj", first_prompt="q")
        assert out["injected"] is True
        assert out["count"] == 1
        assert out["briefing"] != ""
        # 简报含标题与摘要
        assert "排查用户下单失败" in out["briefing"]
        assert out["matches"][0]["story_id"] is not None
        assert out["matches"][0]["similarity"] >= config.PRIME_MIN_SIMILARITY
        assert out["matches"][0]["excerpt"]
        assert out["matches"][0]["keywords"] == ["下单", "超时", "日志"]
        assert out["truncated"] is False

    def test_first_prompt_drives_query(self, fake_embedder):
        _seed("login bug", basis(0))
        fake_embedder.register("fix login bug", basis(0))
        out = prime_module.prime_context(cwd="/x/whatever", first_prompt="fix login bug")
        assert out["injected"] is True
        assert out["query"] == "fix login bug"

    def test_cwd_only_drives_query(self, fake_embedder):
        _seed("payment story", basis(0))
        # build_query 把 "payment-service" -> "payment service"
        fake_embedder.register("payment service", basis(0))
        out = prime_module.prime_context(cwd="/Users/orca/work/payment-service")
        assert out["injected"] is True
        assert out["query"] == "payment service"
        assert out["cwd"] == "/Users/orca/work/payment-service"

    def test_top_k_limits_candidates(self, fake_embedder):
        for i in range(5):
            _seed(f"s{i}", with_cos(0, 0.9 - i * 0.02))   # 0.90/0.88/... 全 >= 0.60
        fake_embedder.register("q", basis(0))
        out = prime_module.prime_context(cwd="/x/proj", first_prompt="q", top_k=2)
        assert out["injected"] is True
        assert out["count"] == 2

    def test_matches_ordered_by_similarity_desc(self, fake_embedder):
        _seed("lo", with_cos(0, 0.65))
        _seed("hi", with_cos(0, 0.95))
        fake_embedder.register("q", basis(0))
        out = prime_module.prime_context(cwd="/x/proj", first_prompt="q")
        sims = [m["similarity"] for m in out["matches"]]
        assert sims == sorted(sims, reverse=True)


# ═══════════════════════════════════════════════
#  token 预算控制
# ═══════════════════════════════════════════════

_LONG_CONTENT = "问题：数据库连接超时排查 " + "步骤细节" * 50  # 远超摘要上限 140 字符


class TestTokenBudget:
    def test_default_budget_fits_all(self, fake_embedder):
        for i in range(3):
            _seed(f"story-{i}", with_cos(0, 0.9 - i * 0.05), content=_LONG_CONTENT)
        fake_embedder.register("q", basis(0))
        out = prime_module.prime_context(cwd="/x/proj", first_prompt="q")
        assert out["count"] == 3
        assert out["truncated"] is False
        # 核心不变式：简报 token 不超预算
        assert prime_module.estimate_tokens(out["briefing"]) <= config.PRIME_TOKEN_BUDGET

    @pytest.mark.parametrize("budget", [200, 300, 500])
    def test_small_budget_truncates_and_stays_within(self, fake_embedder, budget):
        for i in range(3):
            _seed(f"story-{i}", with_cos(0, 0.9 - i * 0.05), content=_LONG_CONTENT)
        fake_embedder.register("q", basis(0))
        out = prime_module.prime_context(cwd="/x/proj", first_prompt="q", token_budget=budget)
        if out["injected"]:
            # 无论裁剪多少，简报必须落在预算内
            assert prime_module.estimate_tokens(out["briefing"]) <= budget
            # 小预算装不下 3 条 -> 必然裁剪
            assert out["truncated"] is True
            assert out["count"] < 3

    def test_tiny_budget_drops_all_silent(self, fake_embedder):
        """预算小到连一条标题行都放不下 -> 静默不注入。"""
        for i in range(3):
            _seed(f"story-{i}", with_cos(0, 0.9 - i * 0.05), content=_LONG_CONTENT)
        fake_embedder.register("q", basis(0))
        out = prime_module.prime_context(cwd="/x/proj", first_prompt="q", token_budget=30)
        assert out["injected"] is False
        assert out["briefing"] == ""
        assert out["truncated"] is True
        assert out["note"] is not None

    def test_excerpt_capped_in_briefing(self, fake_embedder):
        _seed("long", basis(0), content=_LONG_CONTENT)
        fake_embedder.register("q", basis(0))
        out = prime_module.prime_context(cwd="/x/proj", first_prompt="q")
        excerpt = out["matches"][0]["excerpt"]
        assert len(excerpt) <= config.PRIME_CONTENT_EXCERPT_CHARS
