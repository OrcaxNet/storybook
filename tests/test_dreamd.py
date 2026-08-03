"""做梦周期自动化（dreamd）测试：并发锁、单次周期、监听循环、定时守护。

全部 mock Ollama（复用 conftest 的 fake_llm / fake_embedder），并 monkeypatch
统一来源管理器，避免扫描真实 Agent 历史目录。
锁文件随 ``config.DB_PATH`` 重定向到 tmp_path，测试间天然隔离。
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import pytest

from storybook import dreamd, collector, store, processor
from storybook.history_adapters.codex import CodexAdapter
from ._helpers import basis


# ═══════════════════════════════════════════════
#  并发锁
# ═══════════════════════════════════════════════

class TestDreamLock:
    def test_lock_is_exclusive_nonblocking(self):
        """同一进程第二个非阻塞 acquire 应立即失败（flock 不允许跨 fd 重入）。"""
        with dreamd.acquire_dream_lock(blocking=False):
            with pytest.raises(dreamd.DreamLockBusy):
                with dreamd.acquire_dream_lock(blocking=False):
                    pass

    def test_lock_released_after_context(self):
        """上下文退出后锁释放，可再次获取。"""
        with dreamd.acquire_dream_lock(blocking=False):
            pass
        # 不应抛 DreamLockBusy
        with dreamd.acquire_dream_lock(blocking=False):
            pass

    def test_lock_file_created_next_to_db(self):
        """锁文件落在数据库同目录（随 config.DB_PATH 重定向）。"""
        with dreamd.acquire_dream_lock(blocking=False):
            assert dreamd._lock_path().exists()
        assert dreamd._lock_path().parent == store.config.DB_PATH.parent

    def test_lock_holder_pid_diagnostic(self):
        """lock_holder_pid：无锁返回 None；持锁时返回持有者 pid（仅用于诊断日志）。"""
        import os
        assert dreamd.lock_holder_pid() is None
        with dreamd.acquire_dream_lock(blocking=False):
            assert dreamd.lock_holder_pid() == os.getpid()


# ═══════════════════════════════════════════════
#  单次做梦周期
# ═══════════════════════════════════════════════

def _stub_collect(monkeypatch, sessions):
    """把统一来源采集替换为返回固定会话列表的桩。"""
    calls = {"n": 0}

    def _fake(*_a, **_kw):
        calls["n"] += 1
        count = collector.import_sessions(list(sessions))
        return {
            "status": "ok",
            "imported": count,
            "updated": 0,
            "sources": [{"source": "claude", "status": "ok", "imported": count}],
        }

    monkeypatch.setattr(dreamd.source_manager, "import_enabled", _fake)
    return calls


class TestRunDreamCycleOnce:
    def test_imports_and_processes(self, fake_llm, fake_embedder, monkeypatch):
        """import_new=True：采集新会话 -> 导入 -> 加工，返回 ok 摘要。"""
        _stub_collect(monkeypatch, collector.generate_sample_sessions(2))

        result = dreamd.run_dream_cycle_once(import_new=True)

        assert result["status"] == "ok"
        assert result["imported"] == 2
        assert result["total"] == 2
        assert result["success"] == 2
        assert result["failed"] == 0
        assert result["duration_s"] >= 0.0
        # 两条互不相似的会话各创建一条 story
        assert store.count_stories() == 2
        assert len(store.get_pending_sessions()) == 0

    def test_empty_when_no_new_and_no_pending(self, monkeypatch):
        """无新会话、无 pending -> status empty。"""
        _stub_collect(monkeypatch, [])
        result = dreamd.run_dream_cycle_once(import_new=True)
        assert result["status"] == "empty"
        assert result["imported"] == 0
        assert result["total"] == 0

    def test_process_only_skips_collect(self, fake_llm, fake_embedder, monkeypatch):
        """import_new=False：不采集，只加工已导入的 pending 会话。"""
        calls = _stub_collect(monkeypatch, collector.generate_sample_sessions(2))
        # 预先导入 2 条 pending（不经过 collect 桩）
        collector.import_sessions(collector.generate_sample_sessions(2))

        result = dreamd.run_dream_cycle_once(import_new=False)

        assert result["status"] == "ok"
        assert result["imported"] == 0
        assert result["total"] == 2
        assert calls["n"] == 0, "import_new=False 不应调用来源采集"
        assert store.count_stories() == 2

    def test_skipped_when_locked(self, fake_llm, fake_embedder, monkeypatch):
        """锁被占用时，run_dream_cycle_once 应跳过而非阻塞/重叠。"""
        _stub_collect(monkeypatch, collector.generate_sample_sessions(2))
        with dreamd.acquire_dream_lock(blocking=False):
            result = dreamd.run_dream_cycle_once(import_new=True)
        assert result["status"] == "skipped"
        assert result["imported"] == 0
        assert result["total"] == 0
        # 被跳过时不应加工任何会话
        assert store.count_stories() == 0
        assert len(store.get_pending_sessions()) == 0

    def test_concurrent_call_does_not_double_process(self, fake_llm, fake_embedder, monkeypatch):
        """持锁跑一次加工后，第二持锁者的 run_once 被跳过 -> 不重复加工。"""
        _stub_collect(monkeypatch, collector.generate_sample_sessions(1))
        first = dreamd.run_dream_cycle_once(import_new=True)
        assert first["status"] == "ok"

        # 模拟另一触发在第一轮仍在跑时进入：用持锁上下文模拟「正忙」
        with dreamd.acquire_dream_lock(blocking=False):
            second = dreamd.run_dream_cycle_once(import_new=True)
        assert second["status"] == "skipped"
        assert store.count_stories() == 1, "被跳过，story 数不应增长"


# ═══════════════════════════════════════════════
#  快照 / 触发判定
# ═══════════════════════════════════════════════

class TestSnapshot:
    def test_scan_session_files_missing_dir(self, tmp_path):
        assert dreamd.scan_session_files(tmp_path / "nope") == {}

    def test_scan_session_files_records_mtime(self, tmp_path):
        proj = tmp_path / "projects" / "cwd-1"
        proj.mkdir(parents=True)
        f = proj / "abc.jsonl"
        f.write_text("{}", encoding="utf-8")
        snap = dreamd.scan_session_files(tmp_path / "projects")
        assert len(snap) == 1
        key, value = next(iter(snap.items()))
        assert key.startswith("hmac-sha256:")
        assert str(f) not in key
        assert value == f.stat().st_mtime_ns + f.stat().st_size

    def test_snapshot_changed_detects_new_and_modified(self):
        a = {"/p/s1.jsonl": 1.0}
        b = {"/p/s1.jsonl": 1.0, "/p/s2.jsonl": 2.0}  # 新增
        c = {"/p/s1.jsonl": 9.0}                       # 修改
        assert dreamd._snapshot_changed(a, b)
        assert dreamd._snapshot_changed(a, c)
        assert not dreamd._snapshot_changed(a, dict(a))

    def test_adapter_discovery_failure_does_not_hide_healthy_source(
        self, tmp_path, monkeypatch, caplog
    ):
        healthy_file = tmp_path / "good" / "session.jsonl"
        healthy_file.parent.mkdir()
        healthy_file.write_text("{}\n", encoding="utf-8")

        class HealthyAdapter:
            def detect(self):
                return {"available": True, "status": "ready"}

            def discover(self):
                return [healthy_file]

        class DeniedAdapter:
            def detect(self):
                return {"available": True, "status": "ready"}

            def discover(self):
                raise PermissionError("private source path")

        monkeypatch.setattr(
            dreamd.source_manager,
            "adapters",
            lambda: {"good": HealthyAdapter(), "denied": DeniedAdapter()},
        )
        diagnostics = []

        with caplog.at_level(logging.WARNING):
            snapshot = dreamd.scan_session_files(
                sources=["denied", "good"], diagnostics=diagnostics
            )

        assert len(snapshot) == 1
        assert diagnostics == [{
            "source": "denied",
            "code": "SB_SOURCE_PERMISSION_DENIED",
            "hint": "PermissionError",
        }]
        assert "private source path" not in caplog.text
        assert "SB_SOURCE_PERMISSION_DENIED" in caplog.text

    def test_should_run_cycle_first_tick_always_runs(self):
        """首帧总跑（追补未导入会话），其后仅变化时跑。"""
        assert dreamd._should_run_cycle(None, {}) is True
        assert dreamd._should_run_cycle(None, {"/p/s.jsonl": 1.0}) is True
        assert dreamd._should_run_cycle({}, {}) is False
        assert dreamd._should_run_cycle({"/p/s.jsonl": 1.0}, {"/p/s.jsonl": 2.0}) is True


# ═══════════════════════════════════════════════
#  监听循环（process --watch）
# ═══════════════════════════════════════════════

def _noop_sleep(_stop, _secs):
    return


class TestDefaultSleep:
    def test_interruptible_by_stop_event(self):
        """stop_event 已置位时，_default_sleep 立即返回，不真睡。"""
        stop = threading.Event()
        stop.set()
        start = time.monotonic()
        dreamd._default_sleep(stop, 100)
        assert time.monotonic() - start < 1.0

    def test_sleeps_when_not_stopped(self):
        """未置位时正常睡眠指定时长后返回。"""
        stop = threading.Event()
        start = time.monotonic()
        dreamd._default_sleep(stop, 0.2)
        assert time.monotonic() - start >= 0.15


class TestWatchLoop:
    def test_runs_on_first_tick(self, fake_llm, fake_embedder, monkeypatch, tmp_path):
        """首帧追补：发现新会话即采集+加工。"""
        _stub_collect(monkeypatch, collector.generate_sample_sessions(2))
        empty_projects = tmp_path / "projects"  # 不存在 -> 快照稳定为 {}

        out = dreamd.watch_loop(
            poll_interval=1,
            projects_path=empty_projects,
            max_ticks=1,
            sleep_func=_noop_sleep,
        )

        assert out["ticks"] == 1
        assert out["cycles"] == 1
        assert out["results"][0]["status"] == "ok"
        assert store.count_stories() == 2

    def test_skips_when_no_change(self, fake_llm, fake_embedder, monkeypatch, tmp_path):
        """首帧追补后，无新会话文件变化 -> 不再触发加工。"""
        _stub_collect(monkeypatch, [])  # 采集永远空
        empty_projects = tmp_path / "projects"

        out = dreamd.watch_loop(
            poll_interval=1,
            projects_path=empty_projects,
            max_ticks=2,
            sleep_func=_noop_sleep,
        )

        assert out["ticks"] == 2
        assert out["cycles"] == 1, "仅首帧追补跑一次，第二帧无变化不跑"

    def test_triggers_again_on_new_file(self, fake_llm, fake_embedder, monkeypatch, tmp_path):
        """第二轮发现新 jsonl 文件 -> 再次触发加工。"""
        projects = tmp_path / "projects"
        projects.mkdir()
        # 两次采集分别返回 0 条与 2 条，模拟「第二轮才有新会话」
        seq = iter([[], collector.generate_sample_sessions(2)])
        def collect_next(*_a, **_kw):
            sessions = list(next(seq))
            count = collector.import_sessions(sessions)
            return {
                "status": "ok", "imported": count, "updated": 0,
                "sources": [{"source": "claude", "status": "ok", "imported": count}],
            }

        monkeypatch.setattr(dreamd.source_manager, "import_enabled", collect_next)

        # 在第一帧与第二帧之间创建一个新 jsonl，使快照变化 -> 第二帧触发
        def create_file_on_first_sleep(_stop, _secs):
            sub = projects / "cwd-1"
            sub.mkdir(exist_ok=True)
            (sub / "new-session.jsonl").write_text("{}", encoding="utf-8")

        out = dreamd.watch_loop(
            poll_interval=1,
            projects_path=projects,
            max_ticks=2,
            sleep_func=create_file_on_first_sleep,
        )
        # 第一帧追补（0 条）+ 第二帧因「文件新增」触发（2 条）
        assert out["cycles"] == 2
        assert out["results"][1]["status"] == "ok"
        assert out["results"][1]["imported"] == 2
        assert store.count_stories() == 2

    def test_codex_watch_triggers_after_first_tick_and_restart(
        self, fake_llm, fake_embedder, monkeypatch, tmp_path
    ):
        root = tmp_path / ".codex"
        path = root / "sessions/2026/08/01/watch.jsonl"
        path.parent.mkdir(parents=True)
        rows = [
            {
                "timestamp": "2026-08-01T00:00:00Z",
                "type": "session_meta",
                "payload": {
                    "id": "watch-session", "cwd": "/work",
                    "timestamp": "2026-08-01T00:00:00Z",
                    "cli_version": "0.145.0",
                },
            },
            {
                "timestamp": "2026-08-01T00:00:01Z",
                "type": "response_item",
                "payload": {
                    "type": "message", "role": "user",
                    "content": [{"type": "input_text", "text": "watch codex"}],
                },
            },
        ]
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        monkeypatch.setattr(
            dreamd.source_manager,
            "adapters",
            lambda: {"codex": CodexAdapter(root)},
        )
        sequence = iter(["first watch append", "restart watch append"])

        def append_on_sleep(_stop, _secs):
            message = next(sequence)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "timestamp": "2026-08-01T00:00:02Z",
                    "type": "response_item",
                    "payload": {
                        "type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": message}],
                    },
                }) + "\n")

        first = dreamd.watch_loop(
            poll_interval=1, max_ticks=2, sleep_func=append_on_sleep,
            sources=["codex"],
        )
        restarted = dreamd.watch_loop(
            poll_interval=1, max_ticks=2, sleep_func=append_on_sleep,
            sources=["codex"],
        )

        assert first["cycles"] == 2
        assert first["results"][0]["imported"] == 1
        assert first["results"][1]["updated"] == 1
        assert restarted["cycles"] == 2
        assert restarted["results"][0]["updated"] == 0
        assert restarted["results"][1]["updated"] == 1

    def test_skips_cycle_when_locked(self, fake_llm, fake_embedder, monkeypatch, tmp_path):
        """锁被占用时，监听触发的周期被跳过。"""
        _stub_collect(monkeypatch, collector.generate_sample_sessions(2))
        with dreamd.acquire_dream_lock(blocking=False):
            out = dreamd.watch_loop(
                poll_interval=1,
                projects_path=tmp_path / "projects",
                max_ticks=1,
                sleep_func=_noop_sleep,
            )
        assert out["cycles"] == 1
        assert out["results"][0]["status"] == "skipped"
        assert store.count_stories() == 0

    def test_stop_event_exits_loop(self, fake_llm, fake_embedder, monkeypatch, tmp_path):
        """stop_event 在第一帧 sleep 时置位 -> 跑完 1 帧即退，不触达 max_ticks。"""
        _stub_collect(monkeypatch, [])
        stop = threading.Event()
        sleep_calls = [0]

        def stop_after_first_sleep(_stop, _secs):
            sleep_calls[0] += 1
            stop.set()

        out = dreamd.watch_loop(
            poll_interval=1,
            projects_path=tmp_path / "projects",
            stop_event=stop,
            max_ticks=10,
            sleep_func=stop_after_first_sleep,
        )
        assert out["ticks"] == 1, "应只跑 1 帧即退出"
        assert sleep_calls[0] == 1, "只 sleep 一次（随即置位 stop）"


# ═══════════════════════════════════════════════
#  定时守护（dream）
# ═══════════════════════════════════════════════

class TestDreamDaemon:
    def test_one_cycle(self, fake_llm, fake_embedder, monkeypatch):
        """max_cycles=1：跑一轮完整周期后退出（不 sleep）。"""
        _stub_collect(monkeypatch, collector.generate_sample_sessions(2))
        out = dreamd.dream_daemon(
            interval=1, max_cycles=1, sleep_func=_noop_sleep
        )
        assert out["cycles"] == 1
        assert out["results"][0]["status"] == "ok"
        assert store.count_stories() == 2

    def test_skipped_when_locked(self, fake_llm, fake_embedder, monkeypatch):
        _stub_collect(monkeypatch, collector.generate_sample_sessions(1))
        with dreamd.acquire_dream_lock(blocking=False):
            out = dreamd.dream_daemon(
                interval=1, max_cycles=1, sleep_func=_noop_sleep
            )
        assert out["results"][0]["status"] == "skipped"


# ═══════════════════════════════════════════════
#  日志
# ═══════════════════════════════════════════════

def test_setup_dream_logging_idempotent(tmp_path):
    """setup_dream_logging 写文件且幂等（不重复加 handler）。"""
    sb = logging.getLogger("storybook")
    # 清理可能存在的 dream handler，避免跨用例污染
    for h in [h for h in sb.handlers if getattr(h, "_storybook_dream", False)]:
        sb.removeHandler(h)
        h.close()

    log_path = tmp_path / "dream.log"
    try:
        dreamd.setup_dream_logging(log_path)
        logging.getLogger("storybook.x").info("hello dream")
        for h in sb.handlers:
            h.flush()
        assert log_path.exists()
        assert "hello dream" in log_path.read_text(encoding="utf-8")

        n_before = sum(1 for h in sb.handlers if getattr(h, "_storybook_dream", False))
        dreamd.setup_dream_logging(log_path)  # 幂等：不应新增 handler
        n_after = sum(1 for h in sb.handlers if getattr(h, "_storybook_dream", False))
        assert n_after == n_before == 1
    finally:
        for h in [h for h in sb.handlers if getattr(h, "_storybook_dream", False)]:
            sb.removeHandler(h)
            h.close()
