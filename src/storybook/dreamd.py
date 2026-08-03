"""
做梦周期自动化 - 「梦境」后台整理的调度与并发保护。

三类自动化入口（均复用 :func:`run_dream_cycle_once`，受同一把文件锁保护，互不重叠）：

- ``storybook process --watch``  : 反应式监听。轮询启用来源，发现新会话即触发加工（长驻）。
- ``storybook dream --once``     : 单次完整做梦周期（采集 + 加工）后退出；launchd 定时任务用此入口。
- ``storybook dream``            : 定时守护进程（非 macOS 兜底）。每 ``DREAM_INTERVAL`` 秒跑一次周期。

并发保护用 ``fcntl.flock``（非阻塞）锁住 ``dream.lock``；锁与数据库同目录，故测试中随
``config.DB_PATH`` 重定向而隔离。无 ``fcntl`` 的平台（如 Windows）回退到 pid 文件 + 存活探测。
"""
from __future__ import annotations

import logging
import os
import signal
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional

from . import config
from .history_adapters import manager as source_manager
from . import processor
from . import store

logger = logging.getLogger(__name__)

try:
    import fcntl  # Unix: macOS / Linux
    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover - Windows 回退路径，CI 通常在 Unix
    _HAVE_FCNTL = False


class DreamLockBusy(Exception):
    """另一个做梦周期正在运行（锁已被占用）。"""


# ═══════════════════════════════════════════════
#  并发锁
# ═══════════════════════════════════════════════

def _lock_path() -> Path:
    """锁文件路径：与数据库同目录，随 ``config.DB_PATH`` 重定向（测试隔离）。"""
    return config.DB_PATH.parent / "dream.lock"


@contextmanager
def acquire_dream_lock(blocking: bool = False):
    """获取做梦周期独占锁。

    非阻塞（默认）：锁被占用时立即抛 :class:`DreamLockBusy`，调用方据此「跳过」本次触发，
    避免两个 process 同时跑。阻塞模式会等到锁释放。

    主路径用 ``fcntl.flock``：进程退出（含崩溃）时 OS 自动释放，无 stale pid 问题；
    同时把持有者 PID 写入锁文件便于诊断。无 ``fcntl`` 时回退到 pid 文件 + 存活探测。
    """
    path = _lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    if _HAVE_FCNTL:
        fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
        flags = fcntl.LOCK_EX
        if not blocking:
            flags |= fcntl.LOCK_NB
        try:
            fcntl.flock(fd, flags)
        except BlockingIOError:
            os.close(fd)
            raise DreamLockBusy()
        try:
            # 记录持有者 PID 便于排查（不影响锁语义）
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                os.ftruncate(fd, 0)
                os.write(fd, f"{os.getpid()}\n".encode())
            except OSError:
                pass
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        return

    # ── 回退：pid 文件 + 存活探测（无 fcntl 的平台）──  # pragma: no cover
    while True:
        existing = _read_pid(path)
        if existing is not None and _pid_alive(existing):
            if blocking:
                time.sleep(0.2)
                continue
            raise DreamLockBusy()
        # 写入自己的 pid（O_EXCL 防竞争）
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            if blocking:
                time.sleep(0.2)
                continue
            raise DreamLockBusy()
        try:
            os.write(fd, f"{os.getpid()}\n".encode())
            os.close(fd)
            yield
        finally:
            # 仅当文件里仍是自己的 pid 才删除，避免误删后继持有者
            if _read_pid(path) == os.getpid():
                try:
                    path.unlink()
                except OSError:
                    pass


def _read_pid(path: Path) -> Optional[int]:  # pragma: no cover - 仅 Windows 回退路径用
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:  # pragma: no cover - 仅 Windows 回退路径用
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def lock_holder_pid() -> Optional[int]:
    """返回当前锁持有者 PID（诊断用）；无人持有或不可读返回 None。"""
    pid = _read_pid(_lock_path())
    if pid is None:
        return None
    return pid if _pid_alive(pid) else None


# ═══════════════════════════════════════════════
#  日志
# ═══════════════════════════════════════════════

def setup_dream_logging(log_path: Optional[Path] = None) -> Path:
    """把日志写入当前 Profile 的 ``logs/dream.log``（幂等）。

    守护 / 监听 / --once 入口调用；自动周期无 shell，日志落文件便于排查。
    返回日志文件路径。测试不调用此函数，避免触碰真实用户日志目录。
    """
    log_path = Path(log_path) if log_path else config.LOG_DIR / "dream.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sb_logger = logging.getLogger("storybook")
    if not any(getattr(h, "_storybook_dream", False) for h in sb_logger.handlers):
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        fh._storybook_dream = True  # type: ignore[attr-defined]
        sb_logger.addHandler(fh)
    if sb_logger.level == logging.NOTSET or sb_logger.level > logging.INFO:
        sb_logger.setLevel(logging.INFO)
    return log_path


# ═══════════════════════════════════════════════
#  单次做梦周期
# ═══════════════════════════════════════════════

def run_dream_cycle_once(
    import_new: bool = True,
    verbose: bool = False,
    sources: list[str] | tuple[str, ...] | None = None,
) -> dict:
    """跑一次完整做梦周期，全程持有独占锁。

    1. （``import_new=True``）从启用来源增量采集新会话并导入；
    2. 加工所有 pending 会话（``processor.process_all_pending``）。

    返回 ``{"status", "imported", "total", "success", "failed", "duration_s"}``：
    ``status`` 为 ``"ok"``（有加工）/ ``"empty"``（无待加工）/ ``"skipped"``（锁被占用）。
    """
    store.init_db()
    start = time.monotonic()
    try:
        with acquire_dream_lock(blocking=False):
            imported = 0
            updated = 0
            source_results: list[dict] = []
            ingestion_status = "ok"
            if import_new:
                ingestion = source_manager.import_enabled(sources)
                imported = ingestion["imported"]
                updated = ingestion["updated"]
                source_results = ingestion["sources"]
                ingestion_status = ingestion["status"]
                logger.info("多来源采集完成: imported=%d updated=%d", imported, updated)
            summary = processor.process_all_pending(verbose=verbose)
            duration = time.monotonic() - start
            result = {
                "status": (
                    "degraded" if ingestion_status == "degraded"
                    else ("ok" if summary["total"] else "empty")
                ),
                "imported": imported,
                "updated": updated,
                "sources": source_results,
                "total": summary["total"],
                "success": summary["success"],
                "failed": summary["failed"],
                "duration_s": round(duration, 2),
            }
            logger.info("做梦周期完成: %s", result)
            return result
    except DreamLockBusy:
        holder = lock_holder_pid()
        logger.info("跳过做梦周期：另一周期正在运行 (pid=%s)", holder)
        return {
            "status": "skipped",
            "imported": 0,
            "updated": 0,
            "sources": [],
            "total": 0,
            "success": 0,
            "failed": 0,
            "duration_s": 0.0,
        }


# ═══════════════════════════════════════════════
#  反应式监听（process --watch）
# ═══════════════════════════════════════════════

def scan_session_files(
    projects_path: Optional[Path] = None,
    sources: list[str] | tuple[str, ...] | None = None,
) -> dict[str, float]:
    """对启用来源做隐私安全的廉价快照 ``{HMAC 文件键: 版本}``。

    用于 ``--watch`` 判定「是否有变化」，避免每轮都跑完整采集。路径不存在返回 ``{}``。
    """
    if projects_path is not None:
        if not projects_path.exists():
            return {}
        paths = projects_path.glob("*/*.jsonl")
    else:
        selected = sources or [
            name for name, enabled in source_manager.load_settings().items() if enabled
        ]
        registered = source_manager.adapters()
        paths = (
            path
            for name in selected
            if name in registered
            for path in registered[name].discover()
        )
    snap: dict[str, float] = {}
    for path in paths:
        try:
            stat = path.stat()
            key = source_manager.private_file_key("watch", path)
            snap[key] = stat.st_mtime_ns + stat.st_size
        except OSError:
            continue
    return snap


def _snapshot_changed(prev: dict[str, float], curr: dict[str, float]) -> bool:
    """两次快照间是否有新增/修改的会话文件。"""
    if set(curr) != set(prev):
        return True
    return any(curr[k] != prev[k] for k in curr)


def _should_run_cycle(prev: Optional[dict[str, float]], curr: dict[str, float]) -> bool:
    """首帧（``prev is None``）总跑一次以追补未导入会话；其后仅变化时跑。"""
    if prev is None:
        return True
    return _snapshot_changed(prev, curr)


def _default_sleep(stop_event: threading.Event, seconds: float) -> None:
    """可被 stop_event 打断的睡眠（≤1s 延迟响应 SIGINT/SIGTERM）。"""
    slept = 0.0
    while slept < seconds:
        if stop_event.is_set():
            return
        step = min(1.0, seconds - slept)
        time.sleep(step)
        slept += step


def watch_loop(
    poll_interval: Optional[int] = None,
    projects_path: Optional[Path] = None,
    stop_event: Optional[threading.Event] = None,
    max_ticks: Optional[int] = None,
    sleep_func: Optional[Callable[[threading.Event, float], None]] = None,
    verbose: bool = False,
    sources: list[str] | tuple[str, ...] | None = None,
) -> dict:
    """反应式监听：轮询启用来源，有新会话自动触发做梦周期。

    - 首帧追补一次（导入尚未采集的会话）；其后仅当快照变化时触发。
    - 每次触发走 :func:`run_dream_cycle_once`，受文件锁保护；与 launchd / 手动 process 互不重叠。
    - ``max_ticks`` / ``stop_event`` / ``sleep_func`` 供测试注入；生产中由信号处理器置位 stop_event 退出。

    返回 ``{"ticks", "cycles", "results"}``。
    """
    poll_interval = poll_interval if poll_interval is not None else config.WATCH_POLL_INTERVAL
    projects_path = projects_path or config.CLAUDE_PROJECTS_PATH
    stop_event = stop_event or threading.Event()
    sleep_func = sleep_func or _default_sleep

    logger.info("监听启动：每 %ds 轮询 %s", poll_interval, projects_path)
    prev: Optional[dict[str, float]] = None
    ticks = 0
    cycles = 0
    results: list[dict] = []

    while not stop_event.is_set():
        ticks += 1
        curr = scan_session_files(projects_path, sources)
        if _should_run_cycle(prev, curr):
            result = run_dream_cycle_once(
                import_new=True, verbose=verbose, sources=sources
            )
            results.append(result)
            cycles += 1
            logger.info("监听触发: %s", result)
        prev = curr
        if max_ticks is not None and ticks >= max_ticks:
            break
        if stop_event.is_set():
            break
        sleep_func(stop_event, float(poll_interval))

    logger.info("监听结束：共 %d 轮触发 / %d 次轮询", cycles, ticks)
    return {"ticks": ticks, "cycles": cycles, "results": results}


# ═══════════════════════════════════════════════
#  定时守护（dream，非 macOS 兜底）
# ═══════════════════════════════════════════════

def dream_daemon(
    interval: Optional[int] = None,
    stop_event: Optional[threading.Event] = None,
    max_cycles: Optional[int] = None,
    sleep_func: Optional[Callable[[threading.Event, float], None]] = None,
    verbose: bool = False,
    sources: list[str] | tuple[str, ...] | None = None,
) -> dict:
    """定时守护进程：每 ``interval`` 秒跑一次完整做梦周期。

    非 macOS 平台用此替代 launchd（systemd / nohup 托管）。每轮受文件锁保护，
    若上一轮仍在跑则跳过。返回 ``{"cycles", "results"}``。
    """
    interval = interval if interval is not None else config.DREAM_INTERVAL
    stop_event = stop_event or threading.Event()
    sleep_func = sleep_func or _default_sleep

    logger.info("做梦守护进程启动：每 %ds 触发一次", interval)
    cycles = 0
    results: list[dict] = []

    while not stop_event.is_set():
        result = run_dream_cycle_once(
            import_new=True, verbose=verbose, sources=sources
        )
        results.append(result)
        cycles += 1
        if max_cycles is not None and cycles >= max_cycles:
            break
        if stop_event.is_set():
            break
        sleep_func(stop_event, float(interval))

    logger.info("做梦守护进程结束：共 %d 轮", cycles)
    return {"cycles": cycles, "results": results}


# ═══════════════════════════════════════════════
#  信号处理
# ═══════════════════════════════════════════════

def install_signal_handlers(stop_event: threading.Event) -> None:
    """安装 SIGINT/SIGTERM 处理器，置位 stop_event 让长驻循环优雅退出。"""
    def _handler(signum, _frame):
        logger.info("收到信号 %s，准备退出...", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
