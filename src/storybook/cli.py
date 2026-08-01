"""
CLI 入口 — storybook 命令

用法:
  storybook init                    初始化数据库
  storybook profile show|list       查看用户级 Profile
  storybook profile create|switch   创建隔离 Profile / 切换当前 Profile
  storybook sync status             查看本地同步状态（v0.2 为 local-only）
  storybook doctor [--fix]          环境与健康自检（--fix 修复向量双写不一致）
  storybook import <path>           导入会话日志(JSON)
  storybook import                  从 Claude Code 采集（默认数据源）
  storybook import --claude         从 Claude Code 采集（同上，显式写法）
  storybook import --cursor         从 Cursor 自动采集（备用，未用 Cursor 可忽略）
  storybook import --sample [N]     生成N条模拟数据(默认100)
  storybook process                 处理所有pending会话(做梦)
  storybook process --session ID    处理指定会话
  storybook process --watch         监听 ~/.claude/projects，有新会话自动加工（长驻）
  storybook dream --once            跑一次完整做梦周期（采集+加工）后退出；launchd 入口
  storybook dream                   定时守护进程（非 macOS 兜底，每 DREAM_INTERVAL 秒一轮）
  storybook search <query>          搜索记忆
  storybook status --performance    最近查询性能摘要
  storybook benchmark               10k Story warm/cold 查询基准
  storybook stats                   查看统计
  storybook list                    列出所有story
  storybook show <story_id>         查看story详情
  storybook prime [--cwd PATH]      会话启动主动注入（晨间简报），供 SessionStart hook 调用
  storybook mcp                     启动 MCP server（stdio，供 Claude Code 等 agent 召回）
"""
import json
import logging
import os
import threading
from pathlib import Path

import click

from . import (
    collector,
    config,
    dreamd,
    health,
    perf_benchmark,
    performance,
    processor,
    store,
)
from . import eval as eval_module
from . import search as search_module
from .profiles import ProfileError


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="详细日志")
def cli(verbose):
    """🧠 Storybook - 离线 Coding 记忆系统"""
    setup_logging(verbose)


def _profile_payload(profile, *, active: bool) -> dict:
    paths = config.PROFILE_REGISTRY.paths_for(profile)
    return {
        "id": profile.id,
        "display_name": profile.display_name,
        "mode": profile.mode,
        "sync_state": profile.sync_state,
        "active": active,
        "data_dir": str(paths.root),
        "database": str(paths.database),
        "index_dir": str(paths.index_dir),
        "cache_dir": str(paths.cache_dir),
        "log_dir": str(paths.log_dir),
        "created_at": profile.created_at,
    }


@cli.group()
def profile():
    """👤 管理同一 OS 用户共享的 Storybook Profile。"""


@profile.command(name="list")
@click.option("--json", "as_json", is_flag=True, help="输出 JSON")
def profile_list(as_json):
    """列出所有 local / isolated Profile。"""

    active_id = config.PROFILE_REGISTRY.active_profile().id
    items = [
        _profile_payload(item, active=item.id == active_id)
        for item in config.PROFILE_REGISTRY.list_profiles()
    ]
    if as_json:
        click.echo(json.dumps({"profiles": items}, ensure_ascii=False, indent=2))
        return
    for item in items:
        marker = "*" if item["active"] else " "
        click.echo(
            f"{marker} {item['display_name']}  {item['id']}  "
            f"{item['mode']}  {item['sync_state']}"
        )


@profile.command(name="show")
@click.argument("profile_ref", required=False)
@click.option("--json", "as_json", is_flag=True, help="输出 JSON")
def profile_show(profile_ref, as_json):
    """显示当前或指定 Profile 的本地目录与状态。"""

    try:
        item = (
            config.PROFILE_REGISTRY.resolve(profile_ref)
            if profile_ref
            else config.PROFILE_REGISTRY.active_profile()
        )
    except ProfileError as exc:
        raise click.ClickException(str(exc)) from exc
    payload = _profile_payload(
        item, active=item.id == config.PROFILE_REGISTRY.active_profile().id
    )
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    click.echo(f"Profile      {payload['display_name']} ({payload['id']})")
    click.echo(f"Mode         {payload['mode']}")
    click.echo(f"Sync         {payload['sync_state']} (cross-device sync disabled)")
    click.echo(f"Data         {payload['data_dir']}")
    click.echo(f"Database     {payload['database']}")
    click.echo(f"Indexes      {payload['index_dir']}")
    click.echo(f"Cache        {payload['cache_dir']}")
    click.echo(f"Logs         {payload['log_dir']}")


@profile.command(name="create")
@click.argument("display_name")
@click.option(
    "--mode",
    type=click.Choice(["local", "isolated"]),
    default="isolated",
    show_default=True,
    help="isolated 用于客户或敏感环境",
)
@click.option("--switch", "activate", is_flag=True, help="创建后立即切换")
def profile_create(display_name, mode, activate):
    """创建随机 UUID Profile；默认不影响当前 Profile。"""

    try:
        item = config.PROFILE_REGISTRY.create_profile(
            display_name, mode=mode, activate=activate
        )
        if activate:
            config.refresh_profile(item.id)
    except ProfileError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"✅ 已创建 Profile: {item.display_name} ({item.id}) [{item.mode}]")
    if activate:
        click.echo(f"   已切换；数据库: {config.DB_PATH}")


@profile.command(name="switch")
@click.argument("profile_ref")
def profile_switch(profile_ref):
    """按 UUID 或显示名切换当前 Profile。"""

    try:
        item = config.switch_profile(profile_ref)
    except ProfileError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"✅ 当前 Profile: {item.display_name} ({item.id})")
    click.echo(f"   数据目录: {config.DATA_DIR}")


@cli.group()
def sync():
    """🔄 查看同步状态（v0.2 仅提供 local-only 边界）。"""


@sync.command(name="status")
@click.option("--json", "as_json", is_flag=True, help="输出 JSON")
def sync_status(as_json):
    """显示当前 Profile 的跨设备同步状态。"""

    payload = {
        "profile_id": config.PROFILE_ID,
        "sync_state": config.SYNC_STATE,
        "enabled": False,
        "message": "v0.2 stores all memory locally; cross-device sync is not enabled.",
    }
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    click.echo(f"Sync           {payload['sync_state']}")
    click.echo("Cross-device   disabled (v0.2)")
    click.echo("Storage        local user profile")


@cli.command()
def init():
    """初始化数据库"""
    store.init_db()
    click.echo(f"✅ 数据库已初始化: {config.DB_PATH}")


@cli.command()
@click.option("--fix", is_flag=True, help="自动修复向量双写不一致（重建缺失行 / 清除孤立行）")
def doctor(fix):
    """🩺 环境与健康自检

    检查 Ollama 可达性、LLM/Embedding 模型、向量维度、sqlite-vec 扩展与
    story_vectors 虚表、向量双写一致性，逐项给出 ✅/❌ 与修复建议。
    加 --fix 自动修复向量双写不一致。
    """
    health.run_doctor(fix=fix)


@cli.command()
def mcp():
    """🔌 启动 MCP server（stdio），把记忆检索暴露给 MCP-aware agent（如 Claude Code）。

    agent 可在运行时调用 recall / get_story / stats 召回过往记忆，
    实现"跨 session 经验复用"。server 为独立进程，不依赖 CLI 运行态。

    需先安装 MCP 依赖：uv pip install -e ".[mcp]"。
    Claude Code 接入配置见 README「MCP 接入」一节。
    """
    from . import mcp_server
    mcp_server.main()


@cli.command()
@click.option("--cwd", default=None,
              help="当前项目目录（默认取进程 cwd；hook 中传 $CLAUDE_PROJECT_DIR）")
@click.option("--prompt", "first_prompt", default="",
              help="可选的首条提问（agent 路径或手动传入；无则仅用 cwd 派生查询）")
@click.option("--top", "-t", "top_k", type=int, default=None, help="最多召回候选数")
@click.option("--budget", "token_budget", type=int, default=None,
              help="简报 token 预算上限（默认 2000）")
@click.option("--format", "output_format",
              type=click.Choice(["text", "json", "hook"]),
              default="text",
              help="text=纯文本简报到 stdout（供 hook 注入）；json=完整结构化结果；"
                   "hook=SessionStart hookSpecificOutput JSON")
def prime(cwd, first_prompt, top_k, token_budget, output_format):
    """🌅 会话启动主动注入（晨间简报）。

    基于 cwd + 可选首条提问，召回 top-N 相关记忆，生成精简摘要。
    相关度不足或无匹配时静默不输出（供 SessionStart hook 注入：无输出即不污染上下文）。

    \b
    Hook 用法（默认 text 输出到 stdout，被 Claude Code 作为额外上下文注入）:
      storybook prime --cwd "$CLAUDE_PROJECT_DIR"

    \b
    结构化注入（支持 hookSpecificOutput 的 Claude Code 版本）:
      storybook prime --cwd "$CLAUDE_PROJECT_DIR" --format hook

    详见 README「🌅 会话启动注入」一节。
    """
    from . import prime as prime_module

    # CLI 默认用进程 cwd（hook 中应显式传 $CLAUDE_PROJECT_DIR）；prime_context 本身不回退，
    # 以便 cwd/first_prompt 均空时静默不注入的语义可被单测。
    cwd = cwd or os.getcwd()

    # hook 路径须非侵入：DB 未初始化等任何异常都不应报错或向上下文注入错误。
    # 全程兜底 try/except：失败时退化为静默不注入（exit 0、stdout 空）。
    try:
        try:
            store.init_db()
        except Exception as e:  # noqa: BLE001  -- best-effort，工具调用时再按需报错
            logging.getLogger(__name__).warning("prime: init_db 失败（继续）: %s", e)
        result = prime_module.prime_context(
            cwd=cwd, first_prompt=first_prompt,
            top_k=top_k, token_budget=token_budget,
        )
    except Exception as e:  # noqa: BLE001  -- 绝不向 hook 上下文抛错
        result = {
            "cwd": cwd, "query": "", "count": 0, "injected": False,
            "briefing": "", "matches": [], "truncated": False,
            "note": f"prime 异常：{e}",
        }

    briefing = result.get("briefing") or ""

    if output_format == "json":
        # 完整结构化结果（调试 / 程序化用）
        click.echo(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if output_format == "hook":
        # SessionStart hook 结构化注入：仅 additionalContext 被注入，其余 stdout 被忽略
        payload = {
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": briefing,
            }
        }
        click.echo(json.dumps(payload, ensure_ascii=False))
        return

    # text（默认 / hook 兼容）：只把简报写到 stdout（注入上下文）
    if briefing:
        click.echo(briefing)
    # note（环境问题等）只进 stderr，不进上下文，避免污染
    if result.get("note"):
        click.echo(f"[storybook prime] {result['note']}", err=True)


@cli.command()
@click.argument("path", required=False)
@click.option("--claude", is_flag=True, help="从 Claude Code 采集")
@click.option("--cursor", is_flag=True, help="从 Cursor 自动采集")
@click.option("--sample", is_flag=True, help="生成模拟数据")
@click.option("--n", default=100, help="模拟数据数量(配合--sample)")
def import_data(path, claude, cursor, sample, n):
    """导入会话日志"""
    store.init_db()

    if sample:
        click.echo(f"📊 生成 {n} 条模拟会话数据...")
        sessions = collector.generate_sample_sessions(n)
        count = collector.import_sessions(sessions)
        click.echo(f"✅ 导入 {count} 条模拟会话")

    elif cursor:
        click.echo("📥 从 Cursor 采集会话...")
        sessions = collector.collect_cursor_sessions()
        if not sessions:
            click.echo("⚠️  未找到 Cursor 会话数据。使用 --sample 生成模拟数据")
            return
        count = collector.import_sessions(sessions)
        click.echo(f"✅ 导入 {count} 条 Cursor 会话")

    elif claude:
        click.echo("📥 从 Claude Code 采集会话...")
        sessions = collector.collect_claude_sessions()
        if not sessions:
            click.echo("⚠️  未找到新的 Claude Code 会话（可能已全部导入）")
            return
        count = collector.import_sessions(sessions)
        click.echo(f"✅ 导入 {count} 条 Claude Code 会话")

    elif path:
        click.echo(f"📥 从路径导入: {path}")
        sessions = []
        if os.path.isdir(path):
            # 目录：扫描所有 .json 文件
            for fname in sorted(os.listdir(path)):
                if not fname.endswith(".json"):
                    continue
                fpath = os.path.join(path, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        sessions.extend(data)
                    elif isinstance(data, dict) and "sessions" in data:
                        sessions.extend(data["sessions"])
                    elif isinstance(data, dict):
                        sessions.append(data)
                except Exception as e:  # noqa: BLE001 -- 单个坏文件不阻断目录导入
                    click.echo(f"  ⚠️ 跳过 {fname}: {e}")
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                sessions = data
            elif isinstance(data, dict) and "sessions" in data:
                sessions = data["sessions"]
            else:
                sessions = [data]

        # 统一字段格式
        normalized = []
        for s in sessions:
            # 支持测试日志格式：有 messages 字段
            if "messages" in s and "raw_content" not in s:
                msgs = s["messages"]
                combined = "\n".join(
                    f"[{m.get('role','?')}] {m.get('content','')}" for m in msgs
                )
                normalized.append({
                    "source": s.get("source", "json"),
                    "raw_content": combined,
                    "problem_desc": s.get("title", msgs[0]["content"][:200] if msgs else ""),
                    "code_snippets": s.get("code_snippets", "[]"),
                    "conclusion": s.get("conclusion", ""),
                })
            else:
                normalized.append(s)

        count = collector.import_sessions(normalized)
        click.echo(f"✅ 导入 {count} 条会话")

    else:
        # 默认数据源：Claude Code（用 --sample/--cursor/<path> 切换其他来源）
        click.echo("📥 从 Claude Code 采集会话（默认数据源）...")
        sessions = collector.collect_claude_sessions()
        if not sessions:
            click.echo("⚠️  未找到新的 Claude Code 会话（可能已全部导入，或 ~/.claude/projects 不存在）")
            click.echo("   其他来源: --sample 生成模拟数据, --cursor 采 Cursor, 或 <file_path>")
            return
        count = collector.import_sessions(sessions)
        click.echo(f"✅ 导入 {count} 条 Claude Code 会话")


@cli.command(name="process")
@click.option("--session", "-s", type=int, help="处理指定会话ID")
@click.option("--watch", is_flag=True,
              help="监听模式：轮询 ~/.claude/projects，有新会话自动加工（长驻，Ctrl-C 退出）")
@click.option("--interval", default=None, type=int,
              help="--watch 轮询间隔（秒），默认读 config.WATCH_POLL_INTERVAL（60）")
def process_cmd(session, watch, interval):
    """🌙 处理会话(做梦)

    不带 flag：加工所有 pending 会话（受并发锁保护，与 --watch / launchd 互不重叠）。
    --watch：长驻监听，发现新 Claude 会话即自动采集 + 加工。
    """
    store.init_db()

    if watch:
        dreamd.setup_dream_logging()
        stop = threading.Event()
        dreamd.install_signal_handlers(stop)
        poll = interval if interval is not None else config.WATCH_POLL_INTERVAL
        click.echo(f"🌙 监听模式启动：每 {poll}s 轮询 {config.CLAUDE_PROJECTS_PATH}（Ctrl-C 退出）")
        dreamd.watch_loop(poll_interval=poll, stop_event=stop, verbose=True)
        return

    if session:
        # 单会话加工同样受锁保护，避免与正在跑的全量周期撞车
        try:
            with dreamd.acquire_dream_lock(blocking=False):
                click.echo(f"🔄 处理会话 #{session}...")
                result = processor.process_session(session)
                if result:
                    click.echo(f"✅ 完成 -> story #{result}")
                else:
                    click.echo("❌ 处理失败")
        except dreamd.DreamLockBusy:
            click.echo("⏳ 另一个做梦周期正在运行，已跳过")
        return

    # 全量加工（不采集，仅处理 pending），走锁保护的统一路径
    result = dreamd.run_dream_cycle_once(import_new=False, verbose=True)
    if result["status"] == "skipped":
        click.echo("⏳ 另一个做梦周期正在运行，已跳过")
    elif result["total"] == 0:
        click.echo("✅ 没有待处理的会话")


@cli.command()
@click.option("--once", is_flag=True,
              help="只跑一次完整做梦周期（采集+加工）后退出；launchd 定时任务用此入口")
@click.option("--interval", default=None, type=int,
              help="守护进程循环间隔（秒），默认读 config.DREAM_INTERVAL（14400 = 4 小时）")
def dream(once, interval):
    """🌙 做梦周期自动化：定时触发或文件监听，让记忆在后台自动整理。

    --once：单次完整周期（采集 Claude 会话 + 加工 pending）后退出。launchd / cron 调用此入口。
    不带 --once：定时守护进程（非 macOS 兜底），每 interval 秒一轮，Ctrl-C / SIGTERM 退出。
    """
    store.init_db()
    dreamd.setup_dream_logging()

    if once:
        result = dreamd.run_dream_cycle_once(import_new=True, verbose=True)
        if result["status"] == "skipped":
            click.echo("⏳ 另一个做梦周期正在运行，已跳过")
        else:
            click.echo(
                f"🌙 做梦周期完成：采集 {result['imported']} 条，"
                f"加工 {result['total']} 条（成功 {result['success']} / 失败 {result['failed']}），"
                f"用时 {result['duration_s']}s"
            )
        return

    stop = threading.Event()
    dreamd.install_signal_handlers(stop)
    iv = interval if interval is not None else config.DREAM_INTERVAL
    click.echo(f"🌙 做梦守护进程启动：每 {iv}s 触发一次（Ctrl-C 退出）")
    dreamd.dream_daemon(interval=iv, stop_event=stop, verbose=False)


@cli.command()
@click.argument("query")
@click.option("--top", "-t", default=3, help="返回Top N")
def search(query, top):
    """🔍 搜索记忆"""
    store.init_db()
    result = search_module.search(query, top_k=top)
    output = search_module.format_search_result(result)
    click.echo(output)


@cli.command()
def stats():
    """📊 系统统计"""
    store.init_db()
    s = store.get_stats()
    click.echo("\n📊 Storybook 记忆系统统计")
    click.echo("──────────────────────────────")
    click.echo(
        f"  Profile:      {s['profile']['display_name']} "
        f"({s['profile']['mode']})"
    )
    click.echo(f"  Sync:         {s['sync_state']}")
    click.echo(f"  会话总数:   {s['sessions']}")
    click.echo(f"  待处理:     {s['pending']}")
    click.echo(f"  已处理:     {s['processed']}")
    click.echo(f"  Story 记忆: {s['stories']}")
    click.echo(f"  根 Story:   {s['root_stories']}")
    click.echo(f"  子 Story:   {s['child_stories']}")
    click.echo(f"  关联边:     {s['edges']}")
    click.echo("──────────────────────────────\n")


@cli.command()
@click.option("--performance", "include_performance", is_flag=True,
              help="汇总最近 100 次查询的 p50/p95、cache 与 fallback 比例")
@click.option("--json", "as_json", is_flag=True, help="输出结构化 JSON")
def status(include_performance, as_json):
    """📟 查看本地运行状态与可选查询性能摘要。"""
    store.init_db()
    stats_data = store.get_stats()
    payload = {
        "status": "ready",
        "stories": stats_data["stories"],
        "sessions": stats_data["sessions"],
        "pending": stats_data["pending"],
    }
    if include_performance:
        payload["performance"] = performance.summarize_query_performance(limit=100)

    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    click.echo("Overall        READY")
    click.echo(f"Stories        {payload['stories']}")
    click.echo(f"Sessions       {payload['sessions']} ({payload['pending']} pending)")
    if include_performance:
        summary = payload["performance"]
        if not summary["sample_size"]:
            click.echo("Search (0)     暂无查询诊断数据")
            return
        total = summary["latency_ms"]["total"]
        click.echo(
            f"Search ({summary['sample_size']})   "
            f"p50 {total['p50']:.1f}ms · p95 {total['p95']:.1f}ms · "
            f"cache {summary['cache_hit_ratio']:.1%} · "
            f"fallback {summary['fallback_ratio']:.1%}"
        )
        stage_bits = [
            f"{stage} {summary['latency_ms'][stage]['p95']:.1f}"
            for stage in performance.LATENCY_STAGES[:-1]
        ]
        click.echo("Stage p95(ms)  " + " · ".join(stage_bits))


@cli.command(name="list")
@click.option("--limit", "-l", default=20, help="显示数量")
def list_cmd(limit):
    """📋 列出所有 Story"""
    store.init_db()
    stories = store.get_all_stories()
    if not stories:
        click.echo("暂无记忆。使用 storybook import-data --sample 导入测试数据。")
        return

    click.echo(f"\n📋 Story 记忆库 ({len(stories)} 条)\n")
    for s in stories[:limit]:
        parent = f" ← #{s['parent_id']}" if s.get("parent_id") else ""
        click.echo(f"  #{s['id']}{parent} [{s['version']}] {s['title']}")
        click.echo(f"     {s['content'][:80]}...")
        click.echo(f"     关键词: {', '.join(s.get('keywords', []))}")
        click.echo(f"     访问: {s.get('access_count', 0)} 次 | 更新: {s.get('updated_at', '?')}")
        click.echo("")


@cli.command()
@click.argument("story_id")
def show(story_id):
    """🔎 查看 Story 详情"""
    store.init_db()
    story = store.get_story(int(story_id))
    if not story:
        click.echo(f"❌ Story #{story_id} 不存在")
        return

    click.echo(f"\n📌 Story #{story['id']}")
    click.echo(f"   标题: {story['title']}")
    click.echo(f"   版本: v{story['version']}")
    click.echo(f"   内容:\n   {story['content']}")
    click.echo(f"   关键词: {', '.join(story.get('keywords', []))}")
    click.echo(f"   访问: {story.get('access_count', 0)} 次")
    click.echo(f"   来源会话: {story.get('source_session_ids', [])}")
    if story.get("parent_id"):
        click.echo(f"   父 Story: #{story['parent_id']}")

    related = store.get_related_stories(story["id"], limit=10)
    if related:
        click.echo(f"\n   🔗 关联记忆 ({len(related)} 条):")
        for r in related:
            tag = "👨‍👦" if r.get("edge_type") == "parent_child" else "💭"
            click.echo(f"      {tag} [{r.get('weight', 0):.2f}] #{r['id']} {r['title']}")
    click.echo("")


def main():
    cli()


@cli.command()
@click.argument("part", required=False, default="all",
                type=click.Choice(["all", "retrieval", "processing", "split"]))
@click.option("--report", "-r", type=click.Path(dir_okay=False, writable=True),
              help="把完整 JSON 报告写入该路径")
@click.option("--benchmark", "benchmark_path", type=click.Path(exists=True, dir_okay=False),
              help="自定义 benchmark 数据集 JSON（默认 data/retrieval_benchmark.json）")
def eval(part, report, benchmark_path):
    """📐 检索质量评测（benchmark + recall@k + 合并正确率 + 分裂质量）

    PART 取值：retrieval / processing / split / all（默认 all）。

    retrieval 用真实 embedding + 人工标注 story 语料，度量 recall@1/3/5、precision@k、MRR、
    阈值敏感性曲线，并判定是否达 PRD「重复 bug 检索准确率≥70%」(recall@3)。
    processing 用真实 embedding + 确定性 LLM 桩，度量 merge/update 分支是否选对。
    split 度量分裂路径结构正确性。

    需要 Ollama 运行（embedding）。评测在隔离临时库中进行，不污染用户 Profile 数据库。
    用 --report 把可复现的 JSON 报告落盘，便于阈值调整前后量化对比。
    """
    parts = ("retrieval", "processing", "split") if part == "all" else (part,)
    click.echo(f"📐 运行评测: {', '.join(parts)}（embedding 走真实 Ollama）\n")

    bp = benchmark_path
    try:
        rep = eval_module.run_all(parts=parts, benchmark_path=bp)
    except Exception as ex:
        click.echo(f"❌ 评测失败: {ex}")
        raise

    rep.meta["embed_mode"] = "ollama"
    rep.meta["part"] = part
    click.echo(eval_module.format_report(rep))

    if report:
        out = Path(report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        click.echo(f"\n📝 JSON 报告已写入: {out}")


@cli.command(name="benchmark")
@click.option("--model-state", type=click.Choice(["warm", "cold"]), default="warm",
              show_default=True, help="warm 会先预热；cold 每批请求前卸载 embedding 模型")
@click.option("--stories", type=click.IntRange(min=1), default=10_000,
              show_default=True, help="隔离数据集 Story 数")
@click.option("--queries", type=click.IntRange(min=1), default=50,
              show_default=True, help="固定查询数")
@click.option("--repeats", type=click.IntRange(min=1), default=20,
              show_default=True, help="每条查询重复次数")
@click.option("--concurrency", "concurrencies", multiple=True,
              type=click.IntRange(min=1), default=(1, 5), show_default=True,
              help="并发度；可重复指定")
@click.option("--report", "-r", type=click.Path(dir_okay=False, writable=True),
              help="把无原始 query 的完整 JSON 报告写入该路径")
@click.option("--benchmark", "benchmark_path", type=click.Path(exists=True, dir_okay=False),
              help="自定义质量 benchmark JSON")
def benchmark(model_state, stories, queries, repeats, concurrencies, report,
              benchmark_path):
    """📈 运行隔离的查询性能与质量基准。"""
    result = perf_benchmark.run_performance_benchmark(
        story_count=stories,
        query_count=queries,
        repeats=repeats,
        concurrencies=tuple(concurrencies),
        model_state=model_state,
        benchmark_path=benchmark_path,
    )
    click.echo(perf_benchmark.format_benchmark_report(result))
    if report:
        out = Path(report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        click.echo(f"\n📝 JSON 报告已写入: {out}")


if __name__ == "__main__":
    main()
