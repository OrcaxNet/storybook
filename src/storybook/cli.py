"""
CLI 入口 — storybook 命令

用法:
  storybook init                    初始化数据库
  storybook doctor [--fix]          环境与健康自检（--fix 修复向量双写不一致）
  storybook import <path>           导入会话日志(JSON)
  storybook import                  从 Claude Code 采集（默认数据源）
  storybook import --claude         从 Claude Code 采集（同上，显式写法）
  storybook import --cursor         从 Cursor 自动采集（备用，未用 Cursor 可忽略）
  storybook import --sample [N]     生成N条模拟数据(默认100)
  storybook process                 处理所有pending会话(做梦)
  storybook process --session ID    处理指定会话
  storybook search <query>          搜索记忆
  storybook stats                   查看统计
  storybook list                    列出所有story
  storybook show <story_id>         查看story详情
"""
import json
import logging
import sys
from pathlib import Path

import click

from . import config
from . import store
from . import collector
from . import processor
from . import search as search_module
from . import health
from . import eval as eval_module


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
        import os as _os
        click.echo(f"📥 从路径导入: {path}")
        sessions = []
        if _os.path.isdir(path):
            # 目录：扫描所有 .json 文件
            for fname in sorted(_os.listdir(path)):
                if not fname.endswith(".json"):
                    continue
                fpath = _os.path.join(path, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        sessions.extend(data)
                    elif isinstance(data, dict) and "sessions" in data:
                        sessions.extend(data["sessions"])
                    elif isinstance(data, dict):
                        sessions.append(data)
                except Exception as e:
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
def process_cmd(session):
    """🌙 处理会话(做梦)"""
    store.init_db()

    if session:
        click.echo(f"🔄 处理会话 #{session}...")
        result = processor.process_session(session)
        if result:
            click.echo(f"✅ 完成 → story #{result}")
        else:
            click.echo("❌ 处理失败")
    else:
        summary = processor.process_all_pending(verbose=True)
        if summary["total"] == 0:
            click.echo("✅ 没有待处理的会话")


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
    click.echo(f"  会话总数:   {s['sessions']}")
    click.echo(f"  待处理:     {s['pending']}")
    click.echo(f"  已处理:     {s['processed']}")
    click.echo(f"  Story 记忆: {s['stories']}")
    click.echo(f"  根 Story:   {s['root_stories']}")
    click.echo(f"  子 Story:   {s['child_stories']}")
    click.echo(f"  关联边:     {s['edges']}")
    click.echo("──────────────────────────────\n")


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

    需要 Ollama 运行（embedding）。评测在隔离临时库中进行，不污染 data/memory.db。
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


if __name__ == "__main__":
    main()
