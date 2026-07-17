"""环境与健康自检 - ``storybook doctor``

逐项检查 Ollama 可达性 / LLM 与 Embedding 模型 / 向量维度 / sqlite-vec 扩展与虚表 /
向量双写一致性，给出 ✅/❌ 与可操作修复建议；``--fix`` 可修复向量双写不一致。

优先级与依赖：Ollama 不可达时模型/维度检查跳过；sqlite-vec 或虚表缺失时一致性检查跳过。
"""
import logging
from dataclasses import dataclass

import click
import requests

from . import config
from . import store

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """单项检查结果。skipped 表示因上游失败而跳过（不算独立失败）。"""
    name: str
    ok: bool
    skipped: bool = False
    detail: str = ""
    suggestion: str = ""


# ═══════════════════════════════════════════════
#  探测函数
# ═══════════════════════════════════════════════

def _check_ollama_reachable() -> tuple[bool, dict | None, str]:
    """GET {OLLAMA_HOST}/api/tags。返回 (可达, tags JSON, 错误信息)。"""
    try:
        resp = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=5)
        resp.raise_for_status()
        return True, resp.json(), ""
    except Exception as e:
        return False, None, str(e)


def _model_pulled(tags: dict | None, model: str) -> bool:
    """配置的模型是否已拉取（精确匹配 name/model 字段）。"""
    if not tags:
        return False
    for m in tags.get("models", []):
        if m.get("name") == model or m.get("model") == model:
            return True
    return False


def _probe_embed_dim() -> tuple[bool, int, str]:
    """用 EMBED_MODEL 探测实际向量维度。返回 (成功, 维度, 错误信息)。"""
    try:
        resp = requests.post(
            f"{config.OLLAMA_HOST}/api/embeddings",
            json={"model": config.EMBED_MODEL, "prompt": "storybook doctor probe"},
            timeout=30,
        )
        resp.raise_for_status()
        vec = resp.json().get("embedding") or []
        return True, len(vec), ""
    except Exception as e:
        return False, 0, str(e)


# ═══════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════

def run_doctor(fix: bool = False) -> bool:
    """执行全部检查并打印报告。

    fix=True 时对向量双写不一致自动修复，并在修复后复检该项。
    返回整体是否通过（无未跳过的 ❌）。
    """
    results: list[CheckResult] = []

    # [1] Ollama 可达
    ollama_ok, tags, ollama_err = _check_ollama_reachable()
    if ollama_ok:
        n = len(tags.get("models", []))
        results.append(CheckResult(
            "Ollama 服务", True,
            detail=f"{config.OLLAMA_HOST}（已加载 {n} 个模型）"))
    else:
        results.append(CheckResult(
            "Ollama 服务", False,
            detail=f"{config.OLLAMA_HOST} 不可达：{ollama_err}",
            suggestion="启动 Ollama：`ollama serve`（或设置 OLLAMA_HOST 指向正确地址）"))

    embed_pulled = ollama_ok and _model_pulled(tags, config.EMBED_MODEL)

    # [2] LLM 模型已拉取
    if not ollama_ok:
        results.append(CheckResult("LLM 模型", False, skipped=True,
                                   detail=f"{config.LLM_MODEL}（Ollama 不可达，跳过）"))
    elif _model_pulled(tags, config.LLM_MODEL):
        results.append(CheckResult("LLM 模型", True, detail=config.LLM_MODEL))
    else:
        results.append(CheckResult("LLM 模型", False, detail=config.LLM_MODEL,
                                   suggestion=f"`ollama pull {config.LLM_MODEL}`"))

    # [3] Embedding 模型已拉取
    if not ollama_ok:
        results.append(CheckResult("Embedding 模型", False, skipped=True,
                                   detail=f"{config.EMBED_MODEL}（Ollama 不可达，跳过）"))
    elif embed_pulled:
        results.append(CheckResult("Embedding 模型", True, detail=config.EMBED_MODEL))
    else:
        results.append(CheckResult("Embedding 模型", False, detail=config.EMBED_MODEL,
                                   suggestion=f"`ollama pull {config.EMBED_MODEL}`"))

    # [4] Embedding 维度一致
    if not embed_pulled:
        results.append(CheckResult("Embedding 维度", False, skipped=True,
                                   detail=f"期望 {config.EMBED_DIM}（Embedding 模型不可用，跳过）"))
    else:
        dim_ok, actual, dim_err = _probe_embed_dim()
        if dim_ok:
            if actual == config.EMBED_DIM:
                results.append(CheckResult(
                    "Embedding 维度", True,
                    detail=f"期望 {config.EMBED_DIM}，实际 {actual}"))
            else:
                results.append(CheckResult(
                    "Embedding 维度", False,
                    detail=f"期望 {config.EMBED_DIM}，实际 {actual}（不一致）",
                    suggestion=(f"换用 {config.EMBED_DIM} 维的 embedding 模型，"
                                f"或同步调整 config.EMBED_DIM 后 `storybook init` 重建虚表")))
        else:
            results.append(CheckResult(
                "Embedding 维度", False,
                detail=f"探测失败：{dim_err}",
                suggestion="检查 Ollama embedding 接口与模型是否正常"))

    # [5] sqlite-vec 扩展 + story_vectors 虚表
    ext_ok = store.check_vec_extension()
    db_exists = config.DB_PATH.exists()
    stories_exists = store.stories_table_exists()
    vec_exists = store.story_vectors_table_exists()
    if not ext_ok:
        results.append(CheckResult(
            "sqlite-vec 扩展 + story_vectors 虚表", False,
            detail="sqlite-vec 扩展加载失败",
            suggestion="`VIRTUAL_ENV=$(pwd)/.venv uv pip install --force-reinstall sqlite-vec`"))
    elif not db_exists or not stories_exists:
        results.append(CheckResult(
            "sqlite-vec 扩展 + story_vectors 虚表", False,
            detail="数据库未初始化（缺少 stories 表）",
            suggestion="`storybook init`"))
    elif not vec_exists:
        results.append(CheckResult(
            "sqlite-vec 扩展 + story_vectors 虚表", False,
            detail="sqlite-vec ✅，但 story_vectors 虚表缺失",
            suggestion="`storybook init`（幂等，会补建虚表）"))
    else:
        results.append(CheckResult(
            "sqlite-vec 扩展 + story_vectors 虚表", True,
            detail="扩展加载成功，虚表存在"))

    # [6] 向量双写一致性
    can_check = ext_ok and db_exists and stories_exists and vec_exists
    if not can_check:
        results.append(CheckResult("向量双写一致性", False, skipped=True,
                                   detail="依赖未就绪（见上一项），跳过"))
    else:
        cons = store.vector_consistency()
        inconsistent = len(cons["missing_vec"]) + len(cons["orphan_vec"])
        if inconsistent == 0:
            results.append(CheckResult(
                "向量双写一致性", True,
                detail=(f"stories 有 BLOB {cons['blob_count']} 行，"
                        f"story_vectors {cons['vec_count']} 行，一致")))
        else:
            results.append(CheckResult(
                "向量双写一致性", False,
                detail=(f"不一致 {inconsistent} 项：缺 vec0 行 {len(cons['missing_vec'])}，"
                        f"孤立 vec0 行 {len(cons['orphan_vec'])}"
                        f"（BLOB {cons['blob_count']} / vec0 {cons['vec_count']}）"),
                suggestion="`storybook doctor --fix` 重建缺失行 / 清除孤立行"))

    # ── --fix：修复向量双写一致性并复检 ──
    fix_line = None
    if fix:
        if can_check:
            cons = store.vector_consistency()
            if len(cons["missing_vec"]) + len(cons["orphan_vec"]) > 0:
                fr = store.repair_vector_consistency()
                recons = store.vector_consistency()
                remaining = len(recons["missing_vec"]) + len(recons["orphan_vec"])
                parts = [f"重建 {fr['rebuilt']} 行", f"清除 {fr['cleared']} 行"]
                if fr["failed"]:
                    parts.append(f"失败 {len(fr['failed'])}")
                parts.append(f"剩余 {remaining} 项")
                fix_line = "🔧 修复：" + "，".join(parts)
                if remaining == 0:
                    results[-1] = CheckResult(
                        "向量双写一致性", True,
                        detail=(f"已修复：重建 {fr['rebuilt']} 行，清除 {fr['cleared']} 行；"
                                f"现 BLOB {recons['blob_count']} / vec0 {recons['vec_count']}，一致"))
                else:
                    results[-1].detail += f" | 修复后仍剩 {remaining} 项"
            else:
                fix_line = "🔧 无不一致，无需修复"
        else:
            fix_line = "🔧 跳过修复：向量依赖未就绪（见 sqlite-vec / 虚表 检查项）"

    _print_report(results, fix_line)
    return all(r.ok for r in results if not r.skipped)


def _print_report(results: list[CheckResult], fix_line: str | None) -> None:
    n = len(results)
    click.echo("\n🩺 Storybook 环境自检\n")
    for i, r in enumerate(results, 1):
        mark = "⏭️ " if r.skipped else ("✅" if r.ok else "❌")
        click.echo(f"  {mark} [{i}/{n}] {r.name}")
        if r.detail:
            click.echo(f"       {r.detail}")
        if r.suggestion:
            click.echo(f"       -> {r.suggestion}")
    click.echo("")
    if fix_line:
        click.echo(fix_line)
    failed = [r for r in results if not r.ok and not r.skipped]
    if not failed:
        click.echo("✅ 全部通过\n")
    else:
        click.echo(f"❌ {len(failed)} 项未通过\n")
