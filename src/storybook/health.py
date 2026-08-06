"""环境与健康自检 - ``book doctor``

逐项检查 Embedding API / DeepSeek LLM 配置 / Embedding 模型 / 向量维度 /
sqlite-vec 扩展与虚表 / 向量双写一致性，给出 ✅/❌ 与可操作修复建议；
``--fix`` 可修复向量双写不一致。

优先级与依赖：API 不可用时维度检查跳过；sqlite-vec 或虚表缺失时一致性检查跳过。
"""
import logging
from dataclasses import dataclass

import click
import requests

from . import config
from . import embeddings
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
        resp = requests.get(f"{config.EMBED_BASE_URL}/api/tags", timeout=5)
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
    """用统一 API 探测实际向量维度。"""
    result = embeddings.probe()
    return bool(result["ok"]), int(result["dimension"]), str(result["reason"] or "")


def serving_index_compatibility(actual_dimension: int | None) -> dict:
    """Compare configured API target with the immutable active index contract."""

    try:
        state = store.get_embedding_index_state()
        serving_dimension = store.serving_embedding_dimension()
    except Exception:  # schema can be absent during first-run diagnostics
        state, serving_dimension = {}, None
    mismatches: list[str] = []
    active_model = state.get("active_model")
    active_version = state.get("active_version")
    identity = embeddings.serving_route_identity(state)
    active_endpoint = identity["base_url"]
    active_adapter = identity["adapter"]
    active_dimension = state.get("active_dimension")
    active_api_key_env = identity["api_key_env"]
    if state and not identity["credential_known"]:
        mismatches.append("credential reference for active index is unknown")
    if state and active_endpoint != config.EMBED_BASE_URL:
        mismatches.append(
            f"endpoint active={active_endpoint or 'unknown'} target={config.EMBED_BASE_URL}"
        )
    if state and active_adapter != config.EMBED_ADAPTER:
        mismatches.append(
            f"adapter active={active_adapter or 'unknown'} target={config.EMBED_ADAPTER}"
        )
    if state and active_api_key_env != config.EMBED_API_KEY_ENV:
        mismatches.append("credential reference differs from active index")
    if state and active_dimension != serving_dimension:
        mismatches.append(
            f"dimension identity={active_dimension or 'unknown'} active={serving_dimension}"
        )
    if serving_dimension and actual_dimension and serving_dimension != actual_dimension:
        mismatches.append(
            f"dimension active={serving_dimension} api={actual_dimension}"
        )
    if active_model and active_model != config.EMBED_MODEL:
        mismatches.append(f"model active={active_model} target={config.EMBED_MODEL}")
    if active_version and active_version != config.EMBED_VERSION:
        mismatches.append(
            f"version active={active_version} target={config.EMBED_VERSION}"
        )
    return {
        "ok": not mismatches,
        "reason": None if not mismatches else "serving_index_mismatch",
        "serving_dimension": serving_dimension,
        "active_model": active_model,
        "active_version": active_version,
        "active_endpoint": active_endpoint,
        "active_adapter": active_adapter,
        "detail": "; ".join(mismatches),
    }


# ═══════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════

def run_doctor(fix: bool = False) -> bool:
    """执行全部检查并打印报告。

    fix=True 时对向量双写不一致自动修复，并在修复后复检该项。
    返回整体是否通过（无未跳过的 ❌）。
    """
    results: list[CheckResult] = []

    # [1] 顶层始终是 API；只有 Ollama adapter 会枚举本地模型。
    tags = None
    if config.EMBED_ADAPTER == "ollama":
        endpoint_ok, tags, endpoint_err = _check_ollama_reachable()
        detail = f"type=api，adapter=ollama，{config.EMBED_BASE_URL}"
        if endpoint_ok:
            detail += f"（已加载 {len(tags.get('models', []))} 个模型）"
        else:
            detail += f"，reason=endpoint_unreachable：{endpoint_err}"
        results.append(CheckResult(
            "Embedding API", endpoint_ok, detail=detail,
            suggestion="启动 Ollama：`ollama serve`（或设置 STORYBOOK_EMBED_BASE_URL）"
            if not endpoint_ok else ""))
        model_ready = endpoint_ok and _model_pulled(tags, config.EMBED_MODEL)
        probe_result = None
    else:
        probe_result = embeddings.probe()
        endpoint_ok = bool(probe_result["ok"]) or probe_result["reason"] == "dimension_mismatch"
        model_ready = endpoint_ok
        results.append(CheckResult(
            "Embedding API", endpoint_ok,
            detail=(f"type=api，adapter={config.EMBED_ADAPTER}，"
                    f"{config.EMBED_BASE_URL}"
                    + ("" if endpoint_ok else f"，reason={probe_result['reason']}")),
            suggestion="检查 endpoint、凭据环境变量、模型名与响应协议"
            if not endpoint_ok else ""))

    # [2] 云端生成式 LLM 只做无费用的配置就绪检查，不发送生成请求，也不依赖 Ollama。
    if config.LLM_API_KEY:
        results.append(CheckResult(
            "LLM 配置",
            True,
            detail=f"provider={config.LLM_PROVIDER}，model={config.LLM_MODEL}"))
    else:
        results.append(CheckResult(
            "LLM 配置",
            False,
            detail=(f"provider={config.LLM_PROVIDER}，model={config.LLM_MODEL}，"
                    "reason=llm_credentials_missing"),
            suggestion=("设置 ANTHROPIC_AUTH_TOKEN（或 DEEPSEEK_KEY），"
                        "也可通过 STORYBOOK_LLM_ENV_FILE 指定配置文件")))

    # [3] Embedding 模型已拉取
    if not endpoint_ok:
        results.append(CheckResult("Embedding 模型", False, skipped=True,
                                   detail=f"{config.EMBED_MODEL}（API 不可用，跳过）"))
    elif model_ready:
        results.append(CheckResult("Embedding 模型", True, detail=config.EMBED_MODEL))
    else:
        results.append(CheckResult("Embedding 模型", False, detail=config.EMBED_MODEL,
                                   suggestion=f"`ollama pull {config.EMBED_MODEL}`"
                                   if config.EMBED_ADAPTER == "ollama"
                                   else "检查 API 中的模型名与授权"))

    # [4] Embedding 维度一致
    if not model_ready:
        results.append(CheckResult("Embedding 维度", False, skipped=True,
                                   detail=f"期望 {config.EMBED_DIM}（Embedding 模型不可用，跳过）"))
    else:
        if probe_result is None:
            dim_ok, actual, dim_err = _probe_embed_dim()
        else:
            actual = int(probe_result["dimension"])
            dim_ok = bool(probe_result["ok"])
            dim_err = str(probe_result["reason"] or "")
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
                                f"或同步调整 config.EMBED_DIM 后 `book init` 重建虚表")))
        else:
            results.append(CheckResult(
                "Embedding 维度", False,
                detail=f"探测失败：{dim_err}",
                suggestion="检查 embedding endpoint、凭据、模型与响应协议"))
    if config.EMBED_ADAPTER == "ollama":
        results.append(CheckResult(
            "Ollama warm/cold", True,
            detail=f"model_state={embeddings.model_state()}"))

    # [5] Target config must not masquerade as the active serving index.
    compatibility = serving_index_compatibility(actual if model_ready else None)
    if compatibility["ok"]:
        results.append(CheckResult(
            "Serving index 兼容性", True,
            detail=(f"dimension={compatibility['serving_dimension'] or 'uninitialized'}，"
                    f"model={compatibility['active_model'] or 'uninitialized'}，"
                    f"version={compatibility['active_version'] or 'uninitialized'}，"
                    f"adapter={compatibility['active_adapter'] or 'uninitialized'}，"
                    f"endpoint={compatibility['active_endpoint'] or 'uninitialized'}")))
    else:
        results.append(CheckResult(
            "Serving index 兼容性", False,
            detail=f"reason=serving_index_mismatch：{compatibility['detail']}",
            suggestion=("保留当前 serving index；使用 `book admin index` "
                        "完成 shadow 后再原子切换")))

    # [6] sqlite-vec 扩展 + story_vectors 虚表
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
            suggestion="`book init`"))
    elif not vec_exists:
        results.append(CheckResult(
            "sqlite-vec 扩展 + story_vectors 虚表", False,
            detail="sqlite-vec ✅，但 story_vectors 虚表缺失",
            suggestion="`book init`（幂等，会补建虚表）"))
    else:
        results.append(CheckResult(
            "sqlite-vec 扩展 + story_vectors 虚表", True,
            detail="扩展加载成功，虚表存在"))

    # [7] 向量双写一致性
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
                suggestion="`book doctor --fix` 重建缺失行 / 清除孤立行"))

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
