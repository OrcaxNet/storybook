"""
CLI 入口 — canonical ``book`` 命令（兼容 alias：``storybook``）

Canonical 信息架构（FLO-180）:

  高频任务:
    book init                   交互式初始化向导（Profile → 模型 → Agent → schedule → smoke）
    book doctor [--fix]         环境与健康自检（--fix 修复向量双写不一致）
    book run [模式]             采集并形成记忆（默认 / --once / --watch / --daemon / --session）
    book search <query>         搜索记忆
    book status [--performance] 本地运行状态与可选查询性能摘要
    book version                查看当前版本
    book mcp                    启动 MCP server（stdio，供 Claude Code 等 agent 召回）

  分组:
    book memory list|show|forget    记忆管理
    book source list|enable|disable|reset   本机 Agent 历史来源管理
    book profile list|show|create|switch    用户级 Profile
    book admin init-db|migration|index|benchmark|eval|uninstall   低频维护

  兼容 alias（默认从 help 隐藏，保留一个 minor release 后移除）:
    ``storybook``（executable 别名）、``setup``、``process``、``dream``、
    ``import-data``、``sources``、顶层 ``list/show/forget/stats``；
    以及内部/hook 入口 ``sync``、``prime``。
    兼容提示只写 stderr，绝不污染 JSON / MCP stdout 协议。
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
    embeddings,
    health,
    model_config,
    migration as migration_module,
    perf_benchmark,
    performance,
    processor,
    store,
)
from . import eval as eval_module
from . import search as search_module
from . import context as context_module
from .profiles import ProfileError
from .migration import MigrationError
from .setup_manager import SetupError, SetupManager
from .history_adapters import manager as source_manager


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _legacy_alias_hint(canonical: str) -> None:
    """隐藏兼容 alias 的 stderr 提示；绝不写入 stdout（JSON/MCP 协议纯净）。"""
    click.echo(
        "提示: 本命令为兼容 alias，将在一个 minor release 后移除；"
        f"请使用 `book {canonical}`。",
        err=True,
    )


def _legacy_executable_hint(ctx: click.Context) -> None:
    """经 ``storybook`` executable 调用时输出 stderr 提示；stdout 不受影响。"""
    if ctx.find_root().info_name == "storybook":
        click.echo(
            "提示: `storybook` 为兼容入口，将在一个 minor release 后移除；"
            "请使用 `book`。",
            err=True,
        )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="详细日志")
@click.pass_context
def cli(ctx, verbose):
    """🧠 Storybook - 离线 Coding 记忆系统

    \b
    高频任务:  init / doctor / run / search / status / version / mcp
    分组:       memory / source / profile / admin
    """
    setup_logging(verbose)
    _legacy_executable_hint(ctx)


@cli.group(name="memory")
def memory_group():
    """🧠 记忆管理（list / show / forget）。"""


@cli.group(name="source")
def source_group():
    """🔌 管理本机 Agent 历史来源（list / enable / disable / reset）。"""


@cli.group(name="admin")
def admin_group():
    """🔧 低频维护命令（init-db / migration / index / benchmark / eval / uninstall）。"""


@admin_group.group(name="migration")
def migration_group():
    """🧳 发现、迁移和回滚旧项目级数据库。"""


def _emit_setup_error(exc: SetupError, *, as_json: bool) -> None:
    hint = exc.hint
    if exc.code.startswith("SB_MODEL_") and "book doctor" not in (hint or ""):
        hint = "; ".join(
            item for item in (hint, "运行 `book doctor` 诊断 provider 后重试") if item
        )
    if as_json:
        error = exc.as_dict()
        error["hint"] = hint
        click.echo(
            json.dumps(
                {"status": "failed", "error": error},
                ensure_ascii=False,
                indent=2,
            )
        )
        raise click.exceptions.Exit(1)
    rendered_hint = f"\n修复建议: {hint}" if hint else ""
    raise click.ClickException(f"[{exc.code}] {exc}{rendered_hint}") from exc


def _print_setup_plan(plan: dict) -> None:
    profile = plan["profile"]
    click.echo("Storybook setup plan")
    click.echo(
        f"  Profile   {profile['action']} {profile['display_name']} "
        f"({profile['sync_state']})"
    )
    for adapter in plan["adapters"]:
        marker = "configure" if adapter["selected"] and adapter["changed"] else (
            "ready" if adapter["selected"] else "skip"
        )
        click.echo(f"  Agent     {adapter['display_name']}: {marker}")
        for target in adapter["targets"]:
            click.echo(f"            {target}")
    embedding = plan["embedding"]
    label = "Ollama (recommended)" if embedding["preset"] == "ollama" else "Custom API"
    click.echo(
        f"  Embedding API: {label}; adapter={embedding['adapter']}; "
        f"model={embedding['model']}; dimension={embedding['dimension']}"
    )
    if embedding["remote_text_disclosure"]:
        click.echo("            Warning: embedding text is sent to the configured endpoint")
    click.echo(f"  Models    {', '.join(plan['models'])}")
    if plan["legacy_databases"]:
        click.echo("  Legacy    found; migration is not automatic:")
        for path in plan["legacy_databases"]:
            click.echo(f"            {path}")
        click.echo("            run: book admin migration run <path> --dry-run")


@cli.command(name="setup", hidden=True)
@click.option("--yes", "assume_yes", is_flag=True, help="接受计划并非交互执行")
@click.option("--dry-run", is_flag=True, help="只输出计划；零文件、数据库和网络写入")
@click.option("--json", "as_json", is_flag=True, help="输出稳定 JSON 结构")
@click.option(
    "--agent",
    "agents",
    multiple=True,
    type=click.Choice(["claude", "cursor", "codex"]),
    help="只配置指定 Agent；可重复。默认自动检测",
)
@click.option(
    "--embedding-preset",
    type=click.Choice(["ollama", "custom"]),
    help="选择 Ollama（推荐）或自定义 OpenAI-compatible API",
)
@click.option("--embedding-base-url", help="embedding API base URL")
@click.option("--embedding-model", help="embedding 模型名")
@click.option("--embedding-dimension", type=click.IntRange(min=1), help="向量维度")
@click.option("--embedding-version", help="不可变 embedding 版本；切换模型/维度时应使用新值")
@click.option(
    "--embedding-api-key-env",
    help="凭据所在的环境变量名（不接受明文凭据）",
)
@click.option(
    "--provider",
    type=click.Choice(["ollama", "api", "anthropic"]),
    help="模型 provider（旧 shorthand，同时作用于 generation 与 embedding 两个端点）",
)
@click.option("--base-url", help="provider 根 URL（旧 shorthand，同时作用于两个端点）")
@click.option("--llm-model", help="generation 模型名")
@click.option("--api-key-env", help="API key 环境变量名（旧 shorthand，同时作用于两个端点）")
@click.option(
    "--llm-protocol",
    type=click.Choice(["ollama", "openai", "anthropic"]),
    help="generation 端点协议（openai=OpenAI 兼容需 /v1；anthropic 走 /v1/messages）",
)
@click.option("--llm-base-url", help="generation 端点 base URL")
@click.option("--llm-api-key-env", help="generation 端点凭据环境变量名")
@click.option(
    "--embedding-protocol",
    type=click.Choice(["ollama", "openai", "anthropic"]),
    help="embedding 端点协议（默认 ollama）",
)
@click.option("--enable-schedule", is_flag=True, help="配置用户级 watch schedule")
@click.option("--skip-models", "--skip-download", is_flag=True,
              help="不下载缺失 Ollama 模型并进入 degraded")
def setup_command(
    assume_yes, dry_run, as_json, agents, provider, base_url, llm_model,
    embedding_model, api_key_env, embedding_preset, embedding_base_url,
    embedding_dimension, embedding_version, embedding_api_key_env,
    llm_protocol, llm_base_url, llm_api_key_env, embedding_protocol,
    enable_schedule, skip_models,
):
    """兼容 alias；canonical onboarding 命令为 ``book init``。"""
    _legacy_alias_hint("init")

    return _run_onboarding(
        assume_yes=assume_yes,
        dry_run=dry_run,
        as_json=as_json,
        agents=agents,
        provider=provider,
        base_url=base_url,
        llm_model=llm_model,
        embedding_model=embedding_model,
        api_key_env=api_key_env,
        embedding_preset=embedding_preset,
        embedding_base_url=embedding_base_url,
        embedding_dimension=embedding_dimension,
        embedding_version=embedding_version,
        embedding_api_key_env=embedding_api_key_env,
        llm_protocol=llm_protocol,
        llm_base_url=llm_base_url,
        llm_api_key_env=llm_api_key_env,
        embedding_protocol=embedding_protocol,
        enable_schedule=enable_schedule,
        skip_models=skip_models,
        full_onboarding=False,
    )


def _base_url_hint(protocol: str, *, kind: str) -> str:
    """协议标注 + /v1 说明的 baseUrl 提示文案（AC3）。"""

    if protocol == "ollama":
        return f"{kind} base URL（ollama 原生 http://host:11434，无需 /v1）"
    if protocol == "anthropic":
        return (
            f"{kind} base URL（anthropic：Anthropic-compatible 走 /v1/messages，"
            "如 https://api.deepseek.com/anthropic）"
        )
    return (
        f"{kind} base URL（openai：OpenAI-compatible 需含 /v1，"
        "如 http://localhost:11434/v1）"
    )


def _prompt_model_endpoints(
    *, llm_protocol, llm_base_url, llm_model, llm_api_key_env,
    embedding_protocol, embedding_base_url, embedding_model, embedding_api_key_env,
    provider: str | None = None,
) -> tuple:
    """双端点交互提示：LLM → Embedding，embedding baseUrl/secret 默认继承 LLM。

    只对仍未给定的字段提示；已由 flags 提供的字段跳过。secret 留空视为无凭据
    （本地/Ollama 端点不会因空 secret 报错，AC2）。
    """

    # 旧 shorthand `--provider` 提供协议默认值：api→openai、anthropic→anthropic。
    default_llm_protocol = {
        "api": "openai", "anthropic": "anthropic", "ollama": "ollama",
    }.get(provider or "", "openai")
    if llm_protocol is None:
        llm_protocol = click.prompt(
            "LLM protocol (openai, anthropic, ollama)",
            type=click.Choice(["openai", "anthropic", "ollama"]),
            default=default_llm_protocol,
        )
    if llm_base_url is None:
        llm_base_url = click.prompt(
            _base_url_hint(llm_protocol, kind="LLM"),
            default=(
                None if llm_protocol != "ollama" else model_config.DEFAULT_OLLAMA_URL
            ),
        )
    if llm_model is None:
        llm_model = click.prompt(
            "LLM model",
            default=(
                None if llm_protocol != "ollama" else model_config.DEFAULT_LLM_MODEL
            ),
        )
    if llm_api_key_env is None:
        llm_api_key_env = click.prompt(
            "LLM secret env-name (留空表示无凭据，如本地 Ollama)",
            default="",
        ).strip() or None
    if embedding_protocol is None:
        embedding_protocol = click.prompt(
            "Embedding protocol (openai, anthropic, ollama)",
            type=click.Choice(["openai", "anthropic", "ollama"]),
            default=llm_protocol,
        )
    if embedding_base_url is None:
        embedding_base_url = click.prompt(
            _base_url_hint(embedding_protocol, kind="Embedding"),
            default=llm_base_url,
        )
    if embedding_model is None:
        embedding_model = click.prompt(
            "Embedding model",
            default=(
                None
                if embedding_protocol != "ollama"
                else model_config.DEFAULT_EMBED_MODEL
            ),
        )
    if embedding_api_key_env is None:
        embedding_api_key_env = click.prompt(
            "Embedding secret env-name (默认同 LLM secret，留空表示无凭据)",
            default=llm_api_key_env or "",
        ).strip() or None
    return (
        llm_protocol, llm_base_url, llm_model, llm_api_key_env,
        embedding_protocol, embedding_base_url, embedding_model,
        embedding_api_key_env,
    )


def _run_onboarding(
    *, assume_yes, dry_run, as_json, agents, provider, base_url, llm_model,
    embedding_model, api_key_env, embedding_preset, embedding_base_url,
    embedding_dimension, embedding_version, embedding_api_key_env,
    llm_protocol, llm_base_url, llm_api_key_env, embedding_protocol,
    enable_schedule, skip_models, full_onboarding,
):
    """Shared, rerunnable implementation for ``book init`` and setup alias."""

    # 双端点模型配置面（model-config）与 embedding-preset 配置面（setup-state）
    # 是替代接口：LLM/共享 shorthand flags 只能用于 model-config 面。
    if embedding_preset and any(
        value is not None
        for value in (
            provider, base_url, llm_model, api_key_env, llm_protocol,
            llm_base_url, llm_api_key_env, embedding_protocol,
        )
    ):
        raise click.UsageError(
            "--embedding-preset 与 LLM/共享模型端点 flags 是替代的配置面，不能同用"
        )
    if not embedding_preset and any(
        value is not None for value in (embedding_dimension, embedding_version)
    ):
        raise click.UsageError(
            "embedding dimension/version 选项需与 --embedding-preset 同用"
        )
    if embedding_preset:
        try:
            config.apply_embedding_config(
                preset=embedding_preset,
                base_url=embedding_base_url,
                model=embedding_model,
                dimension=embedding_dimension,
                version=embedding_version,
                api_key_env=embedding_api_key_env,
            )
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc

    manager = SetupManager()
    agents = tuple(agents) if agents else None
    model_options_supplied = not embedding_preset and any(
        value is not None
        for value in (
            provider, base_url, llm_model, api_key_env, llm_protocol,
            llm_base_url, llm_api_key_env, embedding_protocol,
            embedding_model, embedding_base_url, embedding_api_key_env,
        )
    )
    interactive = (
        not assume_yes
        and not as_json
        and click.get_text_stream("stdin").isatty()
    )
    if interactive and full_onboarding:
        active_profile = config.PROFILE_REGISTRY.peek_active_profile()
        if active_profile is None:
            click.echo("Profile: create default (local-only)")
        else:
            click.echo(
                f"Profile: reuse {active_profile.display_name} ({active_profile.id})"
            )
    if interactive and not embedding_preset:
        # 双端点交互：LLM → Embedding；baseUrl/secret 默认继承 LLM，均可覆盖。
        (llm_protocol, llm_base_url, llm_model, llm_api_key_env,
         embedding_protocol, embedding_base_url, embedding_model,
         embedding_api_key_env) = _prompt_model_endpoints(
            llm_protocol=llm_protocol, llm_base_url=llm_base_url,
            llm_model=llm_model, llm_api_key_env=llm_api_key_env,
            embedding_protocol=embedding_protocol,
            embedding_base_url=embedding_base_url,
            embedding_model=embedding_model,
            embedding_api_key_env=embedding_api_key_env,
            provider=provider,
        )
        model_options_supplied = True
    selected_model_config = None
    if model_options_supplied:
        # 非交互下远程 provider 必须有 base URL（交互模式由必填提示保证）。
        gen_provider = model_config._provider_from_protocol(llm_protocol) or provider or "ollama"
        if gen_provider in {"api", "anthropic"} and not (llm_base_url or base_url):
            _emit_setup_error(
                SetupError(
                    "SB_MODEL_BASE_URL_REQUIRED",
                    "api/anthropic LLM provider 需要 --llm-base-url 或 --base-url",
                ),
                as_json=as_json,
            )
            return
        emb_provider = (
            model_config._provider_from_protocol(embedding_protocol)
            or provider
            or gen_provider
        )
        if (
            emb_provider in {"api", "anthropic"}
            and not (embedding_base_url or llm_base_url or base_url)
        ):
            _emit_setup_error(
                SetupError(
                    "SB_MODEL_BASE_URL_REQUIRED",
                    "api/anthropic embedding provider 需要 --embedding-base-url 或 --base-url",
                ),
                as_json=as_json,
            )
            return
        try:
            selected_model_config = model_config.build(
                provider=provider,
                base_url=base_url,
                llm_model=llm_model,
                embedding_model=embedding_model,
                api_key_env=api_key_env
                or ("STORYBOOK_API_KEY" if provider == "api" else None),
                llm_protocol=llm_protocol,
                llm_base_url=llm_base_url,
                llm_credential_env=llm_api_key_env,
                embedding_protocol=embedding_protocol,
                embedding_base_url=embedding_base_url,
                embedding_credential_env=embedding_api_key_env,
            )
        except model_config.ModelConfigError as exc:
            _emit_setup_error(
                SetupError("SB_MODEL_CONFIG_INVALID", str(exc)), as_json=as_json
            )
            return
    if interactive and full_onboarding and not agents:
        agent_answer = click.prompt(
            "Agent adapters (auto, skip, back, or comma-separated names)",
            default="auto",
        ).strip()
        if agent_answer.lower() == "back":
            return _run_onboarding(
                assume_yes=False, dry_run=False, as_json=False, agents=None,
                provider=None, base_url=None, llm_model=None,
                embedding_model=None, api_key_env=None, embedding_preset=None,
                embedding_base_url=None, embedding_dimension=None,
                embedding_version=None, embedding_api_key_env=None,
                llm_protocol=None, llm_base_url=None, llm_api_key_env=None,
                embedding_protocol=None,
                enable_schedule=False,
                skip_models=skip_models, full_onboarding=True,
            )
        if agent_answer.lower() == "skip":
            agents = ()
        elif agent_answer.lower() != "auto":
            parsed_agents = tuple(
                item.strip().lower() for item in agent_answer.split(",") if item.strip()
            )
            unknown_agents = set(parsed_agents) - {"claude", "cursor", "codex"}
            if unknown_agents:
                _emit_setup_error(
                    SetupError(
                        "SB_SETUP_AGENT_UNKNOWN",
                        f"未知 adapter: {', '.join(sorted(unknown_agents))}",
                    ),
                    as_json=as_json,
                )
                return
            agents = parsed_agents
    if interactive and full_onboarding and not enable_schedule:
        schedule_answer = click.prompt(
            "Schedule (skip, enable, back)",
            type=click.Choice(["skip", "enable", "back"]),
            default="skip",
        )
        if schedule_answer == "back":
            return _run_onboarding(
                assume_yes=False, dry_run=False, as_json=False, agents=None,
                provider=None, base_url=None, llm_model=None,
                embedding_model=None, api_key_env=None, embedding_preset=None,
                embedding_base_url=None, embedding_dimension=None,
                embedding_version=None, embedding_api_key_env=None,
                llm_protocol=None, llm_base_url=None, llm_api_key_env=None,
                embedding_protocol=None,
                enable_schedule=False,
                skip_models=skip_models, full_onboarding=True,
            )
        enable_schedule = schedule_answer == "enable"
    try:
        plan = manager.plan(
            agents, provider_config=selected_model_config
        ).as_dict()
    except SetupError as exc:
        _emit_setup_error(exc, as_json=as_json)
        return

    plan["model_config"] = (
        selected_model_config or config.MODEL_CONFIG
    ).public_dict(os.environ)
    if selected_model_config is not None:
        plan["models"] = list(dict.fromkeys((
            selected_model_config.generation.model,
            selected_model_config.embedding.model,
        )))
    if dry_run:
        plan["schedule"] = {
            "enabled": bool(enable_schedule),
            "mode": "watch" if enable_schedule else "disabled",
        }
        payload = {"status": "dry_run", "plan": plan, "writes_performed": 0}
        if as_json:
            click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            _print_setup_plan(plan)
            click.echo("\nDry-run complete: no writes performed.")
        return

    if not as_json:
        _print_setup_plan(plan)
    if not assume_yes and not as_json:
        click.confirm("Apply this plan?", abort=True)

    temporary_secrets: list[tuple[str, object]] = []
    secret_marker = object()
    if interactive and selected_model_config is not None:
        prompted_envs: set[str] = set()
        for kind, endpoint in (
            ("LLM", selected_model_config.generation),
            ("Embedding", selected_model_config.embedding),
        ):
            env_name = endpoint.credential_env
            if (
                endpoint.provider != "ollama"
                and env_name
                and not manager.environ.get(env_name)
                and env_name not in prompted_envs
            ):
                temporary_secret = click.prompt(
                    f"{kind} API key (used for this run only)", hide_input=True
                )
                previous = manager.environ.get(env_name, secret_marker)
                manager.environ[env_name] = temporary_secret
                temporary_secrets.append((env_name, previous))
                prompted_envs.add(env_name)

    def progress(event: dict) -> None:
        if as_json:
            return
        model = event.get("model", "")
        status = event.get("status", "")
        suffix = ""
        if event.get("percent") is not None:
            suffix = f" {event['percent']:.1f}%"
        if event.get("size"):
            suffix += f" / {event['size']}"
        click.echo(f"  Model     {model}: {status}{suffix}")

    try:
        try:
            result = manager.execute(
                requested_agents=agents,
                download_models=not skip_models,
                progress=progress,
                provider_config=selected_model_config,
                enable_schedule=enable_schedule,
            )
        except SetupError as exc:
            _emit_setup_error(exc, as_json=as_json)
            return
    finally:
        for env_name, previous in temporary_secrets:
            if previous is secret_marker:
                manager.environ.pop(env_name, None)
            else:
                manager.environ[env_name] = previous

    recall = next(
        (item for item in result["smoke_tests"] if item["name"] == "recall"), None
    )
    if recall and recall["ok"]:
        result["next_command"] = 'book search "what should I remember?"'
    else:
        result["next_command"] = "book doctor"
        result["repair_command"] = "book doctor"
    if as_json:
        click.echo(json.dumps(result, ensure_ascii=False, indent=2))
        return
    click.echo(f"\nSetup status: {result['status']}")
    click.echo(f"  Profile   {result['profile']['id']}")
    click.echo(f"  Database  {result['profile']['database']}")
    for smoke in result["smoke_tests"]:
        click.echo(
            f"  {'PASS' if smoke['ok'] else 'FAIL'}      {smoke['name']}: {smoke['detail']}"
        )
    for history in result.get("history_ingestion", []):
        click.echo(
            f"  HISTORY   {history['name']}: {history['status']} "
            f"({history['adapter_version']})"
        )
    for reason in result["degraded_reasons"]:
        click.echo(f"  DEGRADED  {reason}")
    if recall and recall["ok"]:
        click.echo('\nNext: book search "what should I remember?"')
    else:
        click.echo("\nRecall smoke is degraded. Run: book doctor")


@admin_group.command(name="uninstall")
@click.option("--yes", "assume_yes", is_flag=True, help="非交互确认卸载")
@click.option("--dry-run", is_flag=True, help="只展示卸载计划")
@click.option("--json", "as_json", is_flag=True, help="输出稳定 JSON 结构")
@click.option("--purge-data", is_flag=True, help="同时删除所有 Storybook Profile 与记忆")
@click.option(
    "--confirm-purge",
    is_flag=True,
    help="非交互 purge 的第二重显式确认；必须与 --purge-data 同用",
)
def uninstall_command(assume_yes, dry_run, as_json, purge_data, confirm_purge):
    """恢复 setup 管理的配置；默认 keep-data（低频维护，归入 admin 分组）。"""

    manager = SetupManager()
    plan = {
        "restore_agent_config": True,
        "data": "purge" if purge_data else "keep",
        "state_file": str(manager.state_path),
    }
    if dry_run:
        payload = {"status": "dry_run", "plan": plan, "writes_performed": 0}
        if as_json:
            click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            click.echo("Storybook uninstall plan")
            click.echo("  Agent config  restore managed nodes")
            click.echo(f"  Memory data   {'PURGE' if purge_data else 'KEEP'}")
            click.echo("\nDry-run complete: no writes performed.")
        return

    if confirm_purge and not purge_data:
        _emit_setup_error(
            SetupError(
                "SB_UNINSTALL_CONFIRM_INVALID",
                "--confirm-purge 只能与 --purge-data 同用",
            ),
            as_json=as_json,
        )
        return
    if purge_data and (assume_yes or as_json) and not confirm_purge:
        _emit_setup_error(
            SetupError(
                "SB_UNINSTALL_PURGE_CONFIRM_REQUIRED",
                "非交互 purge 需要同时传 --purge-data --confirm-purge",
                hint="不传 --purge-data 会默认保留全部记忆",
            ),
            as_json=as_json,
        )
        return
    if not assume_yes and not as_json:
        click.confirm("Remove Storybook Agent integrations? Memory is kept by default.", abort=True)
        if purge_data:
            phrase = click.prompt("Type PURGE to permanently delete all memory")
            if phrase != "PURGE":
                raise click.ClickException("purge confirmation did not match")

    try:
        result = manager.uninstall(purge_data=purge_data)
    except SetupError as exc:
        _emit_setup_error(exc, as_json=as_json)
        return
    if as_json:
        click.echo(json.dumps(result, ensure_ascii=False, indent=2))
        return
    click.echo(f"Uninstall status: {result['status']}")
    click.echo(f"Memory data: {result['data']}")
    for item in result["adapters"]:
        click.echo(f"  {item['adapter']}: {item['status']}")


def _profile_payload(profile, *, active: bool) -> dict:
    paths = config.PROFILE_REGISTRY.paths_for(profile)
    return {
        "id": profile.id,
        "display_name": profile.display_name,
        "mode": profile.mode,
        "sync_state": profile.sync_state,
        "active": active,
        "data_dir": str(paths.root),
        "database_ref": profile.database_ref,
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


@cli.group(hidden=True)
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


def _emit_migration_error(exc: MigrationError, *, as_json: bool) -> None:
    if as_json:
        click.echo(json.dumps(
            {"status": "failed", "error": exc.as_dict()},
            ensure_ascii=False,
            indent=2,
        ))
        raise click.exceptions.Exit(1)
    hint = f"\n修复建议: {exc.hint}" if exc.hint else ""
    raise click.ClickException(f"[{exc.code}] {exc}{hint}") from exc


@migration_group.command(name="discover")
@click.option("--json", "as_json", is_flag=True, help="输出 JSON")
def migration_discover(as_json):
    """只读发现当前项目中的 Storybook v1 数据库。"""

    found = migration_module.discover_legacy_databases()
    payload = {"legacy_databases": [str(path) for path in found]}
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if not found:
        click.echo("未发现旧项目级 Storybook v1 数据库")
        return
    click.echo("发现以下旧数据库：")
    for path in found:
        click.echo(f"  {path}")


@migration_group.command(name="run")
@click.argument(
    "source",
    required=False,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--dry-run", is_flag=True, help="只读检查并输出计划；零写入")
@click.option("--json", "as_json", is_flag=True, help="输出稳定 JSON")
def migration_run(source, dry_run, as_json):
    """将一个 v1 数据库安全迁移到当前用户级 Profile。"""

    manager = migration_module.MigrationManager()
    if source is None:
        found = migration_module.discover_legacy_databases()
        if len(found) != 1:
            exc = MigrationError(
                "SB_MIGRATION_SOURCE_REQUIRED",
                f"自动发现 {len(found)} 个候选库，无法唯一选择",
                hint="显式传入 book admin migration run <memory.db>",
            )
            _emit_migration_error(exc, as_json=as_json)
            return
        source = found[0]
    try:
        result = manager.plan(source) if dry_run else manager.run(source)
    except (MigrationError, ProfileError) as exc:
        if isinstance(exc, ProfileError):
            exc = MigrationError("SB_MIGRATION_PROFILE_INVALID", str(exc))
        _emit_migration_error(exc, as_json=as_json)
        return
    if as_json:
        click.echo(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if dry_run:
        click.echo(
            f"Dry-run OK: {result['counts']['sessions']} Session / "
            f"{result['counts']['stories']} Story / {result['counts']['edges']} edge"
        )
        click.echo(f"Migration ID: {result['migration_id']}")
        click.echo("未写入任何文件；移除 --dry-run 执行切换")
        return
    click.echo(f"Migration {result['status']}: {result['migration_id']}")
    click.echo(f"Profile database_ref: {result['generation_ref']}")
    if result.get("retain_until"):
        click.echo(f"v1 backup retained until at least: {result['retain_until']}")


@migration_group.command(name="rollback")
@click.argument("migration_id")
@click.option("--json", "as_json", is_flag=True, help="输出稳定 JSON")
def migration_rollback(migration_id, as_json):
    """原子切回迁移保留的 v1 数据库副本。"""

    try:
        result = migration_module.MigrationManager().rollback(migration_id)
    except (MigrationError, ProfileError) as exc:
        if isinstance(exc, ProfileError):
            exc = MigrationError("SB_MIGRATION_PROFILE_INVALID", str(exc))
        _emit_migration_error(exc, as_json=as_json)
        return
    if as_json:
        click.echo(json.dumps(result, ensure_ascii=False, indent=2))
        return
    click.echo(f"Rollback {result['status']}: {migration_id}")
    click.echo(f"Profile database_ref: {result['database_ref']}")


@migration_group.command(name="status")
@click.option("--json", "as_json", is_flag=True, help="输出稳定 JSON")
def migration_status(as_json):
    """查看当前 Profile 的迁移与备份状态。"""

    try:
        result = migration_module.MigrationManager().status()
    except (MigrationError, ProfileError) as exc:
        if isinstance(exc, ProfileError):
            exc = MigrationError("SB_MIGRATION_PROFILE_INVALID", str(exc))
        _emit_migration_error(exc, as_json=as_json)
        return
    if as_json:
        click.echo(json.dumps(result, ensure_ascii=False, indent=2))
        return
    click.echo(f"Profile database_ref: {result['database_ref']}")
    if not result["migrations"]:
        click.echo("尚无迁移记录")
        return
    for item in result["migrations"]:
        click.echo(
            f"  {item['migration_id']}  {item['status']}  "
            f"retain_until={item['retain_until']}"
        )


@migration_group.command(name="delete-backup")
@click.argument("migration_id")
@click.option("--yes", "assume_yes", is_flag=True, help="确认永久删除保留备份")
@click.option("--json", "as_json", is_flag=True, help="输出稳定 JSON")
def migration_delete_backup(migration_id, assume_yes, as_json):
    """用户主动永久删除一个迁移的 v1 保留备份。"""

    if not assume_yes and not click.confirm("永久删除 v1 备份？此操作不可撤销"):
        raise click.Abort()
    try:
        result = migration_module.MigrationManager().delete_backup(migration_id)
    except (MigrationError, ProfileError) as exc:
        if isinstance(exc, ProfileError):
            exc = MigrationError("SB_MIGRATION_PROFILE_INVALID", str(exc))
        _emit_migration_error(exc, as_json=as_json)
        return
    if as_json:
        click.echo(json.dumps(result, ensure_ascii=False, indent=2))
        return
    click.echo(f"Backup {result['status']}: {migration_id}")


def _legacy_init_db() -> None:
    store.init_db()
    click.echo(f"✅ 数据库已初始化: {config.DB_PATH}")


@cli.command(name="init")
@click.option("--yes", "assume_yes", is_flag=True, help="接受计划并非交互执行")
@click.option("--dry-run", is_flag=True, help="只输出计划；零写入")
@click.option("--json", "as_json", is_flag=True, help="输出稳定 JSON 结构")
@click.option(
    "--agent", "agents", multiple=True,
    type=click.Choice(["claude", "cursor", "codex"]),
    help="只配置指定 Agent；可重复",
)
@click.option(
    "--provider",
    type=click.Choice(["ollama", "api", "anthropic"]),
    help="模型 provider（旧 shorthand，同时作用于 generation 与 embedding 两个端点）",
)
@click.option("--base-url", help="provider 根 URL（旧 shorthand，同时作用于两个端点）")
@click.option("--llm-model", help="generation 模型名")
@click.option("--embedding-model", help="embedding 模型名")
@click.option("--api-key-env", help="API key 环境变量名（旧 shorthand，同时作用于两个端点）")
@click.option(
    "--llm-protocol",
    type=click.Choice(["ollama", "openai", "anthropic"]),
    help="generation 端点协议（openai=OpenAI 兼容需 /v1；anthropic 走 /v1/messages）",
)
@click.option("--llm-base-url", help="generation 端点 base URL")
@click.option("--llm-api-key-env", help="generation 端点凭据环境变量名")
@click.option(
    "--embedding-protocol",
    type=click.Choice(["ollama", "openai", "anthropic"]),
    help="embedding 端点协议（默认 ollama）",
)
@click.option(
    "--embedding-preset",
    type=click.Choice(["ollama", "custom"]),
    help="选择 Ollama（推荐）或自定义 OpenAI-compatible API",
)
@click.option("--embedding-base-url", help="embedding API base URL")
@click.option("--embedding-dimension", type=click.IntRange(min=1), help="向量维度")
@click.option("--embedding-version", help="不可变 embedding 版本")
@click.option("--embedding-api-key-env", help="embedding 凭据环境变量名")
@click.option("--enable-schedule", is_flag=True, help="配置用户级 watch schedule")
@click.option("--skip-models", "--skip-download", is_flag=True, help="跳过模型下载")
@click.pass_context
def init_command(
    ctx, assume_yes, dry_run, as_json, agents, provider, base_url, llm_model,
    embedding_model, api_key_env, embedding_preset, embedding_base_url,
    embedding_dimension, embedding_version, embedding_api_key_env,
    llm_protocol, llm_base_url, llm_api_key_env, embedding_protocol,
    enable_schedule, skip_models,
):
    """Canonical ``book init`` onboarding; legacy ``storybook init`` initializes DB."""

    executable = Path(ctx.find_root().info_name or "").name
    if executable != "book":
        supplied = (
            assume_yes or dry_run or as_json or agents or provider or base_url
            or llm_model or embedding_model or api_key_env or embedding_preset
            or embedding_base_url or embedding_dimension or embedding_version
            or embedding_api_key_env or llm_protocol or llm_base_url
            or llm_api_key_env or embedding_protocol or enable_schedule or skip_models
        )
        if supplied:
            raise click.UsageError(
                "legacy `storybook init` only initializes the database; use `book init`"
            )
        _legacy_init_db()
        return
    return _run_onboarding(
        assume_yes=assume_yes,
        dry_run=dry_run,
        as_json=as_json,
        agents=agents,
        provider=provider,
        base_url=base_url,
        llm_model=llm_model,
        embedding_model=embedding_model,
        api_key_env=api_key_env,
        embedding_preset=embedding_preset,
        embedding_base_url=embedding_base_url,
        embedding_dimension=embedding_dimension,
        embedding_version=embedding_version,
        embedding_api_key_env=embedding_api_key_env,
        llm_protocol=llm_protocol,
        llm_base_url=llm_base_url,
        llm_api_key_env=llm_api_key_env,
        embedding_protocol=embedding_protocol,
        enable_schedule=enable_schedule,
        skip_models=skip_models,
        full_onboarding=True,
    )


@admin_group.command(name="init-db")
def admin_init_db():
    """Initialize the active Profile database."""
    _legacy_init_db()


@cli.command()
@click.option("--fix", is_flag=True, help="自动修复向量双写不一致（重建缺失行 / 清除孤立行）")
def doctor(fix):
    """🩺 环境与健康自检

    检查 DeepSeek LLM 配置、Ollama/Embedding 模型、向量维度、sqlite-vec 扩展与
    story_vectors 虚表、向量双写一致性，逐项给出 ✅/❌ 与修复建议。
    加 --fix 自动修复向量双写不一致。
    """
    health.run_doctor(fix=fix)


@cli.command()
def mcp():
    """🔌 启动 MCP server（stdio），把记忆检索暴露给 MCP-aware agent（如 Claude Code）。

    agent 可在运行时调用 recall / get_story / stats 召回过往记忆，
    实现"跨 session 经验复用"。server 为独立进程，不依赖 CLI 运行态。

    MCP runtime 已包含在基础安装中；Claude Code/Cursor/Codex 可由
    ``book init`` 自动接入。
    """
    from . import mcp_server
    mcp_server.main()


@cli.command(hidden=True)
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
      book prime --cwd "$CLAUDE_PROJECT_DIR"

    \b
    结构化注入（支持 hookSpecificOutput 的 Claude Code 版本）:
      book prime --cwd "$CLAUDE_PROJECT_DIR" --format hook

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
            tool_type="claude_code", integration_mode="hook",
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
        click.echo(f"[book prime] {result['note']}", err=True)


@cli.command(name="import-data", hidden=True)
@click.argument("path", required=False)
@click.option("--claude", is_flag=True, help="从 Claude Code 采集")
@click.option("--cursor", is_flag=True, help="从 Cursor 自动采集")
@click.option("--codex", is_flag=True, help="从 Codex 自动采集")
@click.option(
    "--source",
    type=click.Choice(source_manager.SOURCE_NAMES),
    help="从指定 Agent history adapter 增量采集",
)
@click.option("--sample", is_flag=True, help="生成模拟数据")
@click.option("--n", default=100, help="模拟数据数量(配合--sample)")
@click.option("--json", "as_json", is_flag=True, help="输出稳定 JSON summary")
def import_data(path, claude, cursor, codex, source, sample, n, as_json):
    """兼容 alias；canonical 自动采集入口为 ``book run``。"""
    _legacy_alias_hint("run")
    store.init_db()

    selectors = [bool(path), claude, cursor, codex, bool(source), sample]
    if sum(selectors) > 1:
        raise click.UsageError(
            "<path>, --source, --claude, --cursor, --codex, --sample 互斥"
        )

    selected_source = source or (
        "codex" if codex else "cursor" if cursor else "claude" if claude else None
    )
    if selected_source:
        result = source_manager.import_enabled([selected_source])
        if as_json:
            click.echo(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            item = result["sources"][0]
            click.echo(
                f"✅ {selected_source}: files={item.get('files', 0)} "
                f"scanned={item.get('scanned', 0)} imported={item.get('imported', 0)} "
                f"updated={item.get('updated', 0)} skipped={item.get('skipped', 0)} "
                f"invalid={item.get('invalid', 0)} status={item.get('status')}"
            )
        if result["status"] == "degraded":
            raise click.exceptions.Exit(2)
        return

    if sample:
        click.echo(f"📊 生成 {n} 条模拟会话数据...")
        sessions = collector.generate_sample_sessions(n)
        count = collector.import_sessions(sessions)
        click.echo(f"✅ 导入 {count} 条模拟会话")

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
                    "context": s.get("context"),
                })
            else:
                normalized.append(s)

        count = collector.import_sessions(normalized)
        click.echo(f"✅ 导入 {count} 条会话")

    else:
        # Backward-compatible default remains Claude; dream defaults to all
        # enabled and detected sources.
        result = source_manager.import_enabled(["claude"])
        if as_json:
            click.echo(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            item = result["sources"][0]
            click.echo(
                f"✅ claude: imported={item.get('imported', 0)} "
                f"updated={item.get('updated', 0)} skipped={item.get('skipped', 0)}"
            )


def _source_list_impl(as_json: bool) -> None:
    """显示来源检测、启用、版本和最近导入状态。"""
    store.init_db()
    items = source_manager.list_sources()
    if as_json:
        click.echo(json.dumps({"sources": items}, ensure_ascii=False, indent=2))
        return
    for item in items:
        click.echo(
            f"{item['name']:<8} available={str(item['available']).lower():<5} "
            f"enabled={str(item['enabled']).lower():<5} status={item['status']} "
            f"version={item['adapter_version']}"
        )


def _source_enable_impl(name: str) -> None:
    source_manager.set_enabled(name, True)
    click.echo(f"enabled: {name}")


def _source_disable_impl(name: str) -> None:
    source_manager.set_enabled(name, False)
    click.echo(f"disabled: {name}")


def _source_reset_impl(name: str, yes: bool) -> None:
    if not yes:
        raise click.UsageError("重置 checkpoint 需要 --yes")
    store.init_db()
    count = store.delete_source_checkpoints(name)
    click.echo(f"reset {count} checkpoint(s): {name}")


@source_group.command(name="list")
@click.option("--json", "as_json", is_flag=True, help="输出 JSON")
def source_list(as_json):
    """显示来源检测、启用、版本和最近导入状态。"""
    _source_list_impl(as_json)


@source_group.command(name="enable")
@click.argument("name", type=click.Choice(source_manager.SOURCE_NAMES))
def source_enable(name):
    source_manager.set_enabled(name, True)
    click.echo(f"enabled: {name}")


@source_group.command(name="disable")
@click.argument("name", type=click.Choice(source_manager.SOURCE_NAMES))
def source_disable(name):
    source_manager.set_enabled(name, False)
    click.echo(f"disabled: {name}")


@source_group.command(name="reset")
@click.argument("name", type=click.Choice(source_manager.SOURCE_NAMES))
@click.option("--yes", is_flag=True, help="确认删除来源 checkpoint")
def source_reset(name, yes):
    _source_reset_impl(name, yes)


@cli.group(name="sources", hidden=True)
def sources_alias_group():
    """兼容 alias；canonical 入口为 ``book source``。"""


@sources_alias_group.command(name="list")
@click.option("--json", "as_json", is_flag=True, help="输出 JSON")
def sources_list(as_json):
    """兼容 alias；canonical 入口为 ``book source list``。"""
    _legacy_alias_hint("source list")
    _source_list_impl(as_json)


@sources_alias_group.command(name="enable")
@click.argument("name", type=click.Choice(source_manager.SOURCE_NAMES))
def sources_enable(name):
    """兼容 alias；canonical 入口为 ``book source enable``。"""
    _legacy_alias_hint("source enable")
    _source_enable_impl(name)


@sources_alias_group.command(name="disable")
@click.argument("name", type=click.Choice(source_manager.SOURCE_NAMES))
def sources_disable(name):
    """兼容 alias；canonical 入口为 ``book source disable``。"""
    _legacy_alias_hint("source disable")
    _source_disable_impl(name)


@sources_alias_group.command(name="reset-checkpoint")
@click.argument("name", type=click.Choice(source_manager.SOURCE_NAMES))
@click.option("--yes", is_flag=True, help="确认删除来源 checkpoint")
def sources_reset_checkpoint(name, yes):
    """兼容 alias；canonical 入口为 ``book source reset``。"""
    _legacy_alias_hint("source reset")
    _source_reset_impl(name, yes)


def _resolve_run_mode(once, watch, daemon, session):
    """解析互斥的 run 模式；冲突组合 fail-fast。"""
    long_running = [bool(watch), bool(daemon), session is not None]
    if sum(long_running) > 1:
        raise click.UsageError(
            "--watch / --daemon / --session 互斥：只能选择一种模式"
        )
    if once and (watch or daemon):
        raise click.UsageError("--once 不能与 --watch / --daemon 同用")
    if watch:
        return "watch"
    if daemon:
        return "daemon"
    if session is not None:
        return "session"
    return "once"


def _run_once(source, *, import_new: bool) -> None:
    """单次周期（import_new=True 采集+加工；False 仅加工 pending）后退出。"""
    dreamd.setup_dream_logging()
    result = dreamd.run_dream_cycle_once(
        import_new=import_new,
        verbose=True,
        sources=[source] if source else None,
    )
    if result["status"] == "skipped":
        click.echo("⏳ 另一个做梦周期正在运行，已跳过")
    elif result["total"] == 0 and not import_new:
        click.echo("✅ 没有待处理的会话")
    else:
        click.echo(
            f"🌙 做梦周期完成（{result['status']}）：新增 {result['imported']} 条，"
            f"更新 {result.get('updated', 0)} 条，"
            f"加工 {result['total']} 条（成功 {result['success']} / 失败 {result['failed']}），"
            f"用时 {result['duration_s']}s"
        )
        for item in result.get("sources", []):
            if item.get("detected") or item.get("status") == "degraded":
                click.echo(
                    f"  {item['source']}: {item['status']} "
                    f"imported={item.get('imported', 0)} "
                    f"updated={item.get('updated', 0)} "
                    f"invalid={item.get('invalid', 0)}"
                )


def _run_watch(source, interval) -> None:
    dreamd.setup_dream_logging()
    stop = threading.Event()
    dreamd.install_signal_handlers(stop)
    poll = interval if interval is not None else config.WATCH_POLL_INTERVAL
    scope = source or "all enabled sources"
    click.echo(f"🌙 监听模式启动：每 {poll}s 轮询 {scope}（Ctrl-C 退出）")
    dreamd.watch_loop(
        poll_interval=poll,
        stop_event=stop,
        verbose=True,
        sources=[source] if source else None,
    )


def _run_session(session) -> None:
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


def _run_daemon(source, interval) -> None:
    dreamd.setup_dream_logging()
    stop = threading.Event()
    dreamd.install_signal_handlers(stop)
    iv = interval if interval is not None else config.DREAM_INTERVAL
    click.echo(f"🌙 做梦守护进程启动：每 {iv}s 触发一次（Ctrl-C 退出）")
    dreamd.dream_daemon(
        interval=iv,
        stop_event=stop,
        verbose=False,
        sources=[source] if source else None,
    )


def _run_command(*, once, watch, daemon, session, interval, source, import_new) -> None:
    """统一 ``book run`` 及其 process / dream 兼容 alias 的实现。"""
    store.init_db()
    mode = _resolve_run_mode(once, watch, daemon, session)
    if mode == "watch":
        _run_watch(source, interval)
    elif mode == "daemon":
        _run_daemon(source, interval)
    elif mode == "session":
        _run_session(session)
    else:
        _run_once(source, import_new=import_new)


@cli.command(name="run")
@click.option("--once", is_flag=True,
              help="只跑一次完整周期（采集+加工）后退出；launchd/cron 定时任务用此入口")
@click.option("--watch", is_flag=True,
              help="反应式监听：轮询已启用来源，发现新会话即自动采集 + 加工（长驻，Ctrl-C 退出）")
@click.option("--daemon", is_flag=True,
              help="定时守护循环：每 --interval 秒（默认 config.DREAM_INTERVAL=14400）跑一次完整周期")
@click.option("--session", "-s", type=int, help="只加工指定 Session ID（单次）")
@click.option("--interval", default=None, type=int,
              help="--watch 轮询间隔或 --daemon 循环间隔（秒）")
@click.option("--source", type=click.Choice(source_manager.SOURCE_NAMES), help="只处理指定来源")
def run_cmd(once, watch, daemon, session, interval, source):
    """🔄 采集并形成记忆（collect + process）。

    \b
    默认 / --once   单次完整周期：采集启用来源 + 加工 pending，然后退出
    --watch         反应式监听已启用来源，发现新会话即自动采集 + 加工（长驻）
    --daemon        定时守护循环，每 --interval 秒一轮（默认 4 小时）
    --session ID    只加工指定 Session（单次）

    \b
    --watch / --daemon / --session 互斥；--once 仅与 --session 相容。
    """
    _run_command(
        once=once, watch=watch, daemon=daemon, session=session,
        interval=interval, source=source, import_new=True,
    )


@cli.command(name="process", hidden=True)
@click.option("--session", "-s", type=int, help="处理指定会话ID")
@click.option("--watch", is_flag=True,
              help="监听模式：轮询已启用来源，有新会话自动加工（长驻，Ctrl-C 退出）")
@click.option("--interval", default=None, type=int,
              help="--watch 轮询间隔（秒），默认读 config.WATCH_POLL_INTERVAL（60）")
@click.option("--source", type=click.Choice(source_manager.SOURCE_NAMES), help="只监听指定来源")
def process_alias(session, watch, interval, source):
    """兼容 alias；canonical 入口为 ``book run``。"""
    _legacy_alias_hint("run")
    _run_command(
        once=False, watch=watch, daemon=False, session=session,
        interval=interval, source=source, import_new=False,
    )


@cli.command(name="dream", hidden=True)
@click.option("--once", is_flag=True,
              help="只跑一次完整做梦周期（采集+加工）后退出；launchd 定时任务用此入口")
@click.option("--interval", default=None, type=int,
              help="守护进程循环间隔（秒），默认读 config.DREAM_INTERVAL（14400 = 4 小时）")
@click.option("--source", type=click.Choice(source_manager.SOURCE_NAMES), help="只处理指定来源")
def dream_alias(once, interval, source):
    """兼容 alias；canonical 入口为 ``book run``。"""
    _legacy_alias_hint("run --daemon")
    _run_command(
        once=once, watch=False, daemon=(not once), session=None,
        interval=interval, source=source, import_new=True,
    )


@cli.command()
@click.argument("query")
@click.option("--top", "-t", default=3, help="返回Top N")
@click.option(
    "--context",
    "context_mode",
    type=click.Choice(["auto", "none"]),
    default="auto",
    show_default=True,
    help="自动采集当前环境用于软加权，或禁用环境上下文",
)
@click.option(
    "--scope",
    type=click.Choice(["profile", "project", "strict"]),
    default="profile",
    show_default=True,
    help="profile 全 Profile 软加权；project 仅当前项目；strict 过滤环境冲突",
)
@click.option(
    "--mode",
    "retrieval_mode",
    type=click.Choice(["fast", "auto", "deep"]),
    default=lambda: config.QUERY_DEFAULT_MODE,
    show_default="fast",
    help="fast 无生成式调用；auto 按置信门控；deep 使用显式高预算",
)
@click.option(
    "--transform/--no-transform",
    default=None,
    help="覆盖 Query Transformation/HyDE 总开关",
)
@click.option(
    "--rerank/--no-rerank",
    default=None,
    help="覆盖本地有界 reranker 开关",
)
@click.option("--json", "as_json", is_flag=True, help="输出结构化 JSON")
@click.option("--cwd", type=click.Path(path_type=Path), default=None, hidden=True)
def search(
    query, top, context_mode, scope, retrieval_mode, transform, rerank, as_json, cwd
):
    """🔍 搜索记忆"""
    store.init_db()
    current_context = None
    if context_mode == "auto":
        current_context = context_module.capture_context(
            tool_type="other",
            integration_mode="manual",
            workspace_path=cwd or Path.cwd(),
        )
    result = search_module.search(
        query,
        top_k=top,
        context=current_context,
        scope=scope,
        retrieval_mode=retrieval_mode,
        transform_enabled=transform,
        rerank_enabled=rerank,
    )
    if as_json:
        click.echo(json.dumps(result, ensure_ascii=False, indent=2))
        return
    output = search_module.format_search_result(result)
    click.echo(output)


@admin_group.command(name="index")
@click.option("--model", default=None, help="目标 embedding API 模型")
@click.option("--version", required=True, help="不可变的目标 embedding 版本")
@click.option(
    "--representation",
    type=click.Choice(["default", "full"]),
    default="default",
    show_default=True,
)
@click.option("--batch-size", type=click.IntRange(min=1), default=100,
              show_default=True, help="本次最多处理的 Story 数；重复运行可续跑")
@click.option("--no-activate", is_flag=True,
              help="即使 shadow 已完整也不切换 serving 索引")
def admin_index(model, version, representation, batch_size, no_activate):
    """增量重建 embedding shadow，并在完整后原子切换 serving index（低频维护）。"""

    store.init_db()
    result = embeddings.backfill(
        model=model or config.EMBED_MODEL,
        version=version,
        representation=representation,
        batch_size=batch_size,
        activate=not no_activate,
    )
    click.echo(json.dumps(result, ensure_ascii=False, indent=2))


def _memory_forget_impl(half_life_days, max_access, max_edge_weight, min_age_days,
                        apply, as_json):
    """衰减访问热度，并预览或归档低价值记忆。"""
    store.init_db()
    decay = store.decay_story_access_counts(half_life_days=half_life_days)
    archive = store.archive_low_value_stories(
        max_access_count=max_access,
        max_edge_weight=max_edge_weight,
        min_age_days=min_age_days,
        dry_run=not apply,
    )
    result = {"decay": decay, "archive": archive}
    if as_json:
        click.echo(json.dumps(result, ensure_ascii=False, indent=2))
        return
    action = "已归档" if apply else "预览到"
    click.echo(
        f"访问衰减完成：检查 {decay['examined']} 条，变更 {decay['decayed']} 条；"
        f"{action} {archive['candidate_count']} 条低价值记忆。"
    )
    if not apply and archive["candidate_count"]:
        click.echo("确认后加 --apply 执行；归档保留 Story 与来源证据，但不再参与检索。")


@memory_group.command(name="forget")
@click.option("--half-life-days", type=click.FloatRange(min=0.001), default=30.0,
              show_default=True, help="访问计数指数衰减半衰期")
@click.option("--max-access", type=click.IntRange(min=0), default=0,
              show_default=True, help="衰减后可归档的最大访问计数")
@click.option("--max-edge-weight", type=click.FloatRange(min=0.0), default=0.25,
              show_default=True, help="可归档 Story 的最大关联边权")
@click.option("--min-age-days", type=click.IntRange(min=0), default=90,
              show_default=True, help="距离最近访问/更新的最短天数")
@click.option("--apply", is_flag=True, help="执行归档；默认仅预览候选")
@click.option("--json", "as_json", is_flag=True, help="输出结构化 JSON")
def memory_forget(half_life_days, max_access, max_edge_weight, min_age_days, apply, as_json):
    """衰减访问热度，并预览或归档低价值记忆。"""
    _memory_forget_impl(
        half_life_days, max_access, max_edge_weight, min_age_days, apply, as_json
    )


@cli.command(name="forget", hidden=True)
@click.option("--half-life-days", type=click.FloatRange(min=0.001), default=30.0,
              show_default=True, help="访问计数指数衰减半衰期")
@click.option("--max-access", type=click.IntRange(min=0), default=0,
              show_default=True, help="衰减后可归档的最大访问计数")
@click.option("--max-edge-weight", type=click.FloatRange(min=0.0), default=0.25,
              show_default=True, help="可归档 Story 的最大关联边权")
@click.option("--min-age-days", type=click.IntRange(min=0), default=90,
              show_default=True, help="距离最近访问/更新的最短天数")
@click.option("--apply", is_flag=True, help="执行归档；默认仅预览候选")
@click.option("--json", "as_json", is_flag=True, help="输出结构化 JSON")
def forget_alias(half_life_days, max_access, max_edge_weight, min_age_days, apply, as_json):
    """兼容 alias；canonical 入口为 ``book memory forget``。"""
    _legacy_alias_hint("memory forget")
    _memory_forget_impl(
        half_life_days, max_access, max_edge_weight, min_age_days, apply, as_json
    )


@cli.command(hidden=True)
def stats():
    """兼容入口；运行状态已并入 ``book status``。"""
    _legacy_alias_hint("status")
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
        **SetupManager().runtime_status(),
        "stories": stats_data["stories"],
        "sessions": stats_data["sessions"],
        "pending": stats_data["pending"],
    }
    if include_performance:
        payload["performance"] = performance.summarize_query_performance(limit=100)

    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    click.echo(f"Overall        {payload['status'].upper()}")
    profile = payload["profile"]
    click.echo(
        f"Profile        {profile['display_name']} ({profile['status']})"
    )
    model = payload["model"]
    click.echo(
        f"Models         LLM {model['llm']['provider']}={model['llm']['status']} · "
        f"Embedding {model['embedding']['provider']}={model['embedding']['status']}"
    )
    adapter = payload["adapter"]
    checks = ", ".join(
        f"{item['name']}={item['status']}" for item in adapter["checks"]
    ) or "none configured"
    click.echo(f"Adapters       {checks}")
    click.echo(f"Sync           {payload['sync_state']}")
    for reason in payload["degraded_reasons"]:
        click.echo(f"Degraded       {reason}")
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


@cli.command()
@click.option("--json", "as_json", is_flag=True, help="输出结构化 JSON")
def version(as_json):
    """📦 查看当前版本。"""
    from . import __version__

    if as_json:
        click.echo(
            json.dumps({"name": "storybook", "version": __version__},
                       ensure_ascii=False)
        )
        return
    click.echo(f"storybook {__version__}")


def _memory_list_impl(limit: int) -> None:
    """列出所有 Story。"""
    store.init_db()
    stories = store.get_all_stories()
    if not stories:
        click.echo("暂无记忆。使用 `book run` 采集，或用 `storybook import-data --sample` 生成模拟数据。")
        return

    click.echo(f"\n📋 Story 记忆库 ({len(stories)} 条)\n")
    for s in stories[:limit]:
        parent = f" ← #{s['parent_id']}" if s.get("parent_id") else ""
        click.echo(f"  #{s['id']}{parent} [{s['version']}] {s['title']}")
        click.echo(f"     {s['content'][:80]}...")
        click.echo(f"     关键词: {', '.join(s.get('keywords', []))}")
        click.echo(f"     访问: {s.get('access_count', 0)} 次 | 更新: {s.get('updated_at', '?')}")
        click.echo("")


def _memory_show_impl(story_id: int) -> None:
    """查看 Story 详情。"""
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
    applies, excludes = context_module.applicability_labels(
        story.get("applicability")
    )
    click.echo(f"   适用于: {', '.join(applies) if applies else '未声明'}")
    click.echo(f"   不适用于: {', '.join(excludes) if excludes else '未声明'}")
    environments = story.get("environments", [])
    if environments:
        click.echo(f"   来源环境 ({len(environments)}):")
        for envelope in environments:
            click.echo(f"      - {context_module.environment_label(envelope)}")
    if story.get("parent_id"):
        click.echo(f"   父 Story: #{story['parent_id']}")

    related = store.get_related_stories(story["id"], limit=10)
    if related:
        click.echo(f"\n   🔗 关联记忆 ({len(related)} 条):")
        for r in related:
            tag = "👨‍👦" if r.get("edge_type") == "parent_child" else "💭"
            click.echo(f"      {tag} [{r.get('weight', 0):.2f}] #{r['id']} {r['title']}")
    click.echo("")


@memory_group.command(name="list")
@click.option("--limit", "-l", default=20, help="显示数量")
def memory_list(limit):
    """📋 列出所有 Story"""
    _memory_list_impl(limit)


@memory_group.command(name="show")
@click.argument("story_id")
def memory_show(story_id):
    """🔎 查看 Story 详情"""
    _memory_show_impl(story_id)


@cli.command(name="list", hidden=True)
@click.option("--limit", "-l", default=20, help="显示数量")
def list_alias(limit):
    """兼容 alias；canonical 入口为 ``book memory list``。"""
    _legacy_alias_hint("memory list")
    _memory_list_impl(limit)


@cli.command(name="show", hidden=True)
@click.argument("story_id")
def show_alias(story_id):
    """兼容 alias；canonical 入口为 ``book memory show``。"""
    _legacy_alias_hint("memory show")
    _memory_show_impl(story_id)


def main():
    cli()


@admin_group.command(name="eval")
@click.argument("part", required=False, default="all",
                type=click.Choice([
                    "all", "retrieval", "processing", "split", "ablation",
                    "strategy", "exact-term",
                ]))
@click.option("--report", "-r", type=click.Path(dir_okay=False, writable=True),
              help="把完整 JSON 报告写入该路径")
@click.option("--benchmark", "benchmark_path", type=click.Path(exists=True, dir_okay=False),
              help="自定义 benchmark 数据集 JSON（默认 data/retrieval_benchmark.json）")
@click.option("--transform-cache", type=click.Path(exists=True, dir_okay=False),
              help="query-only 预生成 transformation JSON；仅用于可复现质量证据")
def admin_eval(part, report, benchmark_path, transform_cache):
    """📐 检索、加工、分裂与 Story v2 embedding 表示消融（低频维护）

    PART 取值：retrieval / processing / split / ablation / strategy /
    exact-term / all（默认 all）。

    retrieval 用真实 embedding + 人工标注 story 语料，度量 recall@1/3/5、precision@k、MRR、
    阈值敏感性曲线，并判定是否达 PRD「重复 bug 检索准确率≥70%」(recall@3)。
    processing 用真实 embedding + 确定性 LLM 桩，度量 merge/update 分支是否选对。
    split 度量分裂路径结构正确性。
    ablation 比较 legacy/default/full/multi-vector，并按 exact/synonym/
    cross-tool/cross-language 分组报告质量与时延。
    strategy 比较 direct-vector、hybrid、+graph、+rewrite、+HyDE、+reranker，
    并按 exact/synonym/cross-language/cross-tool/ambiguous 执行默认启用门禁。
    exact-term 隔离度量精确错误码在纯向量与 Hybrid 下的 recall@3。

    需要配置的 embedding API 可用。评测在隔离临时库中进行，不污染用户 Profile 数据库。
    用 --report 把可复现的 JSON 报告落盘，便于阈值调整前后量化对比。
    """
    parts = (
        "retrieval", "processing", "split", "ablation", "strategy",
        "exact_term",
    ) if part == "all" else (part.replace("-", "_"),)
    click.echo(
        f"📐 运行评测: {', '.join(parts)}"
        f"（embedding type=api, adapter={config.EMBED_ADAPTER}）\n"
    )

    bp = benchmark_path
    try:
        transform_provider = None
        transform_source = "live_generated"
        if transform_cache:
            transform_provider = eval_module.pre_generated_transform_provider(
                transform_cache
            )
            transform_source = "query_only_pre_generated"
        rep = eval_module.run_all(
            parts=parts,
            benchmark_path=bp,
            transform_provider=transform_provider,
            transform_source=transform_source,
        )
    except Exception as ex:
        click.echo(f"❌ 评测失败: {ex}")
        raise

    rep.meta["embed_mode"] = "api"
    rep.meta["embedding_adapter"] = config.EMBED_ADAPTER
    rep.meta["embedding_model"] = config.EMBED_MODEL
    rep.meta["embedding_dimension"] = config.EMBED_DIM
    rep.meta["embedding_version"] = config.EMBED_VERSION
    rep.meta["part"] = part
    click.echo(eval_module.format_report(rep))

    if report:
        out = Path(report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        click.echo(f"\n📝 JSON 报告已写入: {out}")


@admin_group.command(name="benchmark")
@click.option("--model-state", type=click.Choice(["warm", "cold"]), default="warm",
              show_default=True, help="warm 会先预热；cold 每批请求前卸载 embedding 模型")
@click.option("--retrieval-mode", type=click.Choice(["fast", "auto", "deep"]),
              default="fast", show_default=True,
              help="要测量的检索模式；deep 会包含 DeepSeek LLM transformation")
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
def admin_benchmark(model_state, retrieval_mode, stories, queries, repeats,
                    concurrencies, report, benchmark_path):
    """📈 运行隔离的查询性能与质量基准（低频维护）。"""
    result = perf_benchmark.run_performance_benchmark(
        story_count=stories,
        query_count=queries,
        repeats=repeats,
        concurrencies=tuple(concurrencies),
        model_state=model_state,
        retrieval_mode=retrieval_mode,
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
