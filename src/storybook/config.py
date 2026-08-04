"""配置管理 — Profile 路径、模型名、阈值常量集中管理。"""
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

# ── 安装根与 .env ──
BASE_DIR = Path(__file__).resolve().parent.parent.parent
_PROCESS_ENV = dict(os.environ)


def _load_env_file(path: Path) -> None:
    """从 .env 文件加载环境变量。

    优先级：命令行/已存在的环境变量 > .env 文件 > 代码默认值。
    因此只在变量尚未由环境/命令行预设时写入（不覆盖），文件不存在时静默跳过。
    支持空行、``#`` 注释、可选的 ``export`` 前缀、成对的单/双引号包裹。
    """
    if not path.is_file():
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        # 不覆盖已存在的环境变量，保证命令行/环境变量优先级高于 .env
        if key not in os.environ:
            os.environ[key] = value


# 启动时自动加载项目根 .env（无则跳过、不报错）；须在读 os.getenv 之前执行
_load_env_file(BASE_DIR / ".env")


def _parse_shell_env_file(path: Path) -> dict[str, str]:
    """Parse simple shell-env assignments without evaluating shell syntax."""

    if not path.is_file():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or not key.replace("_", "a").isalnum() or key[0].isdigit():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[key] = value
    return values


def _expand_home_path(value: str, home: Path) -> Path:
    if value == "~":
        return home
    if value.startswith("~/"):
        return home / value[2:]
    return Path(value)


def resolve_llm_config(
    *,
    process_env: dict[str, str] | None = None,
    project_env_path: Path | None = None,
    home: Path | None = None,
) -> dict[str, str | bool | None]:
    """Resolve cloud LLM settings with source precedence and no shell execution.

    Precedence is process environment, the selected LLM env file, project
    ``.env``, then defaults. Within each source, the documented alias order is
    preserved (for example ``ANTHROPIC_AUTH_TOKEN`` before ``DEEPSEEK_KEY``).
    """

    process = dict(_PROCESS_ENV if process_env is None else process_env)
    project = _parse_shell_env_file(project_env_path or (BASE_DIR / ".env"))
    selected_file = (
        process.get("STORYBOOK_LLM_ENV_FILE")
        or project.get("STORYBOOK_LLM_ENV_FILE")
        or "~/.chrc/dpsk.sh"
    )
    home_path = Path.home() if home is None else Path(home)
    llm_file = _parse_shell_env_file(_expand_home_path(selected_file, home_path))
    layers = (process, llm_file, project)

    def pick(*keys: str, default: str | None = None) -> str | None:
        for layer in layers:
            for key in keys:
                value = layer.get(key)
                if value is not None and value.strip():
                    return value.strip()
        return default

    think_value = pick("STORYBOOK_LLM_THINK", default="0") or "0"
    return {
        "env_file": str(_expand_home_path(selected_file, home_path)),
        "base_url": pick(
            "ANTHROPIC_BASE_URL", default="https://api.deepseek.com/anthropic"
        ),
        "api_key": pick("ANTHROPIC_AUTH_TOKEN", "DEEPSEEK_KEY"),
        "model": pick(
            "STORYBOOK_LLM_MODEL",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL",
            default="deepseek-v4-flash",
        ),
        "think": think_value.strip().lower() in {"1", "true", "yes", "on"},
    }

# ── 用户级 Profile 与运行态路径 ──
# registry 是 CLI、MCP、hook、Cursor/Claude/Codex adapter 的唯一数据目录入口。
# 绝对路径只在本机运行态推导，不写入数据库对象 ID 或 registry 主键。
from . import profiles as profiles_module  # noqa: E402  -- .env 须先加载

PROFILE_REGISTRY = profiles_module.default_registry()

# registry 尚未创建时使用纯内存占位 Profile，让模块导入保持只读。真正需要存储
# 的命令会调用 ``ensure_profile()`` 创建随机 UUID Profile 并刷新全部路径。
_BOOTSTRAP_PROFILE = profiles_module.Profile(
    id=str(uuid.UUID(int=0)),
    display_name="default",
    mode="local",
    sync_state=profiles_module.DEFAULT_SYNC_STATE,
    created_at=datetime.fromtimestamp(0, timezone.utc).isoformat(),
)
_PROFILE_PERSISTED = False


def refresh_profile(profile_ref: str | None = None, *, create: bool = True):
    """重新解析当前 Profile，并原子刷新所有兼容路径常量。"""

    global ACTIVE_PROFILE, PROFILE_PATHS, PROFILE_ID, PROFILE_MODE, SYNC_STATE
    global DATA_DIR, DB_DIR, DB_PATH, INDEX_DIR, CACHE_DIR, LOG_DIR
    global PERFORMANCE_LOG_PATH

    global _PROFILE_PERSISTED

    if profile_ref:
        profile = PROFILE_REGISTRY.resolve(profile_ref)
        _PROFILE_PERSISTED = True
    elif create:
        profile = PROFILE_REGISTRY.active_profile()
        _PROFILE_PERSISTED = True
    else:
        profile = PROFILE_REGISTRY.peek_active_profile() or _BOOTSTRAP_PROFILE
        _PROFILE_PERSISTED = profile is not _BOOTSTRAP_PROFILE
    paths = (
        PROFILE_REGISTRY.ensure_profile_directories(profile)
        if create
        else PROFILE_REGISTRY.paths_for(profile)
    )

    ACTIVE_PROFILE = profile
    PROFILE_PATHS = paths
    PROFILE_ID = profile.id
    PROFILE_MODE = profile.mode
    SYNC_STATE = profile.sync_state
    DATA_DIR = paths.root
    DB_DIR = paths.database_dir
    DB_PATH = paths.database
    INDEX_DIR = paths.index_dir
    CACHE_DIR = paths.cache_dir
    LOG_DIR = paths.log_dir
    PERFORMANCE_LOG_PATH = LOG_DIR / "query_performance.jsonl"
    return profile


def ensure_profile() -> object:
    """在首次真实写操作前创建 Profile；测试重定向 DB 时保持隔离。"""

    if _PROFILE_PERSISTED:
        return ACTIVE_PROFILE
    # 测试/benchmark 会显式重定向 DB_PATH；这种情况下不得把路径刷新回用户目录。
    if DB_PATH != PROFILE_PATHS.database:
        return ACTIVE_PROFILE
    return refresh_profile(create=True)


def refresh_database_pointer() -> object:
    """Follow an atomic registry generation switch in long-running processes.

    Tests and isolated benchmarks deliberately override ``DB_PATH``; preserve
    that boundary instead of refreshing it back to the user Profile.
    """

    if not _PROFILE_PERSISTED or DB_PATH != PROFILE_PATHS.database:
        return ACTIVE_PROFILE
    profile = PROFILE_REGISTRY.peek_active_profile()
    if profile is None:
        return ACTIVE_PROFILE
    if (
        profile.id != ACTIVE_PROFILE.id
        or profile.database_ref != ACTIVE_PROFILE.database_ref
    ):
        return refresh_profile(profile.id)
    return ACTIVE_PROFILE


def switch_profile(profile_ref: str):
    """切换 registry 的 active Profile，并刷新当前进程配置。"""

    profile = PROFILE_REGISTRY.switch_profile(profile_ref)
    refresh_profile(profile.id)
    return profile


try:
    refresh_profile(create=False)
except profiles_module.ProfileError:
    # 损坏的 registry 不应让 CLI 在参数解析前因模块导入直接崩溃。保持只读的
    # bootstrap 路径，交由具体命令在可输出稳定错误格式的边界重新读取并报告。
    ACTIVE_PROFILE = _BOOTSTRAP_PROFILE
    PROFILE_PATHS = PROFILE_REGISTRY.paths_for(_BOOTSTRAP_PROFILE)
    PROFILE_ID = _BOOTSTRAP_PROFILE.id
    PROFILE_MODE = _BOOTSTRAP_PROFILE.mode
    SYNC_STATE = _BOOTSTRAP_PROFILE.sync_state
    DATA_DIR = PROFILE_PATHS.root
    DB_DIR = PROFILE_PATHS.database_dir
    DB_PATH = PROFILE_PATHS.database
    INDEX_DIR = PROFILE_PATHS.index_dir
    CACHE_DIR = PROFILE_PATHS.cache_dir
    LOG_DIR = PROFILE_PATHS.log_dir
    PERFORMANCE_LOG_PATH = LOG_DIR / "query_performance.jsonl"
    _PROFILE_PERSISTED = False

# ── Ollama ──
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_PROVIDER = "deepseek_anthropic"
_LLM_CONFIG = resolve_llm_config()
LLM_BASE_URL = str(_LLM_CONFIG["base_url"])
LLM_API_KEY = _LLM_CONFIG["api_key"]
LLM_MODEL = str(_LLM_CONFIG["model"])
EMBED_MODEL = os.getenv("STORYBOOK_EMBED_MODEL", "qwen3-embedding:0.6b")
EMBED_DIM = 1024
EMBED_VERSION = os.getenv("STORYBOOK_EMBED_VERSION", "story-v2-default-v1")
EMBED_REPRESENTATION = os.getenv(
    "STORYBOOK_EMBED_REPRESENTATION", "default"
)
LLM_THINK = bool(_LLM_CONFIG["think"])

# ── 加工缓存与并行 ──
INFERENCE_CACHE_ENABLED = os.getenv(
    "STORYBOOK_INFERENCE_CACHE_ENABLED", "1"
).strip().lower() not in {"0", "false", "no", "off"}
PROCESS_WORKERS = max(1, int(os.getenv("STORYBOOK_PROCESS_WORKERS", "4")))

# ── 查询快路径 ──
# Ollama 的 keep_alive 由每次 embedding 请求续期；进程内 warm window 用于选择
# 2s/5s 硬超时预算，不作为跨进程主键或持久状态。
EMBED_KEEP_ALIVE = os.getenv("STORYBOOK_EMBED_KEEP_ALIVE", "10m")
EMBED_WARM_WINDOW_SECONDS = float(
    os.getenv("STORYBOOK_EMBED_WARM_WINDOW_SECONDS", "600")
)
QUERY_WARM_TIMEOUT_SECONDS = float(
    os.getenv("STORYBOOK_QUERY_WARM_TIMEOUT_SECONDS", "2")
)
QUERY_COLD_TIMEOUT_SECONDS = float(
    os.getenv("STORYBOOK_QUERY_COLD_TIMEOUT_SECONDS", "5")
)
QUERY_FALLBACK_TIMEOUT_SECONDS = float(
    os.getenv("STORYBOOK_QUERY_FALLBACK_TIMEOUT_SECONDS", "0.5")
)
QUERY_VECTOR_CACHE_SIZE = int(os.getenv("STORYBOOK_QUERY_VECTOR_CACHE_SIZE", "256"))
QUERY_RESULT_CACHE_SIZE = int(os.getenv("STORYBOOK_QUERY_RESULT_CACHE_SIZE", "128"))
QUERY_VECTOR_CACHE_TTL_SECONDS = float(
    os.getenv("STORYBOOK_QUERY_VECTOR_CACHE_TTL_SECONDS", "900")
)
QUERY_RESULT_CACHE_TTL_SECONDS = float(
    os.getenv("STORYBOOK_QUERY_RESULT_CACHE_TTL_SECONDS", "300")
)
QUERY_FEEDBACK_QUEUE_SIZE = int(
    os.getenv("STORYBOOK_QUERY_FEEDBACK_QUEUE_SIZE", "1024")
)

# ── 自适应 Hybrid Search / Query Transformation ──
# ``fast`` 是兼容且可预测的默认值：常态融合 vector + lexical + environment，
# 可继续使用受预算 Memory Graph，但绝不调用生成式 LLM。``auto`` 在 fast 结果
# 低置信或查询复杂时才进入第二阶段；``deep`` 由调用方显式选择。
QUERY_DEFAULT_MODE = os.getenv("STORYBOOK_QUERY_DEFAULT_MODE", "fast").strip().lower()
QUERY_TRANSFORM_ENABLED = os.getenv(
    "STORYBOOK_QUERY_TRANSFORM_ENABLED", "1"
).strip().lower() not in {"0", "false", "no", "off"}
QUERY_AUTO_CONFIDENCE_THRESHOLD = float(os.getenv(
    "STORYBOOK_QUERY_AUTO_CONFIDENCE_THRESHOLD", "0.62"
))
QUERY_AUTO_SCORE_GAP_THRESHOLD = float(os.getenv(
    "STORYBOOK_QUERY_AUTO_SCORE_GAP_THRESHOLD", "0.035"
))
QUERY_AUTO_COMPLEX_CHARS = int(os.getenv(
    "STORYBOOK_QUERY_AUTO_COMPLEX_CHARS", "80"
))
QUERY_AUTO_MAX_TRANSFORMS = int(os.getenv(
    "STORYBOOK_QUERY_AUTO_MAX_TRANSFORMS", "2"
))
QUERY_MULTI_QUERY_LIMIT = int(os.getenv(
    "STORYBOOK_QUERY_MULTI_QUERY_LIMIT", "3"
))
QUERY_AUTO_TRANSFORM_TIMEOUT_SECONDS = float(os.getenv(
    "STORYBOOK_QUERY_AUTO_TRANSFORM_TIMEOUT_SECONDS", "1.2"
))
QUERY_DEEP_TRANSFORM_TIMEOUT_SECONDS = float(os.getenv(
    "STORYBOOK_QUERY_DEEP_TRANSFORM_TIMEOUT_SECONDS", "3.5"
))
QUERY_AUTO_SECOND_STAGE_TIMEOUT_SECONDS = float(os.getenv(
    "STORYBOOK_QUERY_AUTO_SECOND_STAGE_TIMEOUT_SECONDS", "2.0"
))
QUERY_DEEP_SECOND_STAGE_TIMEOUT_SECONDS = float(os.getenv(
    "STORYBOOK_QUERY_DEEP_SECOND_STAGE_TIMEOUT_SECONDS", "4.0"
))
QUERY_DEEP_TOTAL_TIMEOUT_SECONDS = float(os.getenv(
    "STORYBOOK_QUERY_DEEP_TOTAL_TIMEOUT_SECONDS", "5.0"
))

# RRF 把不同量纲的 vector / lexical / transformed-query 排名变成可解释分数。
HYBRID_RRF_K = int(os.getenv("STORYBOOK_HYBRID_RRF_K", "60"))
HYBRID_VECTOR_WEIGHT = float(os.getenv(
    "STORYBOOK_HYBRID_VECTOR_WEIGHT", "1.0"
))
HYBRID_LEXICAL_WEIGHT = float(os.getenv(
    "STORYBOOK_HYBRID_LEXICAL_WEIGHT", "0.8"
))
HYBRID_TRANSFORM_WEIGHT_AUTO = float(os.getenv(
    "STORYBOOK_HYBRID_TRANSFORM_WEIGHT_AUTO", "0.35"
))
HYBRID_TRANSFORM_WEIGHT_DEEP = float(os.getenv(
    "STORYBOOK_HYBRID_TRANSFORM_WEIGHT_DEEP", "0.5"
))

# 本地轻量 reranker 只处理有界 top-N，并拥有独立 deadline 与进程内熔断器。
RERANK_ENABLED = os.getenv(
    "STORYBOOK_RERANK_ENABLED", "1"
).strip().lower() not in {"0", "false", "no", "off"}
RERANK_TOP_N = int(os.getenv("STORYBOOK_RERANK_TOP_N", "20"))
RERANK_TIMEOUT_SECONDS = float(os.getenv(
    "STORYBOOK_RERANK_TIMEOUT_SECONDS", "0.08"
))
RERANK_FAILURE_THRESHOLD = int(os.getenv(
    "STORYBOOK_RERANK_FAILURE_THRESHOLD", "2"
))
RERANK_CIRCUIT_COOLDOWN_SECONDS = float(os.getenv(
    "STORYBOOK_RERANK_CIRCUIT_COOLDOWN_SECONDS", "30"
))

# ── 记忆加工阈值 ──
SIM_THRESHOLD_HIGH = 0.85      # ≥ → 合并/更新
SIM_THRESHOLD_UPDATE_ONLY = 0.92  # ≥ → 仅补充细节（不合并内容）
SIM_THRESHOLD_LOW = 0.75       # ≥ 且 <high → 弱关联新建
SIM_THRESHOLD_SEARCH = 0.50    # 检索最低相似度
TOP_K_RETRIEVAL = 5            # 做梦时检索相似story数量
TOP_K_SEARCH = 3               # 用户搜索返回Top3
# Story v2 only budgets the abstract and recall presentation. Structured detail
# and source evidence are persisted losslessly; this legacy constant remains for
# callers that still import it but no longer controls formation/splitting.
STORY_MAX_CHARS = 400
STORY_ABSTRACT_MAX_CHARS = int(
    os.getenv("STORYBOOK_ABSTRACT_MAX_CHARS", "600")
)
RECALL_SUMMARY_MAX_CHARS = int(
    os.getenv("STORYBOOK_RECALL_SUMMARY_MAX_CHARS", "600")
)
ENVIRONMENT_SCORE_WEIGHT = 0.08  # 环境仅在同语义分桶内作有界次序调节

# ── 关联权重规则 ──
WEIGHT_INCREMENT = 0.1         # 共同调用每次提升
WEIGHT_MAX = 1.0               # 权重上限
WEIGHT_PARENT_CHILD = 1.0      # 父子story默认权重

# ── Memory Graph / Graph RAG ──
# 边类型是对“经历记忆”的关联建模，不把 Story 强行投影为事实知识图谱。
# ``sibling`` 仅为 v0.1 兼容别名；新写入应使用下列七种标准类型。
MEMORY_EDGE_TYPES = (
    "semantic",
    "temporal",
    "causal",
    "same_environment",
    "parent_child",
    "co_recall",
    "supersedes",
)
DIRECTED_EDGE_TYPES = frozenset({
    "temporal", "causal", "parent_child", "supersedes",
})
GRAPH_DEFAULT_ENABLED = os.getenv(
    "STORYBOOK_GRAPH_ENABLED", "1"
).strip().lower() not in {"0", "false", "no", "off"}
GRAPH_MAX_HOPS = int(os.getenv("STORYBOOK_GRAPH_MAX_HOPS", "2"))
GRAPH_MAX_PATHS = int(os.getenv("STORYBOOK_GRAPH_MAX_PATHS", "64"))
GRAPH_FAN_OUT = int(os.getenv("STORYBOOK_GRAPH_FAN_OUT", "8"))
GRAPH_TIME_BUDGET_MS = float(os.getenv("STORYBOOK_GRAPH_TIME_BUDGET_MS", "100"))
GRAPH_TOKEN_BUDGET = int(os.getenv("STORYBOOK_GRAPH_TOKEN_BUDGET", "1600"))
GRAPH_DEEP_MAX_HOPS = int(os.getenv("STORYBOOK_GRAPH_DEEP_MAX_HOPS", "3"))
GRAPH_DEEP_MAX_PATHS = int(os.getenv("STORYBOOK_GRAPH_DEEP_MAX_PATHS", "160"))
GRAPH_DEEP_FAN_OUT = int(os.getenv("STORYBOOK_GRAPH_DEEP_FAN_OUT", "16"))
GRAPH_DEEP_TIME_BUDGET_MS = float(os.getenv(
    "STORYBOOK_GRAPH_DEEP_TIME_BUDGET_MS", "350"
))
GRAPH_DEEP_TOKEN_BUDGET = int(os.getenv(
    "STORYBOOK_GRAPH_DEEP_TOKEN_BUDGET", "3200"
))
GRAPH_HOP_DECAY = float(os.getenv("STORYBOOK_GRAPH_HOP_DECAY", "0.82"))
GRAPH_MIN_SCORE = float(os.getenv("STORYBOOK_GRAPH_MIN_SCORE", "0.55"))
GRAPH_CO_RECALL_HALF_LIFE_DAYS = float(os.getenv(
    "STORYBOOK_GRAPH_CO_RECALL_HALF_LIFE_DAYS", "30"
))
GRAPH_CO_RECALL_MIN_WEIGHT = float(os.getenv(
    "STORYBOOK_GRAPH_CO_RECALL_MIN_WEIGHT", "0.02"
))
GRAPH_EDGE_TYPE_FACTORS = {
    "semantic": 0.85,
    "temporal": 0.80,
    "causal": 1.00,
    "same_environment": 0.90,
    "parent_child": 0.95,
    "co_recall": 0.70,
    "supersedes": 1.00,
    # v0.1 兼容：新边不再写 sibling，但旧库仍可召回。
    "sibling": 0.75,
}

# ── 会话启动主动注入（晨间简报 / 上下文预热）──
# 见 src/storybook/prime.py。相比普通检索，主动注入用更高相关度门槛，
# 避免把弱相关记忆塞进每次会话开头的上下文造成噪声。
PRIME_MIN_SIMILARITY = 0.60    # 主动注入最低相似度（高于 SIM_THRESHOLD_SEARCH=0.50）
PRIME_TOP_K = 5                # 主动注入最多考虑的候选数（再按 token 预算裁剪）
PRIME_TOKEN_BUDGET = 2000      # 注入简报的 token 预算上限（≤2k，避免污染上下文）
PRIME_CONTENT_EXCERPT_CHARS = 140  # 简报中每条 story 摘要的最大字符数

# ── Claude Code 会话路径（主数据源）──
# 每个 .jsonl 文件 = 一个会话；目录名是 cwd 编码（/ -> -）
CLAUDE_PROJECTS_PATH = Path.home() / ".claude" / "projects"

# ── Cursor 日志路径（备用数据源，未用 Cursor 可忽略）──
CURSOR_STORAGE_PATH = Path.home() / "Library" / "Application Support" / "Cursor" / "User" / "workspaceStorage"

# ── 做梦周期自动化 ──
# 定时触发间隔（秒）：launchd / `storybook dream` 守护进程每轮做梦的间隔，默认 4 小时。
# launchd 无 shell 环境，故可通过环境变量在 plist 中覆盖。
DREAM_INTERVAL = int(os.getenv("STORYBOOK_DREAM_INTERVAL", "14400"))
# `storybook process --watch` 轮询 ~/.claude/projects 的间隔（秒），默认 60。
WATCH_POLL_INTERVAL = int(os.getenv("STORYBOOK_WATCH_POLL_INTERVAL", "60"))
