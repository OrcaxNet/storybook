"""配置管理 — Profile 路径、模型名、阈值常量集中管理。"""
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

# ── 安装根与 .env ──
BASE_DIR = Path(__file__).resolve().parent.parent.parent


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
LLM_MODEL = os.getenv("STORYBOOK_LLM_MODEL", "qwythos-hermes:latest")
EMBED_MODEL = os.getenv("STORYBOOK_EMBED_MODEL", "qwen3-embedding:0.6b")
EMBED_DIM = 1024
EMBED_VERSION = os.getenv("STORYBOOK_EMBED_VERSION", "story-v2-default-v1")
EMBED_REPRESENTATION = os.getenv(
    "STORYBOOK_EMBED_REPRESENTATION", "default"
)
LLM_THINK = os.getenv("STORYBOOK_LLM_THINK", "0") == "1"  # Qwen3 思考模式；提取类任务关闭可约 9x 加速，准确率不足时设 1

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
