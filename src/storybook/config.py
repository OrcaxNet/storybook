"""配置管理 — 所有路径、模型名、阈值常量集中管理"""
import os
from pathlib import Path

# ── 路径 ──
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
DB_PATH = DATA_DIR / "memory.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


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

# ── Ollama ──
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("STORYBOOK_LLM_MODEL", "qwythos-hermes:latest")
EMBED_MODEL = os.getenv("STORYBOOK_EMBED_MODEL", "qwen3-embedding:0.6b")
EMBED_DIM = 1024
LLM_THINK = os.getenv("STORYBOOK_LLM_THINK", "0") == "1"  # Qwen3 思考模式；提取类任务关闭可约 9x 加速，准确率不足时设 1

# ── 记忆加工阈值 ──
SIM_THRESHOLD_HIGH = 0.85      # ≥ → 合并/更新
SIM_THRESHOLD_UPDATE_ONLY = 0.92  # ≥ → 仅补充细节（不合并内容）
SIM_THRESHOLD_LOW = 0.75       # ≥ 且 <high → 弱关联新建
SIM_THRESHOLD_SEARCH = 0.50    # 检索最低相似度
TOP_K_RETRIEVAL = 5            # 做梦时检索相似story数量
TOP_K_SEARCH = 3               # 用户搜索返回Top3
STORY_MAX_CHARS = 400          # Story最大字数

# ── 关联权重规则 ──
WEIGHT_INCREMENT = 0.1         # 共同调用每次提升
WEIGHT_MAX = 1.0               # 权重上限
WEIGHT_PARENT_CHILD = 1.0      # 父子story默认权重

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
