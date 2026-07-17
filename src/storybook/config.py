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
