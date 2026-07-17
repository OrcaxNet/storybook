# 离线 Coding 记忆系统 MVP — 技术方案

## 一、环境基线（已确认）

| 组件 | 现状 |
|------|------|
| OS | macOS (Apple Silicon) |
| Python | 3.9.6 (系统) / 3.14 (Homebrew)；用 uv 创建 3.11 venv |
| Ollama | ✅ 已安装运行 |
| LLM 模型 | `qwythos-hermes:latest` (Qwen3架构, 131K上下文) |
| Embedding 模型 | `qwen3-embedding:0.6b` (1024维, 639MB) |
| SQLite | ✅ 系统自带 |
| Cursor | ❌ 未安装（MVP用模拟数据，后续装Cursor后自动适配） |

## 二、技术选型

### 2.1 整体架构

```
┌─────────────────────────────────────────────────┐
│                   CLI (click)                    │
│         import / process / search / stats        │
├─────────────────────────────────────────────────┤
│                                                  │
│  ┌──────────┐  ┌──────────────┐  ┌───────────┐  │
│  │ Collector │  │  Processor   │  │  Search   │  │
│  │ (日志采集)│  │  (做梦加工)  │  │  (检索)   │  │
│  └────┬─────┘  └──────┬───────┘  └─────┬─────┘  │
│       │               │                │         │
│       │         ┌─────┴──────┐         │         │
│       │         │  LLM (摘要  │         │         │
│       │         │  /关键词   │         │         │
│       │         │  /分裂判断)│         │         │
│       │         └─────┬──────┘         │         │
│       │         ┌─────┴──────┐         │         │
│       │         │ Embedding  │         │         │
│       │         │ (语义向量) │         │         │
│       │         └─────┬──────┘         │         │
│       │               │                │         │
│  ┌────┴───────────────┴────────────────┴─────┐  │
│  │            Store (SQLite + sqlite-vec)      │  │
│  │     stories / edges / sessions / queue     │  │
│  └────────────────────────────────────────────┘  │
│                                                  │
│  ┌──────────────────────────────────────────┐    │
│  │          Scheduler (launchd cron)         │    │
│  │          每4小时触发 process              │    │
│  └──────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

### 2.2 核心组件选型

| 层 | 选型 | 理由 |
|----|------|------|
| **存储** | SQLite + `sqlite-vec` | 单文件、零运维；sqlite-vec 是 sqlite-ivm 作者新项目，2024年最热的 SQLite 向量扩展，GitHub 4k+ stars |
| **LLM** | Ollama `qwythos-hermes` | 本地已部署，131K上下文，无需API费用，完全离线 |
| **Embedding** | Ollama `qwen3-embedding:0.6b` | 本地已部署，1024维，中英文表现优秀 |
| **CLI框架** | `click` | Python CLI 事实标准，比 argparse 好用，比 typer 轻量 |
| **定时调度** | macOS `launchd` (plist) | 原生、可靠、不依赖额外进程；也提供 Python 内置 `schedule` 作为备选 |
| **日志** | Python `logging` | 标准库够用 |

### 2.3 为什么选 sqlite-vec

| 对比项 | sqlite-vec | ChromaDB | FAISS |
|--------|-----------|----------|-------|
| 安装复杂度 | `pip install` 即可 | 需要额外依赖 | 需编译 |
| 持久化 | SQLite单文件 | 自带但较重 | 需手动保存 |
| 关系数据 | SQL 原生支持 | 不支持 | 不支持 |
| 适合规模 | <100万向量 | <100万 | <10亿 |
| MVP匹配度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

我们的 story 库规模在千级别，sqlite-vec 完全够用，且能在同一个 SQLite 文件里同时存结构化数据和向量，完美匹配需求。

### 2.4 sqlite-vec 备选方案

如果 `sqlite-vec` 在 Apple Silicon 上编译有问题，备选：
- **numpy 余弦相似度**：把所有 story 向量加载到内存 numpy 数组，暴力计算余弦相似度。千级 story 完全没问题，延迟 <10ms。
- 这个方案零依赖、零编译风险，作为 fallback 非常可靠。

## 三、数据模型设计

### 3.1 SQLite Schema

```sql
-- 会话日志表（原始导入数据）
CREATE TABLE sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,          -- 'cursor' / 'copilot' / 'manual'
    raw_content TEXT NOT NULL,     -- 原始会话JSON
    problem_desc TEXT,             -- 提取的问题描述
    code_snippets TEXT,            -- 代码片段(JSON array)
    conclusion TEXT,               -- 核心结论
    status TEXT DEFAULT 'pending', -- pending / processed / failed
    created_at TEXT DEFAULT (datetime('now')),
    processed_at TEXT
);

-- Story 表（结构化记忆单元）
CREATE TABLE stories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,           -- 简短标题
    content TEXT NOT NULL,         -- ≤400字的"问题-步骤-结果"
    keywords TEXT NOT NULL,        -- 关键词(JSON array)
    embedding TEXT,                -- 语义向量(JSON array, 1024维)
    parent_id INTEGER,             -- 父story ID（分裂时指向原story）
    source_session_ids TEXT,       -- 来源会话ID(JSON array)
    access_count INTEGER DEFAULT 0,-- 被检索命中次数
    version INTEGER DEFAULT 1,     -- 版本号，每次更新+1
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (parent_id) REFERENCES stories(id)
);

-- 关联边表（带权重的关联网络）
CREATE TABLE edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    target_id INTEGER NOT NULL,
    weight REAL DEFAULT 0.0,       -- 0.0~1.0
    edge_type TEXT DEFAULT 'semantic', -- 'semantic' / 'parent_child' / 'co_occur'
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (source_id) REFERENCES stories(id),
    FOREIGN KEY (target_id) REFERENCES stories(id),
    UNIQUE(source_id, target_id)
);

-- 向量表（sqlite-vec 虚拟表，关联到 stories）
CREATE VIRTUAL TABLE story_vectors USING vec0(
    story_id INTEGER PRIMARY KEY,
    embedding FLOAT[1024]
);
```

### 3.2 Story 数据结构

```python
{
    "id": 1,
    "title": "React useEffect 无限循环排查",
    "content": "问题：useEffect触发无限渲染循环。步骤：1.检查依赖数组发现遗漏了setState的回调依赖；2.使用useCallback包裹回调函数；3.将不依赖props的逻辑移出useEffect。结果：渲染次数从无限降为2次。",
    "keywords": ["react", "useEffect", "无限循环", "依赖数组", "useCallback"],
    "embedding": [0.012, -0.034, ...],  # 1024维
    "parent_id": None,
    "access_count": 3,
    "version": 2
}
```

## 四、核心算法设计

### 4.1 「做梦」加工流程

```
输入: 一条 pending 状态的 session
  │
  ▼
Step 1: LLM 提取
  ├── 提取核心技术关键词 (5-10个)
  ├── 生成语义向量 (embedding model)
  └── 提取问题摘要 (≤100字)
  │
  ▼
Step 2: 记忆检索
  ├── 用语义向量在 story_vectors 中搜索 Top5 相似 story
  ├── 相似度阈值: 0.75 (余弦相似度)
  └── 判断匹配等级:
       ├── 高匹配 (sim ≥ 0.85): 合并/更新
       ├── 低匹配 (0.75 ≤ sim < 0.85): 弱关联
       └── 无匹配 (sim < 0.75): 新建story
  │
  ▼
Step 3: 记忆处理 (三种分支)
  │
  ├── 【新建】将会话浓缩为≤400字story
  │     ├── LLM生成"问题-步骤-结果"格式摘要
  │     ├── 存入 stories 表
  │     ├── 向量存入 story_vectors
  │     └── 与Step2找到的弱匹配story建立边(weight=sim)
  │
  ├── 【合并】将新内容并入旧story
  │     ├── LLM合并旧story + 新会话内容 → 新story文本
  │     ├── 检查分裂条件:
  │     │     ├── 合并后 > 400字? → 触发分裂
  │     │     └── LLM判断包含2+独立子步骤? → 触发分裂
  │     ├── 不需分裂: 更新story内容、关键词、向量、version+1
  │     └── 需要分裂: 
  │           ├── LLM拆分为多个子story (每个≤400字)
  │           ├── 子story parent_id 指向原story
  │           ├── 父子边 weight=1.0
  │           └── 子story间建立语义边
  │
  └── 【更新】仅补充细节
        ├── 更新story的keywords (合并去重)
        ├── 重新生成embedding
        └── 关联边weight += 0.1 (上限1.0)
  │
  ▼
Step 4: 关联网络维护
  ├── 新建story: 与Top5相似story建立语义边(weight=sim值)
  ├── 合并story: 强化已有边weight (+0.1, 上限1.0)
  └── session.status → 'processed'
```

### 4.2 检索激活流程

```
用户输入: "useEffect 一直触发怎么办"
  │
  ▼
Step 1: 生成查询向量
  ├── LLM提取关键词
  └── embedding model生成语义向量
  │
  ▼
Step 2: 向量检索 Top3
  ├── 在 story_vectors 中余弦相似度搜索
  └── 返回 Top3 story (sim ≥ 0.5)
  │
  ▼
Step 3: 关联激活
  ├── 对每个Top3 story:
  │     ├── 查询 edges 表，按 weight DESC 取关联story
  │     └── 展示关联story列表 (模拟"下意识联想")
  │
  ▼
Step 4: 返回结果
  {
    "top_matches": [
      {
        "story": {...},
        "similarity": 0.89,
        "related": [
          {"story": {...}, "weight": 0.9},
          {"story": {...}, "weight": 0.7}
        ]
      },
      ...
    ]
  }
```

### 4.3 分裂触发条件

```python
def should_split(merged_text: str, llm_judge: str) -> bool:
    """判断是否需要分裂"""
    # 条件1: 合并后文本超过400字
    if len(merged_text) > 400:
        return True
    # 条件2: LLM判断包含2+独立可复用子步骤
    if "SPLIT:YES" in llm_judge:
        return True
    return False
```

### 4.4 关联权重规则

| 场景 | 初始权重 | 更新规则 |
|------|---------|---------|
| 新建story与相似story | = 语义相似度值 | 每共同被调用 +0.1 |
| 父子story (分裂产生) | 1.0 | 固定不变 |
| 合并更新已有边 | 不变 | +0.1 (上限1.0) |
| 弱关联 (0.75≤sim<0.85) | = sim值 | 每共同被调用 +0.1 |

## 五、模块设计

### 5.1 目录结构

```
coding-memory/
├── pyproject.toml
├── README.md
├── .env.example
├── src/
│   └── coding_memory/
│       ├── __init__.py
│       ├── config.py          # 配置管理
│       ├── store.py           # SQLite存储层
│       ├── embeddings.py      # Ollama embedding封装
│       ├── llm.py             # Ollama LLM封装 (摘要/关键词/分裂)
│       ├── collector.py       # Cursor日志采集
│       ├── processor.py       # 「做梦」核心流程
│       ├── search.py          # 检索激活
│       └── cli.py             # CLI入口
├── scripts/
│   └── com.hermes.coding-memory.plist  # launchd定时任务
├── data/                      # SQLite数据库存放
│   └── memory.db
└── tests/
    ├── test_store.py
    ├── test_processor.py
    └── test_data/             # 模拟Cursor会话日志
        └── sample_sessions.json
```

### 5.2 模块职责

| 模块 | 职责 | 核心接口 |
|------|------|---------|
| `config.py` | 配置管理 | `DB_PATH`, `OLLAMA_HOST`, `LLM_MODEL`, `EMBED_MODEL`, 阈值常量 |
| `store.py` | 存储CRUD | `init_db()`, `add_session()`, `get_pending_sessions()`, `add_story()`, `update_story()`, `add_edge()`, `update_edge_weight()`, `search_vectors()` |
| `embeddings.py` | 语义向量 | `embed(text) -> List[float]`, `cosine_similarity(v1, v2) -> float` |
| `llm.py` | LLM处理 | `extract_keywords(text) -> List[str]`, `summarize_session(session) -> str`, `merge_stories(old, new) -> str`, `judge_split(merged) -> bool`, `split_story(merged) -> List[dict]` |
| `collector.py` | 日志采集 | `collect_cursor_sessions(path) -> List[dict]`, `import_session(session_dict)` |
| `processor.py` | 做梦加工 | `process_session(session_id)`, `process_all_pending()`, `run_dream_cycle()` |
| `search.py` | 检索激活 | `search(query) -> dict`, `get_related_stories(story_id) -> List[dict]` |
| `cli.py` | CLI入口 | `cm import <path>`, `cm process`, `cm search <query>`, `cm stats` |

## 六、关键技术实现细节

### 6.1 LLM Prompt 设计

**摘要生成 Prompt：**
```
你是一个代码记忆管理专家。请将以下AI编程会话浓缩为不超过400字的结构化记忆。

要求：
1. 格式："问题：... 步骤：1.... 2.... 结果：..."
2. 保留核心技术细节和解决方案
3. 去除寒暄、重复、无效内容
4. 聚焦单个coding问题的完整解决逻辑

会话内容：
{session_content}

请直接输出记忆文本，不要额外说明。
```

**关键词提取 Prompt：**
```
从以下技术文本中提取5-10个核心技术关键词。
要求：
1. 包含技术栈名（如React、Python）
2. 包含问题类型（如内存泄漏、类型错误）
3. 包含关键解决方案术语
4. 中英文混合，保持原始语言

文本：
{text}

以JSON数组格式输出，如：["React", "useEffect", "无限循环"]
```

**分裂判断 Prompt：**
```
判断以下技术记忆是否包含两个或以上独立可复用的子步骤。

记忆内容：
{text}

判断标准：
- "配置ESLint规则" 和 "修复useEffect依赖" = 2个独立子步骤 → SPLIT:YES
- "检查依赖数组" 和 "使用useCallback" = 同一问题的连续步骤 → SPLIT:NO

只输出 SPLIT:YES 或 SPLIT:NO
```

### 6.2 Cursor 日志采集策略

Cursor 的会话数据存储在 VS Code 风格的 SQLite 数据库中：
- **路径**：`~/Library/Application Support/Cursor/User/workspaceStorage/*/state.vscdb`
- **关键表**：`cursor_disk_cache` 表中的 `compositeKey` 包含聊天会话
- **解析**：从 `cursor_disk_cache` 表提取 `aiService.prompts` 记录

MVP 阶段策略：
1. 扫描所有 workspaceStorage 子目录
2. 读取每个 `state.vscdb`
3. 从 `cursor_disk_cache` 表提取会话数据
4. 按会话 ID 拆分，提取问题、代码、结论
5. 由于当前机器未装 Cursor，提供 **模拟数据生成器** 用于测试

### 6.3 向量检索实现

优先使用 sqlite-vec：

```python
# sqlite-vec 方式
import sqlite_vec
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)

# 搜索
results = db.execute("""
    SELECT s.*, v.distance
    FROM story_vectors v
    JOIN stories s ON s.id = v.story_id
    WHERE v.embedding MATCH ?
    ORDER BY v.distance
    LIMIT 5
""", [json.dumps(query_vector)])
```

备选 numpy 暴力搜索（零编译风险）：

```python
# numpy fallback
def search_vectors(query_vec, top_k=5):
    all_vecs = db.execute("SELECT story_id, embedding FROM stories WHERE embedding IS NOT NULL")
    # 加载到numpy
    ids, vectors = zip(*[(r[0], np.array(json.loads(r[1]))) for r in all_vecs])
    sims = np.dot(vectors, query_vec) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vec))
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [(ids[i], sims[i]) for i in top_indices]
```

## 七、定时任务配置

### 7.1 launchd plist（macOS 原生）

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/propertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.hermes.coding-memory</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/orca/.local/bin/cm</string>
        <string>process</string>
    </array>
    <key>StartInterval</key>
    <integer>14400</integer>  <!-- 4小时 = 14400秒 -->
    <key>StandardOutPath</key>
    <string>/Users/orca/coding-memory/logs/cron.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/orca/coding-memory/logs/cron-error.log</string>
</dict>
</plist>
```

## 八、MVP 验收标准

| # | 验收项 | 验证方法 |
|---|--------|---------|
| 1 | 100条模拟Cursor会话导入 | `cm import tests/test_data/sample_sessions.json` |
| 2 | 自动加工成story库 | `cm process` → 检查stories表记录数 |
| 3 | 重复bug检索准确率≥70% | `cm search "useEffect无限循环"` → Top3包含正确story |
| 4 | 关联story推送覆盖80% | 检查关联story是否覆盖实际解决步骤搭配 |
| 5 | 每4小时自动触发 | launchd定时任务生效 |

## 九、开发计划

| 阶段 | 内容 | 预计产出 |
|------|------|---------|
| P1 | 项目骨架 + 存储层 + 依赖安装 | 可运行的空壳 + SQLite schema |
| P2 | Embedding + LLM 封装 | 可调用的向量/LLM接口 |
| P3 | 日志采集 + 模拟数据 | 100条测试会话 |
| P4 | 做梦加工流程（核心） | process命令可用 |
| P5 | 检索激活 | search命令可用 |
| P6 | CLI整合 + 端到端测试 | 全流程跑通 |
| P7 | launchd定时任务 | 自动化运行 |
