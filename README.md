# 📖 Storybook

> 离线 Coding 记忆系统 —— 把每一次 AI 编程会话在"梦境"里整理成一条 Story，再连成一张可被联想唤醒的记忆网。
>
> An offline coding memory system that consolidates AI-coding sessions into structured *Stories* via a "dream cycle" and links them into a weighted association graph. All LLM / embedding work runs through a **local Ollama** — fully offline.

## 这是什么 / What it is

Storybook 采集 AI 编程会话日志（Claude Code 会话、Cursor 日志、JSON 文件或内置模拟器），对每条会话跑一遍 **做梦周期（dream cycle）**：用 LLM 抽取关键词、做摘要，把它沉淀成一条 ≤400 字的结构化 *Story*（问题 / 步骤 / 结果），并与已有记忆按相似度**合并 / 更新 / 新建**，同时在 Story 之间建立带权重的关联边。

检索时，先做向量相似度召回，再沿关联边激活相关 Story，共同被召回的 Story 之间的边权重会被强化——像人脑在反复回忆中加深记忆路径。

整个系统**完全离线**：LLM 与 embedding 都走本地 Ollama，不依赖任何云端服务。

## ✨ 特性

- 🧠 **做梦式记忆整理**：每条会话被压缩成一条 Story，相似记忆自动合并/分裂/更新，避免记忆膨胀
- 🔗 **带权关联图**：Story 间有 `semantic` / `parent_child` / `sibling` 三类无向边，检索时沿边激活
- 📐 **双索引存储**：SQLite + sqlite-vec（vec0 向量表），向量同时存于 `stories.embedding` 与 `story_vectors`，L2 归一化后用余弦相似度
- 🔍 **联想检索**：向量召回 + 边图扩散，共同召回会反向增强边权重
- 🔌 **多数据源**：Claude Code 会话（主）、Cursor、JSON 文件/目录、内置模拟器
- 🏠 **完全离线**：只需本地 Ollama，零云端依赖

## 🏗 架构

模块流（均位于 `src/storybook/`）：

```
collector → store → processor (用 llm + embeddings) → search
                    ↑
                  cli.py 串起命令；config.py 集中所有路径/模型/阈值
```

### 做梦周期（`processor.process_session`）

对每条 pending 会话：LLM 抽取关键词 → 对 `关键词 + 问题简述` 做 embedding（聚焦而非用全文）→ 向量检索 top-K 已有 Story → 按最佳相似度分支：

| 分支 | 触发条件 | 动作 |
|------|----------|------|
| **create** | best sim < 0.85 | LLM 摘要成 ≤400 字 Story；与 0.75–0.85 的弱匹配 Story 建边，`weight = sim` |
| **merge** | 0.85 ≤ sim < 0.92 | LLM 合并新旧内容；若结果 >400 字或 LLM 判定 `SPLIT:YES`，则分裂为子 Story（`parent_id`，父子边 1.0，兄弟边 0.5）。分裂后父 Story 的向量从索引删除（不再参与检索），但行保留用于溯源 |
| **update** | sim ≥ 0.92 | 仅合并关键词、重新 embedding、强化已有边权重（+0.1，上限 1.0） |

### 检索（`search.search`）

embed 查询（关键词 + 查询文本）→ vec0 top-K（取 `top_k*2`，按 `SIM_THRESHOLD_SEARCH=0.50` 过滤）→ 对每个命中，沿 `edges` 表（权重降序）浮现相关 Story，并对共同召回的 Story 之间加边权重；命中 Story 的 `access_count` 自增。

### 关联图

`edges` 表，`UNIQUE(source_id, target_id)`。边类型：`semantic`、`parent_child`（固定 1.0）、`sibling`（0.5）。边是**无向**的：`add_or_update_edge` / `increment_edge_weight` 通过 `_edge_pair` 把端点归一化为 `(min_id, max_id)`，因此调用方向无关、`(A,B)` 与 `(B,A)` 不会重复建行。

### 存储层（`store.py`）

单文件 `data/memory.db`（SQLite + sqlite-vec 扩展）。三张表 `sessions` / `stories` / `edges` 外加 `story_vectors` vec0 虚表。**embedding 存两处且必须同步**：`stories.embedding`（float32 BLOB）与 `story_vectors` 一行。`search_by_vector` 用 L2 距离查询并换算为余弦相似度 `1 - dist²/2`（对 L2 归一化向量精确）。

## 🔧 环境要求

- **Python 3.11+**（推荐用 [uv](https://github.com/astral-sh/uv) 建 venv）
- **Ollama** 运行于 `http://localhost:11434`（可用 `OLLAMA_HOST` 覆盖），并拉取两个模型：
  - LLM：`qwythos-hermes:latest`（可用 `STORYBOOK_LLM_MODEL` 覆盖）
  - Embedding：`qwen3-embedding:0.6b`，**1024 维**（可用 `STORYBOOK_EMBED_MODEL` 覆盖）；需与 `config.EMBED_DIM` 一致
- 依赖：`click`、`requests`、`numpy`、`sqlite-vec`

```bash
# 拉模型
ollama pull qwythos-hermes:latest
ollama pull qwen3-embedding:0.6b
```

## 📦 安装

```bash
# 1. 建 venv（uv 创建的 venv 没有 pip）
uv venv .venv

# 2. 以 editable 方式安装，得到 storybook 命令
VIRTUAL_ENV=$(pwd)/.venv uv pip install -e .
# （项目目录移动后 shebang 可能失效，重新跑一次即可）

# 3. 验证
storybook stats
```

不想安装也可直接跑模块：

```bash
PYTHONPATH=src .venv/bin/python -m storybook.cli <command>
```

## 🧪 测试

测试套件覆盖 `store` / `processor` / `search` 三个核心模块的关键路径与边界，
**完全不依赖 Ollama**——所有 LLM / embedding 调用均被 mock 桩替换，本地一键可重复运行。

```bash
# 1. 安装测试依赖（与运行时依赖一并）
VIRTUAL_ENV=$(pwd)/.venv uv pip install -e ".[test]"

# 2. 一键运行全部测试
.venv/bin/pytest

# 带覆盖率报告（聚焦 store/processor/search）
.venv/bin/pytest --cov=storybook --cov-report=term-missing
```

测试不启动 Ollama、不联网：把 `OLLAMA_HOST` 指向任意地址都不影响结果。
用例要点：

- **store**：Session/Story CRUD、`_edge_pair` 无向边归一、`search_by_vector` 的 `1 - dist²/2`
  相似度换算（与 numpy 暴力余弦交叉验证）、双写一致性（`add_story`/`update_story` 后
  `stories.embedding` 与 `story_vectors` 同步；分裂后父 story 向量从索引移除）。
- **processor**：create / merge / update 三分支 + split 路径，mock `llm`/`embeddings` 返回固定值，
  验证分支选择与边建立（弱关联建边、共同召回提权、父子/兄弟边）。
- **search**：阈值过滤、关联激活、共同召回提权（每对每次仅 +0.1 一次）。
- **集成**：用 `generate_sample_sessions` 与 `test_logs/*.json` 跑通 collector → store → processor → search 全链路。

## 🚀 使用

```bash
storybook init                       # 初始化数据库 schema + vec0 虚表（其它命令也会自动初始化）
storybook import-data                # 默认：从 ~/.claude/projects 采集 Claude Code 会话（增量、按 sessionId 去重）
storybook import-data --claude       # 同上（显式写法）
storybook import-data --sample [--n 100]   # 生成并导入模拟会话（无需真实会话即可体验）
storybook import-data --cursor       # 扫描 Cursor 的 workspaceStorage（备用数据源）
storybook import-data <file|dir>     # 导入 JSON（list / {sessions:[...]} / {messages:[...]} 聊天日志）

storybook process [--session ID]     # 做梦周期：处理所有 pending 会话（或指定一条）
storybook search "<query>" [--top 3] # 向量检索 + 关联 Story 激活
storybook stats                      # 系统统计
storybook list [--limit 20]          # 列出所有 Story
storybook show <story_id>            # 查看 Story 详情（含关联记忆）
```

> 命令是 **`import-data`** 而非 `import`（click 把 `import_data` 函数自动连字符化）。无参数/无 flag 时默认走 `--claude`。`--claude` / `--sample` / `--cursor` / `<path>` 四种来源互斥。

### 快速体验（无真实会话）

```bash
storybook import-data --sample --n 50   # 造 50 条模拟会话
storybook process                       # 跑做梦周期
storybook search "如何调试数据库连接"     # 搜一下
storybook stats                         # 看看沉淀了多少 Story
```

## ⚙️ 配置

所有路径、模型名、阈值都集中在 `src/storybook/config.py`。环境变量样例见 `.env.example`（注意：`config.py` 不会自动加载 `.env`，需自行 `export` 或用 direnv 等工具）。

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama 服务地址 |
| `STORYBOOK_LLM_MODEL` | `qwythos-hermes:latest` | 做梦加工用的 LLM |
| `STORYBOOK_EMBED_MODEL` | `qwen3-embedding:0.6b` | embedding 模型（必须 1024 维） |
| `STORYBOOK_LLM_THINK` | `0` | Qwen3 思考模式：`0`=关（提取类任务约 9× 加速），`1`=开（检索准确率不足时再开） |

关键阈值（`config.py`）：

| 常量 | 值 | 含义 |
|------|----|------|
| `SIM_THRESHOLD_HIGH` | 0.85 | ≥ 触发合并/更新 |
| `SIM_THRESHOLD_UPDATE_ONLY` | 0.92 | ≥ 仅补充细节（不合并内容） |
| `SIM_THRESHOLD_LOW` | 0.75 | ≥ 且 <high 触发弱关联新建 |
| `SIM_THRESHOLD_SEARCH` | 0.50 | 检索最低相似度 |
| `TOP_K_RETRIEVAL` / `TOP_K_SEARCH` | 5 / 3 | 做梦召回 / 用户搜索返回数 |
| `STORY_MAX_CHARS` | 400 | Story 最大字数 |
| `WEIGHT_INCREMENT` / `WEIGHT_MAX` | 0.1 / 1.0 | 共同召回提权 / 权重上限 |

LLM 调用参数（temp 0.3、`num_ctx` 8192、120s 超时）硬编码在 `llm._chat` / `_generate` 中。

## 📁 项目结构

```
storybook/
├── src/storybook/
│   ├── cli.py          # CLI 命令入口
│   ├── config.py       # 路径 / 模型 / 阈值常量
│   ├── collector.py    # 会话采集（Claude Code / Cursor / JSON / 模拟）
│   ├── store.py        # SQLite + sqlite-vec 存储层
│   ├── processor.py    # 做梦周期（dream cycle）
│   ├── llm.py          # Ollama LLM 调用
│   ├── embeddings.py   # Ollama embedding 调用
│   └── search.py       # 向量检索 + 关联激活
├── data/               # 运行时数据库 memory.db（不纳入版本管理）
├── logs/               # 运行时日志（不纳入版本管理）
├── docs/TECH_DESIGN.md # 原始设计文档
├── tests/              # pytest 测试套件（store/processor/search + 集成，全 mock Ollama）
├── test_logs/          # 示例 JSON 数据
├── hermes_sessions.json
├── .env.example
└── pyproject.toml
```

## 📝 说明

- **完全离线**：所有 LLM / embedding 走本地 Ollama，不发送任何数据到云端。
- **测试套件**：`tests/` 下 pytest 用例覆盖 store/processor/search 核心路径，全 mock、不依赖 Ollama（见上文「🧪 测试」）。`test_logs/*.json` 与 `hermes_sessions.json` 是 `import-data` 的样例数据源。
- `docs/TECH_DESIGN.md` 是最初的设计文档，其中的目录布局与命令示例早于当前实现（命令为 `import-data`，不存在 `tests/` 或 `scripts/` 目录，也未配置 launchd plist）。
- LLM 输出解析是宽松的：关键词 JSON 在 `[`/`]` 间切片，摘要按 `TITLE:`/`CONTENT:` 标记切分，模型不遵循格式时有字符串切分兜底。

## License

MIT
