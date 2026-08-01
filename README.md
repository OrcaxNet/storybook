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
- 🔍 **低时延联想检索**：MCP 启动预热 + Ollama keep-alive；按 `index_version` 失效的查询向量/结果双缓存；向量不可用或超时时在独立 500ms 预算内切到 FTS/关键词 fallback
- 🧵 **读写解耦**：向量召回与关联读取完成后立即返回，`access_count`/共同召回边权反馈由有界后台队列单事务写入
- 📈 **性能可观察**：每次查询分段记录 cache/embed/vector/fallback/graph/rerank/serialize/total，`status --performance` 汇总最近 100 次 p50/p95；固定 10k Story benchmark 同时守护检索质量
- 🔌 **多数据源**：Claude Code 会话（主）、Cursor、JSON 文件/目录、内置模拟器
- 🏠 **完全离线**：只需本地 Ollama，零云端依赖
- 🤖 **MCP 召回**：通过 MCP server 把记忆检索暴露给 Claude Code 等 agent，新任务可主动 recall 过往经历，实现跨 session 经验复用
- 🌅 **晨间简报**：会话启动时基于 cwd / 首条提问**主动召回**相关记忆并注入上下文（`SessionStart` hook 或 `prime_context` MCP 工具），实现"下意识回忆"--更贴近初衷；token 预算内、相关度不足时**静默不注入**

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

查询响应包含 `request_id`、`mode`、`degraded` 与 `latency_ms.{cache,embed,vector,graph,rerank,serialize,total}`。同一份阶段数据会写入本地 `logs/query_performance.jsonl`，但落盘接口只接受固定白名单字段：不保存原始 query、Story 内容、绝对路径、hostname 或仓库 URL。文件权限为 `0600`，超过大小上限后只保留最近记录。

### 关联图

`edges` 表，`UNIQUE(source_id, target_id)`。边类型：`semantic`、`parent_child`（固定 1.0）、`sibling`（0.5）。边是**无向**的：`add_or_update_edge` / `increment_edge_weight` 通过 `_edge_pair` 把端点归一化为 `(min_id, max_id)`，因此调用方向无关、`(A,B)` 与 `(B,A)` 不会重复建行。

### 存储层（`store.py`）

每个用户 Profile 一份 `profiles/{随机 UUID}/db/memory.db`（SQLite + sqlite-vec 扩展），不再存于仓库。三张表 `sessions` / `stories` / `edges` 外加 `story_vectors` vec0 虚表；持久对象带与路径无关的 `global_id`、`profile_id` 和 `sync_state=local_only`。**embedding 存两处且必须同步**：`stories.embedding`（float32 BLOB）与 `story_vectors` 一行。`search_by_vector` 用 L2 距离查询并换算为余弦相似度 `1 - dist²/2`（对 L2 归一化向量精确）。

## 🔧 环境要求

- **Python 3.11+**（推荐用 [uv](https://github.com/astral-sh/uv) 建 venv）
- **Ollama** 运行于 `http://localhost:11434`（可用 `OLLAMA_HOST` 覆盖），并拉取两个模型：
  - LLM：`qwythos-hermes:latest`（可用 `STORYBOOK_LLM_MODEL` 覆盖）
  - Embedding：`qwen3-embedding:0.6b`，**1024 维**（可用 `STORYBOOK_EMBED_MODEL` 覆盖）；需与 `config.EMBED_DIM` 一致
- 依赖：`click`、`requests`、`numpy`、`sqlite-vec`、`mcp`（Agent 接入所需）

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

# 3. 一键建立 Profile、接入已检测到的 Agent 并运行端到端自检
storybook setup
```

不想安装也可直接跑模块：

```bash
PYTHONPATH=src .venv/bin/python -m storybook.cli <command>
```

### 一键 setup 与安全卸载

`storybook setup` 会先展示完整改动计划，再创建用户级 Profile/schema、检测并接入
Claude Code、Cursor、Codex，检查或下载本地 Ollama 模型，最后执行 schema、embedding、
adapter、recall smoke test。三类 Agent 都复用同一个 `storybook mcp` stdio server；Claude
Code 还会安装幂等的 `SessionStart` recall hook。无需手工编辑 JSON/TOML。

```bash
storybook setup                         # 交互确认
storybook setup --yes                   # 非交互安装
storybook setup --dry-run               # 严格零写入（不建目录/DB、不下载模型）
storybook setup --json                  # 结构化结果，便于自动化
storybook setup --agent codex --yes     # 可重复 --agent，覆盖自动检测
storybook setup --skip-models --yes     # 离线跳过缺失模型，状态为 degraded

storybook uninstall                     # 恢复 setup 写入的节点，默认保留全部记忆
storybook uninstall --dry-run
storybook uninstall --purge-data        # 交互式二次确认后永久删除数据
storybook uninstall --yes --purge-data --confirm-purge  # 非交互双重显式确认
```

配置更新使用同目录原子替换，并在用户 state 目录保存原文件备份与 hash；重复执行不会
重复 MCP 节点或 hook。卸载只恢复名为 `storybook` 的受管节点，保留其他 server、hook
和设置；若节点在安装后被人工修改，会报告 drift 并保留恢复状态，避免覆盖用户改动。
旧项目级 `data/memory.db` 只会在计划/结果中提示，不会由 setup 擅自迁移或删除。

## 👤 用户级 Profile 与共享存储

首次运行会创建随机 UUID 的 `local` Profile。Claude Code 采集、Cursor 采集、CLI、hook 与 MCP（包括 Codex 等 MCP-aware agent）都经同一份 registry 解析当前数据库，因此切换项目 cwd、移动或重命名 Storybook 仓库不会改变记忆归属。

| 平台 | Profile 数据根 |
|------|----------------|
| macOS | `~/Library/Application Support/Storybook/profiles/{profile_id}/` |
| Linux | `${XDG_DATA_HOME:-~/.local/share}/storybook/profiles/{profile_id}/` |
| Windows | `%LOCALAPPDATA%\Storybook\profiles\{profile_id}\` |

数据库/索引位于 Profile 数据根，缓存与日志走各平台的 cache/state 目录；目录权限在 POSIX 上为 `0700`，registry 与数据库为 `0600`。registry 只持久化随机 UUID、显示名、模式和同步状态，不把用户名、hostname 或绝对路径当作主键。

```bash
storybook profile show                    # 当前 Profile、数据目录与 local-only 状态
storybook profile list                    # 列出所有 Profile
storybook profile create client-a         # 默认创建 isolated Profile
storybook profile switch client-a         # UUID 或显示名均可
storybook sync status                     # v0.2 明确显示 local_only、跨设备同步未启用
```

可用 `STORYBOOK_PROFILE=<UUID|NAME>` 为单个进程选择 Profile 而不修改 registry；`STORYBOOK_HOME=/private/path` 可显式收拢/隔离 registry、数据库、缓存和日志。旧仓库 `data/memory.db` 不会被删除或覆盖；安全迁移由独立的 migration 流程负责。

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
  `stories.embedding` 与 `story_vectors` 同步；分裂后父 story 向量从索引移除），以及旧 schema
  对 `global_id` / `profile_id` / `sync_state` 的幂等补齐。
- **profiles**：macOS/Linux/Windows 路径、XDG 覆盖、随机 UUID registry、local/isolated
  切换、跨 cwd 一致性、损坏 registry 拒绝覆盖和最小权限。
- **processor**：create / merge / update 三分支 + split 路径，mock `llm`/`embeddings` 返回固定值，
  验证分支选择与边建立（弱关联建边、共同召回提权、父子/兄弟边）。
- **search**：阈值过滤、关联激活、共同召回提权（每对每次仅 +0.1 一次）。
- **performance**：阶段时钟故障注入、最近窗口百分位、cache/fallback 比例、诊断隐私白名单，以及 warm/cold、并发 1/5 benchmark 编排（测试用小数据集与 mock embedding）。
- **prime**：query 构造（cwd / first_prompt）、主动注入门槛（高于检索）、token 预算裁剪、静默不注入（空库 / 低于门槛 / embedding 失败 / schema 缺失均不抛错）。
- **集成**：用 `generate_sample_sessions` 与 `test_logs/*.json` 跑通 collector → store → processor → search 全链路。
- **dreamd（做梦周期自动化）**：`fcntl.flock` 并发锁互斥与释放、`run_dream_cycle_once` 采集+加工/跳过/空、监听循环首帧追补与变化触发、定时守护、信号退出、`logs/dream.log` 幂等写入。全 mock，不依赖 Ollama。
- **MCP server**：`tests/test_mcp_server.py` 覆盖四个工具（`recall`/`get_story`/`stats`/`prime_context`）的核心逻辑与 FastMCP 装配/端到端调用。

## 📐 检索质量评测（benchmark + recall@k + 合并正确率 + 分裂质量）

PRD 要求「重复 bug 检索准确率≥70%」但原本无任何评测手段。`storybook eval` 建立可重复的检索质量基线，
作为调参与算法改进的度量依据。**需要 Ollama 运行**（embedding 走真实 `qwen3-embedding`），评测在隔离临时库中进行，不污染用户 Profile 数据库。

```bash
storybook eval all                              # 跑全部三轮评测（默认）
storybook eval retrieval                        # 仅检索评测
storybook eval all --report data/eval_reports/baseline.json   # 落盘 JSON 报告，便于阈值调整前后对比
python scripts/eval.py retrieval                # 等价独立脚本（未做 editable 安装时用）
```

三轮评测：

1. **retrieval** — 用 `data/retrieval_benchmark.json`（24 topic × 3 查询变体 = 72 对，含精确术语 / 同义改写 / 跨语言 EN↔ZH + 负例），
   真实 embedding 索引人工标注 story 语料，度量 recall@1/3/5、precision@k、MRR、负例特异性，并判定是否达 recall@3≥70%。
   同时输出 `SIM_THRESHOLD_SEARCH` 阈值敏感性曲线。
2. **processing** — 真实 embedding + 确定性 LLM 桩（人工关键词/摘要），度量 merge/update 分支是否选对
   （duplicate 应并入、distinct 应新建），输出 `SIM_THRESHOLD_HIGH` 阈值敏感性曲线。隔离度量 0.85/0.92 阈值，排除 LLM 关键词质量波动。
3. **split** — 真实 embedding + 确定性 LLM 桩，度量分裂路径结构正确性（父向量移除、父子边 1.0、子向量入索引、子 story 可检索）。

当前基线（2026-07-19，`data/eval_reports/baseline-2026-07-19.json`）：recall@3 = 100% ✅ 达标；
合并正确率 85.7%（`dup-docker-dns` sim 0.83 落在 0.85 阈值下方被误判为 create，阈值敏感性显示 0.82 可达 100%）；
分裂结构正确率 100%。`tests/test_eval.py` 用确定性 mock 覆盖评测逻辑本身，无需 Ollama。

## 📈 查询性能基线与本地诊断

日常查询会自动记录无内容诊断。查看最近 100 次查询：

```bash
storybook status --performance
storybook status --performance --json
```

完整性能基准复用 `data/retrieval_benchmark.json` 的人工 ground truth，并在隔离临时库中构造固定 seed 的 10k Story 数据集。默认跑 50 条固定查询、每条重复 20 次、并发 1 和 5，报告机器/模型状态/规模/重复次数、所有阶段的 p50/p95/p99，以及按 exact/synonym/cross_lang 分组的 recall@1/3/5 和 MRR。基准不会污染用户数据库，也不会把原始 query、Story 内容、绝对路径或仓库 URL 写入报告。

```bash
# warm：先预热模型，再跑 10k × 50 × 20 × concurrency(1,5)
storybook benchmark --model-state warm --report data/perf-warm.json

# cold：每个并发批次前用 Ollama keep_alive=0 卸载 embedding 模型
storybook benchmark --model-state cold --report data/perf-cold.json

# 快速 smoke（报告会如实记录非标准规模）
storybook benchmark --stories 100 --queries 6 --repeats 2 --concurrency 1
```

连续运行时应比较报告中的 machine、embedding model/dim、model_state、dataset seed/size、repeats 与 concurrency；这些字段不同足以解释大多数基线漂移。报告同时按 `cache` / `vector` / `lexical_fallback` lane 给出 p50/p95/p99，分别核验 cache hit ≤80ms、warm ≤1s、cold ≤5s。cold 场景每批先卸载模型并清空进程内缓存，避免把 cache hit 误算为冷启动。

查询快路径不调用生成式 LLM。MCP 启动时 best-effort 预热 embedding，后续每次请求用 `keep_alive` 续期；warm 2s、cold 5s 到达硬超时后立即尝试 FTS5 + 参数化关键词 fallback，fallback 自身最多 500ms。响应中的 `result_state` 明确区分：

- `results` / `no_match`：正常向量或缓存路径；`no_match` 才表示已完成正常检索但没有相关记忆。
- `degraded_results` / `degraded_empty` / `degraded_unavailable`：降级命中、降级空结果、降级自身不可用；这些状态不应被解释为已确认“没有相关记忆”。

## 🚀 使用

```bash
storybook init                       # 初始化数据库 schema + vec0 虚表（其它命令也会自动初始化）
storybook setup [--yes|--dry-run|--json]  # 用户级存储 + 三类 Agent 接入 + smoke test
storybook uninstall [--purge-data]   # 恢复受管配置；默认保留记忆
storybook profile show|list          # 查看用户级 Profile 与数据目录
storybook profile create NAME        # 创建 isolated Profile（可加 --switch）
storybook profile switch ID_OR_NAME  # 切换当前 Profile
storybook sync status                # v0.2 显示 local_only
storybook import-data                # 默认：从 ~/.claude/projects 采集 Claude Code 会话（增量、按 sessionId 去重）
storybook import-data --claude       # 同上（显式写法）
storybook import-data --sample [--n 100]   # 生成并导入模拟会话（无需真实会话即可体验）
storybook import-data --cursor       # 扫描 Cursor 的 workspaceStorage（备用数据源）
storybook import-data <file|dir>     # 导入 JSON（list / {sessions:[...]} / {messages:[...]} 聊天日志）

storybook process [--session ID]     # 做梦周期：处理所有 pending 会话（或指定一条）
storybook process --watch [--interval N]  # 监听模式：轮询 ~/.claude/projects，有新会话自动采集+加工（长驻）
storybook dream --once                # 单次完整做梦周期（采集+加工）后退出；launchd/cron 入口
storybook dream [--interval N]        # 定时守护进程（非 macOS 兜底，每 N 秒一轮，默认 4h）
storybook search "<query>" [--top 3] # 向量检索 + 关联 Story 激活
storybook status --performance       # 最近 100 次查询 p50/p95、cache/fallback 比例
storybook benchmark --model-state warm|cold  # 隔离的 10k Story 性能+质量基准
storybook stats                      # 系统统计
storybook list [--limit 20]          # 列出所有 Story
storybook show <story_id>            # 查看 Story 详情（含关联记忆）
storybook prime [--cwd PATH]         # 会话启动主动注入（晨间简报），供 SessionStart hook 调用
storybook mcp                        # 启动 MCP server（stdio，供 Claude Code 等 agent 运行时召回）
```

> 命令是 **`import-data`** 而非 `import`（click 把 `import_data` 函数自动连字符化）。无参数/无 flag 时默认走 `--claude`。`--claude` / `--sample` / `--cursor` / `<path>` 四种来源互斥。

### 快速体验（无真实会话）

```bash
storybook import-data --sample --n 50   # 造 50 条模拟会话
storybook process                       # 跑做梦周期
storybook search "如何调试数据库连接"     # 搜一下
storybook stats                         # 看看沉淀了多少 Story
```

## 🌙 做梦周期自动化

「做梦」无需手动触发。三种自动化入口（均复用同一把文件锁，互不重叠；运行日志落当前 Profile 日志目录的 `dream.log`）：

| 入口 | 用途 | 平台 |
|------|------|------|
| `storybook process --watch` | 反应式监听：轮询 `~/.claude/projects`，有新会话自动采集 + 加工（长驻，Ctrl-C 退出） | 全平台 |
| `storybook dream --once` | 单次完整周期（采集 + 加工）后退出——**定时调度器的入口** | 全平台 |
| `storybook dream` | 定时守护进程，每 `DREAM_INTERVAL` 秒一轮（Ctrl-C / SIGTERM 退出） | 非 macOS 兜底 |

### macOS：launchd 定时任务

`scripts/` 下提供 plist 模板与一键安装脚本。安装脚本会把模板里的占位符替换为当前 venv 和 Profile 日志目录，写入 `~/Library/LaunchAgents/com.storybook.dream.plist` 并加载；plist 不再把仓库设为工作目录。

```bash
# 安装：每 4 小时（默认）自动跑一次 dream --once
./scripts/install_launchd.sh
# 每 1 小时
./scripts/install_launchd.sh --interval 3600
# 卸载
./scripts/install_launchd.sh --uninstall

# 常用调试命令
launchctl start com.storybook.dream            # 立即触发一次
launchctl print gui/$(id -u)/com.storybook.dream  # 查看状态
storybook profile show                           # 先查看当前 Profile 日志目录
```

plist 触发的是 `<venv>/bin/python -m storybook.cli dream --once`，`StartInterval` 可配置（默认 14400s = 4h），`RunAtLoad=true`（登录时先追补一次离线期间的新会话）。launchd 无 shell 环境，故 `storybook` 必须装在 venv 里、`.env` 由 `config.py` 自动加载——无需手动 `export`。

### Linux / 其它平台：守护进程

非 macOS 用 `storybook dream` 守护进程替代 launchd，由 systemd / nohup 托管：

```bash
# 直接前台 / nohup 后台跑（结构化日志另写当前 Profile 日志目录）
nohup .venv/bin/python -m storybook.cli dream >/dev/null 2>&1 &

# 或用 systemd user 服务（模板：scripts/com.storybook.dream.service）
sed -e "s|__PYTHON_BIN__|$PWD/.venv/bin/python|" \
    scripts/com.storybook.dream.service > ~/.config/systemd/user/storybook-dream.service
systemctl --user daemon-reload
systemctl --user enable --now storybook-dream.service
journalctl --user -u storybook-dream.service -f
```

### 并发保护

所有做梦周期入口（手动 `process` / `--watch` / `dream --once` / `dream` 守护）共用当前 Profile 数据库目录下的 `dream.lock` 文件锁（`fcntl.flock` 非阻塞）。不同 Profile 互不阻塞；同一 Profile 已有周期在跑时，新触发立即跳过、不重复执行。进程崩溃时 OS 自动释放锁，无 stale-pid 问题。

## 🔌 MCP 接入（供 agent 运行时召回）

> 项目北极星是 **agent 跨 session 经验复用**。仅靠人工 `storybook search` 无法让 agent 在运行时自动召回；MCP server 把检索暴露为工具后，Claude Code 等 MCP-aware agent 可在新任务中主动查询记忆库。

MCP server 是一个独立 stdio 进程，**复用** `search.search` / `store.get_story` / `store.get_stats`，不重复实现检索逻辑。

### 安装

MCP SDK 已包含在基础安装中；旧版 `.[mcp]` 安装命令仍兼容：

```bash
VIRTUAL_ENV=$(pwd)/.venv uv pip install -e ".[mcp]"
```

### 启动方式（二选一，均为独立进程，不依赖 CLI 运行态）

```bash
storybook mcp                   # 经 CLI 入口（推荐）
python -m storybook.mcp_server  # 直接跑模块（editable 安装后即可）
```

### 在 Claude Code 中启用

最简方式（命令行注册）：

```bash
claude mcp add storybook -- /绝对路径/storybook/.venv/bin/storybook mcp
```

或手动写入配置（用户级 `~/.claude.json`，或项目级 `.mcp.json`）：

```json
{
  "mcpServers": {
    "storybook": {
      "command": "/绝对路径/storybook/.venv/bin/storybook",
      "args": ["mcp"]
    }
  }
}
```

> ⚠️ 用 **绝对路径** 指向 venv 里的 `storybook`：Claude Code 启动 MCP 进程时不一定继承 shell 的 PATH，相对命令可能找不到。未做 editable 安装时可用 `python -m storybook.mcp_server`（`command` 指向 `.venv/bin/python`，`args` 为 `["-m", "storybook.mcp_server"]`）。

### 暴露的工具

| 工具 | 参数 | 说明 |
|------|------|------|
| `recall` | `query`（必填）, `top_k?`（默认 3） | 快路径召回，返回 `{query,count,matches,request_id,mode,result_state,degraded,degraded_reason,fallback_status,cache_hit,index_version,latency_ms}`。只有 `result_state=no_match` 表示正常检索无匹配；降级空结果不会伪装成“无记忆” |
| `get_story` | `story_id`（必填） | 查看单条记忆详情（含关联记忆），剥离 1024 维 embedding |
| `stats` | - | 记忆库概况（会话/Story/关联边数量） |
| `prime_context` | `cwd?`, `first_prompt?`, `top_k?`（默认 5） | 会话启动主动注入（晨间简报）：基于 cwd + 首条提问召回并生成 ≤2k token 的精简摘要，返回 `{cwd,query,count,injected,briefing,matches,truncated,note}`。`injected=false` 时 `briefing` 为空（无相关记忆 / 相关度不足 / Ollama 不可用），**不报错、静默不注入**。详见下文「🌅 会话启动注入」 |

### 说明

- server、CLI、Claude/Cursor collector 和 Codex 等 MCP 客户端都经 Profile registry 共享同一数据目录（`.env` 自动加载、`OLLAMA_HOST` 等环境变量同样生效）。
- `recall` 复用 CLI `search` 的全部语义；命中记忆的 `access_count` 自增、共同召回边权提权会进入后台反馈队列，不阻塞查询响应。
- `recall` 优先使用本地 Ollama 生成查询向量；Ollama 不可用或超时时返回显式 degraded 状态和 FTS/关键词可用结果，不抛出伪装成“无匹配”的环境错误。`get_story` / `stats` 不依赖 Ollama。
- `prime_context` 同样复用 `search` 的召回与副作用（每次晨间简报即一次"回忆"，会自增 `access_count` / 提权边）；但它**静默不抛错**--Ollama 不可用时返回 `injected=false` + `note`（非异常），因为晨间简报须非侵入。详见下文。
- server 启动时自动 `init_db`：全新环境下 `recall` 返回空、`stats` 返回 0、`get_story` 报不存在、`prime_context` 返回 `injected=false`。

## 🌅 会话启动注入（晨间简报 / 上下文预热）

> 仅暴露 `recall` 等 MCP 工具仍需 agent **主动**调用。更进一步：新会话开始时，基于 cwd / 首条提问**主动 surface** 最相关 story 注入上下文，实现"下意识回忆"--更贴近项目初衷（人脑处理事项时自动想起相关经历）。

`prime_context` 与 `storybook prime` 共享 `src/storybook/prime.py` 的召回 + 预算控制逻辑，**复用 `search.search`**，不重复实现检索。两条触发路径：

| 路径 | 触发时机 | 查询信号 | 接入方式 |
|------|----------|----------|----------|
| **SessionStart hook** | 会话启动（尚无首条提问） | 仅 cwd（项目目录派生项目名） | `storybook prime` CLI，stdout 被注入为额外上下文 |
| **MCP `prime_context`** | agent 读到首条提问后主动调用 | cwd + 首条提问（提问为主信号） | agent 调用工具，拿回 `briefing` 自行呈现 |

### 行为保证（验收标准）

1. **有匹配时自动注入**：召回 ≥ `PRIME_MIN_SIMILARITY`（默认 0.60，高于检索 0.50）的记忆，渲染为精简简报。
2. **无相关记忆时静默不注入、不报错**：召回为空 / 全低于门槛 / Ollama 不可用 / DB 未初始化 -> `injected=false`、`briefing=""`、hook 输出空 stdout（什么都不注入）。
3. **token 预算内、有针对性**：简报 ≤ `PRIME_TOKEN_BUDGET`（默认 2000）token，超额时按相似度从低到高丢弃候选并对单条摘要裁剪（`truncated=true`）；每条摘要 ≤ `PRIME_CONTENT_EXCERPT_CHARS`（默认 140）字符。
4. **hook / 接入说明**：见下文。

### 方式一：Claude Code `SessionStart` hook（推荐，纯自动）

`storybook prime` 默认把简报纯文本写到 stdout（被 Claude Code 作为额外上下文注入）；无匹配时 stdout 为空（不注入）。hook 始终 exit 0、非阻塞，任何环境异常都静默退化（不向上下文注入错误）。

在项目级 `.claude/settings.json` 或用户级 `~/.claude/settings.json` 配置：

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "/绝对路径/storybook/.venv/bin/storybook prime --cwd \"$CLAUDE_PROJECT_DIR\""
          }
        ]
      }
    ]
  }
}
```

> ⚠️ 用**绝对路径**指向 venv 里的 `storybook`（Claude Code 启动 hook 进程时不一定继承 shell 的 PATH）。`$CLAUDE_PROJECT_DIR` 由 Claude Code 注入，即当前项目目录。未做 editable 安装时可用 `python -m storybook.prime` 形式（`command` 指向 `.venv/bin/python`，`args` 为 `["-m", "storybook.prime", "--cwd", "$CLAUDE_PROJECT_DIR"]`）。
>
> 若你的 Claude Code 版本支持 `hookSpecificOutput` 结构化注入，可改用 `--format hook`，仅 `additionalContext` 字段被注入、其余 stdout 被忽略：

```json
"command": "/绝对路径/storybook/.venv/bin/storybook prime --cwd \"$CLAUDE_PROJECT_DIR\" --format hook"
```

调试时可用 `--format json` 查看完整结构化结果（`query` / `count` / `matches` / `truncated` / `note`）：

```bash
storybook prime --cwd "$PWD" --prompt "你的首条提问" --format json
```

### 方式二：MCP `prime_context` 工具（agent 主动调用）

已启用 MCP server（见上文「🔌 MCP 接入」）后，agent 可在读到用户首条提问后调用 `prime_context`，传入自身 cwd 与首条提问，拿回 `briefing` 呈现给用户。适合"提问已到、但想强化主动回忆"的场景，或非 Claude Code 的 MCP-aware agent。

```python
prime_context(cwd="/path/to/project", first_prompt="用户的首条提问", top_k=5)
# -> {"injected": true, "briefing": "📖 Storybook 晨间简报：...", "count": 2, ...}
```

### 简报样例

```
📖 Storybook 晨间简报：以下过往记忆可能与当前任务相关（按相似度排序）

• [82%] 排查用户下单失败
  问题：下单接口超时 步骤：1.查日志定位回调超时 2.复现 结果：调整超时配置并加重试
• [75%] 订单链路需求开发
  问题：支付回调链路改造 步骤：... 结果：...

（来自本地 Storybook 记忆库；调用 recall / get_story 可查看详情。不相关可忽略。）
```


## ⚙️ 配置

所有路径、模型名、阈值都集中在 `src/storybook/config.py`。环境变量样例见 `.env.example`。`config.py` 启动时自动加载项目根 `.env`（无则跳过）；命令行 / 已存在的环境变量优先级高于 `.env`，故 launchd 等无 shell 的场景也无需手动 `export`。

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `STORYBOOK_PROFILE` | registry 当前项 | 仅当前进程选择 Profile（UUID 或显示名），不改 registry |
| `STORYBOOK_HOME` | 平台用户目录 | 显式收拢/隔离 registry、数据、缓存与日志 |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama 服务地址 |
| `STORYBOOK_LLM_MODEL` | `qwythos-hermes:latest` | 做梦加工用的 LLM |
| `STORYBOOK_EMBED_MODEL` | `qwen3-embedding:0.6b` | embedding 模型（必须 1024 维） |
| `STORYBOOK_EMBED_KEEP_ALIVE` | `10m` | 每次 embedding 请求续期的 Ollama 模型驻留时间 |
| `STORYBOOK_QUERY_WARM_TIMEOUT_SECONDS` | `2` | warm embedding 硬超时，超时即降级 |
| `STORYBOOK_QUERY_COLD_TIMEOUT_SECONDS` | `5` | cold embedding 硬超时，超时即降级 |
| `STORYBOOK_QUERY_FALLBACK_TIMEOUT_SECONDS` | `0.5` | FTS/关键词 fallback 独立硬超时 |
| `STORYBOOK_LLM_THINK` | `0` | Qwen3 思考模式：`0`=关（提取类任务约 9× 加速），`1`=开（检索准确率不足时再开） |
| `STORYBOOK_DREAM_INTERVAL` | `14400` | `dream` 守护进程 / launchd 定时间隔（秒），默认 4 小时 |
| `STORYBOOK_WATCH_POLL_INTERVAL` | `60` | `process --watch` 轮询 `~/.claude/projects` 的间隔（秒） |

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
| `PRIME_MIN_SIMILARITY` | 0.60 | 晨间简报主动注入最低相似度（高于检索 0.50，避免噪声） |
| `PRIME_TOP_K` | 5 | 晨间简报最多考虑的候选数（再按 token 预算裁剪） |
| `PRIME_TOKEN_BUDGET` | 2000 | 晨间简报 token 预算上限（≤2k，避免污染上下文） |
| `PRIME_CONTENT_EXCERPT_CHARS` | 140 | 晨间简报中每条 Story 摘要最大字符数 |

LLM 调用参数（temp 0.3、`num_ctx` 8192、120s 超时）硬编码在 `llm._chat` / `_generate` 中。

## 📁 项目结构

```
storybook/
├── src/storybook/
│   ├── cli.py          # CLI 命令入口
│   ├── config.py       # 路径 / 模型 / 阈值常量
│   ├── profiles.py     # 用户级 registry、平台目录、local/isolated Profile
│   ├── collector.py    # 会话采集（Claude Code / Cursor / JSON / 模拟）
│   ├── store.py        # SQLite + sqlite-vec 存储层
│   ├── processor.py    # 做梦周期（dream cycle）
│   ├── llm.py          # Ollama LLM 调用
│   ├── embeddings.py   # Ollama embedding 调用
│   ├── search.py       # 版本化缓存 + 向量/词法降级 + 关联激活
│   ├── query_cache.py  # index_version 隔离的向量/结果 LRU+TTL 缓存
│   ├── feedback.py     # access_count/边权异步反馈队列
│   ├── performance.py  # 隐私安全的查询诊断 JSONL + 最近窗口汇总
│   ├── perf_benchmark.py # 固定数据集 warm/cold 性能与质量基准
│   ├── dreamd.py       # 做梦周期自动化（锁 / 监听 / 定时守护 / 日志）
│   ├── prime.py        # 会话启动主动注入（晨间简报，复用 search）
│   └── mcp_server.py   # MCP server（recall / get_story / stats / prime_context）
├── data/               # benchmark/报告等仓库资源（不再存运行时主数据库）
├── scripts/            # launchd plist 模板 + install_launchd.sh + systemd 单元模板
├── docs/TECH_DESIGN.md # 原始设计文档
├── tests/              # pytest 测试套件（store/processor/search/dreamd + 集成，全 mock Ollama）
├── test_logs/          # 示例 JSON 数据
├── hermes_sessions.json
├── .env.example
└── pyproject.toml
```

## 📝 说明

- **完全离线**：所有 LLM / embedding 走本地 Ollama，不发送任何数据到云端。
- **测试套件**：`tests/` 下 pytest 用例覆盖 store/processor/search/prime/dreamd 核心路径，全 mock、不依赖 Ollama（见上文「🧪 测试」）。`test_logs/*.json` 与 `hermes_sessions.json` 是 `import-data` 的样例数据源。
- **MCP server**：`storybook mcp` 启动独立 stdio 进程，向 Claude Code 等 agent 暴露 `recall`/`get_story`/`stats`/`prime_context`（接入见上文「🔌 MCP 接入」）。
- **晨间简报**：`storybook prime`（SessionStart hook）或 `prime_context` MCP 工具在会话启动时主动召回相关记忆注入上下文，复用 `search` 召回；相关度不足 / 无匹配 / Ollama 不可用时静默不注入（见上文「🌅 会话启动注入」）。
- `docs/TECH_DESIGN.md` 是最初的设计文档，其中的目录布局与命令示例早于当前实现（命令为 `import-data`；`tests/`、`scripts/` 与 launchd plist 已在后续迭代落地，见上文「🧪 测试」与「🌙 做梦周期自动化」）。
- LLM 输出解析是宽松的：关键词 JSON 在 `[`/`]` 间切片，摘要按 `TITLE:`/`CONTENT:` 标记切分，模型不遵循格式时有字符串切分兜底。

## License

MIT
