# 📖 Storybook

> 离线 Coding 记忆系统 —— 把每一次 AI 编程会话在"梦境"里整理成一条 Story，再连成一张可被联想唤醒的记忆网。
>
> A local-first coding memory system that consolidates AI-coding sessions into structured *Stories* via a "dream cycle" and links them into a weighted association graph. Generative work uses DeepSeek; embeddings remain local in Ollama.

## 这是什么 / What it is

Storybook 采集 AI 编程会话日志（Claude Code 会话、Cursor 日志、JSON 文件或内置模拟器），对每条会话跑一遍 **做梦周期（dream cycle）**：按“独立可复用结论 + 环境适用性”形成一个或多个 Story v2。每条 Story 保存 `title + abstract + structured detail + sources`；detail 与证据不硬截断，只有用于检索/展示的 abstract 有预算，并与已有记忆按相似度**合并 / 更新 / 新建**。

检索时，Fast 常态并行使用向量与 FTS/关键词排名，经加权 RRF、环境软信号和本地有界 reranker 融合，再以直接命中为 seed 在 hop、path、fan-out、墙钟时间和 token 预算内扩散 Memory Graph。Auto 仅在 zero/low-confidence、复合、跨语言或强环境歧义时进入独立预算的 Query Transformation/HyDE 第二阶段；Deep 必须由调用方显式选择。每条结果返回来源路径与分数组成；共同召回反馈会强化并衰减独立 `co_recall` 边。

系统采用**混合 provider**：生成式 LLM 通过 DeepSeek Anthropic-compatible Messages API，1024 维 embedding 与索引仍完全留在本机 Ollama。Fast 查询不调用生成式 LLM；只有做梦加工以及门控后的 Auto/Deep transformation 会把对应文本发送给 DeepSeek。

## ✨ 特性

- 🧠 **语义边界记忆整理**：长而不可拆的经历保持完整；短会话中的多个独立结论拆成多条 Story，并共享来源 Session
- 🔗 **可解释 Memory Graph**：`semantic` / `temporal` / `causal` / `same_environment` / `parent_child` / `co_recall` / `supersedes` 多类型边，有明确方向、provenance、版本与软删除规则
- 📐 **可演进双索引**：当前 `story_vectors` 持续服务，模型/版本切换先增量写 shadow，完整后原子切换；失败可续跑
- 🔍 **自适应 Hybrid Search**：Fast 无生成式调用，融合 vector + FTS/关键词 + environment + Graph；Auto 按可解释门控启用 rewrite/multi-query/HyDE；Deep 使用显式高预算；任一组件失败均保留可用 fallback
- 🧵 **读写解耦**：向量召回与关联读取完成后立即返回，`access_count`/共同召回边权反馈由有界后台队列单事务写入
- 📈 **性能可观察**：每次查询分段记录 cache/embed/vector/lexical/fusion/transform/fallback/graph/rerank/serialize/total，`status --performance` 汇总最近 100 次 p50/p95；固定 10k Story benchmark 与离线策略消融同时守护质量/时延
- 🔌 **多数据源**：Claude Code 会话（主）、Cursor、JSON 文件/目录、内置模拟器
- 🏠 **本地优先**：Profile、原始证据、数据库与 embedding 留在本机；云端生成调用可明确门控并快速降级
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

对每条 pending 会话：LLM 按独立结论形成 Story v2 → 默认对 `title + abstract + applicability` 做 embedding → 每个候选分别检索 top-K 已有 Story → 按最佳相似度分支：

| 分支 | 触发条件 | 动作 |
|------|----------|------|
| **create** | best sim < 0.85 | 完整保存 structured detail/source；与 0.75–0.85 的弱匹配 Story 建边，`weight = sim` |
| **merge** | 0.85 ≤ sim < 0.92 | 合并新旧结构化证据；只有存在多个独立结论/适用条件才分裂，不以字符数触发。父行和 revision 链保留用于溯源 |
| **update** | sim ≥ 0.92 | 仅合并关键词、重新 embedding、强化已有边权重（+0.1，上限 1.0） |

### 检索（`search.search`）

Fast：query normalization → vector + FTS/关键词 → 加权 RRF → 环境软加权 → 有界 Graph RAG → 本地 top-N rerank。Fast 不调用生成式 LLM；`graph_enabled=false` 可关闭图扩散。`--scope project` 使用 ContextEnvelope 中隐私安全的 repo/workspace 身份做硬过滤，只返回当前项目来源记忆；`profile` 保持用户级全库召回。

Auto 先完整执行 Fast，再依据 `zero_results`、`low_confidence`、`ambiguous_ranking`、`long_compound_query`、`cross_language`、`environment_ambiguity` 等稳定原因决定是否调用一次 DeepSeek LLM，生成 rewrite、multi-query 或 HyDE 辅助表示。第二阶段有独立 deadline，超时后原 Fast 结果立即作为 fallback 返回。Deep 显式启用三种 transformation、更高 Graph 预算及 5s 总预算。

本地 reranker 只处理有界 top-N，具有独立超时、连续失败熔断与冷却恢复；故障时返回 fusion/graph 排名并标明 `reranker_timeout` / `reranker_unavailable` / `reranker_circuit_open`，不会伪装成“无记忆”。

查询响应保留兼容字段 `mode=cache|vector|lexical_fallback`，并新增 `retrieval_mode=fast|auto|deep`、`transform_used`、`query_plan`、`transform_trace`、`rerank_trace`、`degraded_reasons`。每条 match 返回 `source_paths` 与 `score_components`（vector/lexical/RRF/graph/environment/rerank）。同一份阶段数据会写入本地 `logs/query_performance.jsonl`，但落盘接口只接受固定白名单字段：不保存原始 query、Story 内容、绝对路径、hostname 或仓库 URL。文件权限为 `0600`，超过大小上限后只保留最近记录。

### 关联图

`edges` 以 `UNIQUE(source_id, target_id, edge_type)` 允许同一 Story 对保存多种关系。`temporal`（旧→新）、`causal`（因→果）、`parent_child`（父→子）、`supersedes`（新→旧）是有向边；其余无向。边包含 `provenance_json/version/observations/updated_at/deleted_at`，删除为可审计软删除。Graph RAG 默认从旧 Story 反向跟随 `supersedes` 到新 Story 并抑制旧版；对环、hub、重复路径和噪声共现链做显式抑制。

### 存储层（`store.py`）

每个用户 Profile 一份 `profiles/{随机 UUID}/db/memory.db`（SQLite + sqlite-vec 扩展），不再存于仓库。新 Profile、Session、Story、edge 使用可按时间排序的 UUIDv7 全局 ID。Story v2 增加 `abstract/detail_json/sources_json`、`embedding_model/embedding_version/embedding_content_hash`；`story_revisions` 记录无损本地快照，`memory_events` 以 `event_id/entity_id/base_version/version/device_id/operation/created_at` 记录 create/update/merge/split/delete 的可移植审计链。事件明文 payload 只含固定元数据、关系 UUID 与修订 SHA-256，不复制正文、原始外部 session ID、路径或证据文本，并预留加密 payload 字段。**当前 embedding** 同步存于 `stories.embedding` 与 serving `story_vectors`；`story_embedding_backfill` 是模型切换 shadow，完整后在单事务内切换，部分失败不会影响在线 recall。

`storybook forget` 用持久化浮点热度按半衰期衰减，再将频率无关的整数投影写入 `access_count`，并以最近访问/更新时间、访问计数和最大关联边权共同筛选低价值 Story；归档默认仅预览，`--apply` 后只归档并移出向量/词法/图检索，重复初始化或索引修复也不会重新发布归档向量；高频或强关联记忆受保护，原 Story、embedding 审计数据与 provenance 仍保留。生成式 LLM 与 embedding 的成功结果按 provider/model/schema/输入哈希持久缓存于 Profile 私有 cache；批处理并行执行无数据库写入的 LLM/embedding 准备阶段，再顺序执行 SQLite 合并与写入。

删除 Story 时不物理删行：同一事务清除 serving 向量、追加 delete event 并写入不可变 `memory_tombstones`。查询默认排除 tombstone；本地事件重放采用 delete-wins，即使旧 create/update 事件晚到也不会复活对象。v0.2 的 `storybook sync status` 是纯本地状态查询，不登录、不联网，也没有上传/下载入口。

从 v0.1 升级时，无法审计模型、输入表示与 hash 的旧向量会保留为 `story-v1-unversioned/legacy` 服务窗口，Story 标记为 `stale` 且不冒充 v2 元数据；必须完成可续跑的 shadow backfill 后，才会原子切换为 v2 active 状态。重复初始化不会覆盖 `stale`、`failed` 或 `archived` 等真实状态。

```bash
# 每次最多重建 100 条；重复运行自动跳过 content_hash 未变化的 ready 项
storybook embedding-backfill --model qwen3-embedding:0.6b \
  --version story-v2-default-v2 --batch-size 100
```

## 🔧 环境要求

- **Python 3.11+**（推荐用 [uv](https://github.com/astral-sh/uv) 建 venv）
- **DeepSeek API 凭据**：优先 `ANTHROPIC_AUTH_TOKEN`，兼容 `DEEPSEEK_KEY`；默认读取 `~/.chrc/dpsk.sh`
- **Ollama** 运行于 `http://localhost:11434`（可用 `OLLAMA_HOST` 覆盖），仅需拉取 embedding 模型 `qwen3-embedding:0.6b`，**1024 维**（可用 `STORYBOOK_EMBED_MODEL` 覆盖）；需与 `config.EMBED_DIM` 一致

### 模型 Provider setup

`book` 与 `storybook` 是等价命令。新安装会在 active Profile 内写入版本化的
`model-config.json`，generation 与 embedding 可分别诊断；文件只保存 credential
环境变量名，绝不保存密钥。

```bash
# 本地 Ollama：探测服务，按需拉取 generation/embedding 模型
book setup --provider ollama --llm-model qwen3:8b \
  --embedding-model qwen3-embedding:0.6b

# OpenAI-compatible API：明确使用 /v1/chat/completions 与 /v1/embeddings
export STORYBOOK_API_KEY='...'
book setup --provider api --base-url https://gateway.example/v1-root \
  --llm-model chat-model --embedding-model embedding-model \
  --api-key-env STORYBOOK_API_KEY
```

外部 embedding 必须返回 1024 维，否则 setup 以
`SB_MODEL_EMBED_DIM_MISMATCH` 失败。base URL 中的 userinfo、query 和 fragment
会被拒绝，doctor/status/JSON 输出不会包含 credential 值或 Authorization header。
未生成 Profile 配置的旧安装继续按“Profile 配置 > 旧环境变量 > 默认值”的优先级
只读解析 `ANTHROPIC_*`、`DEEPSEEK_KEY`、`OLLAMA_HOST` 与
`STORYBOOK_EMBED_MODEL`，无需迁移现有数据。

已有 Profile 的 active 向量索引会持久化 provider、base URL、model 与 version
身份。setup 若检测到目标 embedding space 不兼容，会在任何写入和网络探测前以
`SB_MODEL_INDEX_INCOMPATIBLE` 失败；可保持原配置，或先运行
`storybook profile create provider-migration --switch` 创建隔离 Profile 后重新
setup。不同 provider/base URL 即使模型同名，也不会共享 inference/query cache。
- 依赖：`click`、`requests`、`numpy`、`sqlite-vec`、`mcp`（Agent 接入所需）

```bash
# 拉模型
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
Claude Code、Cursor、Codex，只检查或下载本地 Ollama embedding 模型，最后执行 schema、embedding、
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

数据库/索引位于 Profile 数据根，缓存与日志走各平台的 cache/state 目录；目录权限在 POSIX 上为 `0700`，registry 与数据库为 `0600`。registry 只持久化随机 UUID、显示名、模式、同步状态和 Profile 内的相对数据库世代指针，不把用户名、hostname 或绝对路径当作主键。

```bash
storybook profile show                    # 当前 Profile、数据目录与 local-only 状态
storybook profile list                    # 列出所有 Profile
storybook profile create client-a         # 默认创建 isolated Profile
storybook profile switch client-a         # UUID 或显示名均可
storybook sync status                     # v0.2 明确显示 local_only、跨设备同步未启用
```

可用 `STORYBOOK_PROFILE=<UUID|NAME>` 为单个进程选择 Profile 而不修改 registry；`STORYBOOK_HOME=/private/path` 可显式收拢/隔离 registry、数据库、缓存和日志。旧仓库 `data/memory.db` 不会被删除或覆盖；安全迁移由独立的 migration 流程负责。

### v1 → v2 安全迁移与回滚

迁移的发现、dry-run、备份与转换阶段始终只读打开旧项目库，先做一致性备份，再在隔离
的数据库世代内转换和校验。
Session、Story、edge 及关系必须逐项等量；已有 embedding 会进入 sqlite-vec serving
索引并执行真实查询 smoke test。v1 `content` 原样保存在 `legacy_raw`，Story v2 detail/source
同步生成，`abstract_status=pending` 留给后续异步补全。所有检查通过后，才以一次原子
registry CAS 切换 `database_ref`。切换前会在 SQLite 单写者边界内再次核对源库逻辑
hash；backup 后的提交会使迁移明确失败，活动写事务也会阻止切换。对于可写的待退役
世代，同一事务会预置持久拒写触发器；所有退役 fence 与目标 unfence 先持久化，最后才
执行 registry CAS，因此指针落盘后不再有可失败的 SQLite commit。任一 prepare commit
失败或结果不确定时，会在旧指针仍有效时补偿恢复切换前状态。这些旧连接会收到
`SB_MIGRATION_GENERATION_FENCED`，不能在成功切换后向旧世代落入独有数据。只读源库
使用稳定读事务做同一 CAS。rollback 对活动 v2 使用相同协议，副本另存基线 hash；若
回滚后产生新写入，重复迁移会拒绝复用陈旧 v2 世代。因此失败不会改变旧库权威，也不会
静默覆盖回滚后的权威数据。

```bash
storybook migration discover --json
storybook migration run ./data/memory.db --dry-run --json  # 严格零写入
storybook migration run ./data/memory.db --json            # 备份、转换、校验、切换
storybook migration status --json
storybook migration rollback <migration_id> --json         # 原子切回独立 v1 副本
storybook migration delete-backup <migration_id> --yes      # 用户显式永久删除
```

`migration_id` 由目标 Profile 与源库逻辑 SHA-256 确定；重复运行同一源库直接复用已验证
世代，不插入重复对象。当前 Profile 已有 Session/Story/edge 时迁移会拒绝覆盖，应先创建
一个新的空 Profile。原始记忆行不会删除或覆盖；成功切换只在可写旧库中增加世代拒写
触发器，失败切换会回滚该 DDL。受管 v1 只读备份至少保留 30 天，且不会自动提前删除。

## 🧪 测试

测试套件覆盖 `store` / `processor` / `search` 三个核心模块的关键路径与边界，
**完全不依赖真实 DeepSeek/Ollama**——所有 LLM / embedding 调用均被 mock 桩替换，本地一键可重复运行。

```bash
# 1. 安装测试依赖（与运行时依赖一并）
VIRTUAL_ENV=$(pwd)/.venv uv pip install -e ".[test]"

# 2. 一键运行全部测试
.venv/bin/pytest

# 带覆盖率报告（聚焦 store/processor/search）
.venv/bin/pytest --cov=storybook --cov-report=term-missing
```

测试不启动 Ollama、不访问 DeepSeek：HTTP 与 embedding 均使用 mock。
用例要点：

- **store**：Session/Story CRUD、`_edge_pair` 无向边归一、`search_by_vector` 的 `1 - dist²/2`
  相似度换算（与 numpy 暴力余弦交叉验证）、双写一致性（`add_story`/`update_story` 后
  `stories.embedding` 与 `story_vectors` 同步；分裂后父 story 向量从索引移除），以及旧 schema
  对 `global_id` / `profile_id` / `sync_state` 的幂等补齐。
- **profiles**：macOS/Linux/Windows 路径、XDG 覆盖、随机 UUID registry、local/isolated
  切换、跨 cwd 一致性、损坏 registry 拒绝覆盖和最小权限。
- **processor**：create / merge / update 三分支 + split 路径，mock `llm`/`embeddings` 返回固定值，
  验证分支选择与边建立（弱关联建边、共同召回提权、父子/兄弟边）。
- **Story v2**：千 token 原子 Story 无损保存、短会话双结论共享 Session、summary/detail 分层、revision 链，以及 embedding backfill 失败续跑与原子切换。
- **MemoryEvent**：UUIDv7 单调性、create/update/merge/split/delete 版本链、事件/tombstone 不可变、删除重放不复活、payload 隐私白名单，以及 `sync status` 零网络请求。
- **search/graph**：阈值过滤、关联激活、共同召回提权（每对每次仅 +0.1 一次）、单/多跳扩散、路径解释、环/hub/重复抑制、supersedes 替换和独立预算截断。
- **performance**：阶段时钟故障注入、最近窗口百分位、cache/fallback 比例、诊断隐私白名单，以及 warm/cold、并发 1/5 benchmark 编排（测试用小数据集与 mock embedding）。
- **prime**：query 构造（cwd / first_prompt）、主动注入门槛（高于检索）、token 预算裁剪、静默不注入（空库 / 低于门槛 / embedding 失败 / schema 缺失均不抛错）。
- **集成**：用 `generate_sample_sessions` 与 `test_logs/*.json` 跑通 collector → store → processor → search 全链路。
- **dreamd（做梦周期自动化）**：`fcntl.flock` 并发锁互斥与释放、`run_dream_cycle_once` 采集+加工/跳过/空、监听循环首帧追补与变化触发、定时守护、信号退出、`logs/dream.log` 幂等写入。全 mock，不依赖 Ollama。
- **MCP server**：`tests/test_mcp_server.py` 覆盖四个工具（`recall`/`get_story`/`stats`/`prime_context`）的核心逻辑与 FastMCP 装配/端到端调用。

## 📐 检索质量评测（benchmark + recall@k + 合并正确率 + 分裂质量）

PRD 要求「重复 bug 检索准确率≥70%」但原本无任何评测手段。`storybook eval` 建立可重复的检索质量基线，
作为调参与算法改进的度量依据。**需要 Ollama 运行**（embedding 走真实 `qwen3-embedding`），评测在隔离临时库中进行，不污染用户 Profile 数据库。

```bash
storybook eval all                              # 跑全部六轮评测（默认）
storybook eval retrieval                        # 仅检索评测
storybook eval exact-term                       # 精确代码 token：纯向量 vs Hybrid
storybook eval all --report data/eval_reports/baseline.json   # 落盘 JSON 报告，便于阈值调整前后对比
python scripts/eval.py retrieval                # 等价独立脚本（未做 editable 安装时用）
python scripts/generate_eval_transforms.py --variant ambiguous --timeout 30 \
  --output data/eval_reports/query-only-transforms.json
storybook eval strategy --transform-cache data/eval_reports/query-only-transforms.json
```

五轮评测：

1. **retrieval** — 用 `data/retrieval_benchmark.json`（24 topic × 3 查询变体 = 72 对，含精确术语 / 同义改写 / 跨语言 EN↔ZH + 负例），
   真实 embedding 索引人工标注 story 语料，度量 recall@1/3/5、precision@k、MRR、负例特异性，并判定是否达 recall@3≥70%。
   同时输出 `SIM_THRESHOLD_SEARCH` 阈值敏感性曲线。
2. **processing** — 真实 embedding + 确定性 LLM 桩（人工关键词/摘要），度量 merge/update 分支是否选对
   （duplicate 应并入、distinct 应新建），输出 `SIM_THRESHOLD_HIGH` 阈值敏感性曲线。隔离度量 0.85/0.92 阈值，排除 LLM 关键词质量波动。
3. **split** — 真实 embedding + 确定性 LLM 桩，度量分裂路径结构正确性（父向量移除、父子边 1.0、子向量入索引、子 story 可检索）。
4. **ablation** — 比较 legacy、默认 `title+abstract+applicability`、全文单向量、title/abstract/applicability 分字段多向量；按 exact/synonym/cross-tool/cross-language 报告 recall@3/MRR 与索引/查询时延。
5. **strategy** — 比较 direct-vector、hybrid、+graph、raw-query `+reranker` 控制组、+rewrite、+HyDE、+reranker；按 exact/synonym/cross-language/cross-tool/ambiguous 报告 recall@3/MRR/p95。Transformation provider 只能收到原始 query、策略名和 timeout，`topic_id` 只在排序完成后计分；预生成文件按 query SHA-256 索引并拒绝 `topic_id`/目标 Story 字段。报告分别标记 `live_generated`、`query_only_pre_generated`、`oracle_upper_bound`，oracle 永不参与默认选型，预生成质量证据也必须有独立在线时延证据。只有 hard-query 质量提升、overall 非劣、ground-truth 隔离和 fast/deep 在线时延门槛全部通过的策略才标记 `eligible_for_default=true`。
6. **exact-term** — 构造 compact semantic vector 相同、仅 detail/keywords 中精确错误码不同的隔离语料，直接对比 vector-only 与 FTS5/BM25 + vector RRF 的 recall@3，验证代码 token 不被语义向量吞没。

Story v2 固定报告（2026-08-02，`data/eval_reports/story-v2-ablation-2026-08-02.json`）：四种表示在 24 topic × 4 分组上 recall@3/MRR 均为 100%；默认表示相对 legacy 为 `0.00pp`，通过“下降不超过 2pp”门槛。默认单向量索引均值 84.4ms/story，明显低于全文 205.2ms 与多向量 223.7ms；多向量检索 p95 0.94ms，高于默认 0.34ms，因此选择默认表示。

Memory Graph 固定报告（2026-08-02，`data/eval_reports/memory-graph-2026-08-02.json`）覆盖七类边、单/多跳和负例：人工关联子集中 vector-only → Graph RAG 的 recall@5 为 `0% → 100%`，overall recall@3 为 `100% → 100%`；10k active Story、10,063 条边、200 次扩散的 `graph_ms p95=3.030ms`。可用以下命令复现（无需 Ollama）：

```bash
python -m storybook.graph_eval --stories 10000 --repeats 200 \
  --output data/eval_reports/memory-graph.json
```

自适应检索固定报告（2026-08-02，`data/eval_reports/adaptive-retrieval-2026-08-02.json`）使用 24 topic × 5 分组，并引用 query-only 生成物 `data/eval_reports/flo152-query-only-transforms-2026-08-02.json`。ambiguous 组模拟“只记得 outcome/指标、不记得问题名与工具”的经历召回：direct-vector hard recall@3 为 `87.5%`、overall 为 `90%`；不使用任何 transformation 的 `hybrid_graph_reranker` hard/overall recall@3 为 `97.92%/98.33%`（hard `+10.42pp`），MRR `0.8875 → 0.9841`，p95 `99.7ms`，因此是唯一默认候选。reranker 只批量读取有界 top-N 的本地 detail 作为内部打分文本，响应前移除，不默认展开原始证据。24 条 ambiguous query 的 query-only 预生成 rewrite/HyDE 可把 hard/overall recall@3 提升到 `100%/100%`，但本机生成耗时约 10–20s，报告将 `online_latency_evidence=false`，所以这些策略即使离线检索 p95 ≤525ms 也不能默认启用。报告明确记录 artifact SHA、模型、prompt 版本、24 次 cache hit/96 次未触发、ground-truth generation fields 为空；oracle upper bound 未使用且始终无默认资格。

10k Story 固定性能报告为 `data/perf_reports/flo152-warm.json`：并发 1/5 的 cache lane p95 `3.323/4.197ms`，vector/hybrid lane p95 `123.418/480.637ms`，recall@3/MRR 均为 `100%/1.0`。`data/perf_reports/flo152-deep-smoke.json` 是 6 查询×1、并发 1 的 Deep smoke：总 p95 `4.547s`、recall@3 `100%`、degraded ratio `100%`、lexical fallback ratio `16.67%`；本机 9B LLM 在短生成 deadline 内会超时，并与 embedding 模型发生内存换入竞争，响应因此明确降级到 Fast/lexical 结果。该 smoke 证明总预算与 fallback，不把它冒充为生成式质量或成功时延证据；rewrite/HyDE 目前保留为 gated/显式能力，不能在这台机器上默认启用。

当前基线（2026-07-19，`data/eval_reports/baseline-2026-07-19.json`）：recall@3 = 100% ✅ 达标；
合并正确率 85.7%（`dup-docker-dns` sim 0.83 落在 0.85 阈值下方被误判为 create，阈值敏感性显示 0.82 可达 100%）；
分裂结构正确率 100%。`tests/test_eval.py` 用确定性 mock 覆盖评测逻辑本身，无需 Ollama。

## 📈 查询性能基线与本地诊断

日常查询会自动记录无内容诊断。查看最近 100 次查询：

```bash
storybook status --performance
storybook status --performance --json
```

`status --json` 同时返回当前 `profile`、混合 provider `model`、setup 管理的
`adapter`、`sync` 与计数字段。组件全部可用时 `status=ready`；Profile、模型或
已配置 adapter 不可用时返回 `status=ready_degraded`，并通过稳定的
`degraded_reasons`（例如 `llm_credentials_missing`、`ollama_unavailable`、`model_missing:embedding`、
`adapter_unavailable:codex`）解释降级，不把可用的本地数据库误报为整体失败。

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
storybook migration discover         # 只读发现旧项目级 v1 数据库
storybook migration run PATH         # 安全备份、转换、校验并原子切换
storybook migration rollback ID      # 原子切回保留的 v1 副本
storybook sources list --json        # 检测本机来源及启用/版本/最近导入状态
storybook sources disable codex      # 关闭某来源（enable 重新启用）
storybook sources reset-checkpoint codex --yes  # 删除来源 checkpoint 后安全重扫
storybook import-data                # 兼容默认：从 Claude Code 增量采集
storybook import-data --claude       # 同上（显式写法）
storybook import-data --codex --json # Codex 结构化增量导入 summary
storybook import-data --source gemini # 可扩展来源入口
storybook import-data --sample [--n 100]   # 生成并导入模拟会话（无需真实会话即可体验）
storybook import-data --cursor       # 扫描 Cursor 的 workspaceStorage（备用数据源）
storybook import-data <file|dir>     # 导入 JSON（list / {sessions:[...]} / {messages:[...]} 聊天日志）

storybook process [--session ID]     # 做梦周期：处理所有 pending 会话（或指定一条）
storybook process --watch [--source codex] [--interval N]  # 监听全部启用来源或指定单源
storybook dream --once [--source codex] # 单次多来源采集+加工；launchd/cron 入口
storybook dream [--interval N]        # 定时守护进程（非 macOS 兜底，每 N 秒一轮，默认 4h）
storybook search "<query>" [--top 3] [--scope profile|project|strict] [--mode fast|auto|deep] [--json]
storybook forget [--half-life-days 30] [--min-age-days 90] [--apply]  # 默认仅预览
                                    # 默认 fast；auto 门控增强；deep 显式高预算
storybook status --performance       # 最近 100 次查询 p50/p95、cache/fallback 比例
storybook benchmark --model-state warm|cold  # 隔离的 10k Story 性能+质量基准
storybook stats                      # 系统统计
storybook list [--limit 20]          # 列出所有 Story
storybook show <story_id>            # 查看 Story 详情（含关联记忆）
storybook prime [--cwd PATH]         # 会话启动主动注入（晨间简报），供 SessionStart hook 调用
storybook mcp                        # 启动 MCP server（stdio，供 Claude Code 等 agent 运行时召回）
```

文本搜索会在主命中和“联想到的相关记忆”前展示真实 Story ID。可直接用该 ID 展开详情；脚本或 Agent 则可使用 `--json` 获取同一份检索结果（包括主命中与 related 的 `story_id`）：

```bash
storybook search "开发一个语音机器人" --top 1
# 主命中示例：📌 #42 未命名记忆
storybook show 42

storybook search "开发一个语音机器人" --top 1 --json
```

> 命令是 **`import-data`** 而非 `import`（click 把 `import_data` 函数自动连字符化）。无参数/无 flag 时为兼容性默认走 Claude；`dream`/`watch` 无 `--source` 时处理全部已启用且检测到的来源。`--source`、`--claude`、`--cursor`、`--codex`、`--sample` 与 `<path>` 互斥。

Agent history 为 local-first：单来源损坏会在 summary 标记 `degraded`，但不阻断其他来源。MCP 接入与 history ingestion 是两个独立状态。支持矩阵、schema/version 证据及隐私边界见 [Agent History Adapter compatibility](docs/AGENT_HISTORY_ADAPTERS.md)。

Codex JSONL 按 **append-only 增量源**处理：热路径只读取固定上限的文件身份/guard 证据和 checkpoint cursor 后新增的完整记录，复杂度为 `O(delta + C)`，不会为验证全部历史而每轮重读整个文件。文件被原子替换、inode/device 改变、尺寸缩短，或 guard 覆盖的边界发生变化时，会安全回退全量解析。对于同 inode 且继续增长、只改写 guard 未读取的历史中部字节，Storybook 不承诺自动发现；这不属于 supported 来源契约。

若上游工具或用户改写了既有历史，使用以下命令删除该来源的 checkpoint；下一轮 import/dream 会完整重建 checkpoint，其他来源不受影响：

```bash
storybook sources reset-checkpoint codex --yes
storybook import-data --source codex
```

### ContextEnvelope 与环境感知召回

每条新 Session 都保存 `tool/device/session/workspace/runtime/captured_at/provenance`；每个未知叶子字段使用 `null`（`runtime.kind` 使用枚举 `unknown`）并标记 `provenance=unknown`。Claude/Cursor adapter 采集 `detected/reported/inferred/user_confirmed` 来源，原始外部 session ID 使用 Profile 本地 HMAC，绝对路径、hostname、remote host 与 repo URL 只保留哈希或短别名。

Story 合并多个 Session 时会保留全部来源环境，不由最后一次会话覆盖。历史导入和实时采集都会从 cwd 解析 Git 根目录与 origin：远端仓库哈希作为主身份，同时保留根目录的 Profile 本地 HMAC 作为兼容身份，因此旧版仅含路径指纹的 Story 仍可在 `scope=project` 下召回，且绝对路径不会落库。双方都有远端主身份时以远端为准，不允许相同本地路径覆盖 remote 冲突；仅有一侧缺少远端证据时才使用路径兼容身份。搜索的语义相似度始终是主信号：默认 `scope=profile` 仅以 workspace/tool/runtime/OS 等环境信号做有界软加权，冲突结果仍可召回并带 `warnings`；只有调用方显式指定 `scope=strict` 才过滤环境冲突。`storybook show` 展示来源环境以及 `applies_when` / `excludes_when`。

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
| `storybook process --watch` | 反应式监听：轮询全部已启用来源（可用 `--source` 限定），有新会话自动采集 + 加工（长驻，Ctrl-C 退出） | 全平台 |
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
| `recall` | `query`（必填）, `top_k?`（默认 3）, `context?`, `scope?`（`profile\|strict`）, `graph_enabled?` | 返回直接/图扩散命中；图命中含 `seed_story_id/graph_path/score_components`，顶层 `truncated` 表示图预算安全截断 |
| `get_story` | `story_id`（必填） | 查看完整 `detail/sources/revisions` 与兼容 `title/content/version`，剥离 1024 维 embedding |
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

所有路径、模型名、阈值都集中在 `src/storybook/config.py`。环境变量样例见 `.env.example`。生成式 LLM 配置按“进程环境变量 > `STORYBOOK_LLM_ENV_FILE` > 项目 `.env` > 默认值”解析；文件只读取简单 `export KEY=value`/`KEY=value` 文本，绝不 `source` 或执行。未指定文件时默认发现 `~/.chrc/dpsk.sh`，不存在则静默跳过，适用于 launchd 等无 shell 环境。

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `STORYBOOK_PROFILE` | registry 当前项 | 仅当前进程选择 Profile（UUID 或显示名），不改 registry |
| `STORYBOOK_HOME` | 平台用户目录 | 显式收拢/隔离 registry、数据、缓存与日志 |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama 服务地址 |
| `STORYBOOK_LLM_ENV_FILE` | `~/.chrc/dpsk.sh` | DeepSeek shell-env 配置文件（纯文本解析） |
| `ANTHROPIC_BASE_URL` | `https://api.deepseek.com/anthropic` | DeepSeek Anthropic-compatible Base URL |
| `ANTHROPIC_AUTH_TOKEN` / `DEEPSEEK_KEY` | 无 | API key 与兼容回退变量；不会写入日志/status |
| `STORYBOOK_LLM_MODEL` | `deepseek-v4-flash` | 生成模型；其次读取 `ANTHROPIC_DEFAULT_HAIKU_MODEL` |
| `STORYBOOK_EMBED_MODEL` | `qwen3-embedding:0.6b` | embedding 模型（必须 1024 维） |
| `STORYBOOK_EMBED_VERSION` | `story-v2-default-v1` | 活跃表示的不可变版本标识 |
| `STORYBOOK_EMBED_REPRESENTATION` | `default` | 默认 `title + abstract + applicability` |
| `STORYBOOK_INFERENCE_CACHE_ENABLED` | `1` | Profile 私有 LLM/embedding 输入哈希缓存；`0` 禁用 |
| `STORYBOOK_PROCESS_WORKERS` | `4` | 批量加工并行推理准备线程数；SQLite 持久化仍顺序执行 |
| `STORYBOOK_ABSTRACT_MAX_CHARS` | `600` | abstract 预算；不影响 detail/source 持久化 |
| `STORYBOOK_EMBED_KEEP_ALIVE` | `10m` | 每次 embedding 请求续期的 Ollama 模型驻留时间 |
| `STORYBOOK_QUERY_WARM_TIMEOUT_SECONDS` | `2` | warm embedding 硬超时，超时即降级 |
| `STORYBOOK_QUERY_COLD_TIMEOUT_SECONDS` | `5` | cold embedding 硬超时，超时即降级 |
| `STORYBOOK_QUERY_FALLBACK_TIMEOUT_SECONDS` | `0.5` | FTS/关键词 fallback 独立硬超时 |
| `STORYBOOK_QUERY_DEFAULT_MODE` | `fast` | 默认检索模式；Fast 永不调用生成式 LLM |
| `STORYBOOK_QUERY_TRANSFORM_ENABLED` | `1` | Query Transformation/HyDE 总开关 |
| `STORYBOOK_QUERY_AUTO_CONFIDENCE_THRESHOLD` | `0.62` | Auto 低置信第二阶段门槛 |
| `STORYBOOK_QUERY_AUTO_SECOND_STAGE_TIMEOUT_SECONDS` | `2.0` | Auto 生成+扩展检索独立预算 |
| `STORYBOOK_QUERY_DEEP_TOTAL_TIMEOUT_SECONDS` | `5.0` | Deep 总预算，超时保留 Fast fallback |
| `STORYBOOK_RERANK_ENABLED` / `TOP_N` / `TIMEOUT_SECONDS` | `1` / `20` / `0.08` | 本地有界 reranker 与独立超时 |
| `STORYBOOK_GRAPH_ENABLED` | `1` | 默认启用 Graph RAG；`0` 仅返回直接检索 |
| `STORYBOOK_GRAPH_MAX_HOPS` / `MAX_PATHS` / `FAN_OUT` | `2` / `64` / `8` | 图扩散结构预算 |
| `STORYBOOK_GRAPH_TIME_BUDGET_MS` | `100` | 图扩散墙钟预算，用尽时返回 `truncated=true` |
| `STORYBOOK_GRAPH_TOKEN_BUDGET` | `1600` | 图扩散候选摘要与路径预算 |
| `STORYBOOK_LLM_THINK` | `0` | DeepSeek thinking：`0`=关，`1`=显式开启 |
| `STORYBOOK_DREAM_INTERVAL` | `14400` | `dream` 守护进程 / launchd 定时间隔（秒），默认 4 小时 |
| `STORYBOOK_WATCH_POLL_INTERVAL` | `60` | `process --watch` 轮询已启用 Agent history 来源的间隔（秒） |

关键阈值（`config.py`）：

| 常量 | 值 | 含义 |
|------|----|------|
| `SIM_THRESHOLD_HIGH` | 0.85 | ≥ 触发合并/更新 |
| `SIM_THRESHOLD_UPDATE_ONLY` | 0.92 | ≥ 仅补充细节（不合并内容） |
| `SIM_THRESHOLD_LOW` | 0.75 | ≥ 且 <high 触发弱关联新建 |
| `SIM_THRESHOLD_SEARCH` | 0.50 | 检索最低相似度 |
| `ENVIRONMENT_SCORE_WEIGHT` | 0.08 | 环境在同语义分桶内的有界次级权重；不反转语义主排序 |
| `TOP_K_RETRIEVAL` / `TOP_K_SEARCH` | 5 / 3 | 做梦召回 / 用户搜索返回数 |
| `STORY_ABSTRACT_MAX_CHARS` | 600 | abstract 预算；不截断 detail/source |
| `WEIGHT_INCREMENT` / `WEIGHT_MAX` | 0.1 / 1.0 | 共同召回提权 / 权重上限 |
| `PRIME_MIN_SIMILARITY` | 0.60 | 晨间简报主动注入最低相似度（高于检索 0.50，避免噪声） |
| `PRIME_TOP_K` | 5 | 晨间简报最多考虑的候选数（再按 token 预算裁剪） |
| `PRIME_TOKEN_BUDGET` | 2000 | 晨间简报 token 预算上限（≤2k，避免污染上下文） |
| `PRIME_CONTENT_EXCERPT_CHARS` | 140 | 晨间简报中每条 Story 摘要最大字符数 |

记忆形成 LLM 使用 temp 0.3、调用方既有 `max_tokens` 上限与 120s 超时；`extract_keywords`、Story v2 formation、`summarize_session`、`merge_stories`、`judge_split`、`split_story` 与 query transformation 均通过 DeepSeek Anthropic-compatible 的强制命名 tool call + `input_schema` 返回结构化对象，并在本地再次校验类型。旧网关的 JSON 文本仍可兼容解析；401/402/429/5xx、超时、schema 不匹配或空内容均保持原有业务 fallback。

## 📁 项目结构

```
storybook/
├── src/storybook/
│   ├── cli.py          # CLI 命令入口
│   ├── config.py       # 路径 / 模型 / 阈值常量
│   ├── context.py      # ContextEnvelope 采集、隐私归一与环境适配评分
│   ├── profiles.py     # 用户级 registry、平台目录、local/isolated Profile
│   ├── collector.py    # 会话采集（Claude Code / Cursor / JSON / 模拟）
│   ├── store.py        # SQLite + sqlite-vec 存储层
│   ├── processor.py    # 做梦周期（dream cycle）
│   ├── llm.py          # DeepSeek Anthropic-compatible Messages API
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
├── tests/              # pytest 测试套件（store/processor/search/dreamd + 集成，全 mock provider）
├── test_logs/          # 示例 JSON 数据
├── hermes_sessions.json
├── .env.example
└── pyproject.toml
```

## 📝 说明

- **隐私边界**：Profile、数据库、原始证据和 embedding 均留在本机；生成式操作会把其 prompt 发送到 DeepSeek，Fast 查询不会。
- **测试套件**：`tests/` 下 pytest 用例覆盖 store/processor/search/prime/dreamd 核心路径，全 mock、不依赖真实 provider（见上文「🧪 测试」）。`test_logs/*.json` 与 `hermes_sessions.json` 是 `import-data` 的样例数据源。
- **MCP server**：`storybook mcp` 启动独立 stdio 进程，向 Claude Code 等 agent 暴露 `recall`/`get_story`/`stats`/`prime_context`（接入见上文「🔌 MCP 接入」）。
- **晨间简报**：`storybook prime`（SessionStart hook）或 `prime_context` MCP 工具在会话启动时主动召回相关记忆注入上下文，复用 `search` 召回；相关度不足 / 无匹配 / Ollama 不可用时静默不注入（见上文「🌅 会话启动注入」）。
- `docs/TECH_DESIGN.md` 是最初的设计文档，其中的目录布局与命令示例早于当前实现（命令为 `import-data`；`tests/`、`scripts/` 与 launchd plist 已在后续迭代落地，见上文「🧪 测试」与「🌙 做梦周期自动化」）。
- LLM 输出解析是宽松的：关键词 JSON 在 `[`/`]` 间切片，摘要按 `TITLE:`/`CONTENT:` 标记切分，模型不遵循格式时有字符串切分兜底。

## License

MIT
