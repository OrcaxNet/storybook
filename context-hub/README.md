# Context Hub：Mac mini 只读原型

FF-122 的独立 Python 包和独立 SQLite 数据库。从一个明确授权的 Claude 原生会话读取可见历史，通过本机 API / CLI / MCP 召回证据，并用显式更正阻止旧结论继续进入上下文。不依赖 `storybook` 包或旧数据库。

**这是可评审的技术原型，不是已完成全部 v0.2 验收的生产服务。** 实现与测试边界见 [VALIDATION.md](VALIDATION.md)。模型调用、自动后台采集、跨设备和生产安装均不在本 PR 中；读取已有会话不会调用模型或恢复来源会话。

## 安装与运行

目标是当前 macOS，Python 3.14.6。`.python-version` 和 `uv.lock` 固定本轮环境；运行时要求 SQLite ≥3.51.3。Python 版本不代表 SQLite 版本，启动会检查实际链接库。本机 Homebrew Python 3.14.6 链接 3.54.1；uv 自带的另一份 Python 曾链接 3.50.4，不能用于此原型。

```sh
cd context-hub
uv sync --frozen --python /opt/homebrew/bin/python3
uv run --frozen python -c 'import sqlite3; print(sqlite3.sqlite_version)'

# 使用全新目录；init 拒绝覆盖任何已有内容。
uv run --frozen context-hub init --data "$HOME/Library/Application Support/ContextHub-pilot"
uv run --frozen context-hub serve --data "$HOME/Library/Application Support/ContextHub-pilot" --port 7341
```

服务以前台运行，监听 `127.0.0.1`。另一个终端访问 `http://127.0.0.1:7341/health/ready`；停止用前台 Ctrl-C 或服务管理器发送 SIGTERM，先停止接收请求再关闭数据库。数据目录保留供下次启动。操作系统文件锁阻止两个 CLI 服务实例打开同一个权威库。测试程序会关闭自己启动的服务，不留下后台进程。

首次初始化生成 0700 数据目录、0600 数据文件和 `admin.json`。不要将该文件提交 Git、发给 Agent 或贴进对话。日志不记录请求正文或 Authorization。下面的命令在另一个终端运行，`PILOT_DATA` 是任务专用变量：

```sh
PILOT_DATA="$HOME/Library/Application Support/ContextHub-pilot"
printf '%s' '{"project":"pilot","root":"/absolute/approved/project"}' |
  uv run --frozen context-hub call project --credentials "$PILOT_DATA/admin.json"

# 明确指定一个原生会话 ID；不会批量导入整个 ~/.claude。
uv run --frozen context-hub import-claude \
  --credentials "$PILOT_DATA/admin.json" --project pilot \
  --root /absolute/approved/project --session NATIVE_SESSION_UUID

# 这是对消费端的显式正文导出授权。先确认它是否把上下文送往远程模型。
uv run --frozen context-hub grant --credentials "$PILOT_DATA/admin.json" \
  --project pilot --consumer approved-coding-agent --output "$PILOT_DATA/reader.json"

printf '%s' '{"project":"pilot","query":"超时","budget_tokens":2000}' |
  uv run --frozen context-hub call context --credentials "$PILOT_DATA/reader.json"
```

`call` 从标准输入或 `--input request.json` 读取 JSON；错误以稳定错误码返回。读取凭据绑定单一项目与消费端，不能导入、确认、更正或获取另一个项目。`grant` 只将密钥写入指定的新私有文件，终端输出仅含可用于吊销的散列。admin 对数据有管理权限，不要把它当普通 MCP 凭据。

## MCP 接入

仅在受授权 Agent 的配置中添加如下入口，替换两个绝对路径；本 PR 不自动修改已有 Agent 配置。

```json
{
  "mcpServers": {
    "context-hub-pilot": {
      "command": "/absolute/path/context-hub/.venv/bin/context-hub",
      "args": ["mcp", "--endpoint", "http://127.0.0.1:7341", "--credentials", "/absolute/private/reader.json"]
    }
  }
}
```

只暴露 `context_get`、`recall`、`evidence_read`、`sessions_list`。来源正文是资料，不是新指令；原文中的否定、引用旧指令、Agent 建议要由使用者连同前后文核实。MCP 断连或 Hub 停机返回错误，不自行启动另一个数据库。服务每次调用都重新鉴权，撤销凭据会影响已建立的 MCP 连接。

## 原型的实际接口

为减少 P0 路由和权限重复，所有操作采用 `POST /v1/{operation}` + JSON，CLI/MCP 调用相同实现。它是 v0.2 设计接口的原型映射，尚未承诺兼容版本化公开 API；没有实现的接口明确拒绝。

| 操作 | 核心输入 | 权限与结果 |
| --- | --- | --- |
| `project` | project, root | 管理；登记一次固定的授权目录，不能静默换根 |
| `grant` / `revoke` | project+consumer / token_hash | 管理；发放项目读凭据 / 立即吊销 |
| `import` | project, root, session_id, source_version, messages, coverage | 管理；最多 100 条或 1 MiB；事务提交后 ACK |
| `status` | project | 版本、依赖版本、模型出站关闭；管理者另见采集目录 |
| `sessions` | project, cursor?, limit? | 范围内会话、来源覆盖、观测时间和分页游标 |
| `context` / `recall` | project, query, budget_tokens?, limit? | 当前有效的记忆和原文候选、出处、版本、预算与截断标志 |
| `evidence` | project, id, cursor?, limit_bytes?, content_hash?, around? | 规范化 UTF-8 字节半开范围、散列、前后分页、脱敏/失效标志；around 最多返回前后各 5 条相邻消息引用，可继续读取 |
| `propose` | project, text, scope, evidence_ids, expected_revision | 管理；只生成 candidate；不得混合不同项目证据 |
| `confirm` | project, id, expected_revision, reason | 管理；明确声明所有者已核对支持关系，生成 owner_confirmed；不是模型自动验证 |
| `retract` | project, id, expected_revision, reason | 管理；撤回并失效全部共享证据的派生产物，阻止原片段再次被当前召回 |
| `revoke_source` | project, session_ref, expected_revision, reason | 管理；整个来源不可读、所有派生内容失效，禁止重放导入复活 |

`expected_revision` 取 `status.revision` 或上次写入结果；过期请求返回 `STALE_REVISION`，调用者须重新核对，不能自动改成最新版本重发。

更正流程：撤回旧 memory → 导入明确的新更正事件或新的原生会话快照 → 用新证据创建候选 → 所有者确认。撤回后可以手动 `evidence` 查看旧片段并看到 `current=false`，但当前 context 不再选它。没有自动把一段否定句解释成当前有效要求。

当前失效粒度是整条规范化消息块，可能连同该块中的其他有效知识一起隐藏。这是保守取舍；需要继续使用时，以独立、明确的新证据建立新条目，不去掉 tombstone。历史时间查询 `known_at` / `effective_at` 或 `intent=historical` 返回 `AS_OF_UNAVAILABLE`，不会拿最新知识冒充历史。

## 证据、隐私和一致性

- Claude SDK 固定 0.2.179。只使用 `list_sessions` / `get_session_messages`；枚举关闭 worktree 扩展，核对根目录及 cwd，再读取显式选择的会话。SDK 会先读整个文件，因此先限制来源元数据大小 ≤10 MiB，读取后复核修改时间和大小；活跃来源变化时报错重试。
- 只导入 SDK 可见文本、可见工具结果。隐藏推理、工具输入、非文本附件不保存，coverage 标明这些缺口。SDK 可见会话链不是整个原始 JSONL 的完整镜像。
- 稳定来源 ID + 消息 UUID + 分块编号形成证据 ID；同 ID 不同正文冲突并回滚整个批次。长可见文本按 UTF-8 边界分块，保留中段。批次重试幂等；只支持首次快照与相同来源版本重试。来源版本变化返回 SOURCE_VERSION_CHANGED，要求先核实并撤权旧来源，再选择新原生会话；不把旧正文悄悄标成新快照。完整追加、分支、轮转/截断对账需后续实现。
- SQLite WAL、FULL synchronous、单写锁；项目 revision、事件、索引与变更日志同事务提交。返回前在同一锁内核对所有贡献来源，没有跨请求结果缓存或异步模型任务。
- FTS5 使用确定性中文双字切分和英文标识符。支持“超时”“回滚”，不依赖 trigram；不宣称语义召回。返回资料不自动晋升成要求。匹配代码经转义、参数绑定，项目过滤在候选 SQL 内完成。
- 预算使用整个 JSON 的 UTF-8 字节数作保守 token 上界，不伪装成精确 tokenizer 结果；保留引用和适用范围，过小返回 `BUDGET_TOO_SMALL`，不会省略字段凑出可执行结论。全部结果需要核实；未来的强制任务约束组装未实现。
- 内置秘密模式过滤只是防御补充。读取范围与消费端授权仍须由所有者明确决定；本地 MCP 传给远程 Agent 也可能使正文出站。Hub 完全没有远程模型调用代码或供应商 fallback。
- 同一 OS 用户可直接读数据目录，应用 token 不构成同用户进程之间的强隔离。P0 不开放公网、不添加 Tailscale Serve，也不承诺对不可信远端提供服务。

## 备份与 macOS 服务管理

备份命令通过实例锁要求先停止 Hub，避免生产者与恢复操作并行；使用 SQLite Online Backup API 和 integrity_check 创建一致性快照，拒绝覆盖已有文件：

```sh
uv run --frozen context-hub backup --data "$PILOT_DATA" --output /private/backup/hub-2026-11-09.sqlite
```

快照内包含撤回和来源撤权状态，凭据只有散列。原始凭据文件需另行加密保管。旧快照可能早于最新撤权，**不提供自动恢复到在线状态的命令**；恢复时在隔离目录核对最新变更记录、原始凭据、integrity_check、引用及撤回样例后再使用，无法核实最新策略时保持停机。同盘备份不具备整机故障恢复能力；自动备份保留期、独立撤销日志复制、GC/删除与生产恢复演练均待后续实现。

需要常驻时使用 launchd 用户 LaunchAgent；参见 `launchd.plist.example`，先替换绝对路径，Review/QA 后再安装。LaunchAgent 在用户注销后终止；重启、FileVault/Keychain、断电自动恢复均需另验。本 PR 只提供模板，不变更系统电源、登录项或生产服务。

## 验证

```sh
uv run --frozen python -m unittest discover -s tests -v
uv run --frozen python scripts/benchmark.py
# 仅针对你新建的合成验收会话，脚本拒绝缺少专用标记的正文：
uv run --frozen python scripts/native_pilot.py --root /approved/pilot/project --session UUID
```

模型调用不属于上述测试和导入路径。测试覆盖真实 MCP stdio、HTTP、SQLite 子进程崩溃、跨项目访问、撤权、更正、字节范围与中段读取。性能脚本自建临时合成库，不读取个人会话。独立 QA 需另补真实多轮会话、40+10 人工标注题、中文改述/无结果质量、冷启动和持续负载；这些不能从合成精确码基准推导。
