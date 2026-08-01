"""
日志采集模块 — Claude Code 会话日志解析 + 模拟数据生成器（含备用 Cursor 采集器）
"""
import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

from . import config
from . import store
from . import context as context_module

logger = logging.getLogger(__name__)


def collect_cursor_sessions(storage_path: Path = None) -> list[dict]:
    """
    扫描 Cursor workspaceStorage，从 state.vscdb 提取会话数据

    返回: [{source, raw_content, problem_desc, code_snippets, conclusion}, ...]
    """
    storage_path = storage_path or config.CURSOR_STORAGE_PATH
    sessions = []

    if not storage_path.exists():
        logger.warning("Cursor 存储路径不存在: %s", storage_path)
        return sessions

    for workspace_dir in storage_path.iterdir():
        if not workspace_dir.is_dir():
            continue
        vscdb = workspace_dir / "state.vscdb"
        if not vscdb.exists():
            continue

        try:
            sessions.extend(_extract_from_vscdb(vscdb))
        except Exception as e:
            logger.error("解析 %s 失败: %s", vscdb, e)

    logger.info("从 Cursor 采集到 %d 条会话", len(sessions))
    return sessions


def _extract_from_vscdb(vscdb_path: Path) -> list[dict]:
    """从单个 state.vscdb 提取 Cursor 会话"""
    sessions = []
    db = sqlite3.connect(str(vscdb_path))
    db.row_factory = sqlite3.Row
    try:
        # Cursor 的聊天数据通常存在 ItemTable 表中
        # key 形如 "aiService.prompts" 或 "workbench.panel.aichat.view"
        rows = db.execute(
            """SELECT key, value FROM ItemTable
               WHERE key LIKE '%aiService%' OR key LIKE '%aichat%' OR key LIKE '%cursor%chat%'"""
        ).fetchall()

        for row in rows:
            try:
                value = row["value"]
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="ignore")
                data = json.loads(value)
                # 尝试解析会话结构
                adapter_context = context_module.capture_context(
                    tool_type="cursor",
                    integration_mode="log_import",
                    workspace_path=vscdb_path.parent,
                )
                for conv in _parse_cursor_conversation(
                    data, row["key"], adapter_context=adapter_context
                ):
                    sessions.append(conv)
            except (json.JSONDecodeError, TypeError):
                continue
    except sqlite3.OperationalError:
        # 表不存在等
        pass
    finally:
        db.close()

    return sessions


def _parse_cursor_conversation(
    data: dict, source_key: str, *, adapter_context: dict | None = None
) -> list[dict]:
    """解析 Cursor 对话数据为标准格式"""
    sessions = []

    # Cursor 的数据结构可能多种多样，尝试常见格式
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and ("prompt" in item or "message" in item or "text" in item):
                content = item.get("prompt") or item.get("message") or item.get("text", "")
                sessions.append({
                    "source": "cursor",
                    "raw_content": json.dumps(item, ensure_ascii=False),
                    "problem_desc": content[:200] if content else "",
                    "code_snippets": "[]",
                    "conclusion": "",
                    "context": adapter_context,
                })
    elif isinstance(data, dict):
        # 可能是 {conversations: [...]} 或 {messages: [...]}
        conv_list = data.get("conversations") or data.get("messages") or data.get("prompts", [])
        if isinstance(conv_list, list):
            combined = "\n".join(
                str(m.get("text", "") or m.get("content", "") or m.get("prompt", ""))
                for m in conv_list if isinstance(m, dict)
            )
            if combined.strip():
                sessions.append({
                    "source": "cursor",
                    "raw_content": json.dumps(data, ensure_ascii=False),
                    "problem_desc": combined[:200],
                    "code_snippets": "[]",
                    "conclusion": "",
                    "context": adapter_context,
                })

    return sessions


def import_sessions(sessions: list[dict]) -> int:
    """批量导入会话到数据库，返回导入数量"""
    count = 0
    for s in sessions:
        store.add_session(
            source=s.get("source", "manual"),
            raw_content=s.get("raw_content", ""),
            problem_desc=s.get("problem_desc", ""),
            code_snippets=s.get("code_snippets", "[]"),
            conclusion=s.get("conclusion", ""),
            context=s.get("context"),
        )
        count += 1
    logger.info("导入了 %d 条会话", count)
    return count


# ═══════════════════════════════════════════════
#  Claude Code 会话采集
# ═══════════════════════════════════════════════

CLAUDE_SOURCE = "claude_code"
CLAUDE_LEGACY_SOURCE_PREFIX = "claude_code:"
_RAW_CONTENT_CAP = 6000                 # 与 llm.summarize_session 的截断对齐


def collect_claude_sessions(projects_path: Path = None) -> list[dict]:
    """扫描 ~/.claude/projects/*/*.jsonl，解析 Claude Code 会话。

    每个 .jsonl 文件 = 一个会话。按 sessionId 去重（已导入的跳过），
    故可重复运行、增量采集新会话。
    """
    projects_path = projects_path or config.CLAUDE_PROJECTS_PATH
    sessions = []

    if not projects_path.exists():
        logger.warning("Claude Code 项目目录不存在: %s", projects_path)
        return sessions

    existing = _existing_claude_session_keys()
    files = sorted(projects_path.glob("*/*.jsonl"))   # 仅直接子文件，跳过 subagents/
    logger.info("发现 %d 个 Claude 会话文件，已导入 %d 个", len(files), len(existing))

    for jsonl in files:
        session_id = jsonl.stem
        session_hash = context_module.external_session_hash(session_id)
        if session_id in existing or session_hash in existing:
            continue
        try:
            parsed = _parse_claude_jsonl(jsonl, session_id)
            if parsed:
                sessions.append(parsed)
        except Exception as e:
            logger.error("解析 %s 失败: %s", jsonl.name, e)

    logger.info("从 Claude Code 采集到 %d 条新会话", len(sessions))
    return sessions


def _existing_claude_session_keys() -> set:
    """Return legacy raw IDs plus new local-HMAC IDs for incremental dedup."""
    db = store.get_db()
    try:
        rows = db.execute(
            """SELECT source, external_session_hash FROM sessions
               WHERE source = ? OR source LIKE ?""",
            (CLAUDE_SOURCE, CLAUDE_LEGACY_SOURCE_PREFIX + "%"),
        ).fetchall()
        keys = {r["external_session_hash"] for r in rows if r["external_session_hash"]}
        keys.update(
            r["source"][len(CLAUDE_LEGACY_SOURCE_PREFIX):]
            for r in rows if r["source"].startswith(CLAUDE_LEGACY_SOURCE_PREFIX)
        )
        return keys
    finally:
        db.close()


def _parse_claude_jsonl(path: Path, session_id: str) -> Optional[dict]:
    """解析单个 Claude Code 会话 JSONL 为标准 session 格式。

    提取 user 提问 + assistant 文本回复，跳过 thinking/tool_use/元消息。
    无真实 user 提问的文件返回 None（跳过）。
    """
    title = None
    user_turns: list[str] = []
    assistant_texts: list[str] = []
    cwd = None
    tool_version = None
    branch = None
    started_at = None

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            cwd = cwd or obj.get("cwd")
            tool_version = tool_version or obj.get("version")
            branch = branch or obj.get("gitBranch")
            started_at = started_at or obj.get("timestamp")

            t = obj.get("type")
            if t in ("ai-title", "summary") and not title:
                title = obj.get("aiTitle") or obj.get("summary")
                continue

            msg = obj.get("message")
            if not isinstance(msg, dict):
                continue

            if t == "user" and msg.get("role") == "user" and not obj.get("isMeta"):
                text = _extract_user_text(msg.get("content"))
                if text:
                    user_turns.append(text)
            elif t == "assistant" and msg.get("role") == "assistant":
                text = _extract_assistant_text(msg.get("content"))
                if text:
                    assistant_texts.append(text)

    if not user_turns:
        return None   # 无真实用户提问，跳过

    # 首条提问太短（如"你好"）或为空时，用 aiTitle/summary 兜底
    first_user = user_turns[0]
    if len(first_user) < 15 and title:
        problem_desc = title[:200]
    else:
        problem_desc = first_user[:200]
    raw_content = _build_raw_content(user_turns, assistant_texts)

    return {
        "source": CLAUDE_SOURCE,
        "raw_content": raw_content,
        "problem_desc": problem_desc,
        "code_snippets": "[]",
        "conclusion": assistant_texts[-1][:300] if assistant_texts else "",
        "context": context_module.capture_context(
            tool_type="claude_code",
            tool_version=tool_version,
            integration_mode="log_import",
            external_session_id=session_id,
            workspace_path=cwd,
            branch=branch,
            started_at=started_at,
        ),
    }


def _extract_user_text(content) -> str:
    """从 user message content 提取真实提问文本。

    跳过 <local-command-caveat>/<command-name> 等元消息（content 以 '<' 开头）
    和纯 tool_result 的 user 行。
    """
    if isinstance(content, str):
        text = content.strip()
        return "" if text.startswith("<") else text
    if isinstance(content, list):
        parts = [
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        text = "\n".join(p for p in parts if p).strip()
        return "" if text.startswith("<") else text
    return ""


def _extract_assistant_text(content) -> str:
    """从 assistant message content 提取 text 块（跳过 thinking / tool_use）。"""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            b.get("text", "").strip()
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        return "\n".join(p for p in parts if p)
    return ""


def _build_raw_content(user_turns: list[str], assistant_texts: list[str]) -> str:
    """交替拼接 user/assistant 文本；超长则保留首（问题）尾（结论）。"""
    lines = []
    n = max(len(user_turns), len(assistant_texts))
    for i in range(n):
        if i < len(user_turns):
            lines.append(f"[user] {user_turns[i]}")
        if i < len(assistant_texts):
            lines.append(f"[assistant] {assistant_texts[i]}")
    full = "\n".join(lines)
    if len(full) <= _RAW_CONTENT_CAP:
        return full
    half = _RAW_CONTENT_CAP // 2
    return full[:half] + "\n...\n" + full[-half:]


# ═══════════════════════════════════════════════
#  模拟数据生成器（无 Claude Code 会话时用于测试）
# ═══════════════════════════════════════════════

def generate_sample_sessions(n: int = 100) -> list[dict]:
    """生成 n 条模拟 Claude Code 会话日志，覆盖常见编程场景"""
    templates = [
        {
            "problem": "React useEffect 触发无限渲染循环",
            "code": "useEffect(() => { setData(transform(data)); }, [data, setData]);",
            "steps": "1.检查useEffect依赖数组发现setData是useState的setter每次渲染都不同；2.用useCallback包裹setData回调；3.将不依赖props的逻辑移出useEffect到useMemo",
            "conclusion": "将setData用useCallback包裹后渲染次数从无限降为2次",
            "keywords": ["React", "useEffect", "无限循环", "useCallback", "依赖数组"],
        },
        {
            "problem": "Python 列表推导式内存占用过高",
            "code": "result = [expensive_func(x) for x in huge_list]",
            "steps": "1.用内存分析器定位列表推导式占用2GB；2.改为生成器表达式(sum(expensive_func(x) for x in huge_list))；3.对大数据集使用itertools.islice分批处理",
            "conclusion": "改用生成器后内存从2GB降到50MB",
            "keywords": ["Python", "生成器", "内存优化", "列表推导式", "itertools"],
        },
        {
            "problem": "Git rebase 后分支丢失提交记录",
            "code": "git rebase main && git push --force",
            "steps": "1.用git reflog找到丢失的commit hash；2.git reset --hard <hash>恢复；3.重新rebase时使用--preserve-merges避免丢失",
            "conclusion": "通过reflog成功恢复丢失的3个commit",
            "keywords": ["Git", "rebase", "reflog", "force-push", "分支恢复"],
        },
        {
            "problem": "Docker 容器内 DNS 解析失败",
            "code": "docker run --network=bridge myapp  # curl: could not resolve host",
            "steps": "1.检查容器DNS配置cat /etc/resolv.conf发现为空；2.docker run添加--dns 8.8.8.8；3.在daemon.json配置dns字段永久修复",
            "conclusion": "配置daemon.json的dns后容器内DNS解析正常",
            "keywords": ["Docker", "DNS", "网络配置", "daemon.json", "容器网络"],
        },
        {
            "problem": "TypeScript 泛型类型推断失败",
            "code": "function merge<T>(a: T, b: T): T { return {...a, ...b} }",
            "steps": "1.检查泛型约束发现T没有限制为object类型；2.添加extends Record<string, unknown>约束；3.使用泛型推断合并后的类型type ReturnType = T & U",
            "conclusion": "添加泛型约束后类型推断正确，编译通过",
            "keywords": ["TypeScript", "泛型", "类型推断", "Record", "类型约束"],
        },
        {
            "problem": "PostgreSQL 慢查询 N+1 问题",
            "code": "users = User.objects.all(); [u.posts.all() for u in users]",
            "steps": "1.用EXPLAIN ANALYZE发现执行了1001次查询；2.使用select_related('posts')或prefetch_related优化；3.添加db_index到外键字段",
            "conclusion": "使用prefetch_related后查询从1001次降到2次，耗时从3s降到50ms",
            "keywords": ["PostgreSQL", "N+1查询", "ORM优化", "prefetch_related", "EXPLAIN"],
        },
        {
            "problem": "WebSocket 连接频繁断开重连",
            "code": "const ws = new WebSocket('ws://localhost:8080'); ws.onclose = () => reconnect()",
            "steps": "1.检查服务端日志发现keepalive超时；2.客户端添加心跳机制setInterval(() => ws.ping(), 30s)；3.服务端配置ping_timeout=60s",
            "conclusion": "添加心跳机制后WebSocket连接稳定运行超过24小时无断连",
            "keywords": ["WebSocket", "心跳机制", "keepalive", "重连", "连接稳定性"],
        },
        {
            "problem": "React 状态更新不反映到UI",
            "code": "state.items.push(newItem); setState(state)",
            "steps": "1.发现直接修改state引用未变React不会重新渲染；2.改为setState({...state, items: [...state.items, newItem]})；3.使用不可变更新模式或immer库",
            "conclusion": "使用展开运算符创建新引用后UI正确更新",
            "keywords": ["React", "状态管理", "不可变数据", "setState", "immer"],
        },
        {
            "problem": "Python 异步函数阻塞事件循环",
            "code": "async def fetch(): data = requests.get(url); return data.json()",
            "steps": "1.发现使用同步requests库阻塞了asyncio事件循环；2.替换为aiohttp的async/await接口；3.对必须同步的代码用loop.run_in_executor包装",
            "conclusion": "替换为aiohttp后事件循环不再阻塞，并发请求从1提升到50",
            "keywords": ["Python", "asyncio", "aiohttp", "事件循环", "异步编程"],
        },
        {
            "problem": "Kubernetes Pod 一直处于 CrashLoopBackOff",
            "code": "kubectl logs pod-name  # Error: failed to connect to database",
            "steps": "1.kubectl describe pod发现环境变量DB_HOST未设置；2.检查Deployment yaml发现env引用的ConfigMap名称拼写错误；3.修正ConfigMap名称并kubectl apply",
            "conclusion": "修正ConfigMap引用名称后Pod正常运行",
            "keywords": ["Kubernetes", "CrashLoopBackOff", "ConfigMap", "Pod调试", "env"],
        },
        {
            "problem": "Vue 组件 props 类型校验不生效",
            "code": "props: { count: Number }  // 传入 '5' 字符串不报错",
            "steps": "1.发现Vue 2的props类型校验只在开发模式警告不阻止渲染；2.升级props定义为validator函数；3.使用TypeScript + vue-class-component强化类型检查",
            "conclusion": "添加validator函数后在开发环境正确警告类型不匹配",
            "keywords": ["Vue", "props", "类型校验", "validator", "TypeScript"],
        },
        {
            "problem": "Redis 缓存雪崩导致数据库过载",
            "code": "cache.set(key, data, timeout=300)  # 所有缓存同时过期",
            "steps": "1.分析发现所有缓存设置相同TTL导致同时失效；2.在TTL上添加随机抖动timeout=300+random(0,60)；3.实现互斥锁防止缓存重建时大量请求穿透到DB",
            "conclusion": "添加TTL随机抖动和互斥锁后数据库负载峰值下降90%",
            "keywords": ["Redis", "缓存雪崩", "TTL", "互斥锁", "缓存穿透"],
        },
        {
            "problem": "Node.js 内存泄漏导致 OOM",
            "code": "const cache = {}; app.use((req, res, next) => { cache[req.url] = Date.now(); next(); })",
            "steps": "1.用heapdump抓取堆快照对比发现cache对象无限增长；2.改用lru-cache设置max大小；3.添加process.on('warning')监控内存",
            "conclusion": "改用LRU缓存后内存稳定在200MB以内，不再OOM",
            "keywords": ["Node.js", "内存泄漏", "LRU缓存", "heapdump", "OOM"],
        },
        {
            "problem": "CSS flexbox 布局在 Safari 下错位",
            "code": ".container { display: flex; flex: 1; }",
            "steps": "1.检查Safari版本发现flexbox旧语法支持差异；2.添加-webkit-flex和flex-grow: 1兼容前缀；3.使用Autoprefixer自动添加前缀",
            "conclusion": "添加浏览器前缀后Safari布局与Chrome一致",
            "keywords": ["CSS", "flexbox", "Safari兼容", "浏览器前缀", "Autoprefixer"],
        },
        {
            "problem": "Go goroutine 泄漏导致连接数持续增长",
            "code": "for _, url := range urls { go fetch(url) }  // 没有等待完成",
            "steps": "1.用runtime.NumGoroutine()发现goroutine数量持续增长；2.使用sync.WaitGroup等待所有goroutine完成；3.添加context.WithTimeout控制超时",
            "conclusion": "添加WaitGroup和context超时后goroutine数量稳定",
            "keywords": ["Go", "goroutine泄漏", "WaitGroup", "context", "并发控制"],
        },
        {
            "problem": "MySQL 死锁问题排查",
            "code": "BEGIN; UPDATE accounts SET balance = balance - 100 WHERE id = 1; -- 另一事务反向更新",
            "steps": "1.用SHOW ENGINE INNODB STATUS查看死锁日志；2.发现两个事务以不同顺序更新同一批行；3.统一所有事务的加锁顺序，按id ASC排序后更新",
            "conclusion": "统一加锁顺序后死锁问题消除",
            "keywords": ["MySQL", "死锁", "INNODB", "事务隔离", "加锁顺序"],
        },
        {
            "problem": "React Hook 在条件语句中调用导致报错",
            "code": "if (cond) { const [v, setV] = useState(0); }",
            "steps": "1.理解React Hook规则：不能在条件/循环/嵌套函数中调用；2.将条件判断移到Hook内部：const [v, setV] = useState(cond ? 0 : null)；3.使用eslint-plugin-react-hooks自动检测",
            "conclusion": "将条件移入Hook内部后错误消除",
            "keywords": ["React", "Hooks", "useState", "eslint-plugin", "Hook规则"],
        },
        {
            "problem": "Python 多进程共享内存数据同步",
            "code": "from multiprocessing import Process, Value; v = Value('i', 0)",
            "steps": "1.发现多进程写共享Value时数据不一致；2.使用v.get_lock()加锁保护临界区；3.改用multiprocessing.Manager的dict/list实现更复杂的共享状态",
            "conclusion": "使用Manager和锁机制后多进程数据同步正确",
            "keywords": ["Python", "multiprocessing", "共享内存", "锁", "Manager"],
        },
        {
            "problem": "Nginx 反向代理后获取不到真实客户端IP",
            "code": "proxy_pass http://backend;  # $remote_addr 总是 127.0.0.1",
            "steps": "1.Nginx默认不转发客户端IP；2.添加proxy_set_header X-Real-IP $remote_addr；3.后端从X-Forwarded-For头部取第一个IP",
            "conclusion": "配置X-Real-IP头部后后端正确获取客户端真实IP",
            "keywords": ["Nginx", "反向代理", "X-Real-IP", "X-Forwarded-For", "proxy"],
        },
        {
            "problem": "Rust 借用检查器报错 cannot borrow as mutable",
            "code": "let mut v = vec![1,2,3]; let r = &v; v.push(4);  // error",
            "steps": "1.理解Rust所有权规则：不可变借用存在时不能可变借用；2.确保不可变借用使用完毕后再修改；3.使用RefCell在运行时检查借用规则",
            "conclusion": "调整借用顺序后编译通过",
            "keywords": ["Rust", "借用检查器", "所有权", "RefCell", "生命周期"],
        },
    ]

    sessions = []
    sample_context = context_module.capture_context(
        tool_type="claude_code", integration_mode="manual"
    )
    for i in range(n):
        tmpl = templates[i % len(templates)]
        # 添加变化让每条不完全一样
        variation = f" (场景变体{i // len(templates) + 1})" if i >= len(templates) else ""
        raw = {
            "role": "user",
            "content": f"我遇到了一个问题：{tmpl['problem']}{variation}\n代码：{tmpl['code']}\n请问怎么解决？",
            "response": f"解决步骤：{tmpl['steps']}\n结论：{tmpl['conclusion']}",
            "keywords": tmpl["keywords"],
        }
        sessions.append({
            "source": "claude_simulated",
            "raw_content": json.dumps(raw, ensure_ascii=False),
            "problem_desc": tmpl["problem"] + variation,
            "code_snippets": json.dumps([tmpl["code"]]),
            "conclusion": tmpl["conclusion"],
            "context": sample_context,
        })

    return sessions
