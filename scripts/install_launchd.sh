#!/usr/bin/env bash
#
# 安装 / 卸载 Storybook 做梦周期的 macOS launchd 定时任务。
#
# 用法：
#   ./scripts/install_launchd.sh                       # 安装，每 4 小时（14400s）触发
#   ./scripts/install_launchd.sh --interval 3600       # 安装，每 1 小时触发
#   ./scripts/install_launchd.sh --python /opt/.../python   # 指定 python（默认 .venv/bin/python）
#   ./scripts/install_launchd.sh --uninstall          # 卸载
#
# 做的事：把 scripts/com.storybook.dream.plist 模板里的占位符替换为真实路径，
# 写入 ~/Library/LaunchAgents/com.storybook.dream.plist，然后 launchctl bootstrap 加载。
#
# 触发的命令是： <python> -m storybook.cli dream --once   （单次采集+加工，受文件锁保护）
set -euo pipefail

LABEL="com.storybook.dream"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TEMPLATE="$SCRIPT_DIR/com.storybook.dream.plist"
PLIST_DIR="$HOME/Library/LaunchAgents"
PLIST="$PLIST_DIR/$LABEL.plist"

INTERVAL="14400"        # 默认 4 小时
PYTHON_BIN=""
UNINSTALL=0

usage() {
  sed -n '3,12p' "$0" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --interval) INTERVAL="$2"; shift 2 ;;
    --python)   PYTHON_BIN="$2"; shift 2 ;;
    --uninstall) UNINSTALL=1; shift ;;
    -h|--help)  usage ;;
    *) echo "未知参数: $1" >&2; usage ;;
  esac
done

# ── 校验 interval 是正整数 ──
if ! [[ "$INTERVAL" =~ ^[0-9]+$ ]] || [[ "$INTERVAL" -le 0 ]]; then
  echo "❌ --interval 必须是正整数（秒），得到: $INTERVAL" >&2
  exit 1
fi

# ── 卸载 ──
if [[ "$UNINSTALL" -eq 1 ]]; then
  echo "🧹 卸载 $LABEL ..."
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || \
    launchctl unload "$PLIST" 2>/dev/null || true
  rm -f "$PLIST"
  echo "✅ 已卸载（$PLIST 已删除）"
  exit 0
fi

# ── 解析 python 路径 ──
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
  else
    echo "❌ 未找到 $PROJECT_DIR/.venv/bin/python" >&2
    echo "   请先建 venv 并安装：uv venv .venv && VIRTUAL_ENV=\$PWD/.venv uv pip install -e ." >&2
    echo "   或用 --python <path> 指定一个已装 storybook 的 python。" >&2
    exit 1
  fi
fi

# 校验该 python 能 import storybook（launchd 无 shell，必须用装好 storybook 的解释器）
if ! "$PYTHON_BIN" -c "import storybook" >/dev/null 2>&1; then
  echo "❌ $PYTHON_BIN 无法 import storybook。" >&2
  echo "   请在该 venv 里以 editable 方式安装：VIRTUAL_ENV=\$(dirname $PYTHON_BIN) uv pip install -e ." >&2
  exit 1
fi

# 日志目录由当前用户 Profile registry 解析，不绑定仓库位置。
LOG_DIR="$("$PYTHON_BIN" -c 'from storybook import config; print(config.LOG_DIR)')"
mkdir -p "$PLIST_DIR" "$LOG_DIR"

# ── 从模板渲染 ──
echo "📝 渲染 plist -> $PLIST"
# 用 python 做占位符替换，避免 sed 的分隔符/特殊字符问题
"$PYTHON_BIN" - "$TEMPLATE" "$PLIST" "$PYTHON_BIN" "$LOG_DIR" "$INTERVAL" <<'PYEOF'
import sys
src, dst, pybin, logdir, interval = sys.argv[1:6]
text = open(src, encoding="utf-8").read()
text = text.replace("__PYTHON_BIN__", pybin)
text = text.replace("__LOG_DIR__", logdir)
text = text.replace("__START_INTERVAL__", interval)
open(dst, "w", encoding="utf-8").write(text)
print(f"   python={pybin}\n   logdir={logdir}\n   interval={interval}s")
PYEOF

# ── 加载（先卸载旧实例，再 bootstrap；bootstrap 失败回退 load -w）──
echo "🔄 加载 launchd 任务 ..."
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
if launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>/dev/null; then
  echo "✅ 已加载（launchctl bootstrap）"
else
  launchctl unload "$PLIST" 2>/dev/null || true
  launchctl load -w "$PLIST"
  echo "✅ 已加载（launchctl load -w，回退路径）"
fi

echo ""
echo "📋 状态："
launchctl print "gui/$(id -u)/$LABEL" 2>/dev/null | grep -E "state|last exit code|program" | head -5 || true
echo ""
echo "常用命令："
echo "  立即触发一次（调试）: launchctl start $LABEL"
echo "  查看状态:            launchctl print gui/$(id -u)/$LABEL"
echo "  查看日志:            tail -f $LOG_DIR/dream.log"
echo "  卸载:                ./scripts/install_launchd.sh --uninstall"
