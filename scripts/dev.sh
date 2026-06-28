#!/usr/bin/env bash
# Dr.Computer dev/debug runner.
#
# Pre-flight checks (deps, .env, network), then runs the example agent and
# tees output to a timestamped log under .logs/.
#
# Usage:
#   ./scripts/dev.sh                            # default goal
#   ./scripts/dev.sh "Open Safari"              # custom goal
#   ./scripts/dev.sh --verbose "Open Safari"    # verbose + custom goal
#   ./scripts/dev.sh --help
#
# Logs are written to .logs/run_YYYYMMDD_HHMMSS.log (gitignored).

set -euo pipefail

# Resolve project root from this script's location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# --- arg parsing ---
VERBOSE=""
GOAL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -v|--verbose) VERBOSE="--verbose"; shift ;;
        -h|--help)
            # Print the leading comment block (after shebang), stripped of `# `.
            awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "$0"
            exit 0
            ;;
        --*) echo "Unknown flag: $1" >&2; exit 2 ;;
        *) GOAL="$1"; shift ;;
    esac
done

# --- header ---
echo "=== Dr.Computer dev runner ==="
echo "Project: $PROJECT_ROOT"
echo ""

# --- pre-flight checks ---

echo "[1/5] Python"
PY_VER=$(uv run python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$(uname)" == "Darwin" ]]; then
    if [[ "$PY_VER" < "3.11" ]]; then
        echo "  ✗ Python $PY_VER — needs >=3.11"
        exit 1
    fi
fi
echo "  ✓ Python $PY_VER"

echo ""
echo "[2/5] Dependencies"
uv sync --quiet --extra dev
echo "  ✓ All deps installed"

echo ""
echo "[3/5] .env file"
if [[ ! -f .env ]]; then
    if [[ -f .env.example ]]; then
        echo "  ⚠ .env missing — copying from .env.example"
        cp .env.example .env
        echo "  Edit $PROJECT_ROOT/.env and put your DASHSCOPE_API_KEY in it,"
        echo "  then re-run this script."
        exit 1
    else
        echo "  ✗ Neither .env nor .env.example found."
        exit 1
    fi
fi
# Source .env to read DASHSCOPE_API_KEY without leaking it to logs.
set -a
# shellcheck disable=SC1091
source .env
set +a
if [[ -z "${DASHSCOPE_API_KEY:-}" ]] || [[ "$DASHSCOPE_API_KEY" == "sk-replace-me" ]]; then
    echo "  ✗ DASHSCOPE_API_KEY not set (or still 'sk-replace-me') in .env"
    exit 1
fi
echo "  ✓ .env OK (key ${DASHSCOPE_API_KEY:0:8}...${DASHSCOPE_API_KEY: -4})"

echo ""
echo "[4/5] macOS permissions (best-effort check)"
if [[ "$(uname)" == "Darwin" ]]; then
    # Check Screen Recording permission indirectly by attempting a tiny capture.
    PERM_RESULT=$(uv run python - <<'PY' 2>&1 || true
import sys
sys.path.insert(0, 'src')
try:
    from Quartz import (
        CGMainDisplayID, CGDisplayBounds, CGWindowListCreateImage,
        kCGNullWindowID, kCGWindowImageDefault,
    )
    from Quartz import CGImageGetWidth, CGImageGetHeight
    did = CGMainDisplayID()
    img = CGWindowListCreateImage(
        CGDisplayBounds(did),
        kCGNullWindowID, kCGNullWindowID, kCGWindowImageDefault,
    )
    w = CGImageGetWidth(img) if img else 0
    print(f"OK {w}" if w > 100 else f"FAIL {w}")
except Exception as e:
    print(f"FAIL {type(e).__name__}")
PY
)
    if [[ "$PERM_RESULT" == OK* ]]; then
        echo "  ✓ Screen Recording ($PERM_RESULT)"
    else
        echo "  ⚠ Screen Recording may be missing ($PERM_RESULT)"
        echo "    System Settings → Privacy & Security → Screen Recording → add your terminal"
    fi
fi

echo ""
echo "[5/5] Network (Qwen endpoint)"
if ! curl -sS --max-time 5 -o /dev/null -w '%{http_code}' \
        https://dashscope.aliyuncs.com/ > /tmp/dc_netcheck 2>&1; then
    echo "  ✗ Cannot reach https://dashscope.aliyuncs.com/"
    echo "    Check VPN / DNS / firewall. Common fix on macOS:"
    echo "      sudo killall -HUP mDNSResponder"
    exit 1
fi
echo "  ✓ Qwen endpoint reachable"

# --- run ---

echo ""
echo "=== Starting agent ==="

LOG_DIR="$PROJECT_ROOT/.logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/run_${TIMESTAMP}.log"

# Build the python command. Quote the goal carefully.
ARGS=()
[[ -n "$VERBOSE" ]] && ARGS+=("$VERBOSE")
[[ -n "$GOAL" ]] && ARGS+=("$GOAL")
# Expand empty array safely under `set -u`.
if [[ ${#ARGS[@]} -eq 0 ]]; then
    set -- uv run python examples/01_open_notes.py
else
    set -- uv run python examples/01_open_notes.py "${ARGS[@]}"
fi

echo "Log:    $LOG_FILE"
if [[ -n "$GOAL" ]]; then
    echo "Goal:   $GOAL"
else
    echo "Goal:   (script default)"
fi
echo ""

# Run, tee to log. Don't let tee swallow the exit code.
set +e
"$@" 2>&1 | tee "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}
set -e

echo ""
echo "=== Done (exit $EXIT_CODE) ==="
echo "Log: $LOG_FILE"

if [[ $EXIT_CODE -ne 0 ]]; then
    echo ""
    echo "Failed. Tail of log:"
    tail -20 "$LOG_FILE"
fi

exit $EXIT_CODE
