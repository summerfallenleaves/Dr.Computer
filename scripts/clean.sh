#!/usr/bin/env bash
# Clear all generated artifacts: Python caches, log files, pytest cache, mypy.
#
# Usage:
#   ./scripts/clean.sh           # clear caches + logs
#   ./scripts/clean.sh --all     # also clear .venv (force re-install on next uv sync)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

ALL=""
[[ "${1:-}" == "--all" ]] && ALL=1

echo "Clearing Python caches..."
find src tests examples scripts -name __pycache__ -type d -print0 \
    2>/dev/null | xargs -0 rm -rf 2>/dev/null || true
find . -name "*.pyc" -not -path "./.venv/*" -delete 2>/dev/null || true
echo "  done"

echo "Clearing test caches..."
rm -rf .pytest_cache .mypy_cache .ruff_cache 2>/dev/null || true
echo "  done"

echo "Clearing logs..."
# Keep the .logs/ directory itself (dev.sh expects it to exist), only nuke contents.
if [[ -d .logs ]]; then
    rm -f .logs/*.log
    echo "  cleared .logs/*.log"
else
    mkdir -p .logs
    echo "  created empty .logs/"
fi

echo "Clearing build artifacts..."
rm -rf build dist src/dr_computer.egg-info 2>/dev/null || true
echo "  done"

if [[ -n "$ALL" ]]; then
    echo "Removing .venv (run 'uv sync --extra dev' to recreate)..."
    rm -rf .venv
    echo "  done"
fi

echo ""
echo "✓ Clean complete"
