#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
    python3 -m venv .venv
fi
if ! .venv/bin/python -c 'import flowdesk, PySide6, scipy, pyqtgraph' >/dev/null 2>&1; then
    .venv/bin/python -m pip install -r requirements-lock.txt
    .venv/bin/python -m pip install --no-deps -e .
fi
exec .venv/bin/python -m flowdesk
