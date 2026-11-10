#!/bin/bash
# One-time setup, on a login node (needs internet): llm.c, cudnn-frontend and the Python env.
# From the repo root:  source env.sh && bash scripts/setup.sh
set -euo pipefail
: "${ROOT:?source env.sh first}"
cd "$ROOT"

# llm.c is a git submodule pinned at $LLMC_COMMIT. Fetch it if the clone skipped submodules.
if [ ! -f llm.c/Makefile ]; then
  git submodule update --init llm.c
fi

# cudnn-frontend is header-only; llm.c needs it for cuDNN flash attention.
if [ ! -d cudnn-frontend ]; then
  git clone https://github.com/NVIDIA/cudnn-frontend.git cudnn-frontend
  git -C cudnn-frontend checkout --quiet "$CUDNN_FRONTEND_COMMIT"
fi

# Python env with uv (https://docs.astral.sh/uv/).
if [ ! -x "$PY" ]; then
  uv venv --python 3.12 "$VENV"
  uv pip install --python "$PY" -r requirements.txt
fi

mkdir -p logs "$RUNS_DIR"
echo "setup done: llm.c $(git -C llm.c rev-parse --short HEAD), cudnn-frontend $(git -C cudnn-frontend rev-parse --short HEAD), $("$PY" -I --version)"
