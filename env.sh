# Settings shared by every script. From the repo root:  source env.sh
# Put your own values (Slurm account, Python env, caches) in env.local.sh, which git ignores.
# See env.local.example.sh.

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ -f "$ROOT/env.local.sh" ]; then
  source "$ROOT/env.local.sh"
fi
export ROOT

export VENV=${VENV:-$ROOT/.venv}
export PY=${PY:-$VENV/bin/python}
export HF_HOME=${HF_HOME:-$ROOT/.cache/huggingface}
export TIKTOKEN_CACHE_DIR=${TIKTOKEN_CACHE_DIR:-$ROOT/.cache/tiktoken}
export DATA_DIR=${DATA_DIR:-$ROOT/data/edu_fineweb10B}
export RUNS_DIR=${RUNS_DIR:-$ROOT/runs}

# Module names on Northeastern's Explorer cluster. Change them for another cluster.
export CUDA_MODULES=${CUDA_MODULES:-"cuda/12.8.0 cuDNN/9.10.2"}

# Pinned versions, so every input to the run can be fetched again exactly.
export LLMC_COMMIT=f1e2ace651495b74ae22d45d1723443fd00ecd3a            # karpathy/llm.c, 2025-05-10
export CUDNN_FRONTEND_COMMIT=868b3cae44ac053a1d6f0a5ffad801f20a840ff3  # NVIDIA/cudnn-frontend
export FINEWEB_EDU_REVISION=87f09149ef4734204d70ed1d046ddc9ca3f2b8f9   # HuggingFaceFW/fineweb-edu
export HELLASWAG_REVISION=218ec52e09a7e7462a5400043bb9a69a41d06b76     # Rowan/hellaswag
