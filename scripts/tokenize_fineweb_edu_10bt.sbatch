#!/bin/bash
#SBATCH --job-name=gpt2fwe_tokenize
#SBATCH --partition=sharing
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Tokenize FineWeb-Edu sample-10BT with llm.c's dev/data/fineweb.py, unchanged, through
# scripts/run_fineweb.py. Took 32 minutes on 64 CPUs. Runs offline: the parquet files and
# tokenizer files were fetched beforehand. Submit from the repo root after: source env.sh
set -euo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# The Arrow copy of the parquet files (46 GB) is only needed while tokenizing; deleted at the end.
export HF_DATASETS_CACHE=$ROOT/.cache/arrow_tmp
if [ -e "$DATA_DIR" ]; then echo "$DATA_DIR already exists, refusing to overwrite" >&2; exit 1; fi
cd "$ROOT/llm.c"
# -E -s ignores PYTHON* env vars and user site-packages. Not -I, because fineweb.py
# imports data_common from its own directory (llm.c/dev/data).
start=$(date +%s)
"$PY" -E -s "$ROOT/scripts/run_fineweb.py" -t edu -v 10B
echo "tokenize took $(( ($(date +%s) - start) / 60 )) min"
mkdir -p "$(dirname "$DATA_DIR")"
mv dev/data/edu_fineweb10B "$DATA_DIR"
rm -rf "$HF_DATASETS_CACHE"
"$PY" -I "$ROOT/scripts/check_shards.py" "$DATA_DIR"
