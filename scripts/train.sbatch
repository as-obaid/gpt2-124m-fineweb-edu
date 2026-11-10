#!/bin/bash
#SBATCH --job-name=gpt2fwe_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# One 8-hour segment of the 3-epoch run. Each segment resumes from the newest checkpoint (-y 1)
# and is stopped by the time limit; scripts/submit_train.sh chains the segments.
# Submit from the repo root after: source env.sh
set -euo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"
source "$ROOT/scripts/train_flags.sh"
OUT=$RUNS_DIR/gpt2_124m_3ep

if [ -e "$OUT/DONE_$(printf %08d "$MAX_STEPS")" ]; then
  echo "run already finished at step $MAX_STEPS"; exit 0
fi
if [ ! -x "$ROOT/llm.c/train_gpt2cu" ]; then
  echo "train_gpt2cu not built: run scripts/smoke_test.sbatch first" >&2; exit 1
fi
module load $CUDA_MODULES
mkdir -p "$OUT"
cd "$ROOT/llm.c"   # train_gpt2cu reads dev/data/hellaswag/hellaswag_val.bin relative to here
newest=$(ls "$OUT" | grep -o '^DONE_[0-9]*' | sort | tail -1 || true)
echo "segment start $(date -Is) on $(hostname), resuming from: ${newest:-nothing (new run)}"
./train_gpt2cu "${TRAIN_ARGS[@]}" -o "$OUT" -y 1
