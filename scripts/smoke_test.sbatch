#!/bin/bash
#SBATCH --job-name=gpt2fwe_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Smoke test on one H200. Submit from the repo root after: source env.sh
#  1. Build llm.c.
#  2. Score OpenAI's GPT-2 124M on HellaSwag with llm.c's dev/data/hellaswag.py (the baseline).
#     This also writes dev/data/hellaswag/hellaswag_val.bin, which train_gpt2cu evaluates.
#  3. Train 200 steps (run A) with the real run's flags.
#  4. Resume a second run (B) from A's step-100 checkpoint. Steps 101-200 must match A exactly.
set -euo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"
source "$ROOT/scripts/train_flags.sh"
A=$RUNS_DIR/smoke_a
B=$RUNS_DIR/smoke_b
mkdir -p "$RUNS_DIR"

bash "$ROOT/scripts/build_llmc.sh"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

cd "$ROOT/llm.c"
if [ ! -f dev/data/hellaswag/hellaswag_val.bin ]; then
  if HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PY" -E -s dev/data/hellaswag.py -m openai-community/gpt2 -d cuda \
       > "$RUNS_DIR/hellaswag_gpt2_baseline.txt"; then
    echo "HellaSwag baseline, OpenAI GPT-2 124M: $(grep acc_norm "$RUNS_DIR/hellaswag_gpt2_baseline.txt" | tail -1)"
  else
    echo "HellaSwag baseline FAILED, see $RUNS_DIR/hellaswag_gpt2_baseline.txt and the .err log"
  fi
fi

module load $CUDA_MODULES
SMOKE=(-x 200 -n 100 -nk 0 -nm 0 -v 100 -s 0)   # overrides: later flags win in llm.c
rm -rf "$A" "$B"; mkdir -p "$A" "$B"
./train_gpt2cu "${TRAIN_ARGS[@]}" "${SMOKE[@]}" -o "$A" -y 0 | tee "$A/stdout.txt"
cp "$A/model_00000100.bin" "$A/state_00000100_00000.bin" "$A/DONE_00000100" "$B/"
./train_gpt2cu "${TRAIN_ARGS[@]}" "${SMOKE[@]}" -o "$B" -y 1 | tee "$B/stdout.txt"

echo "mean tok/s, run A steps 21-200:"
grep -E '^step ' "$A/stdout.txt" | awk -F'|' 'NR > 20 { split($NF, t, " "); s += t[1]; n++ } END { printf "%.0f over %d steps\n", s/n, n }'
du -sh "$A/model_00000200.bin" "$A/state_00000200_00000.bin"

# "step N/M | loss X (...)| norm Y (...)" -> "N X Y", for the steps after the resume point
pick() { grep -E '^step ' "$1" | sed -E 's/^step +([0-9]+)\/[0-9]+ \| loss ([0-9.]+).*\| norm +([0-9.]+).*/\1 \2 \3/' | awk '$1 > 100'; }
echo "steps compared: $(pick "$B/stdout.txt" | wc -l)"
if diff <(pick "$A/stdout.txt") <(pick "$B/stdout.txt") > "$RUNS_DIR/smoke_resume_diff.txt"; then
  echo "RESUME CHECK PASSED: steps 101-200 identical (loss to 6 decimals, grad norm to 4)"
else
  echo "RESUME CHECK FAILED: see $RUNS_DIR/smoke_resume_diff.txt"; head -20 "$RUNS_DIR/smoke_resume_diff.txt"
  exit 1
fi
