#!/bin/bash
# Submit N chained 8-hour training segments; each starts when the previous one ends.
# Default 4: about 3 are needed, and a segment exits at once if the run has already finished.
# From the repo root:  source env.sh && bash scripts/submit_train.sh [N]
set -euo pipefail
: "${ROOT:?source env.sh first}"
cd "$ROOT"   # Slurm logs and env.sh are found relative to the submit directory
mkdir -p logs
prev=""
for i in $(seq 1 "${1:-4}"); do
  if [ -n "$prev" ]; then
    prev=$(sbatch --parsable --dependency=afterany:"$prev" scripts/train.sbatch)
  else
    prev=$(sbatch --parsable scripts/train.sbatch)
  fi
  echo "segment $i: job $prev"
done
