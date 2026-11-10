#!/bin/bash
# Start 1-hour training chunks on the sharing partition, to keep training while the 8-hour
# segments wait in the gpu queue. A chunk runs train.sbatch, so it resumes from the newest
# checkpoint like a segment. scripts/train_guard.sh does the rest: each chunk queues one successor
# behind itself, a segment that starts cancels the chunks, and no chunk trains while a segment
# runs. Segments never wait for chunks. One chunk is enough to start the chain; N > 1 queues more
# up front, but the sharing QOS allows only 4 submitted jobs per user in total.
# GPU must run the sm_90 build of train_gpt2cu: h100 (default) or h200.
# From the repo root:  source env.sh && bash scripts/submit_chunks.sh [N] [GPU]
set -euo pipefail
: "${ROOT:?source env.sh first}"
gpu=${2:-h100}
case $gpu in
  h100|h200) ;;
  *) echo "GPU must be h100 or h200: train_gpt2cu is built for sm_90 only" >&2; exit 1 ;;
esac
cd "$ROOT"   # Slurm logs and env.sh are found relative to the submit directory
mkdir -p logs
prev=$(squeue --me --name=gpt2fwe_chunk -h -o %i | sort -n | tail -1)
for i in $(seq 1 "${1:-1}"); do
  args=(--job-name=gpt2fwe_chunk --partition=sharing --gres=gpu:"$gpu":1 --time=01:00:00
        --export=ALL,CHUNK_GPU="$gpu")
  if [ -n "$prev" ]; then
    args+=(--dependency=afterany:"$prev")
  fi
  prev=$(sbatch --parsable "${args[@]}" scripts/train.sbatch)
  echo "chunk $i: job $prev"
done
