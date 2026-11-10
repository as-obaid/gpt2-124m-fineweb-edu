#!/bin/bash
# Build llm.c's train_gpt2cu: bf16, cuDNN flash attention, single GPU, for H200 (sm_90).
# Run on a GPU node (inside a Slurm job), after: source env.sh
set -euo pipefail
: "${ROOT:?source env.sh first}"
module load $CUDA_MODULES
# llm.c's Makefile looks for cudnn-frontend only in $HOME or inside llm.c/, so link our copy there.
ln -sfn "$ROOT/cudnn-frontend" "$ROOT/llm.c/cudnn-frontend"
cd "$ROOT/llm.c"
# Recent cudnn-frontend also needs NVRTC at link time, which llm.c's Makefile predates.
# NVCC_LDLIBS is empty when multi-GPU and MPI are off, so setting it only adds -lnvrtc.
make train_gpt2cu USE_CUDNN=1 NO_MULTI_GPU=1 NO_USE_MPI=1 GPU_COMPUTE_CAPABILITY=90 NVCC_LDLIBS=-lnvrtc
ls -la train_gpt2cu
