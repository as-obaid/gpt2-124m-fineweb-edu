# llm.c flags for the 3-epoch GPT-2 124M run. Sourced by train.sbatch and smoke_test.sbatch.
#
# 3 epochs = 56,352 steps. llm.c reads each shard in whole micro-batches of 64 x 1024 tokens and
# skips the rest, so one epoch of the 9.85B training tokens is 98 x 1,525 + 823 = 150,273
# micro-batches, or 18,784 steps of 524,288 tokens. Data is reshuffled every epoch.
MAX_STEPS=56352
TRAIN_ARGS=(
  -i "$DATA_DIR/edu_fineweb_train_*.bin" -j "$DATA_DIR/edu_fineweb_val_*.bin"
  -e d12                               # new GPT-2 124M (12 layers), random init
  -b 64 -t 1024 -d 524288              # micro-batch 64 x 1024 tokens, 8 accumulated per step
  -r 0                                 # no activation recompute: the H200 has the memory
  -k cosine -l 0.0006 -q 0.1 -u 715    # LR 6e-4, 715 warmup steps, cosine down to 6e-5
  -c 0.1                               # weight decay
  -x "$MAX_STEPS"
  -v 500 -m 160 -h 1                   # every 500 steps: val loss and HellaSwag (see below)
  -s 5000                              # print a text sample every 5,000 steps
  -n 500 -nk 2 -nm 10000               # checkpoint every 500 steps; keep 2, plus every 10,000th
)
# Val loss uses the first 160 x 64 x 1024 = 10.5M tokens of the val shard. build-nanogpt evaluates
# 20 batches of 64 x 1024 on each of 8 GPUs, which is exactly these tokens, so the numbers compare.
