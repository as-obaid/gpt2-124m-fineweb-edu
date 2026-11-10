"""Check llm.c token shards: header fields, file sizes and token totals.

Header (llm.c dev/data/data_common.py): 256 int32 = magic 20240520, version 1, token count,
then token count uint16 tokens. Reads only headers and the first token of each shard.

Usage: python -I scripts/check_shards.py "$DATA_DIR"
"""
import glob
import os
import sys

import numpy as np

MAGIC, VERSION, HEADER_BYTES, EOT = 20240520, 1, 256 * 4, 50256

shards = sorted(glob.glob(os.path.join(sys.argv[1], "*.bin")))
totals = {"train": 0, "val": 0}
bad = []
for path in shards:
    header = np.fromfile(path, dtype=np.int32, count=256)
    magic, version, ntok = int(header[0]), int(header[1]), int(header[2])
    size_ok = os.path.getsize(path) == HEADER_BYTES + 2 * ntok
    first = int(np.fromfile(path, dtype=np.uint16, count=1, offset=HEADER_BYTES)[0])
    if magic != MAGIC or version != VERSION or not size_ok:
        bad.append(f"{os.path.basename(path)}: magic={magic} version={version} size_ok={size_ok}")
    split = "val" if "_val_" in os.path.basename(path) else "train"
    totals[split] += ntok
    if split == "val" and first != EOT:
        bad.append(f"{os.path.basename(path)}: first token {first}, expected EOT {EOT}")

print(f"shards: {len(shards)} ({sum('_val_' in p for p in shards)} val)")
print(f"val tokens:   {totals['val']:,}")
print(f"train tokens: {totals['train']:,}")
print(f"steps per epoch at 524,288 tokens/step: {totals['train'] / 524288:,.1f}")
print("problems:", bad if bad else "none")
sys.exit(1 if bad else 0)
