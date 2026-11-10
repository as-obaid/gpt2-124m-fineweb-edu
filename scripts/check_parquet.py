"""Open every downloaded sample-10BT parquet file and count documents and tokens.

Run after: source env.sh   Expected: 14 files, 9,672,101 docs, 9,972,918,177 tokens.
"""
import glob
import os
import sys

import pyarrow.compute as pc
import pyarrow.parquet as pq

revision = os.environ.get("FINEWEB_EDU_REVISION")
if not revision:
    sys.exit("source env.sh first")
snapshot = os.path.join(
    os.environ["HF_HOME"], "hub", "datasets--HuggingFaceFW--fineweb-edu", "snapshots", revision
)
files = sorted(glob.glob(os.path.join(snapshot, "sample", "10BT", "*.parquet")))
docs = tokens = 0
for f in files:
    table = pq.read_table(f, columns=["token_count"])
    docs += table.num_rows
    tokens += pc.sum(table.column("token_count")).as_py()
print(f"{len(files)} files, {docs:,} docs, {tokens:,} tokens")
