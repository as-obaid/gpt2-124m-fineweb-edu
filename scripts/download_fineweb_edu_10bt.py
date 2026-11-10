"""Download FineWeb-Edu sample-10BT (14 raw parquet files, 27 GB) into the Hugging Face cache.

Run on a node with internet (on Explorer, the transfer node), after: source env.sh
Tokenizing happens later, offline, in scripts/tokenize_fineweb_edu_10bt.sbatch.
"""
import os
import sys

from huggingface_hub import snapshot_download

revision = os.environ.get("FINEWEB_EDU_REVISION")
if not revision:
    sys.exit("source env.sh first")

path = snapshot_download(
    repo_id="HuggingFaceFW/fineweb-edu",
    repo_type="dataset",
    revision=revision,
    allow_patterns=["README.md", "sample/10BT/*"],
    max_workers=16,
)
print("downloaded to", path)
