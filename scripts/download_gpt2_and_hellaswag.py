"""Fetch the small inputs (about 0.6 GB) that compute nodes, which have no internet, need.

1. tiktoken's GPT-2 tokenizer files, used by llm.c's dev/data/fineweb.py.
2. OpenAI's GPT-2 124M weights (openai-community/gpt2), the HellaSwag baseline.
3. HellaSwag validation, written as llm.c/dev/data/hellaswag/hellaswag_val.jsonl. llm.c's
   dev/data/hellaswag.py downloads that file from github.com/rowanz/hellaswag, which no longer
   serves it (HTTP 451). The author's Hugging Face copy, Rowan/hellaswag, has the same 10,042
   examples; there the label is a string, in the original file an int.

Run on a login node, after: source env.sh. Not on Explorer's transfer node: it runs CentOS 7,
whose C library is too old for the tiktoken and pyarrow wheels.
"""
import json
import os
import sys

import pyarrow.parquet as pq
import tiktoken
from huggingface_hub import hf_hub_download, snapshot_download

root = os.environ.get("ROOT")
revision = os.environ.get("HELLASWAG_REVISION")
if not (root and revision and os.environ.get("TIKTOKEN_CACHE_DIR")):
    sys.exit("source env.sh first")

tiktoken.get_encoding("gpt2")
print("tiktoken gpt2 files cached in", os.environ["TIKTOKEN_CACHE_DIR"])

path = snapshot_download("openai-community/gpt2", allow_patterns=["*.json", "*.txt", "*.safetensors"])
print("GPT-2 124M weights in", path)

FIELDS = ["ind", "activity_label", "ctx_a", "ctx_b", "ctx", "split", "split_type", "label", "endings", "source_id"]
parquet = hf_hub_download(
    "Rowan/hellaswag", "data/validation-00000-of-00001.parquet", repo_type="dataset", revision=revision
)
rows = pq.read_table(parquet).to_pylist()
if len(rows) != 10042:
    sys.exit(f"expected 10,042 HellaSwag validation examples, got {len(rows):,}")

out = os.path.join(root, "llm.c", "dev", "data", "hellaswag", "hellaswag_val.jsonl")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    for row in rows:
        example = {k: row[k] for k in FIELDS}
        example["label"] = int(example["label"])
        if len(example["endings"]) != 4 or not 0 <= example["label"] < 4:
            sys.exit(f"bad example: {example['ind']}")
        f.write(json.dumps(example) + "\n")
print(f"wrote {len(rows):,} HellaSwag examples to {out}")
