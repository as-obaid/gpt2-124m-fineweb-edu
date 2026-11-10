"""Run llm.c's dev/data/fineweb.py unchanged, with three adjustments made from outside it.

1. Offline loading. Compute nodes have no internet, and load_dataset("HuggingFaceFW/fineweb-edu",
   name="sample-10BT") needs the Hub to resolve that config. On the Hub, sample-10BT is defined
   as the parquet files sample/10BT/* (see the dataset README). We load exactly those files,
   in sorted order, from the local snapshot we downloaded (revision pinned in env.sh).
2. Worker count. fineweb.py starts os.cpu_count() - 2 worker processes. On a shared Slurm
   node that counts the whole machine, so we make os.cpu_count() report this job's CPUs.
3. Parallel Arrow cache. Before fineweb.py runs, we build the Arrow cache with one process
   per parquet file. fineweb.py's own load_dataset call then reuses that cache.

Run from the llm.c root, after source env.sh:  python -E -s ../scripts/run_fineweb.py -t edu -v 10B
"""
import glob
import os
import runpy
import sys

import datasets
from huggingface_hub import snapshot_download

REPO_ID = "HuggingFaceFW/fineweb-edu"
CONFIG = "sample-10BT"
REVISION = os.environ.get("FINEWEB_EDU_REVISION") or sys.exit("source env.sh first")  # pinned in env.sh
N_FILES = 14

DATA_DIR = os.path.abspath(os.path.join("dev", "data"))
SCRIPT = os.path.join(DATA_DIR, "fineweb.py")
if not os.path.isfile(SCRIPT):
    sys.exit(f"run this from the llm.c root: {SCRIPT} not found")
if sys.argv[1:] != ["-t", "edu", "-v", "10B"]:
    sys.exit("this wrapper prepares the sample-10BT cache only; call it with: -t edu -v 10B")

snapshot = snapshot_download(
    repo_id=REPO_ID, repo_type="dataset", revision=REVISION,
    allow_patterns=["README.md", "sample/10BT/*"],  # same files we downloaded; errors if any is missing
    local_files_only=True,
)
files = sorted(glob.glob(os.path.join(snapshot, "sample", "10BT", "*.parquet")))
if len(files) != N_FILES:
    sys.exit(f"expected {N_FILES} parquet files in {snapshot}, found {len(files)}")
print(f"local snapshot: {snapshot} ({len(files)} parquet files)", flush=True)

_load_dataset = datasets.load_dataset


def load_dataset_local(path, name=None, split=None, **kwargs):
    if path != REPO_ID:
        return _load_dataset(path, name=name, split=split, **kwargs)
    if name != CONFIG:
        raise ValueError(f"only {CONFIG} is available offline, got {name}")
    return _load_dataset("parquet", data_files={"train": files}, split=split, **kwargs)


datasets.load_dataset = load_dataset_local  # fineweb.py does "from datasets import load_dataset"

ncpu = len(os.sched_getaffinity(0))
print(f"allocated CPUs: {ncpu}", flush=True)

ds = load_dataset_local(REPO_ID, name=CONFIG, split="train", num_proc=min(ncpu, N_FILES))
print(f"arrow cache ready: {len(ds):,} docs", flush=True)
del ds

os.cpu_count = lambda: ncpu
sys.path.insert(0, DATA_DIR)
sys.argv = [SCRIPT] + sys.argv[1:]
runpy.run_path(SCRIPT, run_name="__main__")
