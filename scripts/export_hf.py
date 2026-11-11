"""Turn an llm.c checkpoint into a Hugging Face model folder, ready to upload, and check it.

Usage: python -I scripts/export_hf.py MODEL_BIN OUT_DIR [--val VAL_SHARD]

The conversion is llm.c's own dev/eval/export_hf.py, unchanged: bf16 weights and OpenAI's GPT-2
tokenizer. This script adds the model card (results/model_card.md), the training figure and a copy
of the llm.c checkpoint under llmc/. Then it loads the folder with Transformers and measures loss
on the first 16 x 1,024 tokens of the val shard, so a broken conversion never gets uploaded.

Needs torch, so run it on a compute node. Those have no internet: set HF_HUB_OFFLINE=1, with the
"gpt2" tokenizer cached under HF_HOME (download_gpt2_and_hellaswag.py does that).
"""
import argparse
import importlib.util
import os
import shutil
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEQ_LEN, N_SEQS = 1024, 16
MAX_LOSS = 3.5  # the run's val loss is 2.97; a wrong conversion lands far above this
SHARD_MAGIC, SHARD_VERSION, HEADER_INTS = 20240520, 1, 256  # llm.c token shard header


def llmc_convert():
    # loaded by path: this file has the same name, and llm.c's dev/eval is not a package
    path = os.path.join(ROOT, "llm.c", "dev", "eval", "export_hf.py")
    spec = importlib.util.spec_from_file_location("llmc_export_hf", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.convert


def val_tokens(path, count):
    header = np.fromfile(path, dtype=np.int32, count=HEADER_INTS)
    if len(header) < 2 or header[0] != SHARD_MAGIC or header[1] != SHARD_VERSION:
        sys.exit(f"{path} is not an llm.c token shard")
    tokens = np.fromfile(path, dtype=np.uint16, count=count, offset=HEADER_INTS * 4)
    if len(tokens) < count:
        sys.exit(f"{path} has fewer than {count:,} tokens")
    return torch.from_numpy(tokens.astype(np.int64))


def check(out_dir, val_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(out_dir, torch_dtype=torch.float32).eval()
    seqs = val_tokens(val_path, N_SEQS * SEQ_LEN).view(N_SEQS, SEQ_LEN)
    with torch.no_grad():
        loss = sum(model(s[None], labels=s[None]).loss.item() for s in seqs) / N_SEQS
    print(f"loss on the first {N_SEQS * SEQ_LEN:,} val tokens: {loss:.3f}")

    tokenizer = AutoTokenizer.from_pretrained(out_dir)
    prompt = tokenizer("During photosynthesis in green plants", return_tensors="pt").input_ids
    torch.manual_seed(0)
    with torch.no_grad():
        sample = model.generate(prompt, max_new_tokens=40, do_sample=True, top_k=50,
                                pad_token_id=tokenizer.eos_token_id)
    print("sample:", tokenizer.decode(sample[0]))
    return loss


def main():
    data_dir = os.environ.get("DATA_DIR", os.path.join(ROOT, "data", "edu_fineweb10B"))
    ap = argparse.ArgumentParser()
    ap.add_argument("model_bin")
    ap.add_argument("out_dir")
    ap.add_argument("--val", default=os.path.join(data_dir, "edu_fineweb_val_000000.bin"))
    args = ap.parse_args()
    if os.path.exists(args.out_dir):
        sys.exit(f"{args.out_dir} exists; remove it first, so no stale files get uploaded")

    llmc_convert()(args.model_bin, args.out_dir, push_to_hub=False, out_dtype="bfloat16")
    shutil.copy(os.path.join(ROOT, "results", "model_card.md"), os.path.join(args.out_dir, "README.md"))
    shutil.copy(os.path.join(ROOT, "results", "figures", "training_curves.png"), args.out_dir)
    os.makedirs(os.path.join(args.out_dir, "llmc"))
    shutil.copy(args.model_bin, os.path.join(args.out_dir, "llmc"))

    loss = check(args.out_dir, args.val)
    if loss > MAX_LOSS:
        sys.exit(f"loss {loss:.3f} is above {MAX_LOSS}: the conversion is broken, don't upload it")
    print("ready to upload:", args.out_dir)


if __name__ == "__main__":
    main()
