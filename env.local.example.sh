# Copy to env.local.sh (git ignores it) and fill in your own values. env.sh sources it.
export SBATCH_ACCOUNT=your_slurm_account        # sbatch reads this; no --account needed in scripts
# Optional: keep the Python env and caches outside the repo, e.g. on scratch storage.
# export VENV=/scratch/$USER/envs/gpt2-124m-fineweb-edu
# export HF_HOME=/scratch/$USER/hf_cache
# export TIKTOKEN_CACHE_DIR=/scratch/$USER/.cache/tiktoken
