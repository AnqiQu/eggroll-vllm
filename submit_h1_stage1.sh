#!/bin/bash
#SBATCH --job-name=eggroll-h1-s1
#SBATCH --output=logs/eggroll-h1-s1-%j.out
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
# #SBATCH --account=brics.u6oz   # optional: you have one account, so it's the default

set -euo pipefail
mkdir -p logs

source "$SCRATCH/uv_envs/vllm_env/.venv/bin/activate"
export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_CACHE_ROOT="$SCRATCH/.cache/vllm_${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$SCRATCH/.triton_${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/.inductor_${SLURM_JOB_ID}"
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

cd "$SCRATCH/eggroll-vllm"

BASE_MODEL=Qwen/Qwen3-1.7B USE_WANDB=1 WANDB_PROJECT=eggroll-h1 \
  POPULATION_SIZE=128 PROMPT_BATCH_SIZE=8 NUM_ITERATIONS=300 \
  STAGE=1 ./run_h1_curriculum.sh
