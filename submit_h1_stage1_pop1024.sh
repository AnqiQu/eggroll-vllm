#!/bin/bash
#SBATCH --job-name=eggroll-h1-s1-pop1024
#SBATCH --output=logs/eggroll-h1-s1-pop1024-%j.out
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00

# Larger-population stage-1 rerun. Only POPULATION_SIZE and OUTPUT_ROOT differ
# from submit_h1_stage1.sh, so results land in a SEPARATE dir and don't clobber
# the pop-128 run. Everything else (sigma/lr/lora-r/prompt-batch/iters) is held
# fixed so the comparison isolates the population effect.
#
# To try a different size, change POP below (must be even and divisible by 4 =
# num GPUs). Runtime scales ~linearly with POP: pop-128 took ~53 min, so expect
# roughly 256 -> ~1.5-2 h, 512 -> ~3-4 h, 1024 -> ~6-8 h. Checkpoints save every
# 50 steps, so a walltime cutoff is recoverable.
POP=1024

set -euo pipefail
mkdir -p logs

source "$SCRATCH/uv_envs/vllm_env/.venv/bin/activate"
export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_CACHE_ROOT="$SCRATCH/.cache/vllm_pop${POP}_${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$SCRATCH/.triton_pop${POP}_${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/.inductor_pop${POP}_${SLURM_JOB_ID}"
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

cd "$SCRATCH/eggroll-vllm"

BASE_MODEL=Qwen/Qwen3-1.7B USE_WANDB=1 WANDB_PROJECT=eggroll-h1 \
  POPULATION_SIZE="$POP" PROMPT_BATCH_SIZE=8 NUM_ITERATIONS=300 \
  OUTPUT_ROOT="runs/h1_curriculum_pop${POP}" \
  STAGE=1 ./run_h1_curriculum.sh
