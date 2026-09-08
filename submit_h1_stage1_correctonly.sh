#!/bin/bash
#SBATCH --job-name=eggroll-h1-s1-correctonly
#SBATCH --output=logs/eggroll-h1-s1-correctonly-%j.out
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00

# Tier-1 diagnostic run: fitness = CORRECTNESS ONLY (h1 format reward dropped
# from the objective, still logged as reward/format). Everything else is held
# fixed vs the earlier runs so this isolates the objective change. Writes to a
# SEPARATE output dir so the pop-128 / pop-1024 (format) checkpoints are kept.
#
# Watch in W&B: reward/correctness (should rise if this works), reward/format
# (may fall -- that's fine, it's no longer optimized), and pair_disagree_rate
# (near 0 => dead gradient => the bottleneck is Tier 2: sigma/lr/lora_r, not the
# objective). You can usually read the verdict within the first ~50-100 steps.
#
# pop-128 is enough to diagnose; bump POP to 1024 for a full run once the
# diagnostic looks promising.
POP=128

set -euo pipefail
mkdir -p logs

source "$SCRATCH/uv_envs/vllm_env/.venv/bin/activate"
export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_CACHE_ROOT="$SCRATCH/.cache/vllm_conly${POP}_${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$SCRATCH/.triton_conly${POP}_${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/.inductor_conly${POP}_${SLURM_JOB_ID}"
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# Tier-1: correctness-only objective (default, set here explicitly so the job
# log records it). Set to "total" to restore h1's correctness+format sum.
export EGGROLL_FITNESS_MODE=correctness

cd "$SCRATCH/eggroll-vllm"

BASE_MODEL=Qwen/Qwen3-1.7B USE_WANDB=1 WANDB_PROJECT=eggroll-h1 \
  POPULATION_SIZE="$POP" PROMPT_BATCH_SIZE=8 NUM_ITERATIONS=300 \
  OUTPUT_ROOT="runs/h1_curriculum_correctonly" \
  STAGE=1 ./run_h1_curriculum.sh
