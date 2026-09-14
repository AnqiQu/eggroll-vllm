#!/bin/bash
#SBATCH --job-name=eggroll-h1-s2-len2-conly
#SBATCH --output=logs/eggroll-h1-s2-len2-conly-%j.out
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00

# The MISSING cell of the experiment matrix: correctness-ONLY fitness on
# horizon-2 (real headroom: base ~37%). The earlier correctness-only run was on
# horizon-1, which is near ceiling (~0.81), so it could not answer the question.
# With no format term in the objective at all, the "format is optimised first,
# correctness later" explanation cannot apply -- if correctness is still flat
# here after 600 steps, iteration count / format masking is not the story.
#
# Also the cleanest place to compare bf16 vs fp32-master updates: run once with
# FP32_MASTER=1 (default here) and once with FP32_MASTER= (empty) and compare
# diag/update/applied_frac and reward/frac_correct.
#
# Watch: reward/frac_correct (the objective), diag/frac_correct/pairs_with_signal
# (how many antithetic pairs disagree on correctness -- the raw ES signal),
# diag/pair_identical_rate + diag/entropy/margin (collapse monitors).
POP=256
FP32_MASTER="${FP32_MASTER:-1}"

set -euo pipefail
mkdir -p logs

source "$SCRATCH/uv_envs/vllm_env/.venv/bin/activate"
export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_CACHE_ROOT="$SCRATCH/.cache/vllm_len2conly${POP}_${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$SCRATCH/.triton_len2conly${POP}_${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/.inductor_len2conly${POP}_${SLURM_JOB_ID}"
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

export EGGROLL_FITNESS_MODE=correctness

cd "$SCRATCH/eggroll-vllm"

SUFFIX="bf16"; [[ -n "$FP32_MASTER" ]] && SUFFIX="fp32master"

# Stage default completion length (1024) is kept: the 2048 budget was inert
# (outputs stayed ~320 tokens). 600 steps from base.
BASE_MODEL=Qwen/Qwen3-1.7B USE_WANDB=1 WANDB_PROJECT=eggroll-h1 \
  POPULATION_SIZE="$POP" PROMPT_BATCH_SIZE=8 NUM_ITERATIONS=600 \
  ENTROPY_TOPK=20 FP32_MASTER="$FP32_MASTER" \
  OUTPUT_ROOT="runs/h1_curriculum_len2_correctonly_${SUFFIX}" \
  STAGE=2 ./run_h1_curriculum.sh
