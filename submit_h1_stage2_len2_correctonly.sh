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
# FP32_MASTER=1 (default here) and once with FP32_MASTER=0 (or FP32_MASTER=
# empty) and compare diag/update/applied_frac and reward/frac_correct.
#
# Each arm lands in its own directory: OUTPUT_ROOT is derived from the arm, e.g.
#   sbatch submit_h1_stage2_len2_correctonly.sh                    -> runs/h1_curriculum_len2_correctonly_fp32master_sigma0.001
#   FP32_MASTER=0 sbatch submit_h1_stage2_len2_correctonly.sh      -> runs/h1_curriculum_len2_correctonly_bf16_sigma0.001
#   SIGMA=0.0003 LEARNING_RATE=0.00006 sbatch submit_h1_stage2_len2_correctonly.sh
#                                                                  -> runs/h1_curriculum_len2_correctonly_fp32master_sigma0.0003
#
# Watch: reward/frac_correct (the objective), diag/frac_correct/pairs_with_signal
# (how many antithetic pairs disagree on correctness -- the raw ES signal),
# diag/pair_identical_rate + diag/entropy/margin (collapse monitors).
POP=256
# NOTE: "${FP32_MASTER-1}" (no colon) so that an explicitly EMPTY value is kept
# (":-" would silently turn FP32_MASTER= back into 1). "0" also means off.
FP32_MASTER="${FP32_MASTER-1}"
[[ "$FP32_MASTER" == "0" ]] && FP32_MASTER=""
# Optional sigma / lr override. The probe (results/probe_len2_base.json) shows
# sigma=1e-3 already LOWERS accuracy under a random perturbation (40.6% ->
# 37.5%) and 3e-3 is destructive (-> 15%), so a SMALLER sigma is the arm worth
# trying, with lr scaled down in proportion so the step size in units of sigma
# is unchanged:   SIGMA=0.0003 LEARNING_RATE=0.00006 sbatch submit_h1_stage2_len2_correctonly.sh
SIGMA="${SIGMA:-0.001}"
LEARNING_RATE="${LEARNING_RATE:-0.0002}"

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
SUFFIX="${SUFFIX}_sigma${SIGMA}"

# Stage default completion length (1024) is kept: the 2048 budget was inert
# (outputs stayed ~320 tokens). 600 steps from base.
BASE_MODEL=Qwen/Qwen3-1.7B USE_WANDB=1 WANDB_PROJECT=eggroll-h1 \
  POPULATION_SIZE="$POP" PROMPT_BATCH_SIZE=8 NUM_ITERATIONS=600 \
  SIGMA="$SIGMA" LEARNING_RATE="$LEARNING_RATE" \
  ENTROPY_TOPK=20 FP32_MASTER="$FP32_MASTER" \
  OUTPUT_ROOT="runs/h1_curriculum_len2_correctonly_${SUFFIX}" \
  STAGE=2 ./run_h1_curriculum.sh
