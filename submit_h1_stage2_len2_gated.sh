#!/bin/bash
#SBATCH --job-name=eggroll-h1-s2-len2-gated
#SBATCH --output=logs/eggroll-h1-s2-len2-gated-%j.out
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00

# Horizon-2 diagnostic with the collaborator's GATED objective + longer CoT.
#   fitness = 1.0 iff (answer correct AND well-formatted) else 0.0   (binary)
#   generation budget 2048 tokens (vs the len-2 default 1024) so the model has
#   room to reason its way to a correct, well-formatted answer.
# Trains train_len_2 (base test acc ~38% -> real headroom, unlike near-ceiling
# len-1) directly from base Qwen3-1.7B with the float-format reward (STAGE=2).
# Separate output dir; nothing prior is clobbered.
#
# WATCH in W&B:
#   reward/frac_gated     -> fraction scoring 1 (both correct AND formatted).
#                            If ~0 the reward is too sparse to learn from.
#   reward/frac_format_ok -> how often the format gate passes (tighten to strict
#                            via EGGROLL_FORMAT_CHECK=strict once this is healthy).
#   reward/frac_correct   -> correctness alone (headroom check).
#   pair_disagree_rate    -> gradient signal (near 0 = dead gradient -> Tier 2).
# ~6-10h expected; walltime 12h with checkpoints every 50 steps (recoverable).
POP=256

set -euo pipefail
mkdir -p logs

source "$SCRATCH/uv_envs/vllm_env/.venv/bin/activate"
export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_CACHE_ROOT="$SCRATCH/.cache/vllm_len2gated${POP}_${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$SCRATCH/.triton_len2gated${POP}_${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/.inductor_len2gated${POP}_${SLURM_JOB_ID}"
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# Objective: binary gated reward (correct AND well-formatted). SOFT gate for this
# first run -- the model currently rarely emits the exact strict XML, so soft
# (reasoning-then-answer order) keeps the reward from being pathologically
# sparse. Tighten to strict (EGGROLL_FORMAT_CHECK=strict) once frac_format_ok
# looks healthy.
export EGGROLL_FITNESS_MODE=gated
export EGGROLL_FORMAT_CHECK=soft

cd "$SCRATCH/eggroll-vllm"

# STAGE=2 -> train_len_2.jsonl + float reward; MAX_TOKENS overrides the stage
# default (1024 -> 2048). BASE_MODEL=base model trains len-2 from scratch (clean
# diagnostic, not the warm-started curriculum).
BASE_MODEL=Qwen/Qwen3-1.7B USE_WANDB=1 WANDB_PROJECT=eggroll-h1 \
  POPULATION_SIZE="$POP" PROMPT_BATCH_SIZE=8 NUM_ITERATIONS=300 \
  MAX_TOKENS=2048 \
  OUTPUT_ROOT="runs/h1_curriculum_len2_gated" \
  STAGE=2 ./run_h1_curriculum.sh
