#!/bin/bash
#SBATCH --job-name=eggroll-h1-s2-len2-gated-resume
#SBATCH --output=logs/eggroll-h1-s2-len2-gated-resume-%j.out
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00

# "Maybe you are not running it for enough iterations" -- the literal test.
#
# CONTINUES the horizon-2 GATED run (train job 6455002) from its last checkpoint
# (step_299) for another 600 ES steps (NUM_ITERATIONS is the ABSOLUTE final step,
# so 900 = steps 300..899). Everything else is identical to
# submit_h1_stage2_len2_gated.sh, plus the new diagnostics (see
# PROFESSOR_FEEDBACK_NOTES.md):
#
#   diag/frac_correct/cos_with_fitness  -> does the ES update carry ANY
#                                           correctness information? (the
#                                           professor's hypothesis predicts this
#                                           rises toward 1 as format saturates)
#   diag/frac_format_ok/pairs_with_signal -> format saturation (falls toward 0)
#   diag/pair_identical_rate            -> entropy-collapse monitor: fraction of
#                                           +/- pairs producing identical text.
#                                           If this climbs toward 1 the gradient
#                                           is dying -- the "be mindful" case.
#   diag/entropy/margin                 -> greedy top1-top2 log-prob margin (up =
#                                           collapsing); needs ENTROPY_TOPK>0
#   diag/update/applied_frac            -> fraction of bf16 weight entries the
#                                           step actually changed (see the bf16
#                                           rounding note in the NOTES file)
#
# ~12h+ for 600 steps at pop-256 (the first 300 took ~6-10h): checkpoints every
# 50 steps, so if the walltime cuts it off, re-submit with RESUME_FROM pointing
# at the newest checkpoint_step_N. Raise --time if your queue allows it.
POP=256
CKPT=runs/h1_curriculum_len2_gated/stage2_len2/checkpoints/checkpoint_step_299

set -euo pipefail
mkdir -p logs

source "$SCRATCH/uv_envs/vllm_env/.venv/bin/activate"
export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_CACHE_ROOT="$SCRATCH/.cache/vllm_len2gatedres${POP}_${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$SCRATCH/.triton_len2gatedres${POP}_${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/.inductor_len2gatedres${POP}_${SLURM_JOB_ID}"
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

export EGGROLL_FITNESS_MODE=gated
export EGGROLL_FORMAT_CHECK=soft

cd "$SCRATCH/eggroll-vllm"
[[ -d "$CKPT" ]] || { echo "checkpoint not found: $CKPT" >&2; exit 1; }

# Same OUTPUT_ROOT as the original run so new checkpoints (step_300, 350, ...)
# land next to the old ones. EVAL_STEPS limits the post-run merge to the NEW
# checkpoints only (the old ones are already merged).
BASE_MODEL=Qwen/Qwen3-1.7B USE_WANDB=1 WANDB_PROJECT=eggroll-h1 \
  POPULATION_SIZE="$POP" PROMPT_BATCH_SIZE=8 NUM_ITERATIONS=900 \
  MAX_TOKENS=2048 ENTROPY_TOPK=20 \
  RESUME_FROM="$CKPT" \
  EVAL_STEPS="400 500 600 700 800 899" \
  OUTPUT_ROOT="runs/h1_curriculum_len2_gated" \
  STAGE=2 ./run_h1_curriculum.sh
