#!/bin/bash
#SBATCH --job-name=eggroll-h1-q25-3b-conly
#SBATCH --output=logs/eggroll-h1-q25-3b-conly-%j.out
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00

# Same-base-model comparison with h1: their table is Qwen2.5-3B-Instruct trained
# with DrGRPO through curriculum stages 1..N. This runs ONE stage with the ES
# recipe that worked on Qwen3-1.7B (correctness-only fitness, fp32 master,
# sigma 1e-3, lr 2e-4, pop 256, prompt batch 8, r 1, 600 steps).
#
#   STAGE=1 sbatch submit_h1_qwen25_3b_stage_correctonly.sh
#   STAGE=2 BASE_MODEL=runs/h1_qwen25_3b_correctonly_fp32master_sigma0.001/stage1_len1/merged/step_<BEST> \
#     sbatch submit_h1_qwen25_3b_stage_correctonly.sh
#   (BEST = highest combined len_1..len_3 held-out accuracy, h1's rule.)
#
# Eval:  PREFIX=q25_3b_len<STAGE> ARM=fp32master_sigma0.001 \
#        MERGED=runs/h1_qwen25_3b_correctonly_fp32master_sigma0.001/stage<STAGE>_len<STAGE>/merged \
#        BASE_LABEL_MODEL=Qwen/Qwen2.5-3B-Instruct sbatch submit_eval_len2_correctonly.sh
# Qwen2.5-3B is ~2x the 1.7B per step; 24 h walltime, checkpoints every 50.
POP=256
STAGE="${STAGE:?Set STAGE=1..5}"
FP32_MASTER="${FP32_MASTER-1}"
[[ "$FP32_MASTER" == "0" ]] && FP32_MASTER=""
SIGMA="${SIGMA:-0.001}"
LEARNING_RATE="${LEARNING_RATE:-0.0002}"

set -euo pipefail
mkdir -p logs

source "$SCRATCH/uv_envs/vllm_env/.venv/bin/activate"
export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_CACHE_ROOT="$SCRATCH/.cache/vllm_q253b_s${STAGE}_${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$SCRATCH/.triton_q253b_s${STAGE}_${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/.inductor_q253b_s${STAGE}_${SLURM_JOB_ID}"
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

export EGGROLL_FITNESS_MODE=correctness

cd "$SCRATCH/eggroll-vllm"

SUFFIX="bf16"; [[ -n "$FP32_MASTER" ]] && SUFFIX="fp32master"
SUFFIX="${SUFFIX}_sigma${SIGMA}"

if [[ "$STAGE" -eq 1 ]]; then
  BASE="Qwen/Qwen2.5-3B-Instruct"
else
  BASE="${BASE_MODEL:?Stage >1 needs BASE_MODEL=<merged dir from the previous stage>}"
  [[ -f "$BASE/config.json" ]] || { echo "base model dir not found: $BASE" >&2; exit 1; }
fi

BASE_MODEL="$BASE" USE_WANDB=1 WANDB_PROJECT=eggroll-h1 \
  POPULATION_SIZE="$POP" PROMPT_BATCH_SIZE=8 NUM_ITERATIONS=600 \
  SIGMA="$SIGMA" LEARNING_RATE="$LEARNING_RATE" \
  ENTROPY_TOPK=20 FP32_MASTER="$FP32_MASTER" \
  OUTPUT_ROOT="runs/h1_qwen25_3b_correctonly_${SUFFIX}" \
  STAGE="$STAGE" ./run_h1_curriculum.sh
