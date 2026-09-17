#!/bin/bash
#SBATCH --job-name=eggroll-h1-s3-len3-conly
#SBATCH --output=logs/eggroll-h1-s3-len3-conly-%j.out
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=14:00:00

# Curriculum STAGE 3 (horizon 3) with the recipe that worked on horizon 2:
# correctness-only fitness, fp32 master, sigma 1e-3, lr 2e-4, pop 256, batch 8,
# 600 steps. Base model = the horizon-2 checkpoint with the best overall
# horizon-1..3 accuracy (h1's selection rule): step_500 of the fp32 arm
# (len_1/2/3 = 76.8 / 50.8 / 31.6, see results/len2_conly_fp32master_sigma0.001_*).
# Stage default completion length is 1280 (h1's recommendation for horizon 3).
# Eval afterwards, including the untrained horizon 4:
#   PREFIX=len3_conly ARM=fp32master_sigma0.001 \
#   MERGED=runs/h1_curriculum_len3_correctonly_fp32master_sigma0.001/stage3_len3/merged \
#   DATASETS="GSM-LongHorizon/test_len_1.jsonl GSM-LongHorizon/test_len_2.jsonl GSM-LongHorizon/test_len_3.jsonl GSM-LongHorizon/test_len_4.jsonl" \
#   sbatch submit_eval_len2_correctonly.sh
POP=256
BASE="${BASE_MODEL:-runs/h1_curriculum_len2_correctonly_fp32master_sigma0.001/stage2_len2/merged/step_500}"
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
export VLLM_CACHE_ROOT="$SCRATCH/.cache/vllm_len3conly${POP}_${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$SCRATCH/.triton_len3conly${POP}_${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/.inductor_len3conly${POP}_${SLURM_JOB_ID}"
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

export EGGROLL_FITNESS_MODE=correctness

cd "$SCRATCH/eggroll-vllm"
[[ -f "$BASE/model.safetensors" || -f "$BASE/config.json" ]] || { echo "base model dir not found: $BASE" >&2; exit 1; }

SUFFIX="bf16"; [[ -n "$FP32_MASTER" ]] && SUFFIX="fp32master"
SUFFIX="${SUFFIX}_sigma${SIGMA}"

BASE_MODEL="$BASE" USE_WANDB=1 WANDB_PROJECT=eggroll-h1 \
  POPULATION_SIZE="$POP" PROMPT_BATCH_SIZE=8 NUM_ITERATIONS=600 \
  SIGMA="$SIGMA" LEARNING_RATE="$LEARNING_RATE" \
  ENTROPY_TOPK=20 FP32_MASTER="$FP32_MASTER" \
  OUTPUT_ROOT="runs/h1_curriculum_len3_correctonly_${SUFFIX}" \
  STAGE=3 ./run_h1_curriculum.sh
