#!/bin/bash
#SBATCH --job-name=eggroll-h1-s1-eval
#SBATCH --output=logs/eval-s1-%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00

set -euo pipefail
mkdir -p logs results

source "$SCRATCH/uv_envs/vllm_env/.venv/bin/activate"
export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_CACHE_ROOT="$SCRATCH/.cache/vllm_eval_${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$SCRATCH/.triton_eval_${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/.inductor_eval_${SLURM_JOB_ID}"
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

cd "$SCRATCH/eggroll-vllm"
MERGED=runs/h1_curriculum/stage1_len1/merged
DATASETS="GSM-LongHorizon/test_len_1.jsonl GSM-LongHorizon/test_len_2.jsonl GSM-LongHorizon/test_len_3.jsonl"

for step in 200 250 299; do
  echo "========== EVAL step ${step} =========="
  python h1_gsm_eval.py \
    --models "${MERGED}/step_${step}" \
    --datasets ${DATASETS} \
    --instruct --tp 1 \
    --out_file "results/stage1_len1_step_${step}.json"
done
echo "ALL EVALS DONE"
