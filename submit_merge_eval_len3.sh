#!/bin/bash
#SBATCH --job-name=eggroll-h1-len3-merge-eval
#SBATCH --output=logs/merge-eval-len3-%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00

# Stage-3 job 6634348 hit its walltime at step 530, after checkpoint_step_500 but
# before run_h1_curriculum.sh's end-of-run merge. This job (1) merges every saved
# stage-3 checkpoint into an HF model dir exactly as the curriculum script would,
# then (2) runs the held-out eval on horizons 1-4 via submit_eval_len2_correctonly.sh
# (invoked as a plain script inside this allocation).
ARM="${ARM:-fp32master_sigma0.001}"
ROOT="runs/h1_curriculum_len3_correctonly_${ARM}/stage3_len3"
BASE_MODEL="${BASE_MODEL:-runs/h1_curriculum_len2_correctonly_fp32master_sigma0.001/stage2_len2/merged/step_500}"

set -euo pipefail
mkdir -p logs results
source "$SCRATCH/uv_envs/vllm_env/.venv/bin/activate"
export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$SCRATCH/eggroll-vllm"

for ckpt in $(ls -d "$ROOT"/checkpoints/checkpoint_step_* | sed 's/.*checkpoint_step_//' | sort -n); do
  src="$ROOT/checkpoints/checkpoint_step_${ckpt}"; out="$ROOT/merged/step_${ckpt}"
  [[ -f "$src/model_weights.safetensors" ]] || { echo "skip $src (no weights)"; continue; }
  [[ -f "$out/config.json" ]] && { echo "already merged: $out"; continue; }
  echo "--- merging $src -> $out ---"
  python merge_checkpoint.py --checkpoint "$src" --model-name "$BASE_MODEL" --output-dir "$out"
done

PREFIX=len3_conly ARM="$ARM" MERGED="$ROOT/merged" \
DATASETS="GSM-LongHorizon/test_len_1.jsonl GSM-LongHorizon/test_len_2.jsonl GSM-LongHorizon/test_len_3.jsonl GSM-LongHorizon/test_len_4.jsonl" \
  bash submit_eval_len2_correctonly.sh
