#!/bin/bash
#SBATCH --job-name=eggroll-pop-eval
#SBATCH --output=logs/eggroll-pop-eval-%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00

# 1 GPU requested as --gpus-per-node=1 + --mem, like submit_eval_rolling.sh: with --gpus=1
# vLLM failed at init with "CUDA-capable device(s) is/are busy or unavailable" (job 7766548).
#
# Population evaluation (population_eval.py): score N EGGROLL-shaped perturbations
# of each model (one vLLM engine, members as rank-1 LoRA adapters) plus N
# temperature samples of the unperturbed model, and summarise coverage /
# rescued / broken / pair signal / GRPO-group signal. Same NOISE_SEED for every
# model, so all models get the same perturbation draws (paired comparison).
#
# MODELS is a space-separated list of label:path. One results file per model:
#   results/pop_<TAG>_<label>.json   (skipped if it exists)
#
# --- Recipe A: direct-jump probe (before submit_h1_qwen25_3b_onlylong.sh) ---
# Is there any learning signal at h=5 from the base model, for ES vs for GRPO?
# Compare, on train_len_5:  sigma block  frac_problems_with_pair_signal / pair_disagree  (ES)
#                           sampled block group_signal@16                              (GRPO)
#   TAG=probe_len5 MODELS="base:Qwen/Qwen2.5-3B-Instruct" \
#   DATASETS="GSM-LongHorizon/train_len_5.jsonl GSM-LongHorizon/train_len_2.jsonl" \
#   NUM_SAMPLES=128 N_MEMBERS=32 SIGMAS="0.001" sbatch submit_population_eval.sh
#
# --- Recipe B: "population as the result" (ES vs GRPO stage-k on horizon k+1) ---
#   R=runs; TAG=s2_on_len3 \
#   MODELS="es_s2:$(cat $R/h1_qwen25_3b_correctonly_fp32master_sigma0.001/stage2_len2/BEST) \
#           grpo_s2:$(cat $R/grpo_h1_qwen25_3b_plain/stage2_len2/BEST) \
#           grponoise_s2:$(cat $R/grpo_h1_qwen25_3b_noise_sigma0.001_r1/stage2_len2/BEST)" \
#   DATASETS="GSM-LongHorizon/test_len_3.jsonl" SIGMAS="0.0005 0.001 0.002" \
#   sbatch submit_population_eval.sh
# Compare coverage@N and rescued at MATCHED member_acc_mean across the sigma sweep
# (a sharper model loses more accuracy at the same sigma).
#
# Env: MODELS (required), DATASETS, TAG (pop), SIGMAS ("0.001"), N_MEMBERS (32),
#      NUM_SAMPLES (all), NOISE_SEED (0), SAMPLE_TEMPERATURE (1.0), MAX_NEW_TOKENS (2048).
# Cost: N_MEMBERS x problems greedy generations per sigma, plus N_MEMBERS samples
# of the centre; roughly 30-60 min per sigma per ~300-problem split for a 3B model.
MODELS="${MODELS:?Set MODELS=\"label:path ...\"}"
DATASETS="${DATASETS:-GSM-LongHorizon/test_len_3.jsonl}"
TAG="${TAG:-pop}"
SIGMAS="${SIGMAS:-0.001}"
N_MEMBERS="${N_MEMBERS:-32}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
NOISE_SEED="${NOISE_SEED:-0}"
SAMPLE_TEMPERATURE="${SAMPLE_TEMPERATURE:-1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"

set -euo pipefail
mkdir -p logs results
source "$SCRATCH/uv_envs/vllm_env/.venv/bin/activate"
export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_CACHE_ROOT="$SCRATCH/.cache/vllm_popeval_${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$SCRATCH/.triton_popeval_${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/.inductor_popeval_${SLURM_JOB_ID}"
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
cd "$SCRATCH/eggroll-vllm"

ns=(); [[ -n "$NUM_SAMPLES" ]] && ns=(--num-samples "$NUM_SAMPLES")
for entry in $MODELS; do
  label="${entry%%:*}"; model="${entry#*:}"
  out="results/pop_${TAG}_${label}.json"
  if [[ -s "$out" ]]; then echo "========== SKIP ${label} (exists: $out) =========="; continue; fi
  echo "========== POPULATION EVAL ${label} (${model}) =========="
  python population_eval.py \
    --model "$model" --label "$label" \
    --datasets $DATASETS \
    --n-members "$N_MEMBERS" --sigmas $SIGMAS --rank 1 --seed "$NOISE_SEED" \
    --sample-temperature "$SAMPLE_TEMPERATURE" --max-new-tokens "$MAX_NEW_TOKENS" \
    "${ns[@]}" --out "$out"
done
echo "POPULATION EVAL DONE"
