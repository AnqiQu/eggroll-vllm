#!/bin/bash
#SBATCH --job-name=eggroll-probe
#SBATCH --output=logs/eggroll-probe-%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00

# "Try it on your own code": es_reference.py is an independent re-implementation
# of EGGROLL in plain PyTorch/transformers (no vLLM, no Ray, no PEFT). This job
# runs its PROBE mode -- no training, it just measures, for the base model and
# for the gated run's final checkpoint, how often a sigma-sized low-rank
# perturbation changes (a) the greedy text, (b) the format, (c) the correctness
# of the answer, and reports the greedy top1-top2 margin / entropy along the
# path. It answers two questions directly:
#   * At sigma=1e-3, is correctness even perturbable, or only format?
#     (P(correct changes) per pair is the raw signal ES has to work with.)
#   * Did the gated run collapse? (margin up / entropy down / P(text changes)
#     down on step_299 vs base = entropy collapse.)
# ~1-2 h per model at these settings on one GH200.

set -euo pipefail
mkdir -p logs results

source "$SCRATCH/uv_envs/vllm_env/.venv/bin/activate"
export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$SCRATCH/eggroll-vllm"
DATA=GSM-LongHorizon/train_len_2.jsonl

for entry in "base:Qwen/Qwen3-1.7B" \
             "gated_step_299:runs/h1_curriculum_len2_gated/stage2_len2/merged/step_299"; do
  label="${entry%%:*}"; model="${entry#*:}"
  [[ -d "$model" || "$label" == "base" ]] || { echo "skip $label ($model missing)"; continue; }
  echo "========== PROBE ${label} (${model}) =========="
  python es_reference.py probe \
    --model "$model" --data "$DATA" --reward-mode float \
    --n-prompts 32 --n-perturb 8 --sigmas 0.001 0.003 0.01 0.03 \
    --rank 1 --max-new-tokens 1024 \
    --out "results/probe_len2_${label}.json"
done
echo "PROBE DONE"
