#!/bin/bash
#SBATCH --job-name=eggroll-h1-len2-gated-eval
#SBATCH --output=logs/eval-len2-gated-%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00

# Held-out eval for the len-2 GATED run (train job 6455002).
# Evaluates BASE Qwen3-1.7B + every merged checkpoint (step_50..299) on the full
# test_len_1/2/3 sets, greedy (temp 0). This answers the question the training
# curves raised: did ANY checkpoint beat base on held-out correctness, or -- as
# frac_correct sitting flat at ~37% the whole run suggested -- did ES only ever
# fix FORMAT while correctness never moved off the base rate?
#
# One FRESH python process per model (same pattern as submit_eval_stage1.sh):
# vLLM does not reliably release GPU memory across models within a single
# process, so we deliberately do NOT pass all models to one --models call.
# Per-model JSON lands in results/; a comparison table is printed at the end of
# the log. ~10 min/model x 7 -> comfortably inside the 3h walltime; --gpus=1
# (tp=1 is right for 1.7B) also schedules far faster than a whole node.

set -euo pipefail
mkdir -p logs results

source "$SCRATCH/uv_envs/vllm_env/.venv/bin/activate"
export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_CACHE_ROOT="$SCRATCH/.cache/vllm_len2eval_${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$SCRATCH/.triton_len2eval_${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/.inductor_len2eval_${SLURM_JOB_ID}"
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

cd "$SCRATCH/eggroll-vllm"
MERGED=runs/h1_curriculum_len2_gated/stage2_len2/merged
DATASETS="GSM-LongHorizon/test_len_1.jsonl GSM-LongHorizon/test_len_2.jsonl GSM-LongHorizon/test_len_3.jsonl"

# label:model_path -- base first as the reference, then checkpoints in order.
MODELS=(
  "base:Qwen/Qwen3-1.7B"
  "step_50:${MERGED}/step_50"
  "step_100:${MERGED}/step_100"
  "step_150:${MERGED}/step_150"
  "step_200:${MERGED}/step_200"
  "step_250:${MERGED}/step_250"
  "step_299:${MERGED}/step_299"
)

for entry in "${MODELS[@]}"; do
  label="${entry%%:*}"
  model="${entry#*:}"
  echo "========== EVAL ${label} (${model}) =========="
  python h1_gsm_eval.py \
    --models "${model}" \
    --datasets ${DATASETS} \
    --instruct --tp 1 \
    --max_new_tokens 2048 \
    --out_file "results/len2_gated_${label}.json"
done

echo
echo "================= COMPARISON TABLE (accuracy %) ================="
python - <<'PYEOF' || echo "(aggregation failed; per-model JSONs are still saved in results/)"
import json, glob, os
rows = {}
for path in sorted(glob.glob("results/len2_gated_*.json")):
    label = os.path.basename(path)[len("len2_gated_"):-len(".json")]
    with open(path) as f:
        data = json.load(f)
    for _model, dsets in data.items():
        rows[label] = {
            os.path.basename(d).replace("test_", "").replace(".jsonl", ""): v.get("accuracy", 0.0) * 100
            for d, v in dsets.items()
        }
order = ["base", "step_50", "step_100", "step_150", "step_200", "step_250", "step_299"]
cols = ["len_1", "len_2", "len_3"]
print(f"{'model':10s} " + " ".join(f"{c:>8s}" for c in cols))
for label in order:
    if label in rows:
        print(f"{label:10s} " + " ".join(f"{rows[label].get(c, float('nan')):8.2f}" for c in cols))
print("\n(base is Qwen3-1.7B; len_2 is the trained horizon. Look for step_* > base on len_2.)")
PYEOF
echo "ALL EVALS DONE"
