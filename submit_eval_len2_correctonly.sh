#!/bin/bash
#SBATCH --job-name=eggroll-h1-len2-conly-eval
#SBATCH --output=logs/eval-len2-conly-%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00

# Held-out eval for one arm of the correctness-only horizon-2 experiment
# (submit_h1_stage2_len2_correctonly.sh). Same pattern as submit_eval_len2_gated.sh:
# BASE Qwen3-1.7B + every merged checkpoint of the arm, on test_len_1/2/3, greedy,
# one fresh python process per model. Per-model JSON -> results/len2_conly_<ARM>_<label>.json
#
#   sbatch submit_eval_len2_correctonly.sh                       # ARM=fp32master_sigma0.001
#   ARM=bf16_sigma0.001 sbatch submit_eval_len2_correctonly.sh
#
# Then:  python analyze_heldout.py --splits len_2 \
#          results/len2_conly_<ARM>_base.json results/len2_conly_<ARM>_step_*.json
#
# The base file is re-evaluated per arm (10 min) so every comparison is paired on
# the same evaluator version; h1_gsm_eval.py's falsy-zero fix is already in.
ARM="${ARM:-fp32master_sigma0.001}"
# Optional overrides to evaluate a different run with the same machinery, e.g. the gated resume:
#   MERGED=runs/h1_curriculum_len2_gated/stage2_len2/merged ARM=gated sbatch submit_eval_len2_correctonly.sh
MERGED_OVERRIDE="${MERGED:-}"

set -euo pipefail
mkdir -p logs results

source "$SCRATCH/uv_envs/vllm_env/.venv/bin/activate"
export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_CACHE_ROOT="$SCRATCH/.cache/vllm_len2conlyeval_${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="$SCRATCH/.triton_len2conlyeval_${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/.inductor_len2conlyeval_${SLURM_JOB_ID}"
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

cd "$SCRATCH/eggroll-vllm"
MERGED="${MERGED_OVERRIDE:-runs/h1_curriculum_len2_correctonly_${ARM}/stage2_len2/merged}"
DATASETS="GSM-LongHorizon/test_len_1.jsonl GSM-LongHorizon/test_len_2.jsonl GSM-LongHorizon/test_len_3.jsonl"
[[ -d "$MERGED" ]] || { echo "merged dir not found: $MERGED" >&2; exit 1; }

MODELS=("base:Qwen/Qwen3-1.7B")
for d in $(ls -d "$MERGED"/step_* | awk -F/ '{print $NF" "$0}' | sort -t_ -k2 -n | cut -d" " -f2); do
  MODELS+=("$(basename "$d"):$d")
done
echo "ARM=$ARM  models: ${#MODELS[@]}"

for entry in "${MODELS[@]}"; do
  label="${entry%%:*}"
  model="${entry#*:}"
  out="results/len2_conly_${ARM}_${label}.json"
  if [[ -s "$out" ]]; then echo "========== SKIP ${label} (exists: $out) =========="; continue; fi
  echo "========== EVAL ${label} (${model}) =========="
  python h1_gsm_eval.py \
    --models "${model}" \
    --datasets ${DATASETS} \
    --instruct --tp 1 \
    --max_new_tokens 2048 \
    --out_file "$out"
done

echo
echo "================= COMPARISON TABLE (accuracy %) ================="
python - "$ARM" <<'PYEOF' || echo "(aggregation failed; per-model JSONs are still saved in results/)"
import json, glob, os, re, sys
arm = sys.argv[1]
prefix = f"results/len2_conly_{arm}_"
rows = {}
for path in glob.glob(prefix + "*.json"):
    label = os.path.basename(path)[len(os.path.basename(prefix)):-len(".json")]
    with open(path) as f:
        data = json.load(f)
    for _model, dsets in data.items():
        rows[label] = {os.path.basename(d).replace("test_", "").replace(".jsonl", ""): v.get("accuracy", 0.0) * 100
                       for d, v in dsets.items()}
def key(l):
    m = re.search(r"step_(\d+)", l); return -1 if not m else int(m.group(1))
cols = ["len_1", "len_2", "len_3"]
print(f"{'model':10s} " + " ".join(f"{c:>8s}" for c in cols))
for label in sorted(rows, key=key):
    print(f"{label:10s} " + " ".join(f"{rows[label].get(c, float('nan')):8.2f}" for c in cols))
print("\n(base is Qwen3-1.7B; len_2 is the trained horizon. Look for step_* > base on len_2.)")
PYEOF
echo
echo "================= PAIRED ANALYSIS (len_2) ================="
python analyze_heldout.py --splits len_2 "results/len2_conly_${ARM}_base.json" results/len2_conly_${ARM}_step_*.json || true
echo "ALL EVALS DONE"
