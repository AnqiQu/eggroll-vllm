#!/bin/bash
#SBATCH --job-name=eggroll-h1-q25-3b-conly
#SBATCH --output=logs/eggroll-h1-q25-3b-conly-%j.out
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=72:00:00

# Same-base-model comparison with h1: their table is Qwen2.5-3B-Instruct trained
# with DrGRPO through curriculum stages 1..N. This job runs ONE stage end to end
# with the ES recipe that worked on Qwen3-1.7B (correctness-only fitness, fp32
# master, sigma 1e-3, lr 2e-4, pop 256, prompt batch 8, r 1, 600 steps):
#
#   1. train (auto-resuming from the newest checkpoint_step_* if the stage was
#      interrupted; up to MAX_ATTEMPTS attempts, so a transient vLLM/Ray crash
#      such as job 6661559's segfault at step 542 does not lose the stage)
#   2. merge every checkpoint into an HF model dir (run_h1_curriculum.sh does this)
#   3. held-out eval of base + every merged step on horizons 1-4
#      -> results/q25_3b_len<STAGE>_fp32master_sigma0.001_<label>.json
#   4. pick the best step by h1's rule (highest combined len_1+len_2+len_3) and
#      write its merged path to <stage dir>/BEST, which the next stage reads.
#
#   STAGE=1 sbatch submit_h1_qwen25_3b_stage_correctonly.sh
#   STAGE=2 sbatch --dependency=afterok:<job1> submit_h1_qwen25_3b_stage_correctonly.sh
#   (or submit_h1_qwen25_3b_chain.sh, which queues stages 1-4 with dependencies)
#
# Env: STAGE (1..5, required); BASE_MODEL (stage >1; default = previous stage's BEST);
#      RESUME_FROM (default: newest checkpoint of this stage, FRESH=1 to ignore);
#      MAX_ATTEMPTS (3); SIGMA, LEARNING_RATE, FP32_MASTER (1).
# Stage 1 on 3B: ~45 min per 50 steps at 768 tokens; later stages are longer
# (more tokens per completion), hence the 72 h walltime (workq has no limit).
POP=256
STAGE="${STAGE:?Set STAGE=1..5}"
FP32_MASTER="${FP32_MASTER-1}"
[[ "$FP32_MASTER" == "0" ]] && FP32_MASTER=""
SIGMA="${SIGMA:-0.001}"
LEARNING_RATE="${LEARNING_RATE:-0.0002}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
NUM_ITERATIONS="${NUM_ITERATIONS:-600}"

set -euo pipefail
mkdir -p logs results

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
ARM="$SUFFIX"
OUTPUT_ROOT="runs/h1_qwen25_3b_correctonly_${SUFFIX}"
STAGE_DIR="$OUTPUT_ROOT/stage${STAGE}_len${STAGE}"
CKPT_DIR="$STAGE_DIR/checkpoints"
MERGED_DIR="$STAGE_DIR/merged"
PREFIX="q25_3b_len${STAGE}"
BASE_LABEL_MODEL="Qwen/Qwen2.5-3B-Instruct"

if [[ "$STAGE" -eq 1 ]]; then
  BASE="$BASE_LABEL_MODEL"
else
  PREV_BEST="$OUTPUT_ROOT/stage$((STAGE-1))_len$((STAGE-1))/BEST"
  if [[ -n "${BASE_MODEL:-}" ]]; then
    BASE="$BASE_MODEL"
  elif [[ -s "$PREV_BEST" ]]; then
    BASE="$(cat "$PREV_BEST")"
    echo "base model from $PREV_BEST: $BASE"
  else
    echo "Stage $STAGE needs BASE_MODEL=<merged dir> or the previous stage's $PREV_BEST" >&2; exit 1
  fi
  [[ -f "$BASE/config.json" ]] || { echo "base model dir not found: $BASE" >&2; exit 1; }
fi

latest_ckpt() {   # newest checkpoint_step_N with weights, or empty
  ls -d "$CKPT_DIR"/checkpoint_step_* 2>/dev/null | sed 's/.*checkpoint_step_//' | sort -n \
    | while read -r s; do [[ -f "$CKPT_DIR/checkpoint_step_$s/model_weights.safetensors" ]] && echo "$s"; done | tail -1
}

# ---- 1+2: train (with retries / auto-resume) and merge ----
FINAL_STEP=$((NUM_ITERATIONS-1))
if [[ -f "$MERGED_DIR/step_${FINAL_STEP}/config.json" ]]; then
  echo "stage $STAGE already trained and merged (found $MERGED_DIR/step_${FINAL_STEP}); skipping to eval"
else
  attempt=1
  while :; do
    resume="${RESUME_FROM:-}"
    if [[ -z "$resume" && "${FRESH:-0}" != "1" ]]; then
      s="$(latest_ckpt)"
      [[ -n "$s" ]] && resume="$CKPT_DIR/checkpoint_step_$s"
    fi
    echo "=== stage $STAGE attempt $attempt/$MAX_ATTEMPTS  resume: ${resume:-none} ==="
    ray stop --force >/dev/null 2>&1 || true
    if RESUME_FROM="$resume" BASE_MODEL="$BASE" USE_WANDB=1 WANDB_PROJECT=eggroll-h1 \
         POPULATION_SIZE="$POP" PROMPT_BATCH_SIZE=8 NUM_ITERATIONS="$NUM_ITERATIONS" \
         SIGMA="$SIGMA" LEARNING_RATE="$LEARNING_RATE" \
         ENTROPY_TOPK=20 FP32_MASTER="$FP32_MASTER" \
         OUTPUT_ROOT="$OUTPUT_ROOT" STAGE="$STAGE" ./run_h1_curriculum.sh; then
      break
    else
      rc=$?
    fi
    echo "!!! stage $STAGE attempt $attempt failed (exit $rc)"
    unset RESUME_FROM FRESH   # next attempt resumes from whatever was saved
    attempt=$((attempt+1))
    [[ "$attempt" -le "$MAX_ATTEMPTS" ]] || { echo "giving up after $MAX_ATTEMPTS attempts" >&2; exit 1; }
    sleep 60
  done
fi
ray stop --force >/dev/null 2>&1 || true

# ---- 3: held-out eval (base + every merged step, horizons 1-4) ----
PREFIX="$PREFIX" ARM="$ARM" MERGED="$MERGED_DIR" BASE_LABEL_MODEL="$BASE_LABEL_MODEL" \
DATASETS="GSM-LongHorizon/test_len_1.jsonl GSM-LongHorizon/test_len_2.jsonl GSM-LongHorizon/test_len_3.jsonl GSM-LongHorizon/test_len_4.jsonl" \
  bash submit_eval_len2_correctonly.sh

# ---- 4: select the best step (h1's rule: highest combined len_1..len_3) ----
python - "$PREFIX" "$ARM" "$MERGED_DIR" "$STAGE_DIR/BEST" <<'PYEOF'
import glob, json, os, sys
prefix, arm, merged, best_file = sys.argv[1:5]
scores = {}
for path in glob.glob(f"results/{prefix}_{arm}_step_*.json"):
    step = int(os.path.basename(path).rsplit("_", 1)[1][:-5])
    acc = {}
    for _m, dsets in json.load(open(path)).items():
        for d, v in dsets.items():
            acc[os.path.basename(d).replace("test_", "").replace(".jsonl", "")] = v["accuracy"] * 100
    if all(k in acc for k in ("len_1", "len_2", "len_3")):
        scores[step] = (sum(acc[k] for k in ("len_1", "len_2", "len_3")), acc)
if not scores:
    sys.exit("no evaluated steps found")
best = max(scores, key=lambda s: (scores[s][0], -s))
path = f"{merged}/step_{best}"
assert os.path.isfile(f"{path}/config.json"), path
with open(best_file, "w") as f:
    f.write(path + "\n")
print(f"BEST step_{best}: combined len_1..3 = {scores[best][0]:.1f}  {scores[best][1]}")
print(f"wrote {best_file} -> {path}")
PYEOF
echo "STAGE $STAGE DONE"
