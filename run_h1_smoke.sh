#!/bin/bash
# run_h1_smoke.sh -- tiny end-to-end debug run of the h1 EGGROLL pipeline.
#
# Purpose: validate the whole stack on your GPU in a few minutes BEFORE
# committing to a real curriculum run. It exercises every moving part at
# minimal size:
#     prompt build -> vLLM multi-LoRA generation -> h1 reward/fitness
#     -> EGGROLL ES update -> checkpoint -> merge to HF -> gsm_eval
#
# It is deliberately cheap: population 8, 2 prompts/iter, 4 ES iterations,
# 256-token completions, on a 32-question subset. Numbers here are NOT
# meaningful -- you are only checking that nothing throws.
#
# Prereqs:
#   * Environment installed (see README "Setup on a single A100").
#   * Model weights cached (default Qwen/Qwen3-1.7B).
#   * GSM-LongHorizon/train_len_1.jsonl and test_len_1.jsonl present
#     (see README "Get the dataset").
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./run_h1_smoke.sh
#
# Override anything via env, e.g.:
#   MODEL=Qwen/Qwen2.5-3B-Instruct POPULATION_SIZE=16 ./run_h1_smoke.sh
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
DATA_DIR="${DATA_DIR:-GSM-LongHorizon}"
OUT="${OUTPUT_ROOT:-runs/smoke}"

# Tiny knobs (all overridable)
POPULATION_SIZE="${POPULATION_SIZE:-8}"    # must be even (antithetic pairs)
PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-2}"
NUM_ITERATIONS="${NUM_ITERATIONS:-4}"
SAVE_FREQ="${SAVE_FREQ:-2}"
MAX_TOKENS="${MAX_TOKENS:-256}"
SUB_DATASET_SIZE="${SUB_DATASET_SIZE:-32}"
EVAL_SAMPLES="${EVAL_SAMPLES:-8}"

echo "############################################################"
echo "# h1 EGGROLL smoke test"
echo "#   model      : $MODEL"
echo "#   data dir   : $DATA_DIR"
echo "#   output     : $OUT"
echo "#   pop/pbs/it : $POPULATION_SIZE / $PROMPT_BATCH_SIZE / $NUM_ITERATIONS"
echo "#   max_tokens : $MAX_TOKENS   subset: $SUB_DATASET_SIZE"
echo "############################################################"

[[ -f "$DATA_DIR/train_len_1.jsonl" ]] || {
    echo "Missing $DATA_DIR/train_len_1.jsonl (see README: get the dataset)" >&2; exit 1; }

# ---- 1) tiny stage-1 train + merge via the real curriculum runner ----------
STAGE=1 \
BASE_MODEL="$MODEL" \
DATA_DIR="$DATA_DIR" \
OUTPUT_ROOT="$OUT" \
POPULATION_SIZE="$POPULATION_SIZE" \
PROMPT_BATCH_SIZE="$PROMPT_BATCH_SIZE" \
NUM_ITERATIONS="$NUM_ITERATIONS" \
SAVE_FREQ="$SAVE_FREQ" \
MAX_TOKENS="$MAX_TOKENS" \
SUB_DATASET_SIZE="$SUB_DATASET_SIZE" \
    ./run_h1_curriculum.sh

# ---- 2) tiny eval of the last merged checkpoint ----------------------------
MERGED_DIR="$OUT/stage1_len1/merged"
last_step="$(ls -d "$MERGED_DIR"/step_* 2>/dev/null | sed 's#.*/step_##' | sort -n | tail -1)"
if [[ -z "$last_step" ]]; then
    echo "No merged checkpoint found under $MERGED_DIR" >&2; exit 1
fi
LAST_MERGED="$MERGED_DIR/step_${last_step}"

echo
echo "############################################################"
echo "# Smoke eval: $LAST_MERGED on test_len_1 (${EVAL_SAMPLES} samples)"
echo "############################################################"
python h1_gsm_eval.py \
    --models "$LAST_MERGED" \
    --datasets "$DATA_DIR/test_len_1.jsonl" \
    --instruct --tp 1 \
    --num_samples "$EVAL_SAMPLES" \
    --max_new_tokens "$MAX_TOKENS" \
    --out_file "$OUT/smoke_eval.json"

echo
echo "############################################################"
echo "# SMOKE TEST PASSED -- the full pipeline runs end to end."
echo "# Now launch a real run: STAGE=1 ./run_h1_curriculum.sh"
echo "############################################################"
