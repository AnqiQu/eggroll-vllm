#!/bin/bash
# "Jump straight to the longest horizon": EGGROLL from the untouched
# Qwen2.5-3B-Instruct directly on train_len_5 (h=Hmax), no curriculum. This is
# the ES counterpart of h1's "Only-Long" GRPO baseline, which trains only on
# h=Hmax with the same compute as the 5-stage curriculum, from the same base:
#
#   h1 Only-Long (GRPO)   82.71 / 43.36 / 20.41 / 3.22 / 1.49   on L1..L5
#   h1 Len-5 curriculum   85.97 / 58.51 / 36.39 / 18.77 / 9.82
#
# Same recipe as the curriculum stages (correctness-only, fp32 master, sigma 1e-3,
# lr 2e-4, pop 256, 8 prompts/step, r 1, 1536-token budget of stage 5).
# NUM_ITERATIONS=600 (2026-10-21): one curriculum stage's budget, evaluated every
# 100 steps. h1's equal-compute rule would be 3000 (= 5 x 600), but the curriculum's
# best steps were 50/500/550/100 and GRPO Only-Long peaked at step 50; the lr is
# constant, so a longer run is this one resumed with a larger NUM_ITERATIONS.
#
# FITNESS picks the objective (two arms, separate run dirs / result prefixes):
#   correctness  h1 correctness only, stripped prompt: the ES curriculum's recipe
#   total        h1's full reward (correctness + xmlcount + soft/strict format +
#                float), unstripped system prompt: the reward GRPO Only-Long uses
#
# Before spending several days of 8 GPUs on this, run the cheap signal probe
# (submit_population_eval.sh, "direct-jump probe" example): if antithetic pairs
# at sigma 1e-3 almost never disagree on horizon-5 problems, this run cannot learn.
#
#   ./submit_h1_qwen25_3b_onlylong.sh                  # correctness arm, one job on 2 nodes
#   FITNESS=total ./submit_h1_qwen25_3b_onlylong.sh    # format arm
#   NUM_ITERATIONS=1200 JOBS=2 ./submit_h1_qwen25_3b_onlylong.sh   # extend (resumes)
#   AFTER=<jobid> ./submit_h1_qwen25_3b_onlylong.sh    # queue behind the curriculum chain
#
# Each job is submit_h1_qwen25_3b_stage_correctonly.sh (idempotent: resumes from
# the newest checkpoint, then merges EVAL_STEPS and evaluates on horizons 1-5
# once training is complete). Results: results/q25_3b_onlylen5[_total]_fp32master_sigma0.001_step_*.json
set -euo pipefail
NODES="${NODES:-2}"
JOBS="${JOBS:-1}"
NUM_ITERATIONS="${NUM_ITERATIONS:-600}"
SIGMA="${SIGMA:-0.001}"
FITNESS="${FITNESS:-correctness}"
case "$FITNESS" in
  correctness) TAG="";       export EGGROLL_FITNESS_MODE=correctness EGGROLL_UNSTRIPPED_PROMPT=0 ;;
  total)       TAG="_total"; export EGGROLL_FITNESS_MODE=total EGGROLL_UNSTRIPPED_PROMPT=1 ;;
  *) echo "FITNESS must be correctness or total" >&2; exit 1 ;;
esac
export STAGE=5 NUM_ITERATIONS SIGMA
export BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"
export OUTPUT_ROOT="${OUTPUT_ROOT:-runs/h1_qwen25_3b_onlylong${TAG}_fp32master_sigma${SIGMA}}"
export PREFIX="${PREFIX:-q25_3b_onlylen5${TAG}}"
export SAVE_FREQ="${SAVE_FREQ:-100}"   # 3B checkpoints are ~6 GB; 100 keeps resume loss small
# the trainer saves at multiples of SAVE_FREQ and at the final step
export EVAL_STEPS="${EVAL_STEPS:-$(seq -s ' ' 100 100 $((NUM_ITERATIONS-1))) $((NUM_ITERATIONS-1))}"
export EVAL_DATASETS="GSM-LongHorizon/test_len_1.jsonl GSM-LongHorizon/test_len_2.jsonl GSM-LongHorizon/test_len_3.jsonl GSM-LongHorizon/test_len_4.jsonl GSM-LongHorizon/test_len_5.jsonl"
export FRESH=0

dep="${AFTER:-}"
for k in $(seq 1 "$JOBS"); do
  args=(--job-name="eggroll-h1-q25-3b-onlylong${TAG}" --nodes="$NODES" --parsable)
  [[ -n "$dep" ]] && args+=(--dependency="afterany:${dep}")
  jid="$(sbatch "${args[@]}" submit_h1_qwen25_3b_stage_correctonly.sh)"
  echo "only-long ($FITNESS) job $k/$JOBS -> $jid ${dep:+(after $dep)}"
  dep="$jid"
done
echo "outputs: $OUTPUT_ROOT  (results/${PREFIX}_*)"
