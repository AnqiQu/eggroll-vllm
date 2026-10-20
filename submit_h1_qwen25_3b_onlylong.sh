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
# NUM_ITERATIONS=3000 = the ES curriculum's 5 x 600 steps (equal compute in ES
# steps / rollouts, like h1's equal-compute baselines). The merged + evaluated
# checkpoints sit at the curriculum's stage boundaries (600, 1200, ...), so the
# direct run can be compared with curriculum stage k at the SAME cumulative budget.
#
# Before spending several days of 8 GPUs on this, run the cheap signal probe
# (submit_population_eval.sh, "direct-jump probe" example): if antithetic pairs
# at sigma 1e-3 almost never disagree on horizon-5 problems, this run cannot learn.
#
#   ./submit_h1_qwen25_3b_onlylong.sh                  # 5 chained 24 h jobs on 2 nodes
#   JOBS=6 NODES=2 ./submit_h1_qwen25_3b_onlylong.sh
#   AFTER=<jobid> ./submit_h1_qwen25_3b_onlylong.sh    # queue behind the curriculum chain
#
# Each job is submit_h1_qwen25_3b_stage_correctonly.sh (idempotent: resumes from
# the newest checkpoint, then merges EVAL_STEPS and evaluates on horizons 1-5
# once training is complete). Results: results/q25_3b_onlylen5_fp32master_sigma0.001_step_*.json
set -euo pipefail
NODES="${NODES:-2}"
JOBS="${JOBS:-5}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3000}"
SIGMA="${SIGMA:-0.001}"
export STAGE=5 NUM_ITERATIONS SIGMA
export BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"
export OUTPUT_ROOT="${OUTPUT_ROOT:-runs/h1_qwen25_3b_onlylong_fp32master_sigma${SIGMA}}"
export PREFIX="${PREFIX:-q25_3b_onlylen5}"
export SAVE_FREQ="${SAVE_FREQ:-100}"   # 3B checkpoints are ~6 GB; 100 keeps resume loss small
# the trainer saves at multiples of SAVE_FREQ and at the final step
export EVAL_STEPS="${EVAL_STEPS:-300 600 1200 1800 2400 $((NUM_ITERATIONS-1))}"
export EVAL_DATASETS="GSM-LongHorizon/test_len_1.jsonl GSM-LongHorizon/test_len_2.jsonl GSM-LongHorizon/test_len_3.jsonl GSM-LongHorizon/test_len_4.jsonl GSM-LongHorizon/test_len_5.jsonl"
export FRESH=0

dep="${AFTER:-}"
for k in $(seq 1 "$JOBS"); do
  args=(--job-name="eggroll-h1-q25-3b-onlylong" --nodes="$NODES" --parsable)
  [[ -n "$dep" ]] && args+=(--dependency="afterany:${dep}")
  jid="$(sbatch "${args[@]}" submit_h1_qwen25_3b_stage_correctonly.sh)"
  echo "only-long job $k/$JOBS -> $jid ${dep:+(after $dep)}"
  dep="$jid"
done
echo "outputs: $OUTPUT_ROOT  (results/${PREFIX}_*)"
