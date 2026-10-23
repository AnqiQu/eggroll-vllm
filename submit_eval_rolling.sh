#!/bin/bash
#SBATCH --job-name=eggroll-rolling-eval
#SBATCH --output=logs/eval-rolling-%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00

# Rolling held-out eval of ES runs that are still training. The stage job
# (submit_h1_qwen25_3b_stage_correctonly.sh) only evaluates after its last step,
# and the pop-5120 Only-Long runs take ~60 h over several 24 h jobs, so this job
# watches their checkpoint dirs and, as each checkpoint_step_N is saved, merges it
# into the stage's merged/step_N and evaluates it (one model per GPU, 4 at a time):
#   -> results/<prefix>_<ARM>_<label>.json, the same files the stage job writes, so
#      it skips them later. After each finished batch of a run it refreshes
#      results/summary and the "<prefix>_<ARM>-heldout" W&B run.
# The final step is left to the stage job (it merges every checkpoint right after
# training), so the two jobs never write the same merged dir at the same time.
#
#   sbatch submit_eval_rolling.sh                       # the two pop-5120 arms
#   RUNS="runs/<dir>:<prefix> ..." FINAL_STEP=599 sbatch submit_eval_rolling.sh
#   DRY_RUN=1 bash submit_eval_rolling.sh               # print what it would evaluate
#
# It polls every POLL seconds, stops when every run has saved FINAL_STEP and all
# earlier checkpoints are evaluated, and resubmits itself when its 24 h run out first.
RUNS="${RUNS:-runs/h1_qwen25_3b_onlylong_pop5120_fp32master_sigma0.001:q25_3b_onlylen5_pop5120 runs/h1_qwen25_3b_onlylong_pop5120_shuffle_fp32master_sigma0.001:q25_3b_onlylen5_pop5120_shuffle}"
ARM="${ARM:-fp32master_sigma0.001}"
STAGE_SUBDIR="${STAGE_SUBDIR:-stage5_len5}"
FINAL_STEP="${FINAL_STEP:-99}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
DATASETS="${DATASETS:-GSM-LongHorizon/test_len_1.jsonl GSM-LongHorizon/test_len_2.jsonl GSM-LongHorizon/test_len_3.jsonl GSM-LongHorizon/test_len_4.jsonl GSM-LongHorizon/test_len_5.jsonl}"
POLL="${POLL:-600}"
BUDGET_S="${BUDGET_S:-$((22*3600))}"   # start no new eval after this (one eval is < 1 h)
NGPU="${NGPU:-4}"
DRY_RUN="${DRY_RUN:-0}"

set -euo pipefail
cd "$SCRATCH/eggroll-vllm"
mkdir -p logs results
source "$SCRATCH/uv_envs/vllm_env/.venv/bin/activate"
export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE="${WANDB_MODE:-online}"
JOB="${SLURM_JOB_ID:-dry}"

# pending items, oldest step first across runs: "<sortkey> <prefix> <label> <ckpt|-> <merged dir|hub model>"
pending() {
  local entry dir prefix ckpt s
  for entry in $RUNS; do
    dir="${entry%%:*}/$STAGE_SUBDIR"; prefix="${entry#*:}"
    [[ -s "results/${prefix}_${ARM}_base.json" ]] || echo "-1 $prefix base - $BASE_MODEL"
    for ckpt in $({ ls -d "$dir"/checkpoints/checkpoint_step_* 2>/dev/null || true; }); do
      s="${ckpt##*_}"
      [[ "$s" -lt "$FINAL_STEP" ]] || continue
      [[ -s "results/${prefix}_${ARM}_step_${s}.json" ]] && continue
      # a checkpoint still being written: weights missing or modified in the last 2 min
      [[ -n "$(find "$ckpt/model_weights.safetensors" -mmin +2 2>/dev/null)" ]] || continue
      echo "$s $prefix step_$s $ckpt $dir/merged/step_$s"
    done
  done | sort -n
}

all_done() {   # every run saved FINAL_STEP and nothing earlier is left (twice-failed items count as done)
  local entry _s prefix label _rest
  for entry in $RUNS; do
    [[ -f "${entry%%:*}/$STAGE_SUBDIR/checkpoints/checkpoint_step_${FINAL_STEP}/model_weights.safetensors" ]] || return 1
  done
  while read -r _s prefix label _rest; do
    [[ -n "$label" && ${fails[$prefix/$label]:-0} -lt 2 ]] && return 1
  done < <(pending)
  return 0
}

declare -A pid_of item_of fails   # gpu -> pid / pending line; "<prefix>/<label>" -> failures
if [[ "$DRY_RUN" == "1" ]]; then pending; all_done && echo "(all done)"; exit 0; fi

run_one() {   # gpu prefix label ckpt model: merge if needed, then evaluate on that GPU
  local gpu="$1" prefix="$2" label="$3" ckpt="$4" model="$5"
  if [[ "$ckpt" != "-" && ! -f "$model/config.json" ]]; then
    rm -rf "$model.tmp_$JOB"
    python merge_checkpoint.py --checkpoint "$ckpt" --model-name "$BASE_MODEL" --output-dir "$model.tmp_$JOB"
    mv "$model.tmp_$JOB" "$model"   # whole dir appears at once, so config.json means complete
  fi
  export CUDA_VISIBLE_DEVICES="$gpu"
  export VLLM_CACHE_ROOT="$SCRATCH/.cache/vllm_rolling_${JOB}_g$gpu"
  export TRITON_CACHE_DIR="$SCRATCH/.triton_rolling_${JOB}_g$gpu"
  export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/.inductor_rolling_${JOB}_g$gpu"
  mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
  python h1_gsm_eval.py --models "$model" --datasets $DATASETS --instruct --tp 1 \
    --max_new_tokens 2048 --out_file "results/${prefix}_${ARM}_${label}.json"
}

start=$(date +%s)
while :; do
  # reap finished evals; refresh a run's summaries/W&B once none of its evals is in flight,
  # so held-out steps reach W&B in increasing order (log_heldout_to_wandb logs only newer ones)
  for g in "${!pid_of[@]}"; do
    kill -0 "${pid_of[$g]}" 2>/dev/null && continue
    read -r _s prefix label _rest <<<"${item_of[$g]}"
    if wait "${pid_of[$g]}"; then echo "=== done $prefix $label (gpu $g)"
    else fails[$prefix/$label]=$(( ${fails[$prefix/$label]:-0} + 1 )); echo "!!! failed $prefix $label (gpu $g, ${fails[$prefix/$label]}x)"; fi
    unset "pid_of[$g]" "item_of[$g]"
    if ! printf '%s\n' "${item_of[@]:-}" | grep -q " $prefix "; then
      python summarize_results.py results/${prefix}_${ARM}_*.json | tail -1 || true
      python log_heldout_to_wandb.py "${prefix}_${ARM}" || echo "(W&B held-out log failed)"
    fi
  done

  all_done && [[ ${#pid_of[@]} -eq 0 ]] && { echo "ALL ROLLING EVALS DONE"; break; }
  over=$(( $(date +%s) - start > BUDGET_S ))
  if [[ "$over" == 1 && ${#pid_of[@]} -eq 0 ]]; then
    echo "time budget used; resubmitting"; sbatch submit_eval_rolling.sh; break
  fi

  # dispatch to free GPUs (not after the budget; skip items that failed twice)
  if [[ "$over" == 0 ]]; then
    while read -r line; do
      [[ -n "$line" ]] || continue
      read -r _s prefix label ckpt model <<<"$line"
      [[ ${fails[$prefix/$label]:-0} -lt 2 ]] || continue
      printf '%s\n' "${item_of[@]:-}" | grep -qxF "$line" && continue
      g=""; for ((i = 0; i < NGPU; i++)); do [[ -z "${pid_of[$i]:-}" ]] && { g=$i; break; }; done
      [[ -n "$g" ]] || break
      echo "=== start $prefix $label on gpu $g"
      run_one "$g" "$prefix" "$label" "$ckpt" "$model" > "logs/eval-rolling-${JOB}-${prefix}-${label}.log" 2>&1 &
      pid_of[$g]=$!; item_of[$g]="$line"
    done < <(pending)
  fi
  sleep $(( ${#pid_of[@]} > 0 ? 60 : POLL ))
done
