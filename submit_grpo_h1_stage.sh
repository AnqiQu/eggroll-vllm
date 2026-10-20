#!/bin/bash
#SBATCH --job-name=grpo-h1-q25-3b
#SBATCH --output=logs/grpo-h1-q25-3b-%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00

# One h1 curriculum stage (GSM-LongHorizon, horizon STAGE) with h1's own GRPO
# trainer on Qwen2.5-3B-Instruct, optionally with EGGROLL-shaped parameter noise
# (grpo_param_noise.py). Two uses:
#
#   NOISE_SIGMA=0      plain h1 GRPO. h1's own checkpoints are on HF
#                      (long-horizon-reasoning/Qwen-3b-GRPO-len-{1..5}); plain runs here
#                      cover what they did not release: untruncated prompts, longer
#                      stages, the Only-Long baseline (submit_grpo_h1_runs.sh).
#   NOISE_SIGMA=0.001  GRPO trained at theta + sigma*eps (fresh eps per step): follows
#                      the same smoothed objective as ES, without a population. The
#                      "smoothing vs population" control.
#
# Defaults follow h1's code and README commands (h1/h1, commit bdb51ab): 300 steps, save
# every 50, warmup 0.1, max_completion_length 768/1024/1280/1536/1536 for horizons 1-5,
# seed 5 at stage 1 and grpo.py's default 42 after, float reward from stage 2, and
# grpo.py's defaults for the rest (max_prompt_length 512, lr 5e-6 cosine, wd 0.1, betas
# (0.9, 0.99), grad norm 0.1, dr_grpo, beta 0, bf16, no shuffle, 1 GPU x 16 samples).
# The released HF checkpoints' training_args.bin differ: 2048 completion tokens at every
# stage and seed 42. Runs with those values (2026-10-13) collapsed to 20-50-token answers
# by step ~100: the model never learned to end with "</answer>\n", so xmlcount's
# 0.001/char penalty kept applying to the whole response.
# MAX_PROMPT_LEN: TRL keeps only the LAST max_prompt_length prompt tokens, so 512 cuts the
# system prompt and often Step 1 of h=5 prompts (median ~590, max 776 tokens); every h<=3
# prompt fits. Use 1024 to avoid it; check_prompt_lengths.py reports cut prompts from
# the training log either way.
#
# Layout (one node, 4 GPUs): GPU 0 trains (h1's 1-GPU setup, 1 question x 16 samples per
# step); GPUs 1-3 evaluate each checkpoint as soon as it is saved (vLLM, greedy, 2048 new
# tokens: h1_gsm_eval.py as in submit_eval_len2_correctonly.sh), so only the last
# checkpoints are left after training. Then BEST by h1's rule (select_best_step.py:
# highest len_1+len_2+len_3), which the next stage reads.
#
#   STAGE=1 sbatch submit_grpo_h1_stage.sh
#   STAGE=2 sbatch --dependency=afterok:<job1> submit_grpo_h1_stage.sh
#   STAGE=1 NOISE_SIGMA=0.001 sbatch submit_grpo_h1_stage.sh
#
# Env: STAGE (1..5; trains on train_len_STAGE), TRAIN_DATASET, BASE_MODEL (stage>1:
#      default previous stage's BEST), OUTPUT_ROOT, PREFIX, NOISE_SIGMA (0), NOISE_RANK (1),
#      NOISE_SEED (0), SEED (5 at stage 1, else 42), MAX_STEPS (300), SAVE_STEPS (50),
#      MAX_COMPLETION (768/1024/1280/1536/1536 by stage), MAX_PROMPT_LEN (512),
#      CORRECTNESS_ONLY (0: h1's format+correctness reward;
#      1: match the ES fitness), NGPUS (1), GRAD_ACCUM (16),
#      EVAL_DATASETS (test_len_1..4, plus test_len_5 at STAGE 5),
#      SELECT_SPLITS ("len_1 len_2 len_3"), PROMPT_CHECK_STRICT (0; 1 = fail if a prompt was cut),
#      RESUMABLE (0).
# Every run needs its OWN OUTPUT_ROOT and PREFIX: results go to results/<PREFIX>_<TAG>_*.json
# and select_best_step.py reads them by pattern, so a shared prefix mixes runs
# (cf. the 2026-09-15 collision).
#
# Batch semantics (TRL 0.21, checked in its GRPOConfig): per_device_train_batch_size
# counts COMPLETIONS, so one optimizer step sees NGPUS x 1 x GRAD_ACCUM completions.
# The same 1 x 16 batch on 4 GPUs: NGPUS=4 GRAD_ACCUM=4 (eval then runs after training).
# Walltime: workq's QoS (workq_qos) caps a job at 24 h, although the partition says
# UNLIMITED. h1 saves model-only checkpoints (no optimizer state), so by default a stage
# must finish in one job, and the job refuses to retrain over existing checkpoints or
# step results. RESUMABLE=1 also saves optimizer/scheduler/RNG state (~3x the 6.2 GB per
# checkpoint) and resumes from the newest complete checkpoint, so a longer run can span
# jobs chained with --dependency=afterany (submit_grpo_h1_runs.sh does this for Only-Long).
# h1's environment ($SCRATCH/uv_envs/h1_env/.venv, Python 3.12): h1/h1/requirements.txt
# with torch 2.7.1 from the cu128 index (aarch64 CUDA wheel), minus vllm/xformers/
# deepspeed/cupy/nvidia-* (unused by GRPO without vLLM generation).
STAGE="${STAGE:?Set STAGE=1..5}"
NOISE_SIGMA="${NOISE_SIGMA:-0}"
NOISE_RANK="${NOISE_RANK:-1}"
NOISE_SEED="${NOISE_SEED:-0}"
SEED="${SEED:-$([[ "$STAGE" -eq 1 ]] && echo 5 || echo 42)}"   # README: --seed 5 at stage 1 only
CORRECTNESS_ONLY="${CORRECTNESS_ONLY:-0}"
NGPUS="${NGPUS:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
MAX_STEPS="${MAX_STEPS:-300}"
SAVE_STEPS="${SAVE_STEPS:-50}"
declare -a MAXTOK=( "" 768 1024 1280 1536 1536 )   # README, per horizon
MAX_COMPLETION="${MAX_COMPLETION:-${MAXTOK[$STAGE]}}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-512}"
TRAIN_DATASET="${TRAIN_DATASET:-GSM-LongHorizon/train_len_${STAGE}.jsonl}"
EVAL_DATASETS="${EVAL_DATASETS:-$(for h in 1 2 3 4 $([[ "$STAGE" -eq 5 ]] && echo 5); do printf 'GSM-LongHorizon/test_len_%s.jsonl ' "$h"; done)}"
SELECT_SPLITS="${SELECT_SPLITS:-len_1 len_2 len_3}"
PROMPT_CHECK_STRICT="${PROMPT_CHECK_STRICT:-0}"
RESUMABLE="${RESUMABLE:-0}"
H1_ENV="${H1_ENV:-$SCRATCH/uv_envs/h1_env/.venv}"
VLLM_ENV="${VLLM_ENV:-$SCRATCH/uv_envs/vllm_env/.venv}"

set -euo pipefail
mkdir -p logs results
export HF_HOME="$SCRATCH/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline   # h1's grpo.py calls wandb.init and reports to wandb
cd "$SCRATCH/eggroll-vllm"

if [[ "$NOISE_SIGMA" == "0" ]]; then TAG="plain"; else TAG="noise_sigma${NOISE_SIGMA}_r${NOISE_RANK}"; fi
[[ "$CORRECTNESS_ONLY" == "1" ]] && TAG="${TAG}_conly"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/grpo_h1_qwen25_3b_${TAG}}"
PREFIX="${PREFIX:-grpo_q25_3b_len${STAGE}}"
STAGE_DIR="$OUTPUT_ROOT/stage${STAGE}_len${STAGE}"
CKPT_DIR="$STAGE_DIR/checkpoints"
MERGED_DIR="$STAGE_DIR/merged"
CLAIMS="$STAGE_DIR/eval_claims"
BASE_LABEL_MODEL="Qwen/Qwen2.5-3B-Instruct"
mkdir -p "$MERGED_DIR" "$STAGE_DIR/eval_logs"
export WANDB_DIR="$STAGE_DIR"

if [[ "$STAGE" -eq 1 ]]; then
  BASE="${BASE_MODEL:-$BASE_LABEL_MODEL}"
else
  PREV_BEST="$OUTPUT_ROOT/stage$((STAGE-1))_len$((STAGE-1))/BEST"
  if [[ -n "${BASE_MODEL:-}" ]]; then BASE="$BASE_MODEL"
  elif [[ -s "$PREV_BEST" ]]; then BASE="$(cat "$PREV_BEST")"; echo "base model from $PREV_BEST: $BASE"
  else echo "Stage $STAGE needs BASE_MODEL=<model dir> or $PREV_BEST" >&2; exit 1; fi
fi

FLOAT_FLAG=""; [[ "$STAGE" -ge 2 ]] && FLOAT_FLAG="--float_reward_func"
EXTRA=(--param_noise_sigma "$NOISE_SIGMA" --param_noise_rank "$NOISE_RANK" --param_noise_seed "$NOISE_SEED")
[[ "$CORRECTNESS_ONLY" == "1" ]] && EXTRA+=(--correctness_only)
[[ "$RESUMABLE" == "1" ]] && EXTRA+=(--resumable)
if [[ "$NGPUS" -gt 1 ]]; then LAUNCH=("$H1_ENV/bin/accelerate" launch --num_processes "$NGPUS")
else LAUNCH=("$H1_ENV/bin/python"); fi
IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)}"
TRAIN_DEVS="$(IFS=,; echo "${GPUS[*]:0:$NGPUS}")"
EVAL_DEVS=("${GPUS[@]:$NGPUS}")

ckpt_steps() { ls "$CKPT_DIR" 2>/dev/null | sed -n 's/^checkpoint-\([0-9]*\)$/\1/p' | sort -n; }

# ---- held-out eval: workers claim unevaluated models (base, then each checkpoint once
# its trainer_state.json, written after the weights, exists) until training has
# finished and nothing is left. Per-model JSON -> results/<PREFIX>_<TAG>_<label>.json ----
eval_one() {  # label model gpu
  ( source "$VLLM_ENV/bin/activate"
    export CUDA_VISIBLE_DEVICES="$3"
    export VLLM_CACHE_ROOT="$SCRATCH/.cache/vllm_grpoeval_${SLURM_JOB_ID}_g$3"
    export TRITON_CACHE_DIR="$SCRATCH/.triton_grpoeval_${SLURM_JOB_ID}_g$3"
    export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/.inductor_grpoeval_${SLURM_JOB_ID}_g$3"
    mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
    python h1_gsm_eval.py --models "$2" --datasets $EVAL_DATASETS --instruct --tp 1 \
      --max_new_tokens 2048 --out_file "results/${PREFIX}_${TAG}_$1.json"
  ) > "$STAGE_DIR/eval_logs/$1.log" 2>&1
}
eval_worker() {  # gpu
  set +e
  local gpu="$1" fin label n
  while :; do
    fin=0; [[ -e "$STAGE_DIR/.train_finished" ]] && fin=1   # read before scanning: no checkpoint is missed
    for label in base $(for n in $(ckpt_steps); do
                          [[ -f "$CKPT_DIR/checkpoint-$n/trainer_state.json" ]] && echo "step_$n"; done); do
      [[ -s "results/${PREFIX}_${TAG}_${label}.json" ]] && continue
      mkdir "$CLAIMS/$label" 2>/dev/null || continue
      if [[ "$label" == base ]]; then model="$BASE_LABEL_MODEL"
      else model="$MERGED_DIR/$label"; ln -sfn "$(realpath "$CKPT_DIR/checkpoint-${label#step_}")" "$model"; fi
      echo "[eval gpu$gpu $(date +%H:%M)] $label"
      eval_one "$label" "$model" "$gpu" || echo "[eval gpu$gpu] $label FAILED, see $STAGE_DIR/eval_logs/$label.log"
    done
    [[ "$fin" == 1 ]] && return 0
    sleep 60
  done
}

# ---- 1: train (GPU 0) while evaluating (GPUs 1-3) ----
TRAINED=0; [[ -f "$CKPT_DIR/checkpoint-${MAX_STEPS}/trainer_state.json" ]] && TRAINED=1
if [[ "$TRAINED" == 0 ]] && { [[ -n "$(ckpt_steps)" ]] || compgen -G "results/${PREFIX}_${TAG}_step_*.json" > /dev/null; }; then
  if [[ "$RESUMABLE" == 1 ]] && compgen -G "$CKPT_DIR/checkpoint-*/optimizer.pt" > /dev/null; then
    echo "resuming stage $STAGE from the newest complete checkpoint in $CKPT_DIR"
  else
    echo "$CKPT_DIR has checkpoints or results/${PREFIX}_${TAG}_step_*.json exist, but stage $STAGE is not trained" \
         "to step $MAX_STEPS and cannot resume (no optimizer state): move them aside or use a new OUTPUT_ROOT/PREFIX." >&2
    exit 1
  fi
fi
rm -rf "$CLAIMS"; mkdir -p "$CLAIMS"; rm -f "$STAGE_DIR/.train_finished"
echo "stage $STAGE  base=$BASE  train=$TRAIN_DATASET  tag=$TAG  prefix=$PREFIX  out=$STAGE_DIR"
echo "steps=$MAX_STEPS save=$SAVE_STEPS seed=$SEED prompt<=$MAX_PROMPT_LEN completion<=$MAX_COMPLETION" \
     "train GPUs=$TRAIN_DEVS eval GPUs=${EVAL_DEVS[*]:-after training}  eval: $EVAL_DATASETS"
TRAIN_PID=""
if [[ "$TRAINED" == 1 ]]; then
  echo "stage $STAGE already trained ($CKPT_DIR/checkpoint-${MAX_STEPS}); evaluating"
else
  CUDA_VISIBLE_DEVICES="$TRAIN_DEVS" "$H1_ENV/bin/python" -c \
    "import torch; assert torch.cuda.is_available(), 'h1_env torch has no CUDA'; print('torch', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
  T0=$(date +%s)
  CUDA_VISIBLE_DEVICES="$TRAIN_DEVS" "${LAUNCH[@]}" grpo_param_noise.py \
    --model "$BASE" \
    --dataset "$TRAIN_DATASET" \
    --output_dir "$CKPT_DIR" \
    --run_name "${PREFIX}_${TAG}" \
    --max_steps "$MAX_STEPS" \
    --max_prompt_length "$MAX_PROMPT_LEN" \
    --max_completion_length "$MAX_COMPLETION" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --save_steps "$SAVE_STEPS" --warmup_ratio 0.1 --seed "$SEED" \
    $FLOAT_FLAG "${EXTRA[@]}" &
  TRAIN_PID=$!
fi
WORKERS=()
for g in "${EVAL_DEVS[@]}"; do eval_worker "$g" & WORKERS+=($!); done
if [[ -n "$TRAIN_PID" ]]; then
  if ! wait "$TRAIN_PID"; then echo "TRAINING FAILED (stage $STAGE)" >&2; exit 1; fi
  echo "training: $(( ($(date +%s) - T0) / 60 )) min for $MAX_STEPS steps"
fi
touch "$STAGE_DIR/.train_finished"

# ---- 2: were prompts cut? (implied prompt length per step from the log vs full length) ----
LAST="$(ckpt_steps | tail -1)"
if ! "$H1_ENV/bin/python" check_prompt_lengths.py --state "$CKPT_DIR/checkpoint-$LAST/trainer_state.json" \
       --dataset "$TRAIN_DATASET" --tokenizer "$BASE" --max_prompt_length "$MAX_PROMPT_LEN"; then
  [[ "$PROMPT_CHECK_STRICT" == 1 ]] && { echo "prompt check failed; stopping" >&2; exit 1; }
  echo "WARNING: prompt check failed (continuing)"
fi

# ---- 3: finish the evals (one retry pass on all GPUs for any that failed) ----
[[ "${#WORKERS[@]}" -gt 0 ]] && wait "${WORKERS[@]}"
rm -rf "$CLAIMS"; mkdir -p "$CLAIMS"; WORKERS=()
for g in "${GPUS[@]}"; do eval_worker "$g" & WORKERS+=($!); done
wait "${WORKERS[@]}"
for n in $(ckpt_steps); do ln -sfn "$(realpath "$CKPT_DIR/checkpoint-$n")" "$MERGED_DIR/step_$n"; done

# ---- 4: table (accuracy %, every evaluated step) and best step by h1's rule ----
source "$VLLM_ENV/bin/activate"
python - "results/${PREFIX}_${TAG}_" "$SELECT_SPLITS" $(ckpt_steps) <<'PYEOF'
import glob, json, os, re, sys
prefix, sel, steps = sys.argv[1], sys.argv[2].split(), sys.argv[3:]
rows = {}
for path in glob.glob(prefix + "*.json"):
    label = path[len(prefix):-len(".json")]
    for _model, dsets in json.load(open(path)).items():
        rows[label] = {os.path.basename(d).replace("test_", "").replace(".jsonl", ""): v["accuracy"] * 100
                       for d, v in dsets.items()}
cols = sorted({c for r in rows.values() for c in r})
key = lambda l: -1 if l == "base" else int(re.sub(r"\D", "", l))
print(f"{'model':10s} " + " ".join(f"{c:>7s}" for c in cols) + f" {'+'.join(sel):>22s}")
for label in sorted(rows, key=key):
    r = rows[label]
    tot = sum(r[c] for c in sel) if all(c in r for c in sel) else float("nan")
    print(f"{label:10s} " + " ".join(f"{r.get(c, float('nan')):7.2f}" for c in cols) + f" {tot:22.2f}")
missing = [f"step_{s}" for s in steps if f"step_{s}" not in rows]
if missing:
    print("WARNING: not evaluated:", " ".join(missing))
PYEOF
python select_best_step.py --prefix "$PREFIX" --arm "$TAG" --merged "$MERGED_DIR" \
  --out "$STAGE_DIR/BEST" --splits $SELECT_SPLITS
echo "GRPO STAGE $STAGE DONE ($TAG, $PREFIX)"
