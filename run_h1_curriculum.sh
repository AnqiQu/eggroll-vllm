#!/bin/bash
# ==========================================================================
# run_h1_curriculum.sh
#
# Replicate the h1 (LongHorizonReasoning) GSM-LongHorizon training run with
# EGGROLL (low-rank ES) replacing DrGRPO. Same model, dataset splits, prompts,
# rewards and stage-wise curriculum -- only the optimiser changes.
#
# The curriculum is a sequence of 5 stages over horizon lengths 1..5. Between
# stages you MERGE the chosen EGGROLL checkpoint into a standard HF model with
# merge_checkpoint.py, evaluate the merged checkpoints EXTERNALLY with h1's
# gsm_eval.py (horizons 1-3), pick the best, and use it as the base model for
# the next stage. This script trains + merges ONE stage per invocation and then
# PAUSES, printing the exact gsm_eval.py commands to run -- it does NOT
# auto-select checkpoints (h1 recommends manual selection).
#
#   Stage 1:  train_len_1.jsonl   max_completion_length 768   (integer reward)
#   Stage 2:  train_len_2.jsonl   max_completion_length 1024  (float reward)
#   Stage 3:  train_len_3.jsonl   max_completion_length 1280  (float reward)
#   Stage 4:  train_len_4.jsonl   max_completion_length 1536  (float reward)
#   Stage 5:  train_len_5.jsonl   max_completion_length 1536  (float reward)
#   Max prompt length 512 throughout (see note below).
#
# USAGE (run one stage at a time):
#   # Stage 1 (base = Qwen/Qwen2.5-3B-Instruct):
#   STAGE=1 ./run_h1_curriculum.sh
#   # ... run the printed gsm_eval.py commands, pick the best merged step ...
#   # Stage 2 (base = the chosen merged HF dir from stage 1):
#   STAGE=2 BASE_MODEL=runs/h1_curriculum/stage1_len1/merged/step_250 ./run_h1_curriculum.sh
#   # ... repeat for stages 3, 4, 5 ...
#
# Env overrides: STAGE (required), BASE_MODEL, DATA_DIR, OUTPUT_ROOT, GSM_EVAL,
#   EVAL_STEPS, MAX_TOKENS, SUB_DATASET_SIZE, USE_WANDB, WANDB_PROJECT, plus any
#   EGGROLL hyperparameter in the config block below.
# ==========================================================================

set -euo pipefail

# ==========================================================================
# ============================  CONFIG BLOCK  ==============================
# Experiment config for the h1 GSM-LongHorizon EGGROLL replication.
# ==========================================================================

# --- Model (same as h1) ---
BASE_MODEL_DEFAULT="Qwen/Qwen2.5-3B-Instruct"   # base for stage 1

# --- EGGROLL hyperparameters ---
# !!! TUNE THESE !!! Defaults are copied from the repo's existing math (GSM-style)
# ES run (slurm_launch_base_n1.sh: math:deepscaler40k) and are NOT tuned for
# GSM-LongHorizon. They are a starting point only.
SIGMA="${SIGMA:-0.001}"                  # perturbation std          (repo math run)
LEARNING_RATE="${LEARNING_RATE:-0.0002}" # ES learning rate          (repo math run)
POPULATION_SIZE="${POPULATION_SIZE:-1024}"   # ES population         (repo math run)
LORA_R="${LORA_R:-1}"                    # LoRA rank of perturbation  (repo math run)
STEPS_PER_ADAPTER="${STEPS_PER_ADAPTER:-4}"
NORMALIZE_WITH_STD="${NORMALIZE_WITH_STD:-normalize-with-std}"  # "" to disable
PROMPT_BATCH_SIZE="${PROMPT_BATCH_SIZE:-16}"   # distinct questions per ES step
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-1}"
TEMPERATURE="${TEMPERATURE:-0.0}"        # ES explores via perturbations, not sampling
NUM_ITERATIONS="${NUM_ITERATIONS:-300}"  # ES steps per stage (h1 uses ~300)
SAVE_FREQ="${SAVE_FREQ:-50}"             # checkpoint every N steps (h1 save_steps 50)

# --- Max prompt length ---
# h1 caps prompts at 512 tokens (TRL max_prompt_length). eggroll-vllm does NOT
# truncate prompts (vLLM sees the full chat-templated prompt), so this is a
# documented parameter only. GSM-LongHorizon prompts generally fit; see README.
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"

# ==========================================================================
# ==================  EQUAL-COMPUTE REFERENCE (per stage)  =================
# h1 GRPO per stage: max_steps=300, num_generations=16, gradient_accumulation
#   =16, per_device_train_batch_size=1. On 1 GPU the prompts consumed per
#   optimizer step ~= per_device_bs * grad_accum * num_gpus = 16, each with 16
#   generations, so:
#       rollouts/stage ~= 300 * 16 * 16 = 76,800   (scales with num_gpus)
#
# EGGROLL per stage:
#       rollouts/stage = NUM_ITERATIONS * POPULATION_SIZE * PROMPT_BATCH_SIZE
#                        * SAMPLES_PER_PROMPT
#   With the defaults above (300 * 1024 * 16 * 1) = 4,915,200 = ~64x h1's budget.
#
# For a BUDGET-MATCHED comparison (~76,800 rollouts/stage at 300 iters, spp=1):
#       POPULATION_SIZE * PROMPT_BATCH_SIZE ~= 256
#   e.g. POPULATION_SIZE=256 PROMPT_BATCH_SIZE=1, or POPULATION_SIZE=128
#   PROMPT_BATCH_SIZE=2. ES typically needs a larger population for a usable
#   gradient estimate, so budget-matching ES to GRPO is a genuine tradeoff --
#   decide whether to compare at equal rollouts or at equal ES steps.
# ==========================================================================

# --- Per-stage curriculum table (index = horizon length) ---
#            horizon:   1     2      3      4      5
declare -a MAXTOK=(  ""  768   1024   1280   1536   1536 )   # max_completion_length
declare -a RMODE=(   ""  int   float  float  float  float )  # int|float reward mode

# --- Paths ---
DATA_DIR="${DATA_DIR:-GSM-LongHorizon}"          # dir with train_len_N.jsonl / test_len_N.jsonl
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/h1_curriculum}" # where stage checkpoints/merges go
GSM_EVAL="${GSM_EVAL:-h1_gsm_eval.py}"           # vendored h1 evaluator (repo root)
RESULTS_DIR="${RESULTS_DIR:-results}"            # lightweight eval metrics JSON -- tracked in git

# ==========================================================================
# ============================  END CONFIG  ================================
# ==========================================================================

STAGE="${STAGE:?Set STAGE=1..5 (train one curriculum stage per invocation)}"
if [[ "$STAGE" -lt 1 || "$STAGE" -gt 5 ]]; then
    echo "STAGE must be 1..5, got $STAGE" >&2; exit 1
fi
HORIZON="$STAGE"
STAGE_MAXTOK="${MAX_TOKENS:-${MAXTOK[$STAGE]}}"   # MAX_TOKENS env overrides the stage default (smoke tests)
STAGE_RMODE="${RMODE[$STAGE]}"
TRAIN_FILE="$DATA_DIR/train_len_${HORIZON}.jsonl"
STAGE_DIR="$OUTPUT_ROOT/stage${STAGE}_len${HORIZON}"
CKPT_DIR="$STAGE_DIR/checkpoints"
MERGED_DIR="$STAGE_DIR/merged"

# Base model: Qwen for stage 1; the chosen merged dir from the previous stage otherwise.
if [[ "$STAGE" -eq 1 ]]; then
    BASE_MODEL="${BASE_MODEL:-$BASE_MODEL_DEFAULT}"
else
    BASE_MODEL="${BASE_MODEL:?Stage >1 needs BASE_MODEL=<merged HF dir chosen from the previous stage>}"
fi

[[ -f "$TRAIN_FILE" ]] || { echo "Missing $TRAIN_FILE (see README: download + split GSM-LongHorizon)" >&2; exit 1; }

NORMALIZE_FLAG=""
[[ -n "$NORMALIZE_WITH_STD" ]] && NORMALIZE_FLAG="--${NORMALIZE_WITH_STD}"

# Optional: cap the number of training questions (handy for smoke tests).
SUBSET_FLAG=""
[[ -n "${SUB_DATASET_SIZE:-}" ]] && SUBSET_FLAG="--sub-dataset-size ${SUB_DATASET_SIZE}"

# Optional: log to Weights & Biases. Set USE_WANDB=1 (run `wandb login` first);
# WANDB_PROJECT overrides the default project name.
WANDB_FLAGS=""
if [[ "${USE_WANDB:-0}" == "1" ]]; then
    WANDB_FLAGS="--use-wandb"
    [[ -n "${WANDB_PROJECT:-}" ]] && WANDB_FLAGS="$WANDB_FLAGS --wandb-project ${WANDB_PROJECT}"
fi

mkdir -p "$RESULTS_DIR"

echo "=========================================================================="
echo " h1 curriculum -- STAGE $STAGE  (horizon $HORIZON)"
echo "   base model            : $BASE_MODEL"
echo "   train file            : $TRAIN_FILE"
echo "   max_completion_length : $STAGE_MAXTOK   reward_mode: $STAGE_RMODE"
echo "   population/sigma/lr/r  : $POPULATION_SIZE / $SIGMA / $LEARNING_RATE / $LORA_R   (TUNE)"
echo "   checkpoints -> $CKPT_DIR"
echo "=========================================================================="

# --- Optional: start a local single-node Ray head if one isn't already up ---
# (Skip on a slurm/multi-node cluster where Ray is started by the launch script.)
if [[ "${START_LOCAL_RAY:-1}" == "1" && -z "${RAY_ADDRESS:-}" ]]; then
    if ! ray status >/dev/null 2>&1; then
        echo "Starting local Ray head..."
        ray start --head --port=6379 --dashboard-host=0.0.0.0 >/dev/null
    fi
fi

# ==========================================================================
# 1) TRAIN this stage with EGGROLL
# ==========================================================================
python es_lora_multinode.py \
    --model-name "$BASE_MODEL" \
    --task "gsm_longhorizon:${STAGE_RMODE}:${TRAIN_FILE}" \
    --max-tokens "$STAGE_MAXTOK" \
    --num-iterations "$NUM_ITERATIONS" \
    --sigma "$SIGMA" \
    --learning-rate "$LEARNING_RATE" \
    --population-size "$POPULATION_SIZE" \
    --lora-r "$LORA_R" \
    --steps-per-adapter "$STEPS_PER_ADAPTER" \
    --prompt-batch-size "$PROMPT_BATCH_SIZE" \
    --samples-per-prompt "$SAMPLES_PER_PROMPT" \
    --temperature "$TEMPERATURE" \
    $NORMALIZE_FLAG \
    $SUBSET_FLAG \
    $WANDB_FLAGS \
    --steps-per-eval -1 \
    --save-freq "$SAVE_FREQ" \
    --checkpoint-dir "$CKPT_DIR" \
    --name-prefix "h1_stage${STAGE}_len${HORIZON}"

# ==========================================================================
# 2) MERGE this stage's EGGROLL checkpoints into standard HF models so they can
#    be evaluated with gsm_eval.py and used as the next stage's base model.
# ==========================================================================
# By default merge every checkpoint_step_* that was saved; override with e.g.
#   EVAL_STEPS="200 250 299"
if [[ -n "${EVAL_STEPS:-}" ]]; then
    STEPS="$EVAL_STEPS"
else
    STEPS="$(ls -d "$CKPT_DIR"/checkpoint_step_* 2>/dev/null \
             | sed 's/.*checkpoint_step_//' | sort -n | tr '\n' ' ')"
fi
[[ -n "$STEPS" ]] || { echo "No checkpoints found under $CKPT_DIR" >&2; exit 1; }

for step in $STEPS; do
    ckpt="$CKPT_DIR/checkpoint_step_${step}"
    out="$MERGED_DIR/step_${step}"
    [[ -f "$ckpt/model_weights.safetensors" ]] || { echo "skip $ckpt (no weights)"; continue; }
    echo "--- merging $ckpt -> $out ---"
    python merge_checkpoint.py \
        --checkpoint "$ckpt" \
        --model-name "$BASE_MODEL" \
        --output-dir "$out"
done

# ==========================================================================
# 3) PAUSE: evaluate merged checkpoints externally and pick the best.
# h1 recommends evaluating each candidate on horizons 1, 2 and 3 and choosing
# the overall best, then using it as the base model for the NEXT stage.
# ==========================================================================
cat <<MSG

==========================================================================
 STAGE $STAGE training + merge done. Now SELECT the best checkpoint.
--------------------------------------------------------------------------
 Run h1's gsm_eval.py on the merged checkpoints (horizons 1-3), e.g.:

   for step in $STEPS; do
     python $GSM_EVAL \\
       --models $MERGED_DIR/step_\${step} \\
       --datasets $DATA_DIR/test_len_1.jsonl $DATA_DIR/test_len_2.jsonl $DATA_DIR/test_len_3.jsonl \\
       --instruct --tp 1 \\
       --out_file $RESULTS_DIR/stage${STAGE}_len${HORIZON}_step_\${step}.json
   done

 Metrics land in $RESULTS_DIR/ (tracked in git); commit them with:
   git add $RESULTS_DIR/ && git commit -m "results: stage $STAGE eval"

 Pick the step with the best overall horizon-1..3 accuracy, then launch the
 next stage using that merged dir as the base model:

   STAGE=$((STAGE+1)) BASE_MODEL=$MERGED_DIR/step_<BEST> ./run_h1_curriculum.sh
==========================================================================
MSG
