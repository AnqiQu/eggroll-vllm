#!/bin/bash
# h1 GRPO baselines on Qwen2.5-3B-Instruct, following h1's code and README commands
# (submit_grpo_h1_stage.sh defaults) except where noted:
#
#   SEED_TEST=1  gate: README stage 1 exactly (300 steps, h=1, 768 tokens) with seeds 5
#                (README), 42 (grpo.py default) and 1, one node each, eval on test_len_1..3.
#                Run before the two baselines: the 2026-10-13 runs collapsed to 20-50-token
#                answers, and the README warns that training is seed-noisy. Ran 2026-10-14
#                (7404520-22) with the stripped prompt and failed: UNSTRIPPED_PROMPT=0 repeats it.
#   FIX_TEST=1   gate for the authors' fix (2026-10-21): SYSTEM_PROMPT without .strip()
#                (submit_grpo_h1_stage.sh's default now), stage 1, 300 steps, eval on
#                test_len_1..3, one job each:
#                  pub_s42     the released len-1 checkpoint's training_args (2048 tokens, seed 42)
#                  readme_s5   README stage 1 (768 tokens, seed 5)
#                  readme_s42  README stage 1 with seed 42
#                Pass: strict-format matches appear within ~30 steps (the authors' len-1
#                trainer_state: 0.37 by step 30), mean length stays ~200-280 tokens, and the
#                best step beats base on len_2.
#   only-long    Only-Long baseline: 1,500 steps on train_len_5 from the base model (the
#                5 x 300-step curriculum's compute, the paper's rule). Exception:
#                MAX_PROMPT_LEN=1024, so h=5 prompts are not cut to their last 512 tokens.
#                RESUMABLE=1 in case it outlasts the 24 h job cap (chained afterany).
#   600-step     h1 curriculum with 600 steps per stage (exception: README uses 300),
#                stages 1..5 chained afterok, each from the previous stage's BEST.
# Both baselines are evaluated on test_len_1..5 at every 50-step checkpoint, with the
# unstripped system prompt (UNSTRIPPED_PROMPT=1, the stage default since 2026-10-21).
# CFG picks their settings: readme (default: README completion lengths 768..1536, seed 5 at
# stage 1, 42 after) or pub (the released checkpoints' training_args: 2048 tokens, seed 42
# at every stage). Output: runs/grpo_h1_fix_<CFG>_{onlylong,600steps}, results/grpo{,600}_fix_<CFG>_*.
#
#   SEED_TEST=1 ./submit_grpo_h1_runs.sh
#   FIX_TEST=1 SB_OPTS="--gpus-per-node=1 --cpus-per-task=16 --mem=32G --time=6:00:00" ./submit_grpo_h1_runs.sh
#   ./submit_grpo_h1_runs.sh                      # the two baselines (CFG=readme)
#   CFG=pub ./submit_grpo_h1_runs.sh              # ... with the released checkpoints' settings
#   AFTER=<jobid> ./submit_grpo_h1_runs.sh        # ... gated on a job (afterok)
#   SB_OPTS="--gpus-per-node=1 --cpus-per-task=16 --mem=32G" ...   # extra sbatch options, e.g. to fit
#                partly used nodes when no node has 4 GPUs free (evals then run after training)
set -euo pipefail
cd "$SCRATCH/eggroll-vllm"
mkdir -p logs
unset BASE_MODEL SEED MAX_COMPLETION MAX_PROMPT_LEN   # stage defaults; stages 2-5 read BEST
ONLYLONG_JOBS="${ONLYLONG_JOBS:-2}"
ALL5="$(printf 'GSM-LongHorizon/test_len_%s.jsonl ' 1 2 3 4 5)"
read -ra SB_OPTS <<< "${SB_OPTS:-}"
sb() { local name="$1"; shift; sbatch --parsable --job-name="$name" --output="logs/$name-%j.out" "${SB_OPTS[@]}" "$@" submit_grpo_h1_stage.sh; }

if [[ "${SEED_TEST:-0}" == 1 ]]; then
  for seed in 5 42 1; do
    j=$(STAGE=1 SEED=$seed OUTPUT_ROOT="runs/grpo_h1_readme_seedtest_s$seed" PREFIX="grpo_readme_s1_seed$seed" \
        EVAL_DATASETS="$(printf 'GSM-LongHorizon/test_len_%s.jsonl ' 1 2 3)" sb "grpo-h1-seedtest-s$seed")
    echo "README stage 1, seed $seed   $j   runs/grpo_h1_readme_seedtest_s$seed  results/grpo_readme_s1_seed${seed}_plain_*"
  done
  exit 0
fi

if [[ "${FIX_TEST:-0}" == 1 ]]; then
  for cfg in "pub_s42 42 2048" "readme_s5 5 768" "readme_s42 42 768"; do
    read -r name seed maxtok <<< "$cfg"
    j=$(STAGE=1 SEED=$seed MAX_COMPLETION=$maxtok UNSTRIPPED_PROMPT=1 \
        OUTPUT_ROOT="runs/grpo_h1_fix_$name" PREFIX="grpo_fix_s1_$name" \
        EVAL_DATASETS="$(printf 'GSM-LongHorizon/test_len_%s.jsonl ' 1 2 3)" sb "grpo-h1-fix-$name")
    echo "prompt fix, $name   $j   runs/grpo_h1_fix_$name  results/grpo_fix_s1_${name}_plain_*"
  done
  exit 0
fi

CFG="${CFG:-readme}"
case "$CFG" in
  readme) ;;
  pub) export SEED=42 MAX_COMPLETION=2048 ;;
  *) echo "CFG must be readme or pub" >&2; exit 1 ;;
esac
DEP=(); [[ -n "${AFTER:-}" ]] && DEP=(--dependency="afterok:$AFTER")
dep=("${DEP[@]}")
for k in $(seq 1 "$ONLYLONG_JOBS"); do
  j=$(STAGE=5 BASE_MODEL=Qwen/Qwen2.5-3B-Instruct MAX_STEPS=1500 MAX_PROMPT_LEN=1024 RESUMABLE=1 \
      OUTPUT_ROOT="runs/grpo_h1_fix_${CFG}_onlylong" PREFIX="grpo_fix_${CFG}_onlylong5" \
      EVAL_DATASETS="$ALL5" sb "grpo-fix-$CFG-onlylong" "${dep[@]}")
  echo "only-long ($CFG) job $k/$ONLYLONG_JOBS   $j   runs/grpo_h1_fix_${CFG}_onlylong  results/grpo_fix_${CFG}_onlylong5_plain_*"
  dep=(--dependency="afterany:$j")
done

dep=("${DEP[@]}")
for s in 1 2 3 4 5; do
  j=$(STAGE=$s MAX_STEPS=600 OUTPUT_ROOT="runs/grpo_h1_fix_${CFG}_600steps" PREFIX="grpo600_fix_${CFG}_len$s" \
      EVAL_DATASETS="$ALL5" sb "grpo600-fix-$CFG-s$s" "${dep[@]}")
  echo "600-step ($CFG) stage $s    $j   runs/grpo_h1_fix_${CFG}_600steps/stage${s}_len${s}  results/grpo600_fix_${CFG}_len${s}_plain_*"
  dep=(--dependency="afterok:$j")
done
