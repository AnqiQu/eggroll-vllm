#!/bin/bash
# Queue the Qwen2.5-3B-Instruct curriculum stages as a dependency chain so the
# full h1 comparison (Len-1 .. Len-4 rows) runs unattended:
#   stage N+1 starts only if stage N finished (train + merge + eval + BEST file).
#
#   ./submit_h1_qwen25_3b_chain.sh            # stages 1..4
#   START_STAGE=2 END_STAGE=4 ./submit_h1_qwen25_3b_chain.sh
#   AFTER=<jobid> START_STAGE=2 ./submit_h1_qwen25_3b_chain.sh   # hang off an existing job
#
# Each stage auto-resumes from its newest checkpoint (see the stage script), so
# stage 1 continues job 6661559's run from checkpoint_step_500 instead of
# restarting. If a stage still fails, the rest of the chain shows
# DependencyNeverSatisfied in squeue: cancel those and re-run this script with
# START_STAGE=<failed stage>.
set -euo pipefail
START_STAGE="${START_STAGE:-1}"
END_STAGE="${END_STAGE:-4}"
dep="${AFTER:-}"
for stage in $(seq "$START_STAGE" "$END_STAGE"); do
  args=(--job-name="eggroll-h1-q25-3b-s${stage}")
  [[ -n "$dep" ]] && args+=(--dependency="afterok:${dep}")
  jid="$(STAGE="$stage" sbatch --parsable "${args[@]}" submit_h1_qwen25_3b_stage_correctonly.sh)"
  echo "stage $stage -> job $jid ${dep:+(after $dep)}"
  dep="$jid"
done
