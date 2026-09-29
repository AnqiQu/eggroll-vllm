#!/bin/bash
# Queue the Qwen2.5-3B-Instruct curriculum stages as a dependency chain so the
# full h1 comparison (Len-1 .. Len-4 rows) runs unattended:
#   stage N+1 starts only if stage N finished (train + merge + eval + BEST file).
#
#   ./submit_h1_qwen25_3b_chain.sh            # stages 1..4 on 2 nodes (8 GPUs)
#   NODES=1 ./submit_h1_qwen25_3b_chain.sh    # 4 GPUs
#   START_STAGE=2 END_STAGE=4 ./submit_h1_qwen25_3b_chain.sh
#   AFTER=<jobid> START_STAGE=2 ./submit_h1_qwen25_3b_chain.sh   # hang off an existing job
#
# The workq QOS caps a job at 24 h, and stages 3-4 on the 3B model can take
# longer, so each stage gets JOBS_PER_STAGE[stage] chained 24 h jobs
# (--dependency=afterany): the stage script is idempotent and resumes from the
# newest checkpoint, so a continuation job either finishes the stage or, if the
# previous job already completed it, just re-prints the eval table and exits.
# The next stage waits (afterok) on the LAST job of the previous stage. If a
# stage fails for real, the later jobs show DependencyNeverSatisfied in squeue:
# scancel them and re-run with START_STAGE=<failed stage>.
set -euo pipefail
START_STAGE="${START_STAGE:-1}"
END_STAGE="${END_STAGE:-4}"
NODES="${NODES:-2}"           # 4 GPUs per node; 8 GPUs = 8 vLLM engines of 32 LoRAs each
# jobs per stage (index = stage). Stage 1 only has ~100 steps left; 2/3/4 grow with tokens.
declare -a JOBS_PER_STAGE=( "" 1 2 2 3 3 )
dep="${AFTER:-}"
for stage in $(seq "$START_STAGE" "$END_STAGE"); do
  n="${JOBS_PER_STAGE[$stage]}"
  for k in $(seq 1 "$n"); do
    args=(--job-name="eggroll-h1-q25-3b-s${stage}" --nodes="$NODES" --parsable)
    if [[ -n "$dep" ]]; then
      if [[ "$k" -eq 1 ]]; then args+=(--dependency="afterok:${dep}"); else args+=(--dependency="afterany:${dep}"); fi
    fi
    jid="$(STAGE="$stage" sbatch "${args[@]}" submit_h1_qwen25_3b_stage_correctonly.sh)"
    echo "stage $stage job $k/$n -> $jid ${dep:+(after $dep)}"
    dep="$jid"
  done
done
