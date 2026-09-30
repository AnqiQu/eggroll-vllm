#!/bin/bash
# slurm_ray_cluster.sh -- source this from an sbatch script to run es_lora_multinode.py
# over every node of the allocation (one vLLM engine per GPU).
#
#   source slurm_ray_cluster.sh     # after `source .venv/bin/activate`
#   start_ray_cluster               # head on node 0, workers on the rest, waits for all GPUs
#   ... python es_lora_multinode.py ... (ray.init(address="auto") picks up RAY_ADDRESS)
#   stop_ray_cluster
#
# Needs --nodes=N --gpus-per-node=G --ntasks-per-node=1. Sets START_LOCAL_RAY=0 so
# run_h1_curriculum.sh uses this cluster instead of starting a local head.
#
# Multi-node NCCL: the trainer's inter-engine weight broadcast is a raw NCCL group
# (PyNcclCommunicator) spanning all engines. Across nodes NCCL must use Slingshot
# through the aws-ofi-nccl (libfabric) plugin; without it the first all-reduce dies
# with "NCCL error: unhandled cuda error" (stage-2 jobs 6952900/6952901). Settings
# follow https://docs.isambard.ac.uk/user-documentation/guides/nccl/ .

NODES_ARR=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
HEAD_NODE="${NODES_ARR[0]}"
GPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-4}"; GPUS_PER_NODE="${GPUS_PER_NODE##*:}"   # "gh200:4" -> 4
EXPECTED_GPUS=$(( GPUS_PER_NODE * SLURM_JOB_NUM_NODES ))
export START_LOCAL_RAY=0

# --- NCCL over Slingshot (env is inherited by `ray start` on every node, hence by the engines) ---
if [[ "$SLURM_JOB_NUM_NODES" -gt 1 ]]; then
  type module >/dev/null 2>&1 || source /opt/cray/pe/lmod/lmod/init/bash   # sbatch shells may lack Lmod
  module load brics/nccl brics/aws-ofi-nccl 2>/dev/null || echo "WARNING: could not load brics/nccl + brics/aws-ofi-nccl modules"
  export NCCL_NET="AWS Libfabric"
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-hsn}"
  export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-PHB}"
  export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"
  export FI_CXI_DISABLE_HOST_REGISTER=1
  export FI_MR_CACHE_MONITOR=userfaultfd
  export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
  echo "multi-node NCCL: NCCL_NET=$NCCL_NET NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME plugin=${AWS_OFI_NCCL_ROOT:-unset}"
fi

shm_cleanup() {
  srun --overlap --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" bash -c '
    chmod -R u+rwx /dev/shm/es_lora_population_async_* /dev/shm/outputs_es_lora 2>/dev/null || true
    rm -rf /dev/shm/es_lora_population_async_* /dev/shm/outputs_es_lora 2>/dev/null || true' || true
}
stop_ray_cluster() {
  srun --overlap --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" bash -c 'ray stop --force >/dev/null 2>&1 || true' || true
  sleep 5
}
start_ray_cluster() {
  stop_ray_cluster
  local head_ip port=6379
  head_ip="$(srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname -I | awk '{print $1}')"
  export RAY_ADDRESS="${head_ip}:${port}"
  echo "Ray head: $HEAD_NODE ($RAY_ADDRESS); nodes: ${NODES_ARR[*]}; expecting $EXPECTED_GPUS GPUs"
  srun --overlap --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    ray start --head --node-ip-address="$head_ip" --port="$port" \
    --num-cpus="$SLURM_CPUS_PER_TASK" --num-gpus="$GPUS_PER_NODE" --block &
  sleep 15
  local i
  for ((i=1; i<SLURM_JOB_NUM_NODES; i++)); do
    srun --overlap --nodes=1 --ntasks=1 -w "${NODES_ARR[$i]}" \
      ray start --address="$RAY_ADDRESS" \
      --num-cpus="$SLURM_CPUS_PER_TASK" --num-gpus="$GPUS_PER_NODE" --block &
  done
  python - "$EXPECTED_GPUS" <<'PYEOF'
import ray, sys, time
want = int(sys.argv[1]); ray.init(address="auto", include_dashboard=False)
for _ in range(60):
    have = int(ray.cluster_resources().get("GPU", 0))
    if have >= want:
        print(f"Ray cluster ready: {have} GPUs"); sys.exit(0)
    time.sleep(5)
sys.exit(f"Ray cluster only has {have} of {want} GPUs")
PYEOF
}
