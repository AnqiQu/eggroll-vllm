#!/bin/bash
#SBATCH --job-name=eggroll-nccl-test
#SBATCH --output=logs/nccl-test-%j.out
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
# 2-node smoke test of the inter-engine NCCL path (see slurm_ray_cluster.sh / nccl_multinode_test.py).
#   sbatch submit_nccl_multinode_test.sh ; grep -E "PASSED|Error|GB/s" logs/nccl-test-<jobid>.out
set -euo pipefail
mkdir -p logs
source "$SCRATCH/uv_envs/vllm_env/.venv/bin/activate"
cd "$SCRATCH/eggroll-vllm"
source slurm_ray_cluster.sh
trap 'stop_ray_cluster' EXIT
start_ray_cluster
python nccl_multinode_test.py
