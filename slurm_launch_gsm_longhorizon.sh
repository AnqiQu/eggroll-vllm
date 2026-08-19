#!/bin/bash

#SBATCH --job-name=h1_gsm_longhorizon
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=1

# --- Create logs directory if it doesn't exist ---
LOG_DIR="./hyperscale-es-vllm/logs/"
mkdir -p "$LOG_DIR"

echo "---------------------------------"
echo "Starting job $SLURM_JOB_ID on $(hostname)"
echo "Nodes involved: $SLURM_JOB_NODELIST"
echo "Running on GPU(s): $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)"
echo "Number of GPUs per node: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo "Log file: $LOG_DIR/multinode_n1-$SLURM_JOB_ID.log"
echo "---------------------------------"

# ==========================================================================
# ===================  h1 GSM-LongHorizon CONFIG BLOCK  ====================
# Replicate h1 (LongHorizonReasoning) with EGGROLL replacing DrGRPO.
# This job trains ONE curriculum stage. Between stages: merge the chosen
# checkpoint (merge_checkpoint.py), evaluate merged checkpoints externally with
# h1's gsm_eval.py on horizons 1-3, pick the best, and pass it as BASE_MODEL to
# the next stage. See run_h1_curriculum.sh + README for the full workflow.
#
# Select the stage with the STAGE env var (1..5), e.g.:
#   STAGE=1 sbatch slurm_launch_gsm_longhorizon.sh
#   STAGE=2 BASE_MODEL=runs/h1_curriculum/stage1_len1/merged/step_250 \
#       sbatch slurm_launch_gsm_longhorizon.sh
#
#   Stage/horizon:  1     2      3      4      5
#   max_tokens   :  768   1024   1280   1536   1536   (max_completion_length)
#   reward_mode  :  int   float  float  float  float  (mirrors --float_reward_func)
#   Max prompt length 512 throughout (eggroll-vllm does not truncate prompts;
#   documented only -- see README).
#
# --- EQUAL-COMPUTE REFERENCE (per stage) ---
#   h1 GRPO: 300 steps x (per_device_bs 1 * grad_accum 16 * num_gpus) prompts
#            x num_generations 16.  On 1 GPU: 300*16*16 = 76,800 rollouts/stage.
#   EGGROLL: NUM_ITERATIONS * POPULATION_SIZE * PROMPT_BATCH_SIZE * SAMPLES_PER_PROMPT.
#            Defaults (300*1024*16*1) = 4,915,200 = ~64x h1's rollout budget.
#   Budget-matched (~76,800 rollouts/stage, 300 iters, spp 1): set
#            POPULATION_SIZE * PROMPT_BATCH_SIZE ~= 256 (e.g. pop 256, pbs 1).
# ==========================================================================

STAGE="${STAGE:?Set STAGE=1..5}"
BASE_MODEL_DEFAULT="Qwen/Qwen2.5-3B-Instruct"   # base for stage 1 (same as h1)
DATA_DIR="${DATA_DIR:-GSM-LongHorizon}"         # dir with train_len_N.jsonl
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/h1_curriculum}"

# Per-stage curriculum table (index = horizon = STAGE)
declare -a STAGE_MAXTOK=( "" 768 1024 1280 1536 1536 )
declare -a STAGE_RMODE=(  "" int float float float float )
horizon="$STAGE"
max_tokens="${STAGE_MAXTOK[$STAGE]}"
reward_mode="${STAGE_RMODE[$STAGE]}"
train_file="$DATA_DIR/train_len_${horizon}.jsonl"

if [[ "$STAGE" -eq 1 ]]; then
    model_name="${BASE_MODEL:-$BASE_MODEL_DEFAULT}"
else
    model_name="${BASE_MODEL:?Stage >1 needs BASE_MODEL=<merged HF dir from previous stage>}"
fi
task="gsm_longhorizon:${reward_mode}:${train_file}"
checkpoint_dir="$OUTPUT_ROOT/stage${STAGE}_len${horizon}/checkpoints"
name_prefix="h1_stage${STAGE}_len${horizon}"

# -----------------------------------------
# EGGROLL hyperparameters -- !!! TUNE !!!
# Defaults copied from the repo's existing math ES run (slurm_launch_base_n1.sh,
# math:deepscaler40k); NOT tuned for GSM-LongHorizon.
# -----------------------------------------
sigma="0.001"
learning_rate="0.0002"
population_size="1024"
lora_r="1"
steps_per_adapter="4"
num_iterations="300"        # ES steps per stage (h1 uses ~300)
save_freq="50"              # checkpoint every N steps (h1 save_steps 50)
normalize_with_std="normalize-with-std"   # ="" to disable
scale_lr_in_grad=""                        # ="scale-lr-in-grad" to enable
prompt_batch_size="16"      # distinct questions per ES step
samples_per_prompt="1"
temperature="0.0"           # ES explores via perturbations, not sampling
pass_at_k="no-pass-at-k"
steps_per_eval="-1"         # no in-loop eval (checkpoints selected externally)
sub_dataset_size="null"

# -----------------------------------------

# --- Echo parameters for logging ---
echo "Parameters:"
echo "  sigma: $sigma"
echo "  learning_rate: $learning_rate"
echo "  max_tokens: $max_tokens"
echo "  model_name: $model_name"
echo "  population_size: $population_size"
echo "  steps_per_adapter: $steps_per_adapter"
echo "  lora_r: $lora_r"
echo "  task: $task"
echo "  normalize_with_std: $normalize_with_std"
echo "  scale_lr_in_grad: $scale_lr_in_grad"
echo "  prompt_batch_size: $prompt_batch_size"
echo "  samples_per_prompt: $samples_per_prompt"
echo "  temperature: $temperature"
echo "  pass_at_k: $pass_at_k"
echo "  steps_per_eval: $steps_per_eval"
echo "  sub_dataset_size: $sub_dataset_size"
echo "  name_prefix: $name_prefix"
echo "---------------------------------"

if [[ "$sub_dataset_size" == "None" ]] || [[ "$sub_dataset_size" == "null" ]] || [[ -z "$sub_dataset_size" ]]; then
    DATASET_SIZE_CMD=""
else
    DATASET_SIZE_CMD="--sub-dataset-size $sub_dataset_size"
fi

# --- Activate Environment ---
echo "Activating virtual environment..."
source "$SCRATCH/uv_envs/vllm_env/.venv/bin/activate"

# --- Set WandB directory ---
export WANDB_DIR="$SCRATCH/for_esvllm/wandb"

# --- Allow expandable PyTorch memory --- 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Force Hugging Face to use offline mode (avoid rate limiting) ---
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --- Conditionally set compile caches based on model size ---
# Extracts the numeric value immediately before a standalone 'B' in the model name
# e.g. "Qwen3-4B" -> 4, "Meta-Llama-3-70B" -> 70, "mistral-7b" -> 7
get_model_size_b() {
    local name="$1"
    local size
    size=$(echo "$name" | grep -oiP '\d+(?=b\b)' | tail -1)
    echo "${size:-0}"
}

model_size_b=$(get_model_size_b "$model_name")

if [ "$model_size_b" -gt 0 ] && [ "$model_size_b" -lt 14 ]; then
    echo "Model size is ${model_size_b}B (< 14B): enabling job-specific compile caches"
    export VLLM_CACHE_ROOT="$SCRATCH/.cache/vllm_${SLURM_JOB_ID}"
    export TRITON_CACHE_DIR="$SCRATCH/.triton_cache_${SLURM_JOB_ID}"
    export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/.inductor_cache_${SLURM_JOB_ID}"
    mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

    cleanup_caches() {
        echo "Cleaning up job-specific caches..."
        rm -rf "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
        echo "Cache cleanup complete"
    }
    trap cleanup_caches EXIT
else
    echo "Model size is ${model_size_b}B (>= 14B or unparseable): skipping job-specific compile caches"
fi

# --- Change to Working Directory ---
echo "Changing to working directory..."
cd "$HOME/hyperscale/hyperscale-es-vllm" || exit 1

# --- Clean up leftover shared memory directories from previous jobs (on all nodes) ---
echo "Cleaning up /dev/shm from previous jobs on all nodes..."
echo "Current job ID: $SLURM_JOB_ID"
srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" bash -c '
    echo "$(hostname): Cleaning /dev/shm..."
    chmod -R u+rwx /dev/shm/es_lora_population_async_* /dev/shm/outputs_es_lora 2>/dev/null || true
    rm -rf /dev/shm/es_lora_population_async_* /dev/shm/outputs_es_lora 2>/dev/null || true
    echo "$(hostname): Cleanup complete"
'
echo "Cleanup complete on all nodes"

# ==========================================
# === RAY CLUSTER SETUP (MULTI-NODE) ===
echo "Setting up Ray Cluster..."

# 1. Get the list of nodes and the head node
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# 2. Port configuration
port=6379
ip_head=$head_node_ip:$port
export RAY_ADDRESS=$ip_head

echo "Head node: $head_node ($head_node_ip)"
echo "Ray Head IP: $ip_head"

# 3. Start Ray Head on the primary node
echo "Starting Ray Head on $head_node..."
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus="${SLURM_CPUS_PER_TASK}" --num-gpus="${SLURM_GPUS_PER_NODE}" --block &

# 4. Wait briefly for head to initialize
sleep 10

# 5. Start Ray Workers on the remaining nodes
worker_num=$((SLURM_JOB_NUM_NODES - 1))
if [ $worker_num -gt 0 ]; then
    for ((i=1; i<=worker_num; i++)); do
        node_i=${nodes_array[$i]}
        echo "Starting Ray Worker on $node_i..."
        srun --nodes=1 --ntasks=1 -w "$node_i" \
            ray start --address "$ip_head" \
            --num-cpus="${SLURM_CPUS_PER_TASK}" --num-gpus="${SLURM_GPUS_PER_NODE}" --block &
    done
fi

# 6. Wait for all nodes to register
echo "Waiting for Ray workers to connect..."
sleep 20
python -c "import ray; ray.init(address='auto'); print('Ray Cluster Resources:', ray.cluster_resources())"
# ==========================================


# --- Run the Python Script (Head Node Only) ---
echo "Starting Python script..."

# Build flag strings for optional flags (only add if non-empty)
NORMALIZE_FLAG=""
if [[ -n "$normalize_with_std" ]]; then
    NORMALIZE_FLAG="--${normalize_with_std}"
fi

SCALE_LR_FLAG=""
if [[ -n "$scale_lr_in_grad" ]]; then
    SCALE_LR_FLAG="--${scale_lr_in_grad}"
fi

PASSATK_FLAG=""
if [[ -n "$pass_at_k" ]]; then
    PASSATK_FLAG="--${pass_at_k}"
fi

python es_lora_multinode.py \
    --sigma "$sigma" \
    --learning-rate "$learning_rate" \
    --max-tokens "$max_tokens" \
    --model-name "$model_name" \
    --population-size "$population_size" \
    --steps-per-adapter "$steps_per_adapter" \
    --lora-r "$lora_r" \
    --task "$task" \
    --num-iterations "$num_iterations" \
    --save-freq "$save_freq" \
    --checkpoint-dir "$checkpoint_dir" \
    $NORMALIZE_FLAG \
    $SCALE_LR_FLAG \
    --prompt-batch-size "$prompt_batch_size" \
    --samples-per-prompt "$samples_per_prompt" \
    --temperature "$temperature" \
    $PASSATK_FLAG \
    --steps-per-eval "$steps_per_eval" \
    $DATASET_SIZE_CMD \
    --name-prefix "$name_prefix" \
    --use-wandb

PYTHON_EXIT_CODE=$?
echo "---------------------------------"
if [ $PYTHON_EXIT_CODE -eq 124 ]; then
    echo "Job timed out after 3 hours"
elif [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "Job finished with error code $PYTHON_EXIT_CODE"
else
    echo "Job finished successfully"
fi
echo "---------------------------------"

# Clean up Ray cluster
echo "Stopping Ray cluster..."
ray stop || true
echo "Ray cluster stopped"

# Clean up shared memory directories on all nodes (best effort)
echo "Cleaning up /dev/shm directories on all nodes..."
if [ -n "$SLURM_JOB_NUM_NODES" ] && [ -n "$SLURM_JOB_NODELIST" ]; then
    srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" bash -c '
        chmod -R u+rwx /dev/shm/es_lora_population_async_* /dev/shm/outputs_es_lora 2>/dev/null || true
        rm -rf /dev/shm/es_lora_population_async_* /dev/shm/outputs_es_lora 2>/dev/null || true
    ' || true
else
    chmod -R u+rwx /dev/shm/es_lora_population_async_* /dev/shm/outputs_es_lora 2>/dev/null || true
    rm -rf /dev/shm/es_lora_population_async_* /dev/shm/outputs_es_lora 2>/dev/null || true
fi
echo "Shared memory cleanup complete"