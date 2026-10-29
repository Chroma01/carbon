#!/bin/bash
#SBATCH --job-name=carbon-x-rl
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=10
#SBATCH --gres=gpu:8
#SBATCH --partition=hopper-b300,hopper-atl,hopper-prod
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue
#SBATCH --time=3-00:00:00

# Run from carbon-x/ after `uv sync` and `venv-ship .venv` on this cluster.
# The last node serves vLLM; preceding nodes train. A single node splits its GPUs.
set -euo pipefail

CONFIG_FILE=""
ACCELERATOR=""
VLLM_ARG_STRING=""
SCRIPT_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      echo 'Usage: sbatch training/posttraining/rl.slurm --config CONFIG --accelerator NAME [--vllm_args "key=value ..."] [-- TRAINING_ARGS...]'
      exit 0 ;;
    --config) CONFIG_FILE="$2"; shift 2 ;;
    --accelerator) ACCELERATOR="$2"; shift 2 ;;
    --vllm_args) VLLM_ARG_STRING="$2"; shift 2 ;;
    --args) read -r -a EXTRA_ARGS <<< "$2"; SCRIPT_ARGS+=("${EXTRA_ARGS[@]}"); shift 2 ;;
    --) shift; SCRIPT_ARGS+=("$@"); break ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done
[[ -n "$CONFIG_FILE" && -n "$ACCELERATOR" ]] || { echo '--config and --accelerator are required.' >&2; exit 1; }

module load cuda/13.0
source "$(venv-load .venv)"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export ACCELERATE_LOG_LEVEL=info
export TRANSFORMERS_VERBOSITY=info
export TRITON_CACHE_DIR="/tmp/triton-$USER-$SLURM_JOB_ID"
export VLLM_USE_AOT_COMPILE=0

MODEL=$(awk '$1 == "model_name_or_path:" {print $2; exit}' "$CONFIG_FILE")
REVISION=$(awk '$1 == "model_revision:" {print $2; exit}' "$CONFIG_FILE")
GPUS_PER_NODE=$SLURM_GPUS_ON_NODE
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
MASTER_ADDR=${NODES[0]}
MASTER_PORT=${MASTER_PORT:-6000}
VLLM_PORT=${VLLM_PORT:-8000}
VLLM_NODE=${NODES[-1]}
VLLM_URL="http://$VLLM_NODE:$VLLM_PORT"
mkdir -p logs
VLLM_LOG="logs/vllm-$SLURM_JOB_ID.log"

VLLM_DP=1
VLLM_TP=1
VLLM_PP=1
VLLM_USER_ARGS=()
read -r -a VLLM_ARG_PAIRS <<< "$VLLM_ARG_STRING"
for arg in "${VLLM_ARG_PAIRS[@]}"; do
  [[ "$arg" == *=* ]] || { echo "Expected key=value: $arg" >&2; exit 1; }
  key=${arg%%=*}
  value=${arg#*=}
  case "$key" in
    host|port|logprobs_mode|weight_transfer_config)
      echo "vLLM argument $key is managed by the launcher." >&2; exit 1 ;;
    data_parallel_size) VLLM_DP=$value ;;
    tensor_parallel_size) VLLM_TP=$value ;;
    pipeline_parallel_size) VLLM_PP=$value ;;
  esac
  flag="--${key//_/-}"
  case "$value" in
    true) VLLM_USER_ARGS+=("$flag") ;;
    false) VLLM_USER_ARGS+=("--no-${flag#--}") ;;
    *) VLLM_USER_ARGS+=("$flag" "$value") ;;
  esac
done
for size in "$VLLM_DP" "$VLLM_TP" "$VLLM_PP"; do
  [[ "$size" =~ ^[1-9][0-9]*$ ]] || { echo 'Parallel sizes must be positive integers.' >&2; exit 1; }
done
VLLM_GPUS=$((VLLM_DP * VLLM_TP * VLLM_PP))
[[ "$VLLM_GPUS" -le "$GPUS_PER_NODE" ]] || { echo 'vLLM needs more GPUs than allocated per node.' >&2; exit 1; }
VLLM_START=0
TRAIN_GPUS=$GPUS_PER_NODE
if [[ "$SLURM_NNODES" -eq 1 ]]; then
  TRAIN_NODES=("${NODES[0]}")
  TRAIN_GPUS=$((GPUS_PER_NODE - VLLM_GPUS))
  [[ "$TRAIN_GPUS" -gt 0 ]] || { echo 'Single-node RL needs a training GPU.' >&2; exit 1; }
  VLLM_START=$TRAIN_GPUS
else
  TRAIN_NODES=("${NODES[@]:0:${#NODES[@]}-1}")
fi
NUM_TRAIN_NODES=${#TRAIN_NODES[@]}
WORLD_SIZE=$((NUM_TRAIN_NODES * TRAIN_GPUS))
TRAIN_NODELIST=$(IFS=,; echo "${TRAIN_NODES[*]}")

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
VLLM_ARGS=(--host 0.0.0.0 --port "$VLLM_PORT" --logprobs-mode processed_logprobs
  --weight-transfer-config '{"backend":"nccl"}' --revision "$REVISION" "${VLLM_USER_ARGS[@]}")
if grep -qE '^trust_remote_code: *true' "$CONFIG_FILE"; then VLLM_ARGS+=(--trust-remote-code); fi

# Slice each node's own allocation: physical GPU IDs may differ between nodes.
srun --overlap --cpus-per-gpu="$SLURM_CPUS_PER_GPU" --nodes=1 --ntasks=1 --nodelist="$VLLM_NODE" \
  --output="$VLLM_LOG" --error="$VLLM_LOG" bash -c '
    IFS=, read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
    export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${devices[*]:$1:$2}")
    shift 2
    export VLLM_SERVER_DEV_MODE=1
    # The HTTP port is a CLI argument; let replicas choose independent internal ports.
    unset VLLM_PORT
    exec vllm serve "$@"
  ' _ "$VLLM_START" "$VLLM_GPUS" "$MODEL" "${VLLM_ARGS[@]}" &
VLLM_PID=$!

for ((attempt=0; attempt<180; attempt++)); do
  if curl -sf --max-time 5 "$VLLM_URL/health" >/dev/null; then break; fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null || [[ "$attempt" -eq 179 ]]; then
    echo 'vLLM failed to become healthy.' >&2
    tail -n 100 "$VLLM_LOG"
    exit 1
  fi
  sleep 10
done
SERVER_WORLD=$(curl -sf --max-time 10 "$VLLM_URL/get_world_size" | python -c 'import json,sys; print(json.load(sys.stdin)["world_size"])')
[[ "$SERVER_WORLD" -eq "$VLLM_GPUS" ]] || { echo "Unexpected vLLM world size: $SERVER_WORLD" >&2; exit 1; }

srun --overlap --cpus-per-gpu="$SLURM_CPUS_PER_GPU" --wait=60 --kill-on-bad-exit=1 \
  --nodes="$NUM_TRAIN_NODES" --ntasks="$NUM_TRAIN_NODES" --nodelist="$TRAIN_NODELIST" bash -c '
    IFS=, read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
    export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${devices[*]:0:$1}")
    shift
    host=$(hostname -s)
    if [[ "$host" == ip-* ]]; then host=${host#ip-}; host=${host//-/.}; fi
    export VLLM_HOST_IP="$host"
    exec accelerate launch --machine_rank "$SLURM_PROCID" "$@"
  ' _ "$TRAIN_GPUS" --config_file "training/posttraining/accelerate_configs/$ACCELERATOR.yaml" \
  --num_machines "$NUM_TRAIN_NODES" --num_processes "$WORLD_SIZE" \
  --main_process_ip "$MASTER_ADDR" --main_process_port "$MASTER_PORT" \
  --rdzv_backend static --max_restarts 0 --tee 3 \
  training/posttraining/rl.py --config "$CONFIG_FILE" "${SCRIPT_ARGS[@]}" --vllm_server_base_url "$VLLM_URL"
