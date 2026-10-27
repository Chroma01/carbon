#!/bin/bash
#SBATCH --job-name=carbon-x-rl
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=hopper-b300,hopper-atl,hopper-prod  # Adjust this for your cluster
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue
#SBATCH --time=3-00:00:00

# Run from the carbon-x/ directory after `uv sync`.

# Slurm launcher for RL training jobs.
#
# In multi-node runs, the last allocated node runs a plain vllm serve generation server and all preceding nodes run
# training. In single-node runs, vLLM and training share the node with disjoint CUDA_VISIBLE_DEVICES assignments.
# AsyncGRPOTrainer talks to vLLM over the OpenAI-compatible API and transfers weights using NCCL.
#
# Usage:
#   sbatch training/posttraining/rl.slurm --config training/posttraining/rl.yaml --accelerator fsdp2 \
#     --vllm_args "data_parallel_size=8 tensor_parallel_size=1 dtype=bfloat16 gpu_memory_utilization=0.85"

if [[ "$*" == *"--help"* ]]; then
  echo "Usage: sbatch training/posttraining/rl.slurm --config CONFIG --accelerator CONFIG [options]"
  echo "Options:"
  echo "  --config PATH             Path to an RL config, e.g. training/posttraining/rl.yaml (required)"
  echo "  --accelerator CONFIG      Accelerator configuration name in training/posttraining/accelerate_configs/ (e.g. fsdp2) (required)"
  echo "  --vllm_args \"ARGS\"        Optional vLLM server args as key=value tokens"
  echo "  --args \"ARGS\"             Optional extra arguments forwarded to training/posttraining/rl.py"
  echo ""
  echo "Submit with sbatch --nodes=1 --gres=gpu:N for a partial-node single-node run. vLLM requests"
  echo "M GPUs, where M is data_parallel_size * tensor_parallel_size *"
  echo "pipeline_parallel_size, and training uses the rest."
  exit 0
fi

module load cuda/13.0
set -x -e

get_local_scratch_root() {
  if [[ -d /scratch ]]; then
    echo "/scratch/${USER}"
    return
  fi
  if [[ -d /local ]]; then
    echo "/local/${USER}"
    return
  fi
  echo "/tmp/${USER}"
}

source ~/.bashrc
source .venv/bin/activate

CONFIG_FILE=""
ACCELERATOR=""
VLLM_ARG_STRING=""
OPTIONAL_ARGS=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --accelerator)
      ACCELERATOR="$2"
      shift 2
      ;;
    --vllm_args)
      VLLM_ARG_STRING="$2"
      shift 2
      ;;
    --args)
      OPTIONAL_ARGS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

if [[ -z "$CONFIG_FILE" ]]; then
  echo "Error: --config is required"
  exit 1
fi
if [[ -z "$ACCELERATOR" ]]; then
  echo "Error: --accelerator is required"
  exit 1
fi
if [[ "${SLURM_NNODES:-0}" -lt 1 ]]; then
  echo "Error: RL must be run under Slurm with at least one allocated node."
  exit 1
fi

LOCAL_SCRATCH_ROOT=$(get_local_scratch_root)
LOCAL_JOB_DIR="$LOCAL_SCRATCH_ROOT/$SLURM_JOB_ID"
LOCAL_HF_HOME="$LOCAL_JOB_DIR/hf"
LOCAL_XET_CACHE="$LOCAL_JOB_DIR/xet"
LOCAL_TMPDIR="$LOCAL_JOB_DIR/tmp"
LOCAL_XDG_CACHE_HOME="$LOCAL_JOB_DIR/.cache"
LOCAL_TORCH_HOME="$LOCAL_JOB_DIR/torch"
LOCAL_TRITON_CACHE_DIR="$LOCAL_JOB_DIR/triton"
LOCAL_VLLM_CACHE_ROOT="$LOCAL_JOB_DIR/vllm"

export TMPDIR="$LOCAL_TMPDIR"
export XDG_CACHE_HOME="$LOCAL_XDG_CACHE_HOME"
export HF_HOME="$LOCAL_HF_HOME"
export HF_HUB_CACHE="$LOCAL_HF_HOME/hub"
export HF_XET_CACHE="$LOCAL_XET_CACHE"
export HF_DATASETS_CACHE="$LOCAL_HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export TORCH_HOME="$LOCAL_TORCH_HOME"
export TRITON_CACHE_DIR="$LOCAL_TRITON_CACHE_DIR"
export VLLM_CACHE_ROOT="$LOCAL_VLLM_CACHE_ROOT"
export VLLM_USE_AOT_COMPILE=0

srun --overlap --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" mkdir -p \
  "$LOCAL_JOB_DIR" \
  "$LOCAL_HF_HOME" \
  "$LOCAL_XET_CACHE" \
  "$LOCAL_TMPDIR" \
  "$LOCAL_XDG_CACHE_HOME" \
  "$LOCAL_TORCH_HOME" \
  "$LOCAL_TRITON_CACHE_DIR" \
  "$LOCAL_VLLM_CACHE_ROOT" \
  "$HF_HUB_CACHE" \
  "$HF_DATASETS_CACHE"

echo "Using local scratch dir: $LOCAL_JOB_DIR"

MODEL=$(grep -E '^\s*model_name_or_path:' "$CONFIG_FILE" | head -n 1 | awk '{print $2}')
REVISION=$(grep -E '^\s*model_revision:' "$CONFIG_FILE" | head -n 1 | awk '{print $2}')

make_gpu_ids() {
  local count="$1"
  local devices=()
  local gpu
  for ((gpu = 0; gpu < count; gpu++)); do
    devices+=("$gpu")
  done
  local IFS=,
  echo "${devices[*]}"
}

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != "NoDevFiles" ]]; then
  IFS=, read -r -a ALLOCATED_GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
elif [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
  IFS=, read -r -a ALLOCATED_GPU_IDS <<< "$SLURM_JOB_GPUS"
elif [[ "${SLURM_GPUS_ON_NODE:-}" =~ ^[0-9]+$ ]]; then
  IFS=, read -r -a ALLOCATED_GPU_IDS <<< "$(make_gpu_ids "$SLURM_GPUS_ON_NODE")"
elif [[ "${SLURM_GPUS_PER_NODE:-}" =~ ^[0-9]+$ ]]; then
  IFS=, read -r -a ALLOCATED_GPU_IDS <<< "$(make_gpu_ids "$SLURM_GPUS_PER_NODE")"
else
  ALLOCATED_GPU_IDS=(0 1 2 3 4 5 6 7)
fi

GPUS_PER_NODE=${#ALLOCATED_GPU_IDS[@]}
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
MASTER_PORT=6000
VLLM_PORT=8000
LOG_DIR="${RL_LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
VLLM_LOG="${LOG_DIR}/vllm-${SLURM_JOB_ID}.log"

VLLM_ARG_PAIRS=()
VLLM_USER_ARGS=()
if [[ -n "$VLLM_ARG_STRING" ]]; then
  read -r -a VLLM_ARG_PAIRS <<< "$VLLM_ARG_STRING"
fi

vllm_arg_value() {
  local key="$1"
  local default="$2"
  local arg
  for arg in "${VLLM_ARG_PAIRS[@]}"; do
    if [[ "$arg" == "${key}="* ]]; then
      echo "${arg#*=}"
      return
    fi
  done
  echo "$default"
}

for arg in "${VLLM_ARG_PAIRS[@]}"; do
  if [[ "$arg" != *=* ]]; then
    echo "Error: --vllm_args entries must use key=value format, got: $arg" >&2
    exit 1
  fi
  key="${arg%%=*}"
  value="${arg#*=}"
  if [[ -z "$key" || -z "$value" ]]; then
    echo "Error: --vllm_args entries must use non-empty key=value format, got: $arg" >&2
    exit 1
  fi
  case "$key" in
    host|port|logprobs_mode|weight_transfer_config)
      echo "Error: vLLM arg '$key' is managed by this launcher and cannot be passed via --vllm_args." >&2
      exit 1
      ;;
  esac
  flag="--${key//_/-}"
  if [[ "$value" == "true" ]]; then
    VLLM_USER_ARGS+=("$flag")
  elif [[ "$value" == "false" ]]; then
    VLLM_USER_ARGS+=("--no-${flag#--}")
  else
    VLLM_USER_ARGS+=("$flag" "$value")
  fi
done

VLLM_DP=$(vllm_arg_value data_parallel_size 1)
VLLM_TP=$(vllm_arg_value tensor_parallel_size 1)
VLLM_PP=$(vllm_arg_value pipeline_parallel_size 1)
for parallel_size in "$VLLM_DP" "$VLLM_TP" "$VLLM_PP"; do
  if ! [[ "$parallel_size" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: data_parallel_size, tensor_parallel_size, and pipeline_parallel_size must be positive integers." >&2
    exit 1
  fi
done
EXPECTED_VLLM_WORLD_SIZE=$((VLLM_DP * VLLM_TP * VLLM_PP))
if [[ "$EXPECTED_VLLM_WORLD_SIZE" -gt "$GPUS_PER_NODE" ]]; then
  echo "Error: vLLM parallel size requires $EXPECTED_VLLM_WORLD_SIZE GPUs, but this launcher supports at most $GPUS_PER_NODE GPUs on the vLLM node." >&2
  exit 1
fi

gpu_ids() {
  local start="$1"
  local count="$2"
  local devices=()
  local index
  for ((index = start; index < start + count; index++)); do
    devices+=("${ALLOCATED_GPU_IDS[$index]}")
  done
  local IFS=,
  echo "${devices[*]}"
}

if [[ "$SLURM_NNODES" -eq 1 ]]; then
  if [[ "$EXPECTED_VLLM_WORLD_SIZE" -ge "$GPUS_PER_NODE" ]]; then
    echo "Error: single-node RL needs at least one training GPU. vLLM requested $EXPECTED_VLLM_WORLD_SIZE of $GPUS_PER_NODE GPUs." >&2
    exit 1
  fi
  VLLM_NODE="${NODES[0]}"
  TRAIN_NODES=("${NODES[0]}")
  TRAIN_GPUS_PER_NODE=$((GPUS_PER_NODE - EXPECTED_VLLM_WORLD_SIZE))
  TRAIN_VISIBLE_DEVICES=$(gpu_ids 0 "$TRAIN_GPUS_PER_NODE")
  VLLM_VISIBLE_DEVICES=$(gpu_ids "$TRAIN_GPUS_PER_NODE" "$EXPECTED_VLLM_WORLD_SIZE")
else
  VLLM_NODE="${NODES[-1]}"
  TRAIN_NODES=("${NODES[@]:0:$((${#NODES[@]} - 1))}")
  TRAIN_GPUS_PER_NODE=$GPUS_PER_NODE
  TRAIN_VISIBLE_DEVICES=$(gpu_ids 0 "$TRAIN_GPUS_PER_NODE")
  VLLM_VISIBLE_DEVICES=$(gpu_ids 0 "$EXPECTED_VLLM_WORLD_SIZE")
fi
NUM_TRAIN_NODES=${#TRAIN_NODES[@]}
WORLD_SIZE=$((NUM_TRAIN_NODES * TRAIN_GPUS_PER_NODE))
MASTER_ADDR="${TRAIN_NODES[0]}"
TRAIN_NODELIST=$(IFS=,; echo "${TRAIN_NODES[*]}")
VLLM_URL="http://${VLLM_NODE}:${VLLM_PORT}"

cleanup() {
  [[ -n "${VLLM_PID:-}" ]] && kill "$VLLM_PID" 2>/dev/null
}
trap cleanup EXIT

echo "Trainer nodes: $TRAIN_NODELIST (GPUs=$TRAIN_VISIBLE_DEVICES)"
echo "vLLM node: $VLLM_NODE (GPUs=$VLLM_VISIBLE_DEVICES, DP=$VLLM_DP, TP=$VLLM_TP, PP=$VLLM_PP; args=${VLLM_ARG_STRING:-<vLLM defaults>})"
VLLM_ARGS=(
  --host 0.0.0.0
  --port "$VLLM_PORT"
  --logprobs-mode processed_logprobs
  --weight-transfer-config '{"backend":"nccl"}'
  "${VLLM_USER_ARGS[@]}"
)
if [[ -n "$REVISION" && -z "$(vllm_arg_value revision "")" ]]; then
  VLLM_ARGS+=(--revision "$REVISION")
fi
if grep -qE '^\s*trust_remote_code:\s*true' "$CONFIG_FILE" && [[ -z "$(vllm_arg_value trust_remote_code "")" ]]; then
  VLLM_ARGS+=(--trust-remote-code)
fi

srun --overlap --nodes=1 --ntasks=1 --nodelist="$VLLM_NODE" \
  --output="$VLLM_LOG" --error="$VLLM_LOG" \
  env CUDA_VISIBLE_DEVICES="$VLLM_VISIBLE_DEVICES" VLLM_SERVER_DEV_MODE=1 vllm serve "$MODEL" "${VLLM_ARGS[@]}" &
VLLM_PID=$!

echo "Waiting for vLLM server at $VLLM_URL..."
for i in $(seq 1 180); do
  if curl -sf --max-time 5 "${VLLM_URL}/health" >/dev/null; then
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM server died during startup."
    tail -n 200 "$VLLM_LOG"
    exit 1
  fi
  if [[ "$i" -eq 180 ]]; then
    echo "vLLM server did not become healthy within 1800 seconds."
    tail -n 200 "$VLLM_LOG"
    exit 1
  fi
  sleep 10
done

WORLD=$(curl -sf --max-time 10 "${VLLM_URL}/get_world_size" || echo "")
EXPECTED_VLLM_WORLD_SIZE=$((VLLM_DP * VLLM_TP * VLLM_PP))
if ! echo "$WORLD" | grep -q "\"world_size\":[[:space:]]*${EXPECTED_VLLM_WORLD_SIZE}"; then
  echo "Expected vLLM world_size $EXPECTED_VLLM_WORLD_SIZE (DP=$VLLM_DP, TP=$VLLM_TP, PP=$VLLM_PP), got: ${WORLD:-<no response>}" >&2
  tail -n 200 "$VLLM_LOG"
  exit 1
fi

echo "Starting training on $NUM_TRAIN_NODES node(s), $WORLD_SIZE GPUs with $ACCELERATOR..."
LAUNCHER="ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch \
  --config_file training/posttraining/accelerate_configs/$ACCELERATOR.yaml \
  --num_machines $NUM_TRAIN_NODES \
  --num_processes $WORLD_SIZE \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  --machine_rank \$SLURM_PROCID \
  --rdzv_backend c10d \
  --max_restarts 0 \
  --tee 3 \
  training/posttraining/rl.py --config $CONFIG_FILE $OPTIONAL_ARGS \
  --vllm_server_base_url $VLLM_URL"

TRAIN_SRUN_ARGS=(srun --wait=60 --kill-on-bad-exit=1)
if [[ "$SLURM_NNODES" -eq 1 ]]; then
  TRAIN_SRUN_ARGS+=(--overlap)
fi
TRAIN_SRUN_ARGS+=(--nodes="$NUM_TRAIN_NODES" --ntasks="$NUM_TRAIN_NODES" --nodelist="$TRAIN_NODELIST")
# Global rank 0 opens the weight-transfer socket that vLLM connects to, and the c10d rendezvous can place it on any
# training node, so each node sets VLLM_HOST_IP to its own address (ip-10-53-196-114 -> 10.53.196.114)
"${TRAIN_SRUN_ARGS[@]}" \
  env CUDA_VISIBLE_DEVICES="$TRAIN_VISIBLE_DEVICES" \
  bash -c "host=\$(hostname -s); if [[ \$host == ip-* ]]; then host=\${host#ip-}; host=\${host//-/.}; fi; VLLM_HOST_IP=\$host $LAUNCHER"

echo "Training finished."
