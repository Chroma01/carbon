#!/bin/bash
#SBATCH --job-name=carbon-x-sft
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --partition=hopper-b300,hopper-atl,hopper-prod  # Adjust this for your cluster
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue
#SBATCH --time=3-00:00:00

# Run from the carbon-x/ directory after `uv sync`.
#
# Usage:
#   sbatch --nodes=2 training/posttraining/sft.slurm --config training/posttraining/sft.yaml --accelerator deepspeed_zero3

if [[ "$*" == *"--help"* ]]; then
  echo "Usage: sbatch training/posttraining/sft.slurm --config CONFIG --accelerator CONFIG [options]"
  echo "Options:"
  echo "  --config PATH            Path to an SFT config, e.g. training/posttraining/sft.yaml (required)"
  echo "  --accelerator CONFIG     Accelerator configuration name in training/posttraining/accelerate_configs/ (e.g. deepspeed_zero3) (required)"
  echo "  --args \"ARGS\"          Optional arguments to pass to the training script"
  exit 0
fi

# Specific configuration optimized for the Hugging Face Compute Cluster
module load cuda/13.0
set -e

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

sync_local_artifacts() {
  local exit_code=$?
  set +e

  echo "SYNC START: $(date)"
  if [[ -n "${LOCAL_OUTPUT_DIR:-}" && -d "${LOCAL_OUTPUT_DIR:-}" && -n "${REQUESTED_OUTPUT_DIR:-}" ]]; then
    mkdir -p "$(dirname "$REQUESTED_OUTPUT_DIR")"
    rsync -a "${LOCAL_OUTPUT_DIR}/" "${REQUESTED_OUTPUT_DIR}/"
  fi
  echo "SYNC END: $(date)"

  trap - EXIT
  exit "$exit_code"
}

source ~/.bashrc
source .venv/bin/activate
START_TIME=$(date +%s)
echo "START TIME: $(date)"

# Default values
CONFIG_FILE=""
ACCELERATOR=""
OPTIONAL_ARGS=""

# Parse command line arguments
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

# Validate required arguments
if [[ -z "$CONFIG_FILE" || -z "$ACCELERATOR" ]]; then
  echo "Error: Missing required arguments"
  echo "Run with --help for usage information"
  exit 1
fi

GRAD_ACC_STEPS=$(grep 'gradient_accumulation_steps' $CONFIG_FILE | awk '{print $2}')
REQUESTED_OUTPUT_DIR=$(grep '^output_dir:' $CONFIG_FILE | awk '{print $2}')

# Split the string into individual arguments
IFS=' ' read -ra ARGS <<< "$OPTIONAL_ARGS"
FILTERED_ARGS=()
# Loop through the arguments and find the one with "--gradient_accumulation_steps"
for ((i = 0; i < ${#ARGS[@]}; i++)); do
    arg="${ARGS[$i]}"
    if [[ "$arg" == "--gradient_accumulation_steps="* ]]; then
        # Extract the value after the equals sign
        GRAD_ACC_STEPS="${arg#*=}"
        FILTERED_ARGS+=("$arg")
        continue
    fi
    if [[ "$arg" == "--gradient_accumulation_steps" ]]; then
        i=$((i + 1))
        GRAD_ACC_STEPS="${ARGS[$i]}"
        FILTERED_ARGS+=("$arg" "$GRAD_ACC_STEPS")
        continue
    fi
    if [[ "$arg" == "--output_dir="* ]]; then
        REQUESTED_OUTPUT_DIR="${arg#*=}"
        continue
    fi
    if [[ "$arg" == "--output_dir" ]]; then
        i=$((i + 1))
        REQUESTED_OUTPUT_DIR="${ARGS[$i]}"
        continue
    fi
    FILTERED_ARGS+=("$arg")
done

echo "Gradient accumulation steps: $GRAD_ACC_STEPS"
echo "Requested output dir: $REQUESTED_OUTPUT_DIR"

LOCAL_SCRATCH_ROOT=$(get_local_scratch_root)
LOCAL_JOB_DIR="$LOCAL_SCRATCH_ROOT/$SLURM_JOB_ID"
LOCAL_OUTPUT_DIR="$LOCAL_JOB_DIR/output/$(basename "$REQUESTED_OUTPUT_DIR")"
LOCAL_HF_HOME="$LOCAL_JOB_DIR/hf"
LOCAL_XET_CACHE="$LOCAL_JOB_DIR/xet"
LOCAL_TMPDIR="$LOCAL_JOB_DIR/tmp"
LOCAL_XDG_CACHE_HOME="$LOCAL_JOB_DIR/.cache"
LOCAL_TORCH_HOME="$LOCAL_JOB_DIR/torch"
LOCAL_TRITON_CACHE_DIR="$LOCAL_JOB_DIR/triton"

mkdir -p \
  "$LOCAL_JOB_DIR" \
  "$LOCAL_OUTPUT_DIR" \
  "$LOCAL_HF_HOME" \
  "$LOCAL_XET_CACHE" \
  "$LOCAL_TMPDIR" \
  "$LOCAL_XDG_CACHE_HOME" \
  "$LOCAL_TORCH_HOME" \
  "$LOCAL_TRITON_CACHE_DIR"

if [[ -d "$REQUESTED_OUTPUT_DIR" ]]; then
  rsync -a "$REQUESTED_OUTPUT_DIR/" "$LOCAL_OUTPUT_DIR/"
fi

export TMPDIR="$LOCAL_TMPDIR"
export XDG_CACHE_HOME="$LOCAL_XDG_CACHE_HOME"
export HF_HOME="$LOCAL_HF_HOME"
export HF_HUB_CACHE="$LOCAL_HF_HOME/hub"
export HF_XET_CACHE="$LOCAL_XET_CACHE"
export HF_DATASETS_CACHE="$LOCAL_HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export TORCH_HOME="$LOCAL_TORCH_HOME"
export TRITON_CACHE_DIR="$LOCAL_TRITON_CACHE_DIR"

mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

OPTIONAL_ARGS="${FILTERED_ARGS[*]} --output_dir=$LOCAL_OUTPUT_DIR"
trap sync_local_artifacts EXIT

echo "Using local scratch dir: $LOCAL_JOB_DIR"
echo "Using local output dir: $LOCAL_OUTPUT_DIR"

# Distributed configuration
NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=8
WORLD_SIZE=$(($NUM_NODES*$GPUS_PER_NODE))
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
MASTER_ADDR=${NODELIST[0]}  # First node for main process
MASTER_PORT=6000

echo "Training nodes: $NUM_NODES"

# force crashing on nccl issues like hanging broadcast
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1

export CMD=" \
    training/posttraining/sft.py --config $CONFIG_FILE $OPTIONAL_ARGS
    "
export LAUNCHER="ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file training/posttraining/accelerate_configs/$ACCELERATOR.yaml  \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --num_machines $NUM_NODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $SLURM_PROCID \
    --rdzv_backend=c10d \
    --max_restarts 0 \
    --tee 3 \
    "
# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --nodes=$NUM_NODES \
    --ntasks=$NUM_NODES
    "
srun $SRUN_ARGS bash -c "$LAUNCHER $CMD" 2>&1

END_TIME=$(date +%s)
echo "END TIME: $(date)"
ELAPSED_SECONDS=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_SECONDS / 3600))
MINUTES=$(( (ELAPSED_SECONDS % 3600) / 60 ))
SECONDS=$((ELAPSED_SECONDS % 60))
echo "TOTAL JOB TIME: ${HOURS}h ${MINUTES}m ${SECONDS}s (${ELAPSED_SECONDS} seconds)"
