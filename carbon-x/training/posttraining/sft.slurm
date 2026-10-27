#!/bin/bash
#SBATCH --job-name=carbon-x-sft
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --partition=hopper-b300,hopper-atl,hopper-prod  # Adjust this for your cluster
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue
#SBATCH --time=3-00:00:00

# Run from the carbon-x/ directory after `uv sync`.
# Usage:
#   sbatch --nodes=2 training/posttraining/sft.slurm --config training/posttraining/sft.yaml --accelerator deepspeed_zero3
set -euo pipefail

CONFIG_FILE=""
ACCELERATOR=""
SCRIPT_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      echo 'Usage: sbatch --nodes=N training/posttraining/sft.slurm --config CONFIG --accelerator NAME [--args "ARGS"] [-- TRAINING_ARGS...]'
      exit 0
      ;;
    --config) CONFIG_FILE="$2"; shift 2 ;;
    --accelerator) ACCELERATOR="$2"; shift 2 ;;
    --args) read -r -a EXTRA_ARGS <<< "$2"; SCRIPT_ARGS+=("${EXTRA_ARGS[@]}"); shift 2 ;;
    --) shift; SCRIPT_ARGS+=("$@"); break ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$CONFIG_FILE" || -z "$ACCELERATOR" ]]; then
  echo "Both --config and --accelerator are required." >&2
  exit 1
fi

module load cuda/13.0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export ACCELERATE_LOG_LEVEL=info
export TRANSFORMERS_VERBOSITY=info

NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-8}
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))
mapfile -t NODELIST < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
MASTER_ADDR=${NODELIST[0]}
MASTER_PORT=${MASTER_PORT:-6000}

# Expand the machine rank inside each Slurm task.
srun --wait=60 --kill-on-bad-exit=1 --nodes="$NUM_NODES" --ntasks="$NUM_NODES" \
  bash -c 'exec .venv/bin/accelerate launch --machine_rank "$SLURM_PROCID" "$@"' _ \
  --config_file "training/posttraining/accelerate_configs/$ACCELERATOR.yaml" \
  --num_machines "$NUM_NODES" --num_processes "$WORLD_SIZE" \
  --main_process_ip "$MASTER_ADDR" --main_process_port "$MASTER_PORT" \
  --rdzv_backend static --max_restarts 0 --tee 3 \
  training/posttraining/sft.py --config "$CONFIG_FILE" "${SCRIPT_ARGS[@]}"
