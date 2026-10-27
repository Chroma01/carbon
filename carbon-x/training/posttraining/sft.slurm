#!/bin/bash
#SBATCH --job-name=carbon-x-sft
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --partition=hopper-prod
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue
#SBATCH --time=3-00:00:00

# Run from carbon-x/. Checkpoints live on shared storage so every rank's
# optimizer shards survive node loss and are available when resuming.
set -euo pipefail

usage() {
  echo 'Usage: sbatch --nodes=N training/posttraining/sft.slurm --config CONFIG --accelerator NAME [--args "ARGS"] [-- TRAINING_ARGS...]'
}

CONFIG_FILE=''
ACCELERATOR=''
SCRIPT_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --config|--accelerator|--args)
      if [[ $# -lt 2 ]]; then echo "Missing value for $1" >&2; exit 1; fi
      case "$1" in
        --config) CONFIG_FILE=$2 ;;
        --accelerator) ACCELERATOR=$2 ;;
        --args)
          # Preserve the legacy --args interface without evaluating shell code.
          PARSED_ARGS=$(.venv/bin/python -c 'import json,shlex,sys; print(json.dumps(shlex.split(sys.argv[1])))' "$2")
          mapfile -d '' -t EXTRA_ARGS < <(.venv/bin/python -c 'import json,sys; [sys.stdout.write(arg + "\0") for arg in json.loads(sys.argv[1])]' "$PARSED_ARGS")
          SCRIPT_ARGS+=("${EXTRA_ARGS[@]}")
          ;;
      esac
      shift 2
      ;;
    --) shift; SCRIPT_ARGS+=("$@"); break ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$CONFIG_FILE" || -z "$ACCELERATOR" ]]; then usage; exit 1; fi
ACCELERATOR_CONFIG="training/posttraining/accelerate_configs/$ACCELERATOR.yaml"
[[ -f "$CONFIG_FILE" && -f "$ACCELERATOR_CONFIG" && -x .venv/bin/accelerate ]] || {
  echo 'Missing config or SFT environment. Run uv sync --locked --no-default-groups --group sft.' >&2
  exit 1
}

# Keep the original login when per-node HF_HOME is redirected to local scratch.
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/token}"
if type module &>/dev/null; then module load "${CARBON_CUDA_MODULE:-cuda/13.0}"; fi
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export ACCELERATE_LOG_LEVEL=info
export TRANSFORMERS_VERBOSITY=info
NUM_NODES=${SLURM_NNODES:?Submit this script with sbatch}
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-8}
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))
mapfile -t NODELIST < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
MASTER_ADDR=${NODELIST[0]}
MASTER_PORT=${MASTER_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 40000))}

# All workers enter the shared working directory. Each prepares its own caches;
# machine_rank is expanded inside srun, where SLURM_PROCID is the node rank.
srun --wait=60 --kill-on-bad-exit=1 --nodes="$NUM_NODES" --ntasks="$NUM_NODES" bash -c '
  set -euo pipefail
  if [[ -d /scratch ]]; then root=/scratch; elif [[ -d /local ]]; then root=/local; else root=/tmp; fi
  job_dir="$root/$USER/carbon-x-$SLURM_JOB_ID"
  export TMPDIR="$job_dir/tmp"
  export XDG_CACHE_HOME="$job_dir/cache"
  export HF_HOME="$job_dir/hf"
  export HF_HUB_CACHE="$HF_HOME/hub"
  export HF_DATASETS_CACHE="$HF_HOME/datasets"
  export HF_XET_CACHE="$job_dir/xet"
  export TORCH_HOME="$job_dir/torch"
  export TRITON_CACHE_DIR="$job_dir/triton"
  mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_XET_CACHE" "$TORCH_HOME" "$TRITON_CACHE_DIR"
  python_executable=.venv/bin/python
  if [[ "${CARBON_STAGE_VENV:-0}" == 1 ]]; then
    runtime_dir="${CARBON_RUNTIME_DIR:-$job_dir/runtime}"
    if [[ ! -f "$runtime_dir/.ready" ]]; then
      mkdir -p "$runtime_dir"
      echo "Staging SFT environment to $runtime_dir on $(hostname)"
      if [[ -n "${CARBON_VENV_ARCHIVE:-}" ]]; then
        tar -C "$runtime_dir" -xf "$CARBON_VENV_ARCHIVE"
      else
        tar -C .venv -cf - . | tar -C "$runtime_dir" -xf -
      fi
      touch "$runtime_dir/.ready"
    fi
    # Invoke the copied venv directly: shared entry-point shebangs would keep
    # shared site-packages on sys.path and trigger slow metadata scans.
    python_executable="$runtime_dir/bin/python"
    export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
    export PYTHONDONTWRITEBYTECODE=1
  fi
  exec "$python_executable" -m accelerate.commands.launch \
    --config_file "$1" --num_machines "$2" --num_processes "$3" \
    --main_process_ip "$4" --main_process_port "$5" \
    --machine_rank "$SLURM_PROCID" --rdzv_backend static --max_restarts 0 --tee 3 \
    training/posttraining/sft.py "${@:6}"
' _ "$ACCELERATOR_CONFIG" "$NUM_NODES" "$WORLD_SIZE" "$MASTER_ADDR" "$MASTER_PORT" \
  --config "$CONFIG_FILE" "${SCRIPT_ARGS[@]}"
