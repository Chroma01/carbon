#!/bin/bash
#SBATCH --job-name=carbon-x-sft-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --partition=hopper-prod
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:00:00

# Run from carbon-x/. The same allocation trains, resumes, and reloads the model.
set -euo pipefail
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export CARBON_STAGE_VENV=${CARBON_STAGE_VENV:-1}
PYTHON=.venv/bin/python
if [[ "$CARBON_STAGE_VENV" == 1 ]]; then
  if [[ -d /scratch ]]; then root=/scratch; elif [[ -d /local ]]; then root=/local; else root=/tmp; fi
  export CARBON_RUNTIME_DIR=${CARBON_RUNTIME_DIR:-$root/$USER/carbon-x-$SLURM_JOB_ID/runtime}
  PYTHON="$CARBON_RUNTIME_DIR/bin/python"
  export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
  export PYTHONDONTWRITEBYTECODE=1
fi
OUTPUT_DIR="${CARBON_SMOKE_OUTPUT_DIR:-$PWD/scratch/sft-smoke-${SLURM_JOB_ID:?Submit with sbatch}}"
# Freeze the nested launcher too: sbatch snapshots this wrapper, but Bash would
# otherwise keep reading a mutable repository file after each long srun call.
mkdir -p "$OUTPUT_DIR"
LAUNCHER_SNAPSHOT="$OUTPUT_DIR/sft-launcher-${SLURM_JOB_ID}.slurm"
cp training/posttraining/sft.slurm "$LAUNCHER_SNAPSHOT"
COMMON_ARGS=(
  --dataset_train_split 'train[:32]' --dataset_test_split 'test[:8]'
  --dataset_num_proc 1 --per_device_train_batch_size 1 --per_device_eval_batch_size 1
  --gradient_accumulation_steps 1 --save_strategy steps --save_steps 2 --save_total_limit 2
  --eval_strategy no --do_eval false --report_to none --push_to_hub false
  --cleanup_after_push false --output_dir "$OUTPUT_DIR" --warmup_steps 0
  "$@"
)
if [[ "${CARBON_SMOKE_RESUME_ONLY:-0}" != 1 ]]; then
  bash "$LAUNCHER_SNAPSHOT" --config training/posttraining/sft.yaml \
    --accelerator deepspeed_zero3 -- "${COMMON_ARGS[@]}" --max_steps 3
elif [[ ! -f "$OUTPUT_DIR/checkpoint-2/trainer_state.json" ]]; then
  echo "Resume-only validation requires a complete checkpoint 2 in $OUTPUT_DIR" >&2
  exit 1
fi
bash "$LAUNCHER_SNAPSHOT" --config training/posttraining/sft.yaml \
  --accelerator deepspeed_zero3 -- "${COMMON_ARGS[@]}" --max_steps 4 \
  --resume_from_checkpoint "$OUTPUT_DIR/checkpoint-2"
"$PYTHON" training/posttraining/validate_sft_smoke.py "$OUTPUT_DIR"
