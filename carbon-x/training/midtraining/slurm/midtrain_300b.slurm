#!/bin/bash
# Carbon-X midtraining: 300B tokens on Qwen3-14B-Instruct with the hybrid DNA tokenizer.
#
# Mixture: 55% DNA / 15% bio text / 20% English / 10% instruction SFT.
# Schedule: cosine, LR 5e-5 -> 5e-6, warmup 2000, GBS 512, sequence length 8192, TP 4, 32 nodes x 8 GPUs.
# Starts from the instruct freeze-body-DNA checkpoint and resumes from SAVE_PATH if one is already there.
#
#   bash slurm/midtrain_300b.slurm --dry-run           # print the blend, weight sum and iterations, no GPUs
#   SMOKE=1 MIXTURE=$PWD/configs/mixture_300b.sh sbatch slurm/midtrain_300b.slurm   # 20-iter check, no ckpt
#   MIXTURE=$PWD/configs/mixture_300b.sh sbatch slurm/midtrain_300b.slurm           # the full run
#
# Partition and QoS below are specific to our cluster; change them for yours.
#SBATCH --job-name=carbon-x-midtrain-300b-14b
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=5-00:00:00

DRYRUN=0; SMOKE=${SMOKE:-0}
for a in "$@"; do [ "$a" = "--dry-run" ] && DRYRUN=1; [ "$a" = "--smoke" ] && SMOKE=1; done
[ $DRYRUN -eq 0 ] && set -x

# ----------------------------------------------------------------------------
# Point these at your own environment and data before running.
# ----------------------------------------------------------------------------
VENV="${VENV:-.venv}"                                  # environment with Megatron-LM and Transformer Engine
MEGATRON_PATH="${MEGATRON_PATH:?set MEGATRON_PATH to your Megatron-LM-Carbon checkout}"
CKPT_ROOT="${CKPT_ROOT:-checkpoints}"                  # where checkpoints are written
PRETRAINED_MEGATRON_CKPT="${PRETRAINED_MEGATRON_CKPT:?set PRETRAINED_MEGATRON_CKPT to the freeze-body-DNA checkpoint}"

# Megatron-preprocessed shards. Every --data-path prefix needs BOTH a .bin and a .idx:
# a missing .idx kills rank 0 at startup and the job looks like it is hanging.
DNA_DIR="${DNA_DIR:?set DNA_DIR}"
BIO_DIR="${BIO_DIR:?set BIO_DIR}"
ENGLISH_DIR="${ENGLISH_DIR:?set ENGLISH_DIR}"

WANDB_PROJECT="${WANDB_PROJECT:-carbon-x}"
# ----------------------------------------------------------------------------

[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
export CUDA_DEVICE_MAX_CONNECTIONS=1 NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_NVLS_ENABLE=0 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800 FI_EFA_USE_HUGE_PAGE=0

TAIL=npad
TOKENIZER="${TOKENIZER:-HuggingFaceBio/Carbon-X}"
TRAINING_SCRIPT="${MEGATRON_PATH}/pretrain_gpt.py"

TP=4; PP=1; MBS=1; GBS=512; SEQ=8192
WARMUP=2000
if [ "$SMOKE" -eq 1 ]; then
  TRAIN_ITERS=20; WARMUP=2; SAVE_INTERVAL=1000000
  SAVE_PATH="${CKPT_ROOT}/carbon-x-midtrain-smoke-${SLURM_JOB_ID:-manual}"
  RUN_NAME="carbon-x-midtrain-smoke"
else
  TRAIN_ITERS=72000                                    # 302B tokens at GBS 512 x 8192
  SAVE_INTERVAL=4000
  SAVE_PATH="${CKPT_ROOT}/carbon-x-midtrain-300b-14b"
  RUN_NAME="carbon-x-midtrain-300b-14b"
fi

# The mixture lives in its own file so it can be read and diffed on its own; it sets DATA_PATH.
# sbatch copies this script into the SLURM spool before running it, so $BASH_SOURCE points at the spool and
# not at the repo. Pass MIXTURE=/path/to/configs/mixture_300b.sh when submitting; the relative guess below
# only works when the script is run directly with bash from the repo.
MIXTURE="${MIXTURE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)/../configs/mixture_300b.sh}"
[ -f "$MIXTURE" ] || { echo "ERROR: mixture not found at '$MIXTURE'. Set MIXTURE to configs/mixture_300b.sh" >&2; exit 1; }
source "$MIXTURE"

# A bare `python` exists in the NeMo container but not on every login/compute node, and WSUM=$(python ...) hides
# the failure from set -e: the blend would then carry no weights AND the sum check below would be skipped. Resolve
# the interpreter up front and fail if there is none.
PYBIN="${PYBIN:-$(command -v python || command -v python3)}"
[ -n "$PYBIN" ] || { echo "ERROR: no python found; set PYBIN=/path/to/python" >&2; exit 1; }
WSUM=$("$PYBIN" -c "w=[float(x) for i,x in enumerate('$DATA_PATH'.split()) if i%2==0]; print(f'{sum(w):.4f} ({len(w)} shards)')") \
  || { echo "ERROR: could not compute the data weight sum" >&2; exit 1; }
[ -n "$WSUM" ] || { echo "ERROR: empty weight sum -- the mixture produced no data paths" >&2; exit 1; }
echo "smoke=$SMOKE  weight_sum=$WSUM  train_iters=$TRAIN_ITERS  warmup=$WARMUP  lr 5e-5 cosine min-lr 5e-6  (GBS=$GBS seq=$SEQ TP=$TP)"
echo "save=$SAVE_PATH  init=$PRETRAINED_MEGATRON_CKPT"
# This script has no `set -e` (the Megatron launch below manages its own failures), so the check has to abort
# explicitly. Without the `||` the error would print and training would start on the wrong mixture anyway.
"$PYBIN" - "$WSUM" <<'PYCHK' || exit 1
import sys
total = float(sys.argv[1].split()[0])
if abs(total - 1.0) > 1e-3:
    sys.exit(f"ERROR: data weights sum to {total:.4f}, expected 1.0 (a source is missing or mis-weighted)")
PYCHK
[ $DRYRUN -eq 1 ] && exit 0
[ -d "${PRETRAINED_MEGATRON_CKPT}" ] || { echo "missing init checkpoint ${PRETRAINED_MEGATRON_CKPT}"; exit 1; }

if [ -f "${SAVE_PATH}/latest_checkpointed_iteration.txt" ]; then
  LOAD_ARGS="--load ${SAVE_PATH} --no-load-rng"
  echo "resuming from ${SAVE_PATH} (iteration $(cat ${SAVE_PATH}/latest_checkpointed_iteration.txt))"
else
  LOAD_ARGS="--load ${PRETRAINED_MEGATRON_CKPT} --finetune --no-load-optim --no-load-rng"
  echo "fresh start from ${PRETRAINED_MEGATRON_CKPT}"
fi

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST"|head -n1); export MASTER_PORT=6000
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST"|wc -l)

# --untie-embeddings-and-output-weights is required: Qwen3-14B has untied embeddings, and without this flag
# Megatron ties them and the lm_head is dropped at conversion with no error.
TRAINING_PARAMS=" --distributed-timeout-minutes 180 \
  --tensor-model-parallel-size ${TP} --pipeline-model-parallel-size ${PP} \
  --use-distributed-optimizer --use-mcore-models --disable-bias-linear --untie-embeddings-and-output-weights \
  --micro-batch-size ${MBS} --global-batch-size ${GBS} --train-iters ${TRAIN_ITERS} \
  --manual-gc --manual-gc-interval 5 --transformer-impl transformer_engine --no-gradient-accumulation-fusion \
  --tokenizer-type HuggingFaceTokenizer --tokenizer-model ${TOKENIZER} --trust-remote-code \
  --data-path ${DATA_PATH} --split 100,0,0 --no-mmap-bin-files --num-workers 6 --dataloader-type cyclic \
  --num-layers 40 --hidden-size 5120 --ffn-hidden-size 17408 --num-attention-heads 40 --kv-channels 128 \
  --group-query-attention --num-query-groups 8 --seq-length ${SEQ} --max-position-embeddings ${SEQ} \
  --make-vocab-size-divisible-by 1 --position-embedding-type rope --rotary-percent 1.0 --rotary-base 1000000 \
  --rotary-seq-len-interpolation-factor 1 --normalization RMSNorm --swiglu --norm-epsilon 1e-6 --qk-layernorm \
  --attention-dropout 0.0 --hidden-dropout 0.0 --clip-grad 1.0 --weight-decay 0.1 \
  --lr 5.0e-5 --min-lr 5.0e-6 --lr-warmup-iters ${WARMUP} --lr-decay-iters ${TRAIN_ITERS} \
  --lr-decay-style cosine \
  --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 --eval-iters 0 --eval-interval 2000 \
  --auto-detect-ckpt-format ${LOAD_ARGS} --save ${SAVE_PATH} --save-interval ${SAVE_INTERVAL} \
  --log-throughput --log-timers-to-tensorboard --tensorboard-dir ${SAVE_PATH}/tensorboard --log-interval 1 \
  --wandb-project ${WANDB_PROJECT} --wandb-exp-name ${RUN_NAME} \
  --bf16 --overlap-grad-reduce --overlap-param-gather "

srun -u bash -c "torchrun --nproc_per_node 8 --nnodes ${COUNT_NODE} --node_rank \$SLURM_PROCID --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} ${TRAINING_SCRIPT} ${TRAINING_PARAMS}"
