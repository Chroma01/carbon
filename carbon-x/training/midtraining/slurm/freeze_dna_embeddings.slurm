#!/bin/bash
# Carbon-X stage 1 of 2: train the new DNA embeddings while the rest of the model is frozen.
#
# The hybrid tokenizer adds DNA rows to Qwen3's vocabulary, and those rows start out randomly initialised.
# Midtraining straight away would let that noise propagate through the whole network, so this stage trains only
# the DNA embedding rows on ~10B tokens of DNA, with the body and the English embeddings held fixed.
# Stage 2 is slurm/midtrain_300b.slurm, which unfreezes everything.
#
# Qwen3-14B has untied embeddings, so this uses pretrain_gpt_freeze_body_untied.py, which masks gradients on both
# the input embedding and the output layer for rows below ORIGINAL_VOCAB_SIZE.
#
#   sbatch slurm/freeze_dna_embeddings.slurm
#
# Partition and QoS below are specific to our cluster; change them for yours.

#SBATCH --job-name=carbon-x-freeze-body-14b
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --partition=hopper-prod
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --qos=normal
#SBATCH --time=3-00:00:00

set -x
VENV="${VENV:-.venv}"
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_NVLS_ENABLE=0
# cuDNN ships inside the python env here; Megatron needs it on LD_LIBRARY_PATH. A bare `python` exists in the
# container but not everywhere, and $(...) hides the failure, which would leave LD_LIBRARY_PATH short and fail
# later with an opaque loader error instead of here.
PYBIN="${PYBIN:-$(command -v python || command -v python3)}"
[ -n "$PYBIN" ] || { echo "ERROR: no python found; set PYBIN=/path/to/python" >&2; exit 1; }
NVIDIA_LIBS=$("$PYBIN" -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__) + '/lib')") \
  || { echo "ERROR: could not locate nvidia.cudnn via $PYBIN" >&2; exit 1; }
export LD_LIBRARY_PATH=${NVIDIA_LIBS}:${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/opt/aws-ofi-nccl/lib:$LD_LIBRARY_PATH
export FI_EFA_USE_HUGE_PAGE=0

# Rows below this index are frozen; rows at or above it are the DNA vocabulary and are trained.
# 151669 is the hybrid tokenizer's dna_start_id (<dna>/<oov> + 4096 k-mers + the N-pad tail).
# This is the tokenizer dna_start_id, the first DNA row. A larger value such as Qwen3 151936
# would freeze part of the DNA vocabulary you are trying to train.
export ORIGINAL_VOCAB_SIZE=151669

MEGATRON_PATH="${MEGATRON_PATH:?set MEGATRON_PATH to your Megatron-LM-Carbon checkout}"
TRAINING_SCRIPT="${MEGATRON_PATH}/pretrain_gpt_freeze_body_untied.py"   # shipped alongside this script

TOKENIZER="${TOKENIZER:-HuggingFaceBio/Carbon-X}"
# This stage trains on eukaryote DNA only, reading the same tokenized shards step 5 uses, so there is nothing
# extra to tokenize: point MEG_DATA at the eukaryote directory from step 1 ($DNA_DIR/eukaryote). It reads ~10B
# tokens of it, far less than the sources hold, sampled according to the species weights below.
MEG_DATA="${MEG_DATA:?set MEG_DATA to the tokenized eukaryote directory from step 1 (\$DNA_DIR/eukaryote)}"
PRETRAINED_MEGATRON_CKPT="${PRETRAINED_MEGATRON_CKPT:?set it to the vocab-extended Qwen3-14B-Instruct Megatron checkpoint}"
SAVE_PATH="${SAVE_PATH:-checkpoints/qwen3-14b-instruct-freeze-body-dna-10B-npad}"
RUN_NAME="carbon-x-freeze-body-14b"

if [ ! -d "${PRETRAINED_MEGATRON_CKPT}" ]; then
    echo "Missing pretrained Megatron checkpoint dir: ${PRETRAINED_MEGATRON_CKPT}"; exit 1
fi

# ---- Data mix (100% eukaryote with metadata), weights = the natural species distribution of the corpus ----
# Shards are discovered, not assumed: a species ends up with as many as it had parquet files, since datatrove
# hands whole files to tasks. Same add() as configs/mixture_300b.sh: skip 0-byte shards, weight the rest by size.
DATA_PATH=""
add(){ mapfile -t P < <(for f in "$1"/*_tokens.bin; do [ -s "$f" ] && [ -s "${f%.bin}.idx" ] && printf '%s\n' "${f%.bin}"; done)
  if [ ${#P[@]} -eq 0 ]; then echo "ERROR: no usable shards in $1 (need non-empty *_tokens.bin with a matching .idx)" >&2; exit 1; fi
  W=$("$PYBIN" -c "
import os, sys
w, paths = float(sys.argv[1]), sys.argv[2:]
sizes = [os.path.getsize(p + '.bin') for p in paths]
print(' '.join(f'{w * s / sum(sizes)} {p}' for s, p in zip(sizes, paths)))" "$2" "${P[@]}") \
    || { echo "ERROR: weight calc failed for $1" >&2; exit 1; }
  DATA_PATH="$DATA_PATH $W"; }

D="${MEG_DATA}/gener_metadata"
add "${D}_fungi_qwen3h_npad"                0.029
add "${D}_protozoa_qwen3h_npad"             0.005
add "${D}_plant_qwen3h_npad"                0.080
add "${D}_invertebrate_qwen3h_npad"         0.197
add "${D}_vertebrate_mammalian_qwen3h_npad" 0.268
add "${D}_vertebrate_other_qwen3h_npad"     0.421

# ---- Parallelism (14B: TP=4, 4 nodes -> DP=8; MBS=1 for memory headroom) ----
TP=4; PP=1; MBS=1; GBS=256; TRAIN_ITERS=4800  # ~10B tokens at seq_len=8192, GBS=256

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=6000
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)

nvidia-smi
echo "14B NPAD freeze (UNTIED) | tok ${TOKENIZER} | ORIGINAL_VOCAB_SIZE ${ORIGINAL_VOCAB_SIZE} | TP${TP} nodes ${COUNT_NODE} | START $(date)"

TRAINING_PARAMS=" \
    --distributed-timeout-minutes 120 \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --use-distributed-optimizer \
    --use-mcore-models \
    --disable-bias-linear \
    --untie-embeddings-and-output-weights \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --train-iters ${TRAIN_ITERS} \
    --manual-gc \
    --manual-gc-interval 5 \
    --transformer-impl transformer_engine \
    --no-gradient-accumulation-fusion \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER} \
    --trust-remote-code \
    --data-path ${DATA_PATH} \
    --split 100,0,0 \
    --no-mmap-bin-files \
    --num-workers 6 \
    --dataloader-type cyclic \
    --num-layers 40 \
    --hidden-size 5120 \
    --ffn-hidden-size 17408 \
    --num-attention-heads 40 \
    --kv-channels 128 \
    --group-query-attention \
    --num-query-groups 8 \
    --seq-length 8192 \
    --max-position-embeddings 8192 \
    --make-vocab-size-divisible-by 1 \
    --position-embedding-type rope \
    --rotary-percent 1.0 \
    --rotary-base 1000000 \
    --rotary-seq-len-interpolation-factor 1 \
    --normalization RMSNorm \
    --swiglu \
    --norm-epsilon 1e-6 \
    --qk-layernorm \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --lr 1.0e-4 \
    --min-lr 1.0e-5 \
    --lr-warmup-iters 400 \
    --lr-decay-iters ${TRAIN_ITERS} \
    --lr-decay-style cosine \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --eval-iters 0 \
    --eval-interval 500 \
    --auto-detect-ckpt-format \
    --load ${PRETRAINED_MEGATRON_CKPT} \
    --finetune \
    --no-load-optim \
    --no-load-rng \
    --save ${SAVE_PATH} \
    --save-interval 500 \
    --log-throughput \
    --log-timers-to-tensorboard \
    --tensorboard-dir ${SAVE_PATH}/tensorboard \
    --log-interval 1 \
    --wandb-project ${WANDB_PROJECT:-carbon-x} \
    --wandb-exp-name ${RUN_NAME} \
        --bf16 \
    --overlap-grad-reduce \
    --overlap-param-gather \
"

srun -u bash -c "
    torchrun \
        --nproc_per_node 8 \
        --nnodes ${COUNT_NODE} \
        --node_rank \$SLURM_PROCID \
        --master_addr ${MASTER_ADDR} \
        --master_port ${MASTER_PORT} \
        ${TRAINING_SCRIPT} ${TRAINING_PARAMS}
"
# Capture srun's status before anything else runs: a bare `echo` after it would reset $? and the job would exit 0
# even though training died, which makes a failed multi-node run look like a successful one in sacct and in the log.
rc=$?
echo "END: $(date)  rc=${rc}"
[ $rc -eq 0 ] || echo "ERROR: training failed with rc=${rc}" >&2
exit $rc
