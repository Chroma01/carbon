#!/usr/bin/env bash
# Evaluate a Carbon-X checkpoint on the DNA tasks reported for midtraining, plus the English suite.
#
#   ./run_evals.sh HuggingFaceBio/Carbon-X-Base carbon-x-base            # midtrained checkpoint
#   ./run_evals.sh HuggingFaceBio/Carbon-X carbon-x                              # the released chat model
#   SR_USE_TAGS=0 SR_GEN_LEN=40 SR_BP_PER_TOKEN=1 SR_NO_PREFIX=1 \
#     ./run_evals.sh Qwen/Qwen3-14B qwen3-14b                                    # plain-BPE baseline
#
# There is deliberately no DNA-tagging switch for the MCQ tasks. Carbon's HybridDNATokenizer inserts <dna> tags
# itself (auto_dna_tags), and it agrees with the tag_dna_sequences helper on every case we could construct,
# including the 6 bp threshold and IUPAC runs, where neither tags. Pre-tagging therefore produces byte-identical
# token ids, and scoring SeqQA and GenomeQA with and without it gives identical numbers. A plain-BPE model gets no
# tags because nothing inserts them. So both are scored on the text as it comes.
#
# SR_USE_TAGS is a different decision despite the similar name, which is why it survives: it selects the open
# "<dna>" generation prefix instead of "<s>", changing the continuation setup, so runs with and without it are not
# comparable. Leave it on for Carbon, set SR_USE_TAGS=0 for a BPE baseline.
#
# All scoring runs on vLLM. The scorers in scripts/ are self-contained reimplementations of the MCF/CF primitive
# and the sequence-recovery metric: no lighteval. Their numbers were checked against the transformers
# implementations they replace, on the released checkpoint (see README).
set -euo pipefail

MODEL="${1:?usage: run_evals.sh MODEL LABEL}"
LABEL="${2:?usage: run_evals.sh MODEL LABEL}"
OUT="${OUT:-results/$LABEL}"
PY="${PY:-python}"
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts"
SR_USE_TAGS="${SR_USE_TAGS:-1}"
SR_TAG_FLAG=""; [ "$SR_USE_TAGS" = "1" ] && SR_TAG_FLAG="--use_dna_tags"
mkdir -p "$OUT"

# A task whose result file is already in $OUT is skipped, so a run can be resumed after a timeout or a crash without
# redoing the finished tasks. ONLY="seqqa euk" restricts the run to the named tasks; the names are the ones printed
# in the "== [label] ..." headers below.
ONLY="${ONLY:-}"
# A non-empty file is not the same as a finished one: a scorer killed mid-write leaves truncated JSON. Both the
# resume check and the post-run check go through here, so neither can accept a half-written result.
valid_json() { "$PY" -c "import json,sys; json.load(open(sys.argv[1]))" "$1" >/dev/null 2>&1; }

want() {  # want <task-name> <result-glob>
  if [ -n "$ONLY" ] && ! printf '%s\n' $ONLY | grep -qxF "$1"; then return 1; fi
  for f in $2; do
    [ -s "$f" ] || continue
    if valid_json "$f"; then
      echo "== [$LABEL] $1 already done ($f), skipping"; return 1
    fi
    echo "== [$LABEL] $1: $f is not valid JSON (truncated run?), redoing it" >&2
    rm -f "$f"
  done
  return 0
}

# vLLM can abort at interpreter shutdown ("terminate called without an active exception", SIGABRT) AFTER a scorer has
# already written its result. The result file, not the exit status, is therefore what decides whether a task
# succeeded. Each task below runs with `|| true` and is then checked here, so a teardown abort does not stop the
# suite, while a task that genuinely produced nothing is recorded and reported at the end instead of silently
# aborting the five tasks queued behind it.
FAILED=""
check() {  # check <task-name> <result-glob>
  for f in $2; do
    [ -s "$f" ] || continue
    if valid_json "$f"; then echo "== [$LABEL] $1 -> $f"; return 0; fi
    # A scorer that died partway can leave a truncated file behind. Treating that as success would both report a
    # passing run and let the next resume skip the task, so delete it and record the failure.
    echo "ERROR: [$LABEL] $1 wrote $f but it is not valid JSON; removing it" >&2
    rm -f "$f"
  done
  echo "ERROR: [$LABEL] $1 produced no usable result" >&2
  FAILED="$FAILED $1"
}

if [ "${RUN_DNA:-1}" = "1" ]; then
# SeqQA, the three curated tasks. max-length 6144 because ORF-transeff's options are ~1 kb of RNA each and the
# common 2048 default truncates them, costing that task about 25 points.
if want seqqa "$OUT/seqqa_$LABEL.json"; then
echo "== [$LABEL] SeqQA (3 curated tasks)"
$PY "$SELF/mcq_mcf_cf_vllm.py" --model-path "$MODEL" --run-name "$LABEL" \
  --dataset HuggingFaceBio/seqqa-carbon-curated-3tasks --split test --subtask-field subtask \
  --max-length 6144 --out "$OUT/seqqa_$LABEL.json" || true
check seqqa "$OUT/seqqa_$LABEL.json"
fi

if want euk "$OUT/euk_$LABEL.json"; then
echo "== [$LABEL] GenomeQA eukaryote classification"
$PY "$SELF/euk_mcf_cf_vllm.py" --model "$MODEL" --run-name "$LABEL" --out "$OUT/euk_$LABEL.json" || true
check euk "$OUT/euk_$LABEL.json"
fi

# gen_len is in TOKENS, not bases, and the right value depends on the tokenizer:
#   hybrid 6-mer tail : 8 tokens x 6 bp = ~48 bp, enough for the 30 bp scored  (the default here)
#   plain BPE         : ~1 bp per token, so 8 tokens would score 8 bases against a 30 bp denominator and collapse.
#                       Use SR_GEN_LEN=40 SR_BP_PER_TOKEN=1 SR_NO_PREFIX=1 for a BPE model such as Qwen3-14B.
if want sequence_recovery "$OUT/${LABEL}_*.json"; then
echo "== [$LABEL] sequence recovery"
SR_PREFIX_FLAG=""; [ "${SR_NO_PREFIX:-0}" = "1" ] && SR_PREFIX_FLAG="--no_prefix"
$PY "$SELF/sequence_recovery_eval.py" --model "$MODEL" --model_name "$LABEL" \
  --use_vllm $SR_TAG_FLAG $SR_PREFIX_FLAG --gen_len "${SR_GEN_LEN:-8}" --bp_per_token "${SR_BP_PER_TOKEN:-6}" \
  --data_path "${SR_DATA:-hf://datasets/GenerTeam/sequence-recovery}" --data_type eukaryote \
  --max_samples "${SR_SAMPLES:-2000}" \
  --accuracy_mode fixed_bp --score_len_bp 30 --label_source dataset \
  --bf16 --batch_size "${SR_BS:-64}" --max_seq_len 6144 --output_dir "$OUT" || true
check sequence_recovery "$OUT/${LABEL}_*.json"
fi
fi

# English, scored by the same MCF/CF primitive with the field names remapped per dataset. Never tag DNA here.
# The two MCQ tasks use --prompt-style lighteval: the release post's English panels came from lighteval, whose
# option rendering differs from the in-house one and is worth ~4 points on an instruction-tuned model.
if [ "${RUN_ENGLISH:-1}" = "1" ]; then
  if want mmlu_pro "$OUT/mmlu_pro_$LABEL.json"; then
  echo "== [$LABEL] MMLU-Pro biology"
  $PY "$SELF/mcq_mcf_cf_vllm.py" --model-path "$MODEL" --run-name "$LABEL" \
    --dataset TIGER-Lab/MMLU-Pro --split test \
    --filter-field category --filter-value biology --prompt-style lighteval \
    --question-field question --options-field options --answer-field answer_index \
    --max-length 4096 --out "$OUT/mmlu_pro_$LABEL.json" || true
  check mmlu_pro "$OUT/mmlu_pro_$LABEL.json"
  fi

  if want openbookqa "$OUT/openbookqa_$LABEL.json"; then
  echo "== [$LABEL] OpenBookQA"
  $PY "$SELF/mcq_mcf_cf_vllm.py" --model-path "$MODEL" --run-name "$LABEL" \
    --dataset allenai/openbookqa:main --split test \
    --question-field question_stem --options-field choices --answer-field answerKey --prompt-style lighteval \
    --max-length 2048 --out "$OUT/openbookqa_$LABEL.json" || true
  check openbookqa "$OUT/openbookqa_$LABEL.json"
  fi

  if want hellaswag "$OUT/hellaswag_$LABEL.json"; then
  echo "== [$LABEL] HellaSwag"
  $PY "$SELF/mcq_mcf_cf_vllm.py" --model-path "$MODEL" --run-name "$LABEL" \
    --dataset Rowan/hellaswag --split validation \
    --question-field ctx --options-field endings --answer-field label \
    --cf-style completion \
    --max-length 2048 --out "$OUT/hellaswag_$LABEL.json" || true
  check hellaswag "$OUT/hellaswag_$LABEL.json"
  fi
fi

if [ -n "$FAILED" ]; then
  echo "FAILED tasks:$FAILED" >&2
  exit 1
fi
echo "done -> $OUT"
