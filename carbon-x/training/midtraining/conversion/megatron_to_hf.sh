#!/bin/bash
# Export Megatron checkpoints to HuggingFace format, for evaluation or release. Runs INSIDE the NeMo 26.02
# container, which is where Megatron-Bridge lives.
#
# The export itself is one AutoBridge call per checkpoint. The work around it is the fiddly part:
#   * Bridge needs three small source patches to get past assertions and a trust_remote_code that is not
#     forwarded. Check whether your Bridge version still needs them before applying.
#   * patch_exported_config.py then fixes config.json. It is not optional: Bridge writes the reference model's
#     vocab_size rather than the padded one, and can leave rope_theta null, which makes transformers silently fall
#     back to 10000 and breaks positional encoding on Qwen3 (correct value 1000000).
#   * the hybrid tokenizer files are copied in from the reference, since Bridge does not carry them.
#
# Required:
#   MEGATRON_CKPT_DIR  directory holding iter_XXXXXXX checkpoints
#   RUN_NAME           name for the export
#   HF_MODEL           HF reference the Megatron run was built from (local path or repo id)
#   HF_EXPORT_BASE     where to write step_XXXX directories
# Optional:
#   STEPS              space-separated steps to export (default: all found)

set -ex

: "${MEGATRON_CKPT_DIR:?Set MEGATRON_CKPT_DIR}"
: "${RUN_NAME:?Set RUN_NAME}"

HF_MODEL="${HF_MODEL:?Set HF_MODEL (char/npad random-init HF ref)}"
HF_EXPORT_BASE="${HF_EXPORT_BASE:?Set HF_EXPORT_BASE}"
PATCH_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/patch_exported_config.py"

export HF_HUB_TRUST_REMOTE_CODE=1

# ── Bridge patches ────────────────────────────────────────────────────
# These patch Megatron-Bridge's own source in place, at the location it has inside the NeMo container. Outside the
# container Bridge lives somewhere else, so point BRIDGE_SRC at your checkout's src/ directory, or set
# APPLY_BRIDGE_PATCHES=0 if your Bridge version no longer needs them (check before assuming it does).
BRIDGE_SRC="${BRIDGE_SRC:-/opt/Megatron-Bridge/src}"
if [ "${APPLY_BRIDGE_PATCHES:-1}" = "1" ]; then
[ -d "$BRIDGE_SRC" ] || { echo "ERROR: Megatron-Bridge source not found at '$BRIDGE_SRC'. Set BRIDGE_SRC=/path/to/Megatron-Bridge/src, or APPLY_BRIDGE_PATCHES=0 to skip." >&2; exit 1; }
ARGS_FILE="${BRIDGE_SRC}/megatron/bridge/training/mlm_compat/arguments.py"
sed -i "s/args\.apply_layernorm_1p/getattr(args, 'apply_layernorm_1p', False)/g" "${ARGS_FILE}"
sed -i "s/kw_args\[\"layernorm_epsilon\"\] = args\.norm_epsilon/kw_args[\"layernorm_epsilon\"] = getattr(args, \"norm_epsilon\", getattr(args, \"layernorm_epsilon\", 1e-5))/g" "${ARGS_FILE}"

HF_TOK_FILE=$(python -c "import megatron.core.tokenizers.text.libraries.huggingface_tokenizer as m; print(m.__file__)")
sed -i 's/trust_remote_code=trust_remote_code/trust_remote_code=True/g' "${HF_TOK_FILE}"

MODEL_FILE="${BRIDGE_SRC}/megatron/bridge/training/model_load_save.py"
sed -i '/assert model_type in ("gpt", "mamba")/d' "${MODEL_FILE}"
sed -i '/assert megatron_args is not None/d' "${MODEL_FILE}"
sed -i 's/provider = _gpt_provider if model_type == "gpt" else _mamba_provider/provider = _gpt_provider/g' "${MODEL_FILE}"

fi
"${PYBIN:-python}" -c "from megatron.bridge import AutoBridge; print('Bridge OK')"

# ── Export checkpoints ────────────────────────────────────────────────
echo "START TIME: $(date)"
mkdir -p "${HF_EXPORT_BASE}"
CONVERTED=""   # only patch/copy steps THIS job converted -> avoids concurrent-job safetensors race

for iter_dir in "${MEGATRON_CKPT_DIR}"/iter_*; do
    iter_name=$(basename "${iter_dir}")
    step=$(echo "${iter_name}" | sed 's/iter_0*//')

    # Filter by STEPS if set
    if [ -n "${STEPS:-}" ]; then
        if ! echo " ${STEPS} " | grep -q " ${step} "; then
            continue
        fi
    fi

    hf_path="${HF_EXPORT_BASE}/step_${step}"

    if [ -f "${hf_path}/.export_complete" ]; then
        echo "Skipping step ${step} (already exported)"
        continue
    fi

    echo "=== Exporting step ${step}: ${iter_dir} -> ${hf_path} ==="

    python -c "
from megatron.bridge import AutoBridge

bridge = AutoBridge.from_hf_pretrained('${HF_MODEL}', trust_remote_code=True)
bridge.export_ckpt(
    megatron_path='${iter_dir}',
    hf_path='${hf_path}',
    show_progress=True,
    strict=False,
)
print('Done: step ${step}')
"
    echo "Exported step ${step}: $(ls ${hf_path}/*.safetensors 2>/dev/null | wc -l) safetensor files"
    CONVERTED="${CONVERTED} ${hf_path}"
done

# ── Patch vocab_size in config.json ───────────────────────────────────
echo ""
echo "=== Patching vocab_size in config.json (only steps converted by THIS job) ==="
# `|| echo` would hand set -e a success and leave an unpatched config.json behind: wrong vocab_size, and a null
# rope_theta that makes transformers fall back to 10000 and silently break Qwen3's positional encoding. The resume
# check above keys on config.json, so a half-patched directory would then be skipped forever. Fail instead.
if [ -n "${CONVERTED}" ]; then
    "${PYBIN:-python}" "${PATCH_SCRIPT}" ${CONVERTED} || { echo "ERROR: config patch failed; exported checkpoints are NOT usable" >&2; exit 1; }
else
    echo "no new steps to patch"
fi

echo ""
echo "=== All exports complete ==="
echo "END TIME: $(date)"
ls -d "${HF_EXPORT_BASE}"/step_*/

# ── Copy hybrid tokenizer files from the HF reference into each exported step ──
if [ -d "${HF_MODEL}" ] && [ -n "${CONVERTED}" ]; then
  for sd in ${CONVERTED}; do
    for tf in tokenizer.py tokenizer.json tokenizer_config.json dna_config.json special_tokens_map.json added_tokens.json vocab.json merges.txt chat_template.jinja; do
      [ -f "${HF_MODEL}/${tf}" ] && cp -n "${HF_MODEL}/${tf}" "${sd}/" 2>/dev/null
    done
  done
  for sd in ${CONVERTED}; do touch "${sd}/.export_complete"; done
  echo "copied hybrid tokenizer files into exported steps"
fi
