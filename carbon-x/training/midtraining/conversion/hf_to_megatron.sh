#!/bin/bash
# Convert the vocabulary-extended Hugging Face checkpoint from step 1 into Megatron format, so steps 3 and 4
# can train it.
#
#   bash conversion/hf_to_megatron.sh <HF_MODEL_OR_PATH> <MEGATRON_OUTPUT_PATH>
#
# The conversion itself is six lines of Megatron-Bridge at the bottom of this file. Everything above it is
# environment setup for the container we ran it in.
set -euo pipefail

HF_MODEL="${1:?usage: hf_to_megatron.sh <HF_MODEL_OR_PATH> <MEGATRON_OUTPUT_PATH>}"
MEGATRON_OUT="${2:?usage: hf_to_megatron.sh <HF_MODEL_OR_PATH> <MEGATRON_OUTPUT_PATH>}"

export HF_HUB_TRUST_REMOTE_CODE=1 TRANSFORMERS_TRUST_REMOTE_CODE=1

# Workarounds for Megatron-Bridge as of NeMo 26.02. Check whether you still need them before applying: they edit
# library source in place and the line they match may not exist in your version.
#   1-3. Bridge asserts model_type in ("gpt", "mamba") and that megatron_args is present, which both fail for a
#        vocabulary-extended Qwen3.
#   4.   Bridge's tokenizer loader does not pass trust_remote_code through, which the hybrid DNA tokenizer needs.
# BRIDGE_SRC points at your Megatron-Bridge checkout's src/ directory, the same variable megatron_to_hf.sh uses.
# BRIDGE_MODEL_FILE and BRIDGE_TOK_FILE override the individual files if your layout differs.
BRIDGE_ROOT="${BRIDGE_ROOT:-${BRIDGE_SRC:+$(dirname "$BRIDGE_SRC")}}"
BRIDGE_ROOT="${BRIDGE_ROOT:-/opt/Megatron-Bridge}"
BRIDGE_MODEL_FILE="${BRIDGE_MODEL_FILE:-${BRIDGE_SRC:-$BRIDGE_ROOT/src}/megatron/bridge/training/model_load_save.py}"
BRIDGE_TOK_FILE="${BRIDGE_TOK_FILE:-$BRIDGE_ROOT/3rdparty/Megatron-LM/megatron/core/tokenizers/text/libraries/huggingface_tokenizer.py}"
if [ "${APPLY_BRIDGE_PATCHES:-1}" = "1" ]; then
  [ -f "$BRIDGE_MODEL_FILE" ] || { echo "ERROR: Megatron-Bridge not found at '$BRIDGE_MODEL_FILE'. Set BRIDGE_SRC=/path/to/Megatron-Bridge/src, or APPLY_BRIDGE_PATCHES=0 to skip." >&2; exit 1; }
  [ -f "$BRIDGE_TOK_FILE" ] || { echo "ERROR: Bridge tokenizer file not found at '$BRIDGE_TOK_FILE'. Set BRIDGE_TOK_FILE, or APPLY_BRIDGE_PATCHES=0 to skip." >&2; exit 1; }
  sed -i '/assert model_type in ("gpt", "mamba")/d' "$BRIDGE_MODEL_FILE"
  sed -i '/assert megatron_args is not None/d' "$BRIDGE_MODEL_FILE"
  sed -i 's/provider = _gpt_provider if model_type == "gpt" else _mamba_provider/provider = _gpt_provider/g' "$BRIDGE_MODEL_FILE"
  sed -i 's/trust_remote_code=trust_remote_code/trust_remote_code=True/g' "$BRIDGE_TOK_FILE"
  echo "applied Megatron-Bridge compatibility patches"
fi

python -c "
import torch
from megatron.bridge import AutoBridge

bridge = AutoBridge.from_hf_pretrained('${HF_MODEL}')
bridge.import_ckpt(hf_model_id='${HF_MODEL}', megatron_path='${MEGATRON_OUT}', torch_dtype=torch.bfloat16)
print('import complete')
"
ls -la "${MEGATRON_OUT}/"
