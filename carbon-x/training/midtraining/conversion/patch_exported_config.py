#!/usr/bin/env python3
"""Patch config.json after AutoBridge export.

Fixes:
1. vocab_size: match actual embedding tensor shape (AutoBridge may write the
   original HF vocab_size which differs from Megatron's padded vocab).
2. rope_theta: AutoBridge sometimes writes null for rope_theta while the
   correct value is only in rope_parameters.rope_theta. Transformers falls
   back to 10000.0 when rope_theta is null/missing, breaking positional
   encodings for models that use a different value (e.g. Qwen3 uses 1000000).

Usage:
    python patch_vocab_size.py /path/to/exported/step_XXXX
    python patch_vocab_size.py /path/to/export_base/step_*   # multiple dirs
"""
import json
import glob
import sys

def patch_config(hf_path):
    try:
        import safetensors.torch as st
    except ImportError:
        sys.exit("safetensors not installed. Run: pip install safetensors")

    cfg_path = f"{hf_path}/config.json"
    try:
        cfg = json.load(open(cfg_path))
    except FileNotFoundError:
        print(f"  SKIP {hf_path}: no config.json found")
        return

    changed = False

    # --- Fix vocab_size ---
    safetensor_files = sorted(glob.glob(f"{hf_path}/model-*.safetensors"))
    if not safetensor_files:
        safetensor_files = sorted(glob.glob(f"{hf_path}/model.safetensors"))
    if safetensor_files:
        actual_vocab = None
        for sf in safetensor_files:
            tensors = st.load_file(sf)
            for k, v in tensors.items():
                if "embed_tokens" in k:
                    actual_vocab = v.shape[0]
                    break
            if actual_vocab is not None:
                break

        if actual_vocab is not None:
            old_vocab = cfg.get("vocab_size")
            if old_vocab != actual_vocab:
                cfg["vocab_size"] = actual_vocab
                changed = True
                print(f"  FIXED {hf_path}: vocab_size {old_vocab} -> {actual_vocab}")
            else:
                print(f"  OK   {hf_path}: vocab_size={actual_vocab}")
    else:
        print(f"  SKIP {hf_path}: no safetensor files found for vocab check")

    # --- Fix rope_theta ---
    rope_theta = cfg.get("rope_theta")
    rope_params = cfg.get("rope_parameters") or {}
    rope_params_theta = rope_params.get("rope_theta")
    if rope_params_theta and (rope_theta is None or rope_theta != rope_params_theta):
        cfg["rope_theta"] = float(rope_params_theta)
        changed = True
        print(f"  FIXED {hf_path}: rope_theta {rope_theta} -> {rope_params_theta}")
    elif rope_theta is not None:
        print(f"  OK   {hf_path}: rope_theta={rope_theta}")

    if changed:
        json.dump(cfg, open(cfg_path, "w"), indent=2)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <hf_export_dir> [<hf_export_dir2> ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        patch_config(path)
