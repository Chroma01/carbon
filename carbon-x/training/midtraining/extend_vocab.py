"""
Expand a Qwen3 model's embeddings for the hybrid DNA tokenizer with RANDOM init N(0,std),
handling BOTH tied (Qwen3-4B) and UNTIED (Qwen3-14B) embeddings.

Carbon-X starts from `Qwen/Qwen3-14B`, which is the instruction-tuned release; `Qwen/Qwen3-14B-Base` is a
different checkpoint and is NOT what this recipe used.

- Tied  : resize handles input==lm_head; init the new input rows.
- Untied: resize handles input AND output_layer separately; init the new rows of BOTH
          (input embeddings and lm_head) with N(0,std). Old rows preserved for both.

Usage:
    python extend_vocab.py \
      --base_model Qwen/Qwen3-14B \
      --tokenizer HuggingFaceBio/Carbon-X \
      --output_dir ./qwen3-14b-hybrid-npad-random-init \
      --std 0.02 --seed 42 --dtype bfloat16
"""
import argparse, shutil
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--hub_repo", default=None)
    p.add_argument("--std", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", default="bfloat16", choices=["float32","bfloat16","float16"])
    return p.parse_args()

def main():
    a = parse_args()
    td = {"float32":torch.float32,"bfloat16":torch.bfloat16,"float16":torch.float16}[a.dtype]
    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    target = max(tok.get_vocab().values()) + 1
    print(f"target vocab: {target}")
    model = AutoModelForCausalLM.from_pretrained(a.base_model, torch_dtype=td, device_map="cpu")
    orig = model.config.vocab_size; H = model.config.hidden_size
    tied = bool(model.config.tie_word_embeddings)
    print(f"orig vocab={orig} hidden={H} tied={tied}")
    n_new = target - orig
    assert n_new > 0 and target % 128 == 0, f"bad target {target}"

    in_old = model.get_input_embeddings().weight.data.clone()
    out_old = None if tied else model.get_output_embeddings().weight.data.clone()

    model.resize_token_embeddings(target, mean_resizing=False)
    torch.manual_seed(a.seed)
    model.get_input_embeddings().weight.data[orig:] = torch.randn(n_new, H, dtype=td) * a.std
    if not tied:
        # init the NEW lm_head rows too (separate tensor); different seed offset for independence
        torch.manual_seed(a.seed + 1)
        model.get_output_embeddings().weight.data[orig:] = torch.randn(n_new, H, dtype=td) * a.std

    # ---- verification ----
    assert model.get_input_embeddings().weight.shape == (target, H)
    assert (model.get_input_embeddings().weight.data[:orig] - in_old).abs().max().item() == 0.0, "input old rows changed!"
    print(f"[PASS] input emb resized {model.get_input_embeddings().weight.shape}, old rows preserved")
    if tied:
        assert model.get_output_embeddings().weight.data_ptr() == model.get_input_embeddings().weight.data_ptr(), "tied but not shared!"
        print("[PASS] tied: lm_head shares input embedding")
    else:
        assert model.get_output_embeddings().weight.data_ptr() != model.get_input_embeddings().weight.data_ptr(), "untied but shared!"
        assert (model.get_output_embeddings().weight.data[:orig] - out_old).abs().max().item() == 0.0, "lm_head old rows changed!"
        nn = model.get_output_embeddings().weight.data[orig:]
        print(f"[PASS] untied: lm_head resized + old rows preserved; new lm_head row_norm~{nn.norm(dim=1).mean().item():.4f}")
    assert model.config.vocab_size == target
    print(f"[PASS] config.vocab_size={target}")

    print(f"saving to {a.output_dir}")
    model.save_pretrained(a.output_dir); tok.save_pretrained(a.output_dir)
    outpy = Path(a.output_dir)/"tokenizer.py"
    if not outpy.exists():
        from huggingface_hub import hf_hub_download
        shutil.copy2(Path(hf_hub_download(a.tokenizer,"tokenizer.py")), outpy); print("copied tokenizer.py")

    # reload + english sanity
    m2 = AutoModelForCausalLM.from_pretrained(a.output_dir, torch_dtype=td, device_map="cpu")
    assert m2.config.vocab_size == target
    t2 = AutoTokenizer.from_pretrained(a.output_dir, trust_remote_code=True)
    ins = t2("The capital of France is", return_tensors="pt")
    with torch.no_grad(): out = m2.generate(**ins, max_new_tokens=15, do_sample=False)
    print("english sanity:", repr(t2.decode(out[0], skip_special_tokens=True)))
    if a.hub_repo:
        m2.push_to_hub(a.hub_repo, private=True); t2.push_to_hub(a.hub_repo, private=True)
        if outpy.exists():
            from huggingface_hub import HfApi
            HfApi().upload_file(path_or_fileobj=str(outpy), path_in_repo="tokenizer.py", repo_id=a.hub_repo)
        print(f"pushed to {a.hub_repo}")
    print("DONE expand untied")

if __name__ == "__main__":
    main()
