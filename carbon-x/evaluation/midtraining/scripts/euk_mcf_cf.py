"""GenomeQA eukaryote classification, MCF + CF, scored with transformers.

This is the reference implementation that euk_mcf_cf_vllm.py reproduces; keep the two in step. See that file for
what MCF and CF mean and for the row filtering, which is shared.
"""

import argparse
import json
import math
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

LETTERS = ["A", "B", "C", "D"]
LN2 = math.log(2)
SEQ = re.compile(r"[ACGTN]{50,}")

_TAGGER = None


def tag(text, enabled):
    """Wrap DNA runs in <dna>...</dna>. A no-op on Carbon, whose tokenizer already tags; see run_evals.sh."""
    global _TAGGER
    if not enabled or not text:
        return text
    if _TAGGER is None:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from tag_dna_sequences import tag_dna_sequences

        _TAGGER = tag_dna_sequences
    return _TAGGER(str(text))[0]


@torch.no_grad()
def sumnll(model, tok, prompt, continuation, device, maxlen):
    """Summed NLL of `continuation` given `prompt`, in nats.

    The concatenated ids are truncated from the left to maxlen, so the continuation is always scored in full.
    Logit i predicts token i+1, hence the [-n-1:-1] slice.
    """
    prompt_ids = tok(prompt, add_special_tokens=False).input_ids
    cont_ids = tok(continuation, add_special_tokens=False).input_ids
    ids = (prompt_ids + cont_ids)[-maxlen:]
    logits = model(torch.tensor([ids], device=device)).logits[0]
    n = len(cont_ids)
    logprobs = torch.log_softmax(logits[-n - 1 : -1].float(), -1)
    return -logprobs[range(n), torch.tensor(ids[-n:], device=device)].sum().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--L", type=int, default=8192, help="truncate the DNA run to this many characters")
    ap.add_argument("--maxlen", type=int, default=8192, help="truncate the tokenized prompt to this many tokens")
    ap.add_argument("--tag-dna", action="store_true")
    args = ap.parse_args()
    device = "cuda"

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, dtype=torch.bfloat16)
        .to(device)
        .eval()
    )

    path = hf_hub_download("HuggingFaceBio/genomeqa", "mcq/eukaryote_cls_mcq.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    df = df[df["answer_text"].astype(str).str.len() <= 30].reset_index(drop=True)

    items = []
    for _, row in df.iterrows():
        question = row["question"]
        match = SEQ.search(question.upper())
        if not match:
            continue
        items.append(
            (
                question[: match.start()].rstrip(),
                question[match.start() : match.end()],
                list(row["options"]),
                int(row["answer_index"]),
            )
        )

    mcf_ok, cf_ok = [], []
    for stem, seq, opts, gold in items:
        tagged = tag(seq[: args.L], args.tag_dna)

        # MCF: score " A", " B", ... as continuations of the option list and take the likeliest letter.
        letters_prompt = f"{stem}\n{tagged}\n" + "\n".join(f"{LETTERS[j]}. {o}" for j, o in enumerate(opts)) + "\nAnswer:"
        letter_nll = [sumnll(model, tok, letters_prompt, f" {LETTERS[j]}", device, args.maxlen) for j in range(len(opts))]
        mcf_ok.append(int(np.argmin(letter_nll) == gold))

        # CF: score each option text and take the lowest bits per byte.
        cloze_prompt = f"{stem}\n{tagged}\nAnswer:"
        option_bpb = [
            sumnll(model, tok, cloze_prompt, f" {o}", device, args.maxlen) / (len(str(o).encode()) * LN2) for o in opts
        ]
        cf_ok.append(int(np.argmin(option_bpb) == gold))

    out = {
        "run": args.run_name,
        "L": args.L,
        "mcf": float(np.mean(mcf_ok)),
        "cf": float(np.mean(cf_ok)),
        "n": len(mcf_ok),
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"{args.run_name}: MCF={out['mcf']:.3f} CF={out['cf']:.3f} n={out['n']}")


if __name__ == "__main__":
    main()
