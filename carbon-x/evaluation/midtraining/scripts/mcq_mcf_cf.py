"""MCF + CF accuracy on a curated MCQ set (SeqQA-v2 3 subtasks, or basic-dna), one checkpoint -> JSON.
  MCF = letter-argmax (predict the option letter with highest logprob).
  CF  = pick the option with the lowest answer bits-per-byte (cloze / likelihood).
Reports per-subtask + AVG. Self-contained: only torch / transformers / datasets + local hybrid_tag.
WARNING: this is a standalone reimplementation of the lighteval MCF primitive (letter-argmax logprob).
Its MCF scores SHOULD MATCH lighteval's MCF — sanity-check they agree on one checkpoint before trusting.
--tag-dna wraps DNA runs (>=6 A/C/G/T) in <dna>...</dna> at eval time, so a HYBRID model routes them
through the 6-mer path (matches training). Leave OFF for BPE models.

Usage:
  python mcq_mcf_cf.py --model-path <hf_ckpt> --run-name <label> --out <json> \
      --dataset HuggingFaceBio/seqqa-carbon-curated-v2 --subtask-field subtask --tag-dna --max-length 6144
"""
import argparse, json, os, math, sys
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid_tag import tag                      # local: wraps DNA in <dna>..</dna> when enabled

LN2 = math.log(2.0)
LETTERS = [chr(ord("A") + i) for i in range(12)]

@torch.no_grad()
def continuation_nll(model, tok, context, continuation, max_length, device):
    """Summed NLL (nats) of `continuation` given `context`, and its token count.
    logit at pos i predicts token i+1; context left-truncated if the pair overflows max_length."""
    ctx_ids = tok(context, add_special_tokens=True)["input_ids"]
    cont_ids = tok(continuation, add_special_tokens=False)["input_ids"]
    if len(cont_ids) == 0:
        return None
    max_ctx = max(1, max_length - len(cont_ids) - 1)
    if len(ctx_ids) > max_ctx:
        ctx_ids = ctx_ids[-max_ctx:]
    input_ids = torch.tensor([ctx_ids + cont_ids], device=device)
    logits = model(input_ids).logits[0].float()
    logprobs = torch.log_softmax(logits, dim=-1)
    start = len(ctx_ids) - 1
    target_logprobs = logprobs[start : len(ctx_ids) + len(cont_ids) - 1]
    targets = torch.tensor(cont_ids, device=device)
    token_lp = target_logprobs.gather(-1, targets[:, None]).squeeze(-1)
    return float(-token_lp.sum().item()), len(cont_ids)

def build_mcf_prompt(question, options, include_question=True):
    lines = []
    if include_question:
        lines.append(f"Question: {question}")
    for i, opt in enumerate(options):
        lines.append(f"{LETTERS[i]}. {opt}")
    lines.append("Answer:")
    return "\n".join(lines)

@torch.no_grad()
def mcf_predict(model, tok, prompt, n_options, max_length, device):
    """Argmax over letter-continuation logprob -> predicted option index."""
    best_i, best_lp = 0, -1e30
    for i in range(n_options):
        res = continuation_nll(model, tok, prompt, " " + LETTERS[i], max_length, device)
        lp = -res[0] if res else -1e30
        if lp > best_lp:
            best_lp, best_i = lp, i
    return best_i

ap = argparse.ArgumentParser()
ap.add_argument("--model-path", required=True); ap.add_argument("--run-name", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--max-length", type=int, default=6144)
ap.add_argument("--tag-dna", action="store_true")
ap.add_argument("--dataset", default="HuggingFaceBio/seqqa-carbon-curated-v2")
ap.add_argument("--subtask-field", default="")
ap.add_argument("--split", default="train")
# Field names default to the DNA datasets' schema. English MCQ sets use different names, so they can be
# remapped instead of needing a second scorer: the scoring primitive below is identical either way.
ap.add_argument("--question-field", default="question")
ap.add_argument("--options-field", default="options")
ap.add_argument("--answer-field", default="answer_index")
# --subtask-field only GROUPS the report. To evaluate a subset (e.g. MMLU-Pro biology) the rows must be filtered.
ap.add_argument("--filter-field", default="")
ap.add_argument("--filter-value", default="")
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(a.model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(a.model_path, trust_remote_code=True, dtype=torch.bfloat16).to(dev).eval()
_p = a.dataset.split(":"); ds = load_dataset(_p[0], _p[1] if len(_p) > 1 else None, split=a.split)

if a.filter_field:
    before = len(ds)
    ds = ds.filter(lambda r: str(r[a.filter_field]) == a.filter_value)
    print(f"filtered {a.filter_field}=={a.filter_value}: {before} -> {len(ds)} rows")
    if len(ds) == 0:
        raise SystemExit(f"ERROR: no rows with {a.filter_field}=={a.filter_value}")

from collections import defaultdict
agg = defaultdict(lambda: [0, 0, 0])  # subtask -> [mcf_correct, cf_correct, n]
for r in ds:
    raw_opts = r[a.options_field]
    opts = [str(o) for o in (raw_opts["text"] if isinstance(raw_opts, dict) and "text" in raw_opts else raw_opts)]
    gold = r[a.answer_field]
    # answer may be an index (int or digit string) or a letter key such as "C"
    gi = (int(gold) if str(gold).lstrip("-").isdigit()
          else LETTERS.index(str(gold).strip().upper()))
    sub = (r.get(a.subtask_field, "all") if a.subtask_field else "all")
    q = tag(str(r[a.question_field]), a.tag_dna); opts_t = [tag(o, a.tag_dna) for o in opts]
    mpred = mcf_predict(model, tok, build_mcf_prompt(q, opts_t, include_question=True), len(opts), a.max_length, dev)
    ctx = q + "\nAnswer: "
    bpbs = []
    for o, ot in zip(opts, opts_t):
        res = continuation_nll(model, tok, ctx, ot, a.max_length, dev)
        bpbs.append(res[0] / (len(o.encode("utf-8")) * LN2) if res else float("inf"))
    cpred = min(range(len(bpbs)), key=lambda i: bpbs[i])
    agg[sub][0] += int(mpred == gi); agg[sub][1] += int(cpred == gi); agg[sub][2] += 1

out = {"run": a.run_name, "subtasks": {}}
for sub, (mc, cf, n) in agg.items():
    out["subtasks"][sub] = {"mcf": mc / n, "cf": cf / n, "n": n}
tot_n = sum(v[2] for v in agg.values()); tot_m = sum(v[0] for v in agg.values()); tot_c = sum(v[1] for v in agg.values())
out["AVG"] = {"mcf": tot_m / tot_n, "cf": tot_c / tot_n, "n": tot_n}
os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)
print(f"{a.run_name}: AVG mcf={tot_m/tot_n:.3f} cf={tot_c/tot_n:.3f} | " +
      " ".join(f"{k.split('-v1')[0]}:m{v[0]/v[2]:.2f}/c{v[1]/v[2]:.2f}" for k, v in agg.items()))
