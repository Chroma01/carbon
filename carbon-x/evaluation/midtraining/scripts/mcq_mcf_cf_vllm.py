"""MCF + CF accuracy on a multiple-choice set, scored with vLLM.

  MCF = letter-argmax: score " A", " B", ... as continuations of the option list, take the highest logprob.
  CF  = cloze: score each option text as a continuation of "Question ...\\nAnswer: ", take the lowest
        bits-per-byte (summed NLL / (utf8 bytes * ln2)).

Scoring goes through scripts/vllm_scoring.py; see that file for how a scoring-only pass is done on vLLM. Contexts
are left-truncated with one token reserved (reserve=1), matching the scorer these numbers were first produced with.

--tag-dna wraps DNA runs in <dna>...</dna> so a hybrid model routes them down the 6-mer path. Carbon's tokenizer
inserts those tags itself and does so idempotently, so the flag is a no-op there; leave it off for plain BPE models,
where the tags would otherwise be scored as literal text.

Usage:
  python mcq_mcf_cf_vllm.py --model-path <hf_ckpt> --run-name <label> --out <json> \\
      --dataset HuggingFaceBio/seqqa-carbon-curated-3tasks --split test --subtask-field subtask \\
      --tag-dna --max-length 6144
"""
import argparse, json, math, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid_tag import tag
import vllm_scoring

LN2 = math.log(2.0)
LETTERS = [chr(ord("A") + i) for i in range(12)]


def _cap(s):
    return s[0].upper() + s[1:] if s else s


def build_mcf_prompt(question, options, style="plain"):
    """The option list a model is asked to pick a letter from.

    Two renderings, because the published numbers were produced by two different harnesses and the choice is worth
    about four points on an instruction-tuned model:

      plain      "Question: {q}" then "A. {opt}" per line. This is what the in-house MCQ scorer used, and it is what
                 the DNA tasks (SeqQA, GenomeQA) were scored with.
      lighteval  "Question: {Q}" then " A. {Opt}" per line: each option line opens with a space and each option and
                 the question are capitalised. This reproduces lighteval's MCF template byte for byte (checked
                 against its own `get_mcq_prompt_function` on all 717 MMLU-Pro biology and 500 OpenBookQA rows), and
                 it is what the English panels of the release post used.
    """
    if style == "lighteval":
        lines = [f"Question: {_cap(str(question))}"]
        lines += [f" {LETTERS[i]}. {_cap(str(opt))}" for i, opt in enumerate(options)]
    else:
        lines = [f"Question: {question}"]
        lines += [f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options)]
    lines.append("Answer:")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-length", type=int, default=6144)
    ap.add_argument("--tag-dna", action="store_true")
    ap.add_argument("--dataset", default="HuggingFaceBio/seqqa-carbon-curated-3tasks")
    ap.add_argument("--split", default="test")
    ap.add_argument("--subtask-field", default="")
    ap.add_argument("--question-field", default="question")
    ap.add_argument("--options-field", default="options")
    ap.add_argument("--answer-field", default="answer_index")
    ap.add_argument("--filter-field", default="")
    ap.add_argument("--filter-value", default="")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--prompt-style", choices=["plain", "lighteval"], default="plain",
                    help="MCF option-list rendering; see build_mcf_prompt. The English tasks of the release post\n                         used lighteval, the DNA tasks used plain.")
    ap.add_argument("--cf-style", choices=["qa", "completion"], default="qa",
                    help="qa: score the option after 'Answer: ' (knowledge questions). "
                         "completion: score the option as a direct continuation of the context (HellaSwag).")
    ap.add_argument("--no-prefix-caching", action="store_true",
                    help="disable vLLM prefix caching (slower; used to check it does not change the scores)")
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    _p = a.dataset.split(":")
    ds = load_dataset(_p[0], _p[1] if len(_p) > 1 else None, split=a.split)
    if a.filter_field:
        before = len(ds)
        ds = ds.filter(lambda r: str(r[a.filter_field]) == a.filter_value)
        print(f"filtered {a.filter_field}=={a.filter_value}: {before} -> {len(ds)} rows", flush=True)
        if len(ds) == 0:
            raise SystemExit(f"ERROR: no rows with {a.filter_field}=={a.filter_value}")

    tok = AutoTokenizer.from_pretrained(a.model_path, trust_remote_code=True)

    # Build every (context, continuation) pair up front: MCF letters and CF option texts for all rows.
    jobs, rows = [], []
    for r in ds:
        raw = r[a.options_field]
        opts = [str(o) for o in (raw["text"] if isinstance(raw, dict) and "text" in raw else raw)]
        gold = r[a.answer_field]
        gi = int(gold) if str(gold).lstrip("-").isdigit() else LETTERS.index(str(gold).strip().upper())
        sub = (r.get(a.subtask_field, "all") if a.subtask_field else "all")
        q = tag(str(r[a.question_field]), a.tag_dna)
        opts_t = [tag(o, a.tag_dna) for o in opts]

        rec = {"sub": sub, "gi": gi, "n_opt": len(opts), "mcf": [], "cf": []}
        mcf_ctx = tok(build_mcf_prompt(q, opts_t, a.prompt_style), add_special_tokens=True)["input_ids"]
        for i in range(len(opts)):
            cont = tok(" " + LETTERS[i], add_special_tokens=False)["input_ids"]
            rec["mcf"].append(len(jobs)); jobs.append((mcf_ctx, cont, None))
        if a.cf_style == "completion":
            # Sentence completion: the option continues the context directly, with no Answer: scaffold. This is how
            # HellaSwag is conventionally scored, and the scaffold costs it about 9 points, so the framing is not a
            # cosmetic choice. The leading space is part of the scored continuation and of its byte count.
            cf_ctx = tok(q, add_special_tokens=True)["input_ids"]
            texts = [" " + o for o in opts_t]
            norms = [" " + o for o in opts]
        else:
            cf_ctx = tok(q + "\nAnswer: ", add_special_tokens=True)["input_ids"]
            texts, norms = opts_t, opts
        for ot, o in zip(texts, norms):
            cont = tok(ot, add_special_tokens=False)["input_ids"]
            rec["cf"].append(len(jobs)); jobs.append((cf_ctx, cont, len(o.encode("utf-8"))))
        rows.append(rec)

    llm = vllm_scoring.build_llm(a.model_path, a.max_length, a.tensor_parallel_size, a.gpu_memory_utilization,
                                 prefix_caching=not a.no_prefix_caching)
    nll = vllm_scoring.score(llm, [(c, k) for c, k, _ in jobs], a.max_length, reserve=1)

    agg = defaultdict(lambda: [0, 0, 0])
    for rec in rows:
        mpred = min(range(rec["n_opt"]), key=lambda i: nll[rec["mcf"][i]])
        bpbs = [nll[j] / (jobs[j][2] * LN2) if jobs[j][2] else float("inf") for j in rec["cf"]]
        cpred = min(range(len(bpbs)), key=lambda i: bpbs[i])
        agg[rec["sub"]][0] += int(mpred == rec["gi"])
        agg[rec["sub"]][1] += int(cpred == rec["gi"])
        agg[rec["sub"]][2] += 1

    out = {"run": a.run_name, "backend": "vllm", "subtasks": {}}
    for sub, (mc, cf, n) in agg.items():
        out["subtasks"][sub] = {"mcf": mc / n, "cf": cf / n, "n": n}
    tn = sum(v[2] for v in agg.values()); tm = sum(v[0] for v in agg.values()); tc = sum(v[1] for v in agg.values())
    out["AVG"] = {"mcf": tm / tn, "cf": tc / tn, "n": tn}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"{a.run_name}: AVG mcf={tm/tn:.3f} cf={tc/tn:.3f} | " +
          " ".join(f"{k.split('-v1')[0]}:m{v[0]/v[2]:.3f}/c{v[1]/v[2]:.3f}" for k, v in agg.items()), flush=True)
    vllm_scoring.shutdown(llm)


if __name__ == "__main__":
    main()
