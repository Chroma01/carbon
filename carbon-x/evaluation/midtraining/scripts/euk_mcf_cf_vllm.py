"""GenomeQA eukaryote classification, MCF + CF, scored with vLLM. Drop-in replacement for euk_mcf_cf.py.

Mirrors the transformers implementation exactly so the numbers reproduce:
  * rows kept only when the answer text is <= 30 chars and the question contains a run of >= 50 A/C/G/T/N;
  * the DNA run is truncated to --L characters and tagged on its own, the stem is never tagged;
  * MCF prompt = "{stem}\\n{seq}\\n{A. opt}...\\nAnswer:" scored against " A", " B", ...;
  * CF prompt   = "{stem}\\n{seq}\\nAnswer:" scored against " {option}", byte-normalised to bits per byte;
  * contexts are tokenized with add_special_tokens=False and the CONCATENATED ids are left-truncated to --maxlen.

Usage:
  python euk_mcf_cf_vllm.py --model <hf_ckpt> --run-name <label> --out <json> [--tag-dna]
"""
import argparse, json, math, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vllm_scoring

LETTERS = ["A", "B", "C", "D"]
LN2 = math.log(2)
SEQ = re.compile(r"[ACGTN]{50,}")
_T = None


def tag(t, en):
    global _T
    if not en or not t:
        return t
    if _T is None:
        from tag_dna_sequences import tag_dna_sequences
        _T = tag_dna_sequences
    return _T(str(t))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--L", type=int, default=8192)
    ap.add_argument("--maxlen", type=int, default=8192)
    ap.add_argument("--tag-dna", action="store_true")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--no-prefix-caching", action="store_true",
                    help="disable vLLM prefix caching (slower; used to check it does not change the scores)")
    a = ap.parse_args()

    import pandas as pd
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    f = hf_hub_download("HuggingFaceBio/genomeqa", "mcq/eukaryote_cls_mcq.parquet", repo_type="dataset")
    df = pd.read_parquet(f)
    df = df[df["answer_text"].astype(str).str.len() <= 30].reset_index(drop=True)

    items = []
    for _, r in df.iterrows():
        q = r["question"]
        m = SEQ.search(q.upper())
        if not m:
            continue
        items.append((q[:m.start()].rstrip(), q[m.start():m.end()], list(r["options"]), int(r["answer_index"])))
    print(f"{len(items)} items", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    jobs, rows = [], []     # jobs: (ctx_ids, cont_ids, bytes_or_None)
    for stem, seq, opts, gi in items:
        s = tag(seq[:a.L], a.tag_dna)
        base_l = f"{stem}\n{s}\n" + "\n".join(f"{LETTERS[j]}. {o}" for j, o in enumerate(opts)) + "\nAnswer:"
        base_c = f"{stem}\n{s}\nAnswer:"
        l_ids = tok(base_l, add_special_tokens=False).input_ids
        c_ids = tok(base_c, add_special_tokens=False).input_ids
        rec = {"gi": gi, "n_opt": len(opts), "mcf": [], "cf": []}
        for j in range(len(opts)):
            rec["mcf"].append(len(jobs))
            jobs.append((l_ids, tok(f" {LETTERS[j]}", add_special_tokens=False).input_ids, None))
        for o in opts:
            rec["cf"].append(len(jobs))
            jobs.append((c_ids, tok(f" {o}", add_special_tokens=False).input_ids, len(str(o).encode())))
        rows.append(rec)

    # the reference truncates the CONCATENATED ids from the left, with nothing reserved, hence reserve=0
    llm = vllm_scoring.build_llm(a.model, a.maxlen, a.tensor_parallel_size, a.gpu_memory_utilization,
                                 prefix_caching=not a.no_prefix_caching)
    nll = vllm_scoring.score(llm, [(c, k) for c, k, _ in jobs], a.maxlen, reserve=0)

    mcf_ok, cf_ok = [], []
    for rec in rows:
        mpred = min(range(rec["n_opt"]), key=lambda i: nll[rec["mcf"][i]])
        bpbs = [nll[j] / (jobs[j][2] * LN2) for j in rec["cf"]]
        cpred = min(range(len(bpbs)), key=lambda i: bpbs[i])
        mcf_ok.append(int(mpred == rec["gi"]))
        cf_ok.append(int(cpred == rec["gi"]))

    out = {"run": a.run_name, "backend": "vllm", "L": a.L,
           "mcf": sum(mcf_ok) / len(mcf_ok), "cf": sum(cf_ok) / len(cf_ok), "n": len(mcf_ok)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"{a.run_name}: MCF={out['mcf']:.3f} CF={out['cf']:.3f} n={out['n']}", flush=True)
    vllm_scoring.shutdown(llm)


if __name__ == "__main__":
    main()
