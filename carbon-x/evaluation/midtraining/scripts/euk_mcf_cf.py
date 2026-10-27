"""GenomeQA Eukaryote classification @8192: MCF (letter) AND CF (class-name likelihood, byte-norm)."""
import argparse, json, math, re, collections, sys
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
import pandas as pd
LETTERS=["A","B","C","D"]; LN2=math.log(2)
_T=None
def tag(t,en):
    global _T
    if not en or not t: return t
    if _T is None:
        import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from tag_dna_sequences import tag_dna_sequences; _T=tag_dna_sequences
    return _T(str(t))[0]
@torch.no_grad()
def sumnll(model,tok,prompt,cont,dev,maxlen):
    pids=tok(prompt,add_special_tokens=False).input_ids; cids=tok(cont,add_special_tokens=False).input_ids
    ids=(pids+cids)[-maxlen:]; x=torch.tensor([ids],device=dev); lg=model(x).logits[0]
    n=len(cids); lp=torch.log_softmax(lg[-n-1:-1].float(),-1)
    return -lp[range(n),torch.tensor(ids[-n:],device=dev)].sum().item()
ap=argparse.ArgumentParser()
ap.add_argument("--model",required=True); ap.add_argument("--run-name",required=True); ap.add_argument("--out",required=True)
ap.add_argument("--L",type=int,default=8192); ap.add_argument("--maxlen",type=int,default=8192); ap.add_argument("--tag-dna",action="store_true")
a=ap.parse_args(); dev="cuda"
tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True)
model=AutoModelForCausalLM.from_pretrained(a.model,trust_remote_code=True,dtype=torch.bfloat16).to(dev).eval()
f=hf_hub_download("HuggingFaceBio/genomeqa","mcq/eukaryote_cls_mcq.parquet",repo_type="dataset"); df=pd.read_parquet(f)
df=df[df["answer_text"].astype(str).str.len()<=30].reset_index(drop=True)
SEQ=re.compile(r"[ACGTN]{50,}"); items=[]
for _,r in df.iterrows():
    q=r["question"]; m=SEQ.search(q.upper())
    if not m: continue
    items.append((q[:m.start()].rstrip(), q[m.start():m.end()], list(r["options"]), int(r["answer_index"])))
mcf_ok=[]; cf_ok=[]
for stem,seq,opts,gi in items:
    s=tag(seq[:a.L],a.tag_dna)
    base_l=f"{stem}\n{s}\n"+"\n".join(f"{LETTERS[j]}. {o}" for j,o in enumerate(opts))+"\nAnswer:"
    mcf=int(np.argmin([sumnll(model,tok,base_l,f" {LETTERS[j]}",dev,a.maxlen) for j in range(len(opts))])==gi)
    base_c=f"{stem}\n{s}\nAnswer:"
    cf=int(np.argmin([sumnll(model,tok,base_c,f" {o}",dev,a.maxlen)/(len(str(o).encode())*LN2) for o in opts])==gi)
    mcf_ok.append(mcf); cf_ok.append(cf)
out={"run":a.run_name,"L":a.L,"mcf":float(np.mean(mcf_ok)),"cf":float(np.mean(cf_ok)),"n":len(mcf_ok)}
json.dump(out,open(a.out,"w"),indent=2); print(f"{a.run_name}: MCF={out['mcf']:.3f} CF={out['cf']:.3f} n={out['n']}")
