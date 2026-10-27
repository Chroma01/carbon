"""Freeze verification (per-vector, per-param).

Run this after step 4 and before launching step 5, on the two Megatron dist-checkpoints that stage sits between:
the init scaffold that `conversion/hf_to_megatron.sh` wrote, and an `iter_*` directory from the freeze run.

Compares two Megatron dist-checkpoints (init scaffold vs a freeze run) via torch DCP, loading
tensors key-by-key (memory-safe, works on a small node). Replicates the 4B validation philosophy
(torch.equal exact checks) at per-vector granularity:

  A. BODY (every non-embedding/non-output_layer param): must be BITWISE identical (frozen).
  B. INPUT EMB + LM_HEAD original rows [0:OVS]: EVERY row bitwise identical (frozen).
  C. INPUT EMB + LM_HEAD DNA rows [OVS:]: report how many moved / how many are exactly 0.
     (lm_head should be 100% moved => mask enables every DNA row; input-emb moved == tokens seen.)

Usage: python verify_freeze.py <scaffold_iter_dir> <trained_iter_dir> [OVS=151669]
  OVS is the tokenizer's dna_start_id, 151669 for HuggingFaceBio/Carbon-X -- the same value step 4 freezes below.
"""
import sys, torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader

SCAF, TRN = sys.argv[1], sys.argv[2]
OVS = int(sys.argv[3]) if len(sys.argv) > 3 else 151669
EMB_KEYS = ["embedding.word_embeddings.weight", "output_layer.weight"]

def meta(path): return FileSystemReader(path).read_metadata().state_dict_metadata
def load_one(path, key, m):
    sd = {key: torch.empty(tuple(m.size), dtype=m.properties.dtype)}
    dcp.load(sd, storage_reader=FileSystemReader(path)); return sd[key]

ms, mt = meta(SCAF), meta(TRN)
keys = [k for k in ms.keys() if k in mt and hasattr(ms[k], "size") and len(ms[k].size) > 0]
print(f"scaffold={SCAF}\ntrained={TRN}\nOVS={OVS}\nparams to check: {len(keys)}\n")

# Nothing to compare is a setup error, not a pass. Without this, pointing the script at two wrong directories
# reports PASS: no body params means nothing changed, and absent embeddings are skipped below.
if not keys:
    sys.exit(f"ERROR: no comparable tensors found. Are both paths Megatron dist-checkpoint iter_* dirs?\n"
             f"  scaffold: {SCAF}\n  trained:  {TRN}")
missing = [k for k in EMB_KEYS if k not in keys]
if len(missing) == len(EMB_KEYS):
    sys.exit(f"ERROR: neither embedding tensor found in both checkpoints ({EMB_KEYS}); nothing to verify.")

# ---- A. BODY: every non-emb param bitwise identical ----
body_keys = [k for k in keys if k not in EMB_KEYS]
body_changed = []
for k in body_keys:
    a = load_one(SCAF, k, ms[k]); b = load_one(TRN, k, mt[k])
    if not torch.equal(a, b):
        d = (a.float() - b.float()).abs()
        body_changed.append((k, d.max().item()))
    del a, b
print(f"A. BODY frozen check: {len(body_keys)} params, {len(body_changed)} changed")
if body_changed:
    for k, mx in body_changed[:10]: print(f"     CHANGED {k}  max|Δ|={mx:.3e}")
    print("   ==> FAIL: body not fully frozen")
else:
    print("   ==> PASS: every body param bitwise identical")

# ---- B & C. embeddings, per-vector ----
allok = not body_changed
unverified = []
for k in EMB_KEYS:
    if k not in keys:
        # PP can legitimately split these across ranks, but on this checkpoint pair it means we cannot verify that
        # tensor, so say so rather than letting it pass unexamined.
        print(f"[{k}] NOT PRESENT in both checkpoints -- not verified")
        unverified.append(k)
        continue
    a = load_one(SCAF, k, ms[k]).float(); b = load_one(TRN, k, mt[k]).float()
    d = (a - b).abs()
    row_changed = (d.sum(dim=1) > 0)            # per-row: did this vector move at all?
    fro = row_changed[:OVS]; dna = row_changed[OVS:]
    fro_n = int(fro.sum()); dna_moved = int(dna.sum()); dna_tot = dna.numel()
    fro_max = d[:OVS].max().item()
    print(f"\n[{k}]  shape={tuple(a.shape)}")
    print(f"   B. frozen rows [0:{OVS}]: {fro_n} of {OVS} moved  | max|Δ|={fro_max:.3e}  "
          f"==> {'PASS (every frozen vector identical)' if fro_n==0 else 'FAIL: '+str(fro_n)+' frozen vectors moved!'}")
    print(f"   C. DNA rows [{OVS}:{a.shape[0]}]: {dna_moved} of {dna_tot} moved ({100*dna_moved/dna_tot:.1f}%)  "
          f"| unmoved={dna_tot-dna_moved}")
    if dna_moved < dna_tot:
        idx = (~dna).nonzero(as_tuple=True)[0][:20].tolist()
        print(f"      unmoved DNA row indices (global): {[OVS+i for i in idx]}{' ...' if dna_tot-dna_moved>20 else ''}")
    allok &= (fro_n == 0)
    del a, b, d

print("\n=== VERDICT:", "PASS" if allok else "FAIL",
      "- body+original-emb frozen bitwise; DNA trainability per above ===")
if unverified:
    print(f"    NOTE: not verified: {unverified}")
# Exit status matters: this runs between two expensive stages, often from a script, and a FAIL that exits 0 is
# exactly the silent pass this tool exists to catch.
sys.exit(0 if allok else 1)
