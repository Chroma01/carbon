"""
Remove training rows that contain an evaluation DNA sequence in full, or an evaluation SNV with its context.

Complements `decontaminate_carbon.py`, which removes exact whole-span DNA matches. Windows of different lengths or
offsets around the same locus (for example a 8,192 bp VEPQA training window containing a 4,096 bp BioReason test
window) are never whole-span matches; this script catches them.

A training row is removed when, in either orientation, one of its DNA runs (in any string field):

- contains an evaluation sequence in full (an exact identical stretch as long as the evaluation sequence), or
- contains the alternate allele of an evaluation SNV with 25 bp of context on each side. SNVs are read from evaluation
  records holding a reference and an alternate sequence of equal length that differ at one base.

`--min_overlap_bp N` additionally removes rows sharing any exact stretch of at least N bases (off by default).
Candidates come from content-sampled canonical 32-mers shared with the evaluation sequences; each candidate is
extended base by base to the exact overlap length, so interspersed repeats (Alu, ~300 bp) never qualify.

Evaluation sets and their revision come from `decontaminate_carbon.py`. Reports are saved to
`outputs/decontamination/<dataset>/<config>/<split>/dna_overlap_report.json`.

Usage:

uv run data/decontaminate_dna_overlap.py \
    --dataset HuggingFaceBio/FineDNA \
    --config sft_mix \
    --split train \
    --num_proc 16
"""

import argparse
import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from datasets import load_dataset
from decontaminate_carbon import EVAL_DATASET, EVAL_REVISION, EVALS


K = 32
SAMPLING = np.uint64(0x9E3779B97F4A7C15)
SHIFTS = np.arange(2 * (K - 1), -1, -2, dtype=np.uint64)
CODES = np.full(256, 0, np.uint8)
for code, base in enumerate("ACGT"):
    CODES[ord(base)] = code
COMPLEMENT = str.maketrans("ACGT", "TGCA")
MIN_EVAL_LENGTH = 100
DNA_RUN = re.compile(rf"[ACGT]{{{MIN_EVAL_LENGTH},}}")
SNV_CONTEXT = 25


def reverse_complement(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):  # `datasets.map` passes rows as Mappings that are not dicts
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def sampled_kmers(seq: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Canonical 2-bit 32-mers of an A/C/G/T run, kept when their hash falls in 1/16, with their start positions.

    Args:
        seq (`str`):
            Uppercase DNA run without other characters.

    Returns:
        `tuple` of the kept k-mer values (`np.ndarray` of `uint64`) and their start positions (`np.ndarray`).
    """
    windows = np.lib.stride_tricks.sliding_window_view(
        CODES[np.frombuffer(seq.encode(), np.uint8)].astype(np.uint64), K
    )
    forward = (windows << SHIFTS).sum(axis=1, dtype=np.uint64)
    backward = ((3 - windows[:, ::-1]) << SHIFTS).sum(axis=1, dtype=np.uint64)
    canonical = np.minimum(forward, backward)
    keep = (canonical * SAMPLING) >> np.uint64(60) == 0
    return canonical[keep], np.nonzero(keep)[0]


def exact_overlap(seq: str, eval_seq: str, seeds: list[int]) -> int:
    """Longest exact identical stretch between `seq` and `eval_seq`, in either orientation, through the seed positions."""
    best = 0
    for start in seeds:
        for query, position in ((seq, start), (reverse_complement(seq), len(seq) - start - K)):
            kmer = query[position : position + K]
            hit = eval_seq.find(kmer)
            while hit >= 0:
                left = 0
                while (
                    position - left > 0 and hit - left > 0 and query[position - left - 1] == eval_seq[hit - left - 1]
                ):
                    left += 1
                right = 0
                while (
                    position + K + right < len(query)
                    and hit + K + right < len(eval_seq)
                    and query[position + K + right] == eval_seq[hit + K + right]
                ):
                    right += 1
                best = max(best, left + K + right)
                hit = eval_seq.find(kmer, hit + 1)
    return best


def load_evals(names: list[str]) -> list[dict]:
    """One entry per evaluation DNA sequence: benchmark, record id, sequence, and the alternate-allele context of SNVs."""
    entries = []
    for name in names:
        dataset = load_dataset(EVAL_DATASET, name, split="train", revision=EVAL_REVISION)
        for index, record in enumerate(dataset):
            runs = list(dict.fromkeys(m.group() for text in strings(record) for m in DNA_RUN.finditer(text.upper())))
            snv_context = None
            pairs = [(a, b) for a in runs for b in runs if a < b and len(a) == len(b)]
            for a, b in pairs:
                diff = [i for i in range(len(a)) if a[i] != b[i]]
                if len(diff) == 1:
                    # The alternate sequence is the second DNA run of the record (reference first).
                    alt = b if runs.index(b) > runs.index(a) else a
                    snv_context = alt[max(0, diff[0] - SNV_CONTEXT) : diff[0] + SNV_CONTEXT + 1]
            record_id = f"{name}:{record.get('ID') or record.get('id') or index}"
            entries += [{"benchmark": name, "id": record_id, "seq": run, "snv_context": snv_context} for run in runs]
    return entries


def find_overlaps(
    row: dict, evals: list[dict], values: np.ndarray, owners: np.ndarray, min_overlap_bp: int | None
) -> dict:
    """Evaluation records whose sequence the row contains in full (or shares `min_overlap_bp` bases with), or whose SNV it contains."""
    text = "\n".join(strings(row)).upper()
    overlaps = {}
    for match in DNA_RUN.finditer(text):
        seq = match.group()
        if len(seq) < K:
            continue
        kmers, positions = sampled_kmers(seq)
        index = np.searchsorted(values, kmers)
        index[index == len(values)] = 0
        seeds = {}
        for position, start in zip(positions[values[index] == kmers], index[values[index] == kmers], strict=True):
            end = start
            while end < len(values) and values[end] == values[start]:
                seeds.setdefault(int(owners[end]), []).append(int(position))
                end += 1
        for owner, owner_seeds in seeds.items():
            entry = evals[owner]
            # A fully contained sequence shares about len / 16 sampled k-mers; fewer seeds cannot be a full match.
            if min_overlap_bp is None and len(set(owner_seeds)) < len(entry["seq"]) // 32:
                continue
            overlap = exact_overlap(seq, entry["seq"], sorted(set(owner_seeds))[:8])
            full = overlap >= len(entry["seq"])
            long_stretch = min_overlap_bp is not None and overlap >= min_overlap_bp
            if full or long_stretch:
                previous = overlaps.get(entry["id"], {"overlap_bp": 0, "same_variant": False})
                overlaps[entry["id"]] = {**previous, "overlap_bp": max(overlap, previous["overlap_bp"])}
        # A row holding an evaluation SNV with its context overlaps that locus, so only seeded candidates are checked.
        for owner in seeds:
            entry = evals[owner]
            context = entry["snv_context"]
            if context and (context in text or reverse_complement(context) in text):
                previous = overlaps.get(entry["id"], {"overlap_bp": 0, "same_variant": False})
                overlaps[entry["id"]] = {**previous, "same_variant": True}
    return {"dna_overlaps": json.dumps(overlaps)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--evals", nargs="+", default=list(EVALS), choices=list(EVALS))
    parser.add_argument("--min_overlap_bp", type=int, default=None)
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--push_to_hub", action="store_true")
    args = parser.parse_args()

    evals = load_evals(args.evals)
    values, owners = [], []
    for index, entry in enumerate(evals):
        kmers, _ = sampled_kmers(entry["seq"])
        values.append(kmers)
        owners.append(np.full(len(kmers), index))
    values, owners = np.concatenate(values), np.concatenate(owners)
    order = np.argsort(values)
    values, owners = values[order], owners[order]
    print(f"{len(evals)} evaluation sequences from {len(args.evals)} benchmarks, {len(values)} sampled k-mers")

    dataset = load_dataset(args.dataset, args.config, split=args.split, num_proc=args.num_proc)
    flagged = dataset.map(
        find_overlaps,
        fn_kwargs={"evals": evals, "values": values, "owners": owners, "min_overlap_bp": args.min_overlap_bp},
        num_proc=args.num_proc,
        load_from_cache_file=False,
        desc="Finding DNA overlaps",
    )
    matches = [json.loads(value) for value in flagged["dna_overlaps"]]
    removed = [index for index, match in enumerate(matches) if match]

    by_benchmark = Counter(eval_id.split(":")[0] for match in matches for eval_id in match)
    same_variant = Counter(
        eval_id.split(":")[0] for match in matches for eval_id, item in match.items() if item["same_variant"]
    )
    print(
        f"{args.dataset} {args.config}/{args.split}: {len(removed)} of {len(dataset)} rows overlap an evaluation sequence"
    )
    print(f"{'benchmark':32s} {'row matches':>12s} {'same SNV':>9s}")
    for name in args.evals:
        print(f"{name:32s} {by_benchmark[name]:12d} {same_variant[name]:9d}")

    output_dir = Path(
        args.output_dir or Path("outputs/decontamination") / args.dataset / (args.config or "default") / args.split
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "min_overlap_bp": args.min_overlap_bp,
        "evals": {
            name: {"dataset": EVAL_DATASET, "config": name, "split": "train", "revision": EVAL_REVISION}
            for name in args.evals
        },
        "input_rows": len(dataset),
        "removed_rows": len(removed),
        "matches": [{"row": index, "overlaps": matches[index]} for index in removed],
    }
    (output_dir / "dna_overlap_report.json").write_text(json.dumps(report, indent=1))
    print(f"Report: {output_dir / 'dna_overlap_report.json'}")

    if args.push_to_hub:
        # Chain onto earlier steps, e.g. all -> all_decontaminated_ngram_dna_homology.
        config = args.config or "default"
        prefix = config if "decontaminated" in config else f"{config}_decontaminated"
        config_name = f"{prefix}_dna"
        keep = sorted(set(range(len(dataset))) - set(removed))
        dataset.select(keep).push_to_hub(args.dataset, config_name=config_name, split=args.split)
        print(f"Pushed {len(keep)} rows to {args.dataset} config {config_name}")


if __name__ == "__main__":
    main()
