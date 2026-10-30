"""
Remove training rows whose DNA is homologous to an evaluation sequence, using BLAST+ megablast.

Complements `decontaminate_carbon.py`, which removes exact whole-span DNA matches. Training windows that cover the
same locus as an evaluation window at a different offset or length, or a near-identical copy of it, are not
whole-span matches; this script finds them by alignment.

Every distinct DNA run (at least `--min_query_bp` bases) in the training rows is aligned against every evaluation
DNA sequence with `blastn -task megablast` on both strands. A training row is removed when one of its runs has a hit
of at least `--min_identity` percent identity whose aligned length is at least `min(--min_aligned_bp, 90% of the
evaluation sequence length)`, so short evaluation sequences (such as 400 bp GenomeQA windows) count when they are
almost fully contained.

BLAST+ is not a Python package. The script uses `blastn` and `makeblastdb` from `--blast_bin_dir` or `PATH`; when
neither has them, it downloads the pinned BLAST+ 2.17.0 Linux x64 release from NCBI, checks its MD5, and unpacks it
under `outputs/blast/`, where later runs reuse it.
Evaluation sets and their revision come from `decontaminate_carbon.py`. Reports are saved to
`outputs/decontamination/<dataset>/<config>/<split>/homology_report.json`.

Usage:

uv run data/decontaminate_homology.py \
    --dataset HuggingFaceBio/FineDNA \
    --config sft_mix \
    --split train \
    --evals frontier_vep \
    --num_threads 16
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path

from datasets import Features, List, Value, load_dataset
from decontaminate_carbon import EVAL_DATASET, EVAL_REVISION, EVALS


DNA_RUN = re.compile(r"[ACGT]{100,}")
BLAST_VERSION = "2.17.0"
BLAST_URL = (
    f"https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/{BLAST_VERSION}/"
    f"ncbi-blast-{BLAST_VERSION}+-x64-linux.tar.gz"
)
BLAST_MD5 = "bdec166721de3b55f90a3badc83538e8"
BLAST_DIR = Path("outputs/blast")


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def dna_runs(record) -> list[str]:
    """Distinct uppercase A/C/G/T runs of at least 100 bases in any string field, in order of appearance."""
    return list(dict.fromkeys(m.group() for text in strings(record) for m in DNA_RUN.finditer(text.upper())))


def blast_bin_dir(bin_dir: str | None) -> Path:
    """
    Directory holding `blastn` and `makeblastdb`: `bin_dir`, else `PATH`, else the pinned release from NCBI (downloaded once).

    Args:
        bin_dir (`str`, *optional*):
            Directory with the BLAST+ binaries.

    Returns:
        `Path` to the directory containing `blastn` and `makeblastdb`.
    """
    if bin_dir:
        return Path(bin_dir)
    if shutil.which("blastn") and shutil.which("makeblastdb"):
        return Path(shutil.which("blastn")).parent
    local = BLAST_DIR / f"ncbi-blast-{BLAST_VERSION}+" / "bin"
    if not (local / "blastn").exists():
        print(f"Downloading BLAST+ {BLAST_VERSION} from {BLAST_URL} to {BLAST_DIR}")
        BLAST_DIR.mkdir(parents=True, exist_ok=True)
        archive = BLAST_DIR / BLAST_URL.rsplit("/", 1)[1]
        urllib.request.urlretrieve(BLAST_URL, archive)
        digest = hashlib.md5(archive.read_bytes()).hexdigest()
        if digest != BLAST_MD5:
            raise SystemExit(f"BLAST+ download failed its MD5 check ({digest} != {BLAST_MD5})")
        with tarfile.open(archive) as tar:
            tar.extractall(BLAST_DIR, filter="data")
        archive.unlink()
    return local


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--evals", nargs="+", default=list(EVALS), choices=list(EVALS))
    parser.add_argument("--min_identity", type=float, default=90.0)
    parser.add_argument("--min_aligned_bp", type=int, default=1000)
    parser.add_argument("--min_query_bp", type=int, default=100)
    parser.add_argument("--evalue", type=float, default=1e-20)
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument("--blast_bin_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--push_to_hub", action="store_true")
    args = parser.parse_args()
    bin_dir = blast_bin_dir(args.blast_bin_dir)
    blastn, makeblastdb = str(bin_dir / "blastn"), str(bin_dir / "makeblastdb")
    print(subprocess.run([blastn, "-version"], capture_output=True, text=True, check=True).stdout.splitlines()[0])

    output_dir = Path(
        args.output_dir or Path("outputs/decontamination") / args.dataset / (args.config or "default") / args.split
    )
    work_dir = output_dir / "homology_blast"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Evaluation sequences: one FASTA record per DNA run, named <benchmark>:<record id>:<record index>:<run>.
    eval_lengths = {}
    with open(work_dir / "eval.fa", "w") as fasta:
        for name in args.evals:
            dataset = load_dataset(EVAL_DATASET, name, split="train", revision=EVAL_REVISION)
            for index, record in enumerate(dataset):
                record_id = record.get("ID") or record.get("id") or index
                for run_index, run in enumerate(dna_runs(record)):
                    # Record ids can repeat (GenomeQA reuses them across window lengths); the index keeps lengths apart.
                    seq_id = f"{name}:{record_id}:{index}:{run_index}"
                    eval_lengths[seq_id] = len(run)
                    fasta.write(f">{seq_id}\n{run}\n")
    print(f"{len(eval_lengths)} evaluation sequences from {len(args.evals)} benchmarks")

    # Training sequences: one FASTA record per distinct DNA run, keyed by its SHA-1; keep the rows using each run.
    dataset = load_dataset(args.dataset, args.config, split=args.split, num_proc=args.num_proc)
    runs = dataset.map(
        lambda row: {"dna_runs": [run for run in dna_runs(row) if len(run) >= args.min_query_bp]},
        remove_columns=dataset.column_names,
        # Declare the type: a shard whose first rows have no DNA runs would otherwise infer list<null> and fail later.
        features=Features({"dna_runs": List(Value("string"))}),
        num_proc=args.num_proc,
        load_from_cache_file=False,
        desc="Extracting DNA",
    )["dna_runs"]
    rows_of = defaultdict(list)
    with open(work_dir / "train.fa", "w") as fasta:
        for row, row_runs in enumerate(runs):
            for run in row_runs:
                key = hashlib.sha1(run.encode()).hexdigest()
                if key not in rows_of:
                    fasta.write(f">{key}\n{run}\n")
                rows_of[key].append(row)
    print(f"{len(rows_of)} distinct training DNA runs from {len(dataset)} rows")

    subprocess.run(
        [makeblastdb, "-in", str(work_dir / "eval.fa"), "-dbtype", "nucl", "-out", str(work_dir / "evaldb")],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            blastn,
            "-task",
            "megablast",
            "-query",
            str(work_dir / "train.fa"),
            "-db",
            str(work_dir / "evaldb"),
            "-strand",
            "both",
            "-perc_identity",
            str(args.min_identity),
            "-evalue",
            str(args.evalue),
            # Each evaluation locus can appear several times (reference and alternate windows), so keep room for them.
            "-num_threads",
            str(args.num_threads),
            "-max_target_seqs",
            "50",
            "-outfmt",
            "6 qseqid sseqid pident length",
            "-out",
            str(work_dir / "hits.tsv"),
        ],
        check=True,
    )

    # A hit counts when its aligned length reaches min(--min_aligned_bp, 90% of the evaluation sequence).
    matches = defaultdict(dict)
    for line in open(work_dir / "hits.tsv"):
        query, subject, identity, length = line.split("\t")[:4]
        needed = min(args.min_aligned_bp, int(0.9 * eval_lengths[subject]))
        if float(identity) >= args.min_identity and int(length) >= needed:
            eval_id = subject.rsplit(":", 2)[0]
            for row in rows_of[query]:
                matches[row][eval_id] = max(int(length), matches[row].get(eval_id, 0))
    removed = sorted(matches)

    by_benchmark = Counter(eval_id.split(":")[0] for row in removed for eval_id in matches[row])
    print(
        f"{args.dataset} {args.config}/{args.split}: {len(removed)} of {len(dataset)} rows are homologous to an evaluation sequence"
    )
    print(f"{'benchmark':32s} {'row matches':>12s}")
    for name in args.evals:
        print(f"{name:32s} {by_benchmark[name]:12d}")

    report = {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "min_identity": args.min_identity,
        "min_aligned_bp": args.min_aligned_bp,
        "evalue": args.evalue,
        "evals": {
            name: {"dataset": EVAL_DATASET, "config": name, "split": "train", "revision": EVAL_REVISION}
            for name in args.evals
        },
        "input_rows": len(dataset),
        "removed_rows": len(removed),
        "matches": [{"row": row, "aligned_bp": matches[row]} for row in removed],
    }
    (output_dir / "homology_report.json").write_text(json.dumps(report, indent=1))
    print(f"Report: {output_dir / 'homology_report.json'}")

    if args.push_to_hub:
        # Chain onto earlier steps, e.g. all -> all_decontaminated_ngram_dna_homology.
        config = args.config or "default"
        prefix = config if "decontaminated" in config else f"{config}_decontaminated"
        config_name = f"{prefix}_homology"
        keep = sorted(set(range(len(dataset))) - set(removed))
        dataset.select(keep).push_to_hub(args.dataset, config_name=config_name, split=args.split)
        print(f"Pushed {len(keep)} rows to {args.dataset} config {config_name}")


if __name__ == "__main__":
    main()
