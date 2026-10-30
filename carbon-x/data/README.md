# Decontamination

Scripts that remove training examples overlapping the Carbon-X evaluation benchmarks: the five subsets of
[`HuggingFaceBio/DNAIntelligenceIndex`](https://huggingface.co/datasets/HuggingFaceBio/DNAIntelligenceIndex) evaluated
in [`evaluation/`](../evaluation). Each script checks one kind of overlap:

| Script | Removes training rows that | Config suffix |
|---|---|---|
| [`decontaminate_carbon.py`](decontaminate_carbon.py) | share a complete DNA span of at least six DNA tokens with an evaluation prompt, or cover more than half of an evaluation prompt with shared 8-word n-grams | `_ngram` |
| [`decontaminate_dna_overlap.py`](decontaminate_dna_overlap.py) | contain an evaluation sequence in full, at any offset and in either orientation, or the alternate allele of an evaluation SNV with 25 bp of context on each side | `_dna` |
| [`decontaminate_homology.py`](decontaminate_homology.py) | hold DNA that aligns to an evaluation sequence with BLAST+ megablast at 90% identity or more, over 1,000 bp or 90% of a shorter evaluation sequence | `_homology` |

`decontaminate_carbon.py` checks the prompt only (the first user message for chat data); the other two check every
string field of a row. The evaluation dataset and Carbon-X tokenizer revisions are pinned at the top of
`decontaminate_carbon.py`.

Run commands from the `carbon-x/` directory. Install dependencies with:

```sh
uv sync
```

Without `--push_to_hub`, a script only reports. For example:

```sh
uv run data/decontaminate_carbon.py --dataset HuggingFaceBio/FineDNA --config sft_mix --prompt_column messages
```

Reports go to `outputs/decontamination/<dataset>/<config>/<split>/`. `decontaminate_carbon.py` also writes the flagged
rows and their evaluation matches to an Inspect log, `decontamination.eval`, which you can browse with `inspect view`.

To clean a dataset, run the three scripts in order with `--push_to_hub`. Each one publishes the cleaned split as a new
config of the source repository and the next one reads it, so `<config>` ends up as
`<config>_decontaminated_ngram_dna_homology`:

```sh
uv run data/decontaminate_carbon.py --dataset <repo> --config <config> --prompt_column messages --push_to_hub
uv run data/decontaminate_dna_overlap.py --dataset <repo> --config <config>_decontaminated_ngram --push_to_hub
uv run data/decontaminate_homology.py --dataset <repo> --config <config>_decontaminated_ngram_dna --push_to_hub
```

Clean one config of a repository at a time: concurrent pushes to the same repository overwrite each other's config
entries in its README.

`decontaminate_homology.py` uses `blastn` and `makeblastdb` from `--blast_bin_dir` or `PATH`. If neither has them, it
downloads the pinned BLAST+ release for Linux x64 from NCBI into `outputs/blast/`.
