# Carbon-X midtraining evaluations

Six tasks reporting what midtraining did: three DNA tasks for what the model gained, three English tasks for what it
kept. They are the midtraining numbers in the release post, and a different set from the post-training chat
evaluations. Run them on the midtrained checkpoint, on the released model, and on the `Qwen/Qwen3-14B` starting
point, since the comparison against the starting point is what the numbers mean.

> **Access.** `HuggingFaceBio/Carbon-X-Base` is gated: you need an approved HF token (`hf auth login`) to fetch
> it. `HuggingFaceBio/Carbon-X` and all the evaluation datasets are public.

Run everything from this directory (`carbon-x/evaluation/midtraining`); `OUT` defaults to `results/<label>`
relative to the working directory. A single 80 GB GPU is enough: each task starts its own vLLM engine at TP=1 on a
14B model, so expect ~3 minutes of engine startup per task and a 28 GB weight download the first time.

```bash
# sequence recovery reads GenerTeam/sequence-recovery straight from the Hub by default;
# set SR_DATA only to point at a local copy.

# any Carbon checkpoint: the tokenizer handles the DNA tags itself, so there is nothing to configure
./run_evals.sh HuggingFaceBio/Carbon-X-Base carbon-x-base
./run_evals.sh HuggingFaceBio/Carbon-X carbon-x

# plain-BPE baseline: sequence recovery needs BPE settings and no <dna> generation prefix
SR_USE_TAGS=0 SR_GEN_LEN=40 SR_BP_PER_TOKEN=1 SR_NO_PREFIX=1 ./run_evals.sh Qwen/Qwen3-14B qwen3-14b
```

### Environment variables

| variable | default | what it does |
|---|---|---|
| `OUT` | `results/<label>` | where result JSON is written, relative to the working directory |
| `PY` | `python` | interpreter for the scorers, so activate your venv or set this to its python |
| `ONLY` | all | space-separated task names to run, e.g. `ONLY="seqqa euk"` |
| `RUN_DNA` / `RUN_ENGLISH` | 1 | run half the suite, for splitting across GPUs |
| `SR_DATA` | the Hub dataset | local copy of the sequence-recovery data |
| `SR_SAMPLES` | 2000 | sequence-recovery subsample size; the table below uses 2000 |
| `SR_BS` | 64 | sequence-recovery batch size; lower it if you hit OOM |
| `SR_USE_TAGS`, `SR_GEN_LEN`, `SR_BP_PER_TOKEN`, `SR_NO_PREFIX` | Carbon values | tokenizer-dependent, see the settings table |

Two things worth setting in the job itself: `VLLM_CACHE_ROOT`, because vLLM writes its torch.compile and
FlashInfer caches under `$HOME/.cache/vllm` regardless of `HF_HOME` and those can be gigabytes; and
`HF_HOME`, so the 28 GB of weights do not land on a home quota. If you only want to evaluate,
`uv sync --no-default-groups --group evaluation` skips DeepSpeed and the post-training dependencies.

A task whose result JSON is already in the output directory is skipped, so a run can be resumed after a crash or a
timeout. `ONLY="seqqa euk"` restricts a run to the named tasks, and `RUN_DNA` / `RUN_ENGLISH` select a half each, so
the suite can be spread over several GPUs.

## The tasks

| task | what it answers | dataset |
|---|---|---|
| MMLU-Pro Biology | did midtraining damage reasoning in biology? | `TIGER-Lab/MMLU-Pro`, rows with `category == biology` |
| HellaSwag | did it damage general English? | `Rowan/hellaswag`, validation |
| OpenBookQA | did it damage general English? | `allenai/openbookqa`, config `main`, test |
| SeqQA (3 tasks) | can it answer questions that require reading the DNA? | `HuggingFaceBio/seqqa-carbon-curated-3tasks` |
| GenomeQA eukaryote classification | can it classify a genomic sequence? | `HuggingFaceBio/genomeqa`, file `mcq/eukaryote_cls_mcq.parquet`; rows are kept when the answer is <=30 chars and the question holds a run of >=50 bases, which is 600 of the 1,000 rows |
| Sequence recovery | can it reconstruct DNA it has not seen? per-base accuracy over the first 30 bases | `GenerTeam/sequence-recovery` |

Metrics: MCF (accuracy by option-letter logprob) and CF (accuracy by lowest answer bits-per-byte) for the
multiple-choice tasks, and per-base accuracy over the first 30 recovered bases for sequence recovery (one wrong base
scores 29/30, not zero). Each scorer writes a JSON with per-subtask and average scores.

## Settings

`run_evals.sh` sets these; they are not defaults and changing them changes the numbers.

| setting | value | why |
|---|---|---|
| `--max-length` on SeqQA | 6144 | ORF-transeff's options are ~1 kb of RNA each, so the option list is long. How long depends on the tokenizer: on Carbon's 6-mer tail the longest prompt measures 1,388 tokens, but on Qwen3's BPE the same prompt is 4,217. A 2048 default therefore leaves Carbon untouched while silently truncating the baseline's options, which is how this task once lost about 25 points. 6144 clears both |
| DNA tagging | none | the MCQ tasks score the text as it comes. Carbon's tokenizer inserts `<dna>` tags itself and agrees with the `tag_dna_sequences` helper on every case we could construct, including the 6 bp threshold and IUPAC runs where neither tags, so pre-tagging gives byte-identical ids and identical scores (checked on SeqQA and GenomeQA). A BPE model gets no tags because nothing inserts them |
| `SR_USE_TAGS` | 1 on Carbon, 0 on a plain-BPE model | sequence recovery only. Despite the name this is not input tagging: it picks the open `<dna>` generation prefix instead of `<s>`, which changes the continuation setup, so runs with and without it are not comparable |
| `--gen_len` | 8 tokens | counted in tokens, not bases; 8 tokens is ~48 bp on the 6-mer tail, enough for the 30 bp scored |
| `--prompt-style` on the English MCQ tasks | `lighteval` | how the option list is rendered. lighteval opens each option line with a space and capitalises the question and every option (` A. Cross section`), the in-house scorer does neither (`A. cross section`). Worth 4 points on Qwen3-14B (0.764 vs 0.725 on MMLU-Pro biology) though only 0.6 on Qwen3-14B-Base, so it is model-dependent. The release post's English panels came from lighteval, so they are scored this way; the DNA tasks came from the in-house scorer and keep `plain` |
| `--cf-style` on HellaSwag | `completion` | HellaSwag is a sentence completion, so the ending is scored as a direct continuation of the context. Scoring it through the same `Answer: ` scaffold the knowledge tasks use costs it about 9 points, so this is not a cosmetic choice. The other tasks use the default `qa` style |

## What to expect

Our numbers for the three checkpoints, from these scorers. MCF unless noted; sequence recovery is per-base
accuracy over the first 30 bases.

| task | n | Qwen3-14B | Carbon-X-Base | Carbon-X |
|---|--:|--:|--:|--:|
| SeqQA (3 tasks) | 120 | 0.658 | 0.733 | 0.858 |
| GenomeQA eukaryote cls | 600 | 0.275 | 0.373 | 0.933 |
| Sequence recovery | 2,000 | 0.295 | 0.563 | 0.564 |
| MMLU-Pro biology | 717 | 0.725 | 0.727 | 0.736 |
| OpenBookQA | 500 | 0.882 | 0.868 | 0.874 |
| HellaSwag (CF) | 10,042 | 0.782 | 0.785 | 0.766 |

Midtraining is what creates DNA ability: Qwen3-14B is at chance on GenomeQA (0.25) and on sequence recovery
(~0.25 per base), and Carbon-X-Base moves both. English is preserved across the three rows below the line.
Post-training then lifts the DNA tasks further.

Expect small differences from these: the scorers run in bf16, so results vary slightly across GPUs and vLLM
builds, and at n=40 per SeqQA subtask one question is 2.5 points.
