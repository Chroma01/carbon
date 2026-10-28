# Carbon-X midtraining evaluations

The six evaluations reported for midtraining: SeqQA, GenomeQA eukaryote classification and sequence recovery for
DNA, MMLU-Pro biology, OpenBookQA and HellaSwag for English. All scored with vLLM.

```bash
# any Carbon checkpoint
./run_evals.sh HuggingFaceBio/Carbon-X-Base carbon-x-base
./run_evals.sh HuggingFaceBio/Carbon-X carbon-x

# plain-BPE baseline: sequence recovery needs BPE settings and no <dna> generation prefix
SR_USE_TAGS=0 SR_GEN_LEN=40 SR_BP_PER_TOKEN=1 SR_NO_PREFIX=1 ./run_evals.sh Qwen/Qwen3-14B qwen3-14b
```

Run from this directory. One GPU is enough. Results go to `results/<label>`, one JSON per task, and a task whose
JSON is already there is skipped, so a run resumes after a crash. `ONLY="seqqa euk"` restricts a run to named
tasks; `RUN_DNA` / `RUN_ENGLISH` select a half each to spread the suite over several GPUs. `PY` points at the
interpreter if you have not activated the venv.

## The tasks

| task | what it answers | dataset |
|---|---|---|
| MMLU-Pro Biology | did midtraining damage reasoning in biology? | `TIGER-Lab/MMLU-Pro`, rows with `category == biology` |
| HellaSwag | did it damage general English? | `Rowan/hellaswag`, validation |
| OpenBookQA | did it damage general English? | `allenai/openbookqa`, config `main`, test |
| SeqQA (3 tasks) | can it answer questions that require reading the DNA? | `HuggingFaceBio/seqqa-carbon-curated-3tasks` |
| GenomeQA eukaryote classification | can it classify a genomic sequence? | `HuggingFaceBio/genomeqa`, file `mcq/eukaryote_cls_mcq.parquet`; the file mixes three question types, so we score the 600 `metadata.dim == "basic_class"` rows (the other 400 ask which of four 8 kb sequences is the target, which is retrieval, not classification) |
| Sequence recovery | can it reconstruct DNA it has not seen? per-base accuracy over the first 30 bases | `GenerTeam/sequence-recovery` |

Metrics: MCF (accuracy by option-letter logprob) and CF (accuracy by lowest answer bits-per-byte) for the
multiple-choice tasks, and per-base accuracy over the first 30 recovered bases for sequence recovery (one wrong base
scores 29/30, not zero). Each scorer writes a JSON with per-subtask and average scores.

## Settings

`run_evals.sh` sets these, and changing them changes the numbers.

| setting | value | why |
|---|---|---|
| `--max-length` on SeqQA | 6144 | ORF-transeff's options are ~1 kb of RNA each. The option list is 1,388 tokens on Carbon's 6-mer tail but 4,217 on Qwen3's BPE, so a 2048 default silently truncates the baseline |
| DNA tagging | none | Carbon's tokenizer inserts `<dna>` tags itself, idempotently, and a BPE model should not get them. Nothing to set |
| `SR_USE_TAGS` | 1 on Carbon, 0 on plain BPE | sequence recovery only, and not input tagging: it selects the open `<dna>` generation prefix instead of `<s>` |
| `--gen_len` | 8 tokens | tokens, not bases: 8 tokens is ~48 bp on the 6-mer tail, covering the 30 bp scored |
| `--prompt-style` on English MCQ | `lighteval` | lighteval opens each option line with a space and capitalises the question and options (` A. Cross section`). Worth ~4 points on an instruction-tuned model, so it has to match how the English numbers were produced |
| `--cf-style` on HellaSwag | `completion` | HellaSwag is a sentence completion; scoring it through the `Answer: ` scaffold costs it ~9 points |

## What to expect

MCF unless noted; sequence recovery is per-base accuracy over the first 30 bases.

| task | n | Qwen3-14B | Carbon-X-Base | Carbon-X |
|---|--:|--:|--:|--:|
| SeqQA (3 tasks) | 120 | 0.658 | 0.733 | 0.858 |
| GenomeQA eukaryote cls | 600 | 0.275 | 0.373 | 0.933 |
| Sequence recovery | 2,000 | 0.295 | 0.563 | 0.564 |
| MMLU-Pro biology | 717 | 0.725 | 0.727 | 0.736 |
| OpenBookQA | 500 | 0.882 | 0.868 | 0.874 |
| HellaSwag (CF) | 10,042 | 0.782 | 0.785 | 0.766 |

Qwen3-14B is at chance on GenomeQA and sequence recovery (0.25 and ~0.25 per base); midtraining moves both while
English holds. Post-training lifts the DNA tasks further. Expect small differences run to run: the scorers are
bf16, and one SeqQA question is 2.5 points at n=40 per subtask.
