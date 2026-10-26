# Carbon

Genomic foundation models from Hugging Face. Carbon is a family of causal
language models trained on **1T tokens of DNA / 6T DNA base pairs** from the
[Carbon Pretraining Corpus](https://huggingface.co/datasets/HuggingFaceBio/carbon-pretraining-corpus),
a curated mix of DNA & RNA sequences.

This directory contains:

* the eval code for Carbon tasks: sequence recovery, variant
effect prediction, and perturbations. We put this together because the zero-shot DNA eval landscape is
currently scattered — useful tasks live in different repos, often buried
alongside evals that need finetuning or that are already saturated, which
makes reproducibility harder.
* scripts for fine-tuning the Carbon models on downstream tasks.
* the embedding representation analysis of Carbon-3B.

## Contents

- [Models](#models)
- [Installation](#installation)
- [Inference](#inference)
- [Pretraining](#pretraining)
- [Evaluation](#evaluation)
- [Finetuning](#finetuning)
- [Analysis](#analysis)

## Models

| Model | Params | Notes |
|---|---|---|
| [`HuggingFaceBio/Carbon-500M`](https://huggingface.co/HuggingFaceBio/Carbon-500M) | 500M | Draft model for speculative decoding. |
| **[`HuggingFaceBio/Carbon-3B`](https://huggingface.co/HuggingFaceBio/Carbon-3B)** | 3B | **Flagship.** Matches or beats Evo2 7B. |
| [`HuggingFaceBio/Carbon-8B`](https://huggingface.co/HuggingFaceBio/Carbon-8B) | 8B | Larger model for more performance. |

The Carbon checkpoints use a **hybrid tokenizer**: BPE for English text and 6-mer
for DNA, switched by a `<dna>` tag mid-sequence. That's why every inference
or eval snippet below wraps DNA inputs with `<dna>` — see
[evaluation/README.md](evaluation/README.md) for the full DNA-tag explanation.

## Installation

Run all commands from this directory. `uv sync` installs the core runtime plus
the `evaluation`, `finetuning`, and `analysis` dependency groups:

```bash
cd carbon
uv sync
```

For Evo2-backed evaluation, also install the Evo2 dependency group:

```bash
uv sync --group evo2
```

To install a single group only, e.g. for finetuning:

```bash
uv sync --no-default-groups --group finetuning
```

## Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "HuggingFaceBio/Carbon-3B"
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
  model_id, 
  trust_remote_code=True,
  torch_dtype="bfloat16"
).to("cuda")

# DNA generation: wrap the prompt with <dna> so the tokenizer routes to 6-mer mode.
context = "ATGGCCTCGAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAG"
prompt = f"<dna>{context}"
inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")

out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
print(tok.decode(out[0]))
```

For zero-shot variant scoring, just feed the model the full sequence and read
the log-likelihood — see [`evaluation/vep_eval.py`](evaluation/vep_eval.py).

## Pretraining

Carbon was trained on **1 T tokens (≈ 6 T DNA base pairs)** from the
[Carbon Pretraining Corpus](https://huggingface.co/datasets/HuggingFaceBio/carbon-pretraining-corpus)
with our Megatron-LM fork,
[**huggingface/Megatron-LM-Carbon**](https://github.com/huggingface/Megatron-LM-Carbon).
See [`training/pretraining/`](training/pretraining) for the data mixture and
what the fork adds.

## Evaluation

This repo ships a **suite of seven zero-shot DNA evaluations** with
reproducible code. The benchmark datasets are available in this [collection](https://huggingface.co/collections/HuggingFaceBio/dna-benchmarks).

The suite covers four modes of zero-shot evaluation:

- **Variant effect prediction**, with three established benchmarks spanning
  both coding (BRCA2) and non-coding regulatory variants (TraitGym
  Mendelian), plus ClinVar for broad pathogenic-vs-benign coverage.
- **A generative task** — sequence recovery, ported from the GENERator paper.
- **Two perturbation tasks** we built — CAG repeat insertion and
  synonymous-codon substitution — to probe regulatory-motif awareness and
  codon-usage structure.
- **Long-context retrieval**  we built — Genome-NIAH, a needle-in-a-haystack eval
  adapted to DNA (four tasks × six context lengths up to 786 kbp).

All eval scripts live in [`evaluation/`](evaluation). Each one runs on Carbon,
GENERator, or Evo2 via a single backend flag, so numbers are directly
comparable across model families.

| Benchmark | What it measures | Script |
|---|---|---|
| **Sequence recovery** | Given a DNA context, generate the next 30 bp; score per-base accuracy against the held-out continuation. Training-free generative eval from the GENERator paper. | [`sequence_recovery.py`](evaluation/sequence_recovery.py) |
| **CAG repeat insertion** | A 30 bp codon-aligned region 60 bp into the CDS exon is replaced with 10 consecutive CAG triplets, mimicking polyglutamine expansion disorders (HD, SCAs, DRPLA). The patch is length- and reading-frame-preserving; all sequence outside is identical. The model should assign higher likelihood to the intact native sequence. | [`perturbation_tasks.py`](evaluation/perturbation_tasks.py) `--task motif_human` |
| **Synonymous codon substitution** | CDS codons are replaced with the highest-frequency synonym for the target species (human or mouse); amino acid identity is preserved by construction. The model should prefer native codon usage over the codon-optimised variant. Probes coding-sequence structure and species-specific codon bias. | [`perturbation_tasks.py`](evaluation/perturbation_tasks.py) `--task syn_human` / `--task syn_mouse` |
| **BRCA2 VEP** | Zero-shot VEP on saturation-mutagenesis BRCA2 ([Huang 2025](https://www.nature.com/articles/s41586-024-08388-8)). Centered 8 kb window + full-LL delta. | [`vep_eval.py`](evaluation/vep_eval.py) |
| **TraitGym Mendelian** | 3,380 fine-mapped non-coding regulatory variants for 113 Mendelian diseases ([Benegas et al. 2025](https://www.biorxiv.org/content/10.1101/2025.02.11.637758v1)). Centered 8 kb window + full-LL delta. | [`vep_eval.py`](evaluation/vep_eval.py) |
| **ClinVar** | Pathogenic vs benign on curated coding + noncoding ClinVar variants. Right-end / next-token scoring with 24 kb left context. | [`clinvar_vep_eval.py`](evaluation/clinvar_vep_eval.py) (uses [`HuggingFaceBio/clinvar-vep-final`](https://huggingface.co/datasets/HuggingFaceBio/clinvar-vep-final) directly) |
| **Genome-NIAH** | Long-context retrieval: insert a (key, value) pair in a real-genome haystack, ask the model to retrieve the value. Four tasks × six context lengths (up to 786 kbp). | [`genome_niah_eval.py`](evaluation/genome_niah_eval.py) |

See [`evaluation/README.md`](evaluation/README.md) for run commands, DNA-tag
flags, and per-benchmark details.

## Finetuning

The [`training/finetuning/`](training/finetuning) directory contains task-specific fine-tuning
recipes for Carbon models:

### Classification Tasks

A minimal end-to-end finetuning example (promoter detection from the
Nucleotide Transformer downstream benchmark) uses the standard 🤗 Transformers
`Trainer` with `AutoModelForSequenceClassification` on top of the Carbon
backbone — swap in any other classification dataset by changing one flag.

See [`training/finetuning/promoter_detection/finetune_promoter.py`](training/finetuning/promoter_detection/finetune_promoter.py) for
the full example.

### Supervised Fine-Tuning with FNS

For autoregressive DNA sequence modeling, we provide
[`training/finetuning/sft/finetune_sft.py`](training/finetuning/sft/finetune_sft.py), which uses
**Factorized Nucleotide Supervision (FNS)** via the custom `FNSTrainer`.
FNS applies base-pair level loss for DNA k-mer tokens, providing finer-grained
supervision than standard token-level loss.

See [`training/finetuning/README.md`](training/finetuning/README.md) for usage examples and
detailed documentation.

### Continual Pretraining

To specialise Carbon on a new clade (e.g. a specific bacterium or protist
that wasn't well represented in the pretraining mix), the same scaffolding
works for **continual pretraining**: load the model with
`AutoModelForCausalLM`, feed it sequences with the `<dna>` tag, and continue
training on next-token loss. The ~10 % prokaryote slice in the pretraining
data means the model already has a reasonable starting point even for
bacterial sequences.

## Analysis

The [`analysis/`](analysis) directory contains training-free embedding probes
of Carbon-3B, showing how the model organizes sequences by taxonomy, strand
orientation, and codon phase. See [`analysis/README.md`](analysis/README.md)
for the pipelines and figures.
