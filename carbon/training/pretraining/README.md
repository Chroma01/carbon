# Carbon Pretraining

## Training data

Carbon was trained on **1 T tokens (≈ 6 T DNA base pairs)** drawn from the
[Carbon Pretraining Corpus](https://huggingface.co/datasets/HuggingFaceBio/carbon-pretraining-corpus) mix of:

- **Eukaryote genes** (animals, plants, fungi, protists) — functional genomic regions, extracted from refSeq from Generator training mix.
- **mRNA transcripts** — processed, spliced mRNA from OpenGenome2.
- **Prokaryote genomes** — long chromosomal chunks from bacteria and archaea
  (GTDB v220 + IMG/PR), included as a smaller fraction (~10 % of the
  training mixture).

The mixture is **eukaryote-heavy by design**. Carbon's target use case is
eukaryote. The
prokaryote share is 10% of the pretraining mixture, so the model can be continually pretrained on prokaryote species.

## Pretraining code

Carbon was trained with our Megatron-LM fork:
[**huggingface/Megatron-LM-Carbon**](https://github.com/huggingface/Megatron-LM-Carbon).
The fork adds:

- Hybrid loss: the loss for bridging coarse 6-mer tokenization and single-nucleotide resolution.
- Carbon training scripts
