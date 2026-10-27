# Carbon-X evaluation

The five DNA benchmarks from the
[release blog post](https://huggingface.co/spaces/HuggingFaceBio/Carbon-X-Release),
implemented as [Inspect](https://inspect.aisi.org.uk/) tasks in
[`tasks/`](tasks).

| Benchmark | Inspect task | Dataset |
|---|---|---|
| BioReason Coding | `bioreason_coding_sequence_only` | `HuggingFaceBio/bioreason_vep` |
| BioReason Non-SNV | `bioreason_non_snv` | `HuggingFaceBio/bioreason_vep` |
| GenomeQA | `genomeqa_mcq_carbon` | `HuggingFaceBio/genomeqa` |
| NextBase | `seqrecqa_varlen` | `HuggingFaceBio/sequence-recovery-qa` (`varlen`) |
| FrontierVEP | `vepqa_single_1k` | `HuggingFaceBio/vepqa` (`binary_single_1k`) |
