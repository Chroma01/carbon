# Carbon-X evaluation

These are the post-training benchmarks. The separate suite used to track *midtraining*, before the model was
post-trained, is in [`midtraining/`](midtraining): SeqQA, GenomeQA eukaryote classification, sequence recovery,
MMLU-Pro biology, HellaSwag and OpenBookQA, scored by logprob rather than by generation.

The five DNA benchmarks from the
[release blog post](https://huggingface.co/spaces/HuggingFaceBio/Carbon-X-Release),
implemented as [Inspect](https://inspect.aisi.org.uk/) tasks in
[`tasks/`](tasks).

| Benchmark | Inspect task | Dataset |
|---|---|---|
| BioReason Coding | [`bioreason.py@bioreason_coding_sequence_only`](tasks/bioreason.py) | `HuggingFaceBio/bioreason_vep` |
| BioReason Non-SNV | [`bioreason.py@bioreason_non_snv`](tasks/bioreason.py) | `HuggingFaceBio/bioreason_vep` |
| GenomeQA | [`genomeqa.py@genomeqa_mcq_carbon`](tasks/genomeqa.py) | `HuggingFaceBio/genomeqa` |
| NextBase | [`seqrecqa.py@seqrecqa_varlen`](tasks/seqrecqa.py) | `HuggingFaceBio/sequence-recovery-qa` (`varlen`) |
| FrontierVEP | [`vepqa.py@vepqa_single_1k`](tasks/vepqa.py) | `HuggingFaceBio/vepqa` (`binary_single_1k`) |

Run from the `carbon-x/` directory, e.g. against a model served with vLLM:

```sh
inspect eval evaluation/tasks/vepqa.py@vepqa_single_1k \
  --model vllm/HuggingFaceBio/Carbon-X \
  -M trust_remote_code=true
```
