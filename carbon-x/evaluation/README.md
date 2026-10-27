# Carbon-X evaluation

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

## Run on Hugging Face Jobs

[`evaluate.py`](evaluate.py) launches one [Hugging Face Job](https://huggingface.co/docs/huggingface_hub/guides/jobs)
per benchmark. Log in with `hf auth login` first. Run from the `carbon-x/` directory:

```sh
uv run evaluation/evaluate.py --model HuggingFaceBio/Carbon-X-SFT
```

`--backend` sets where the model runs:

- `vllm` (default): `--model` is a Hub repo ID, served with vLLM on a GPU Job (`rtx-pro-6000` by default). The
  defaults reproduce the model card settings: thinking disabled, sampling from the model's `generation_config.json`
  and one sample per problem. Select a checkpoint with `--revision` and override `vllm serve` options with
  `--vllm_args`.
- `endpoint`: `--model` is an [Inspect model string](https://inspect.aisi.org.uk/models.html), called from a CPU Job
  (`cpu-basic` by default). `HF_TOKEN`, together with `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` and `GOOGLE_API_KEY` when
  they are set locally, is forwarded as a secret.

```sh
# Smoke test on two benchmarks
uv run evaluation/evaluate.py --model HuggingFaceBio/Carbon-X-SFT \
  --benchmarks vepqa_single_1k genomeqa_mcq_carbon \
  --limit 8

# An API model
uv run evaluation/evaluate.py --backend endpoint \
  --model anthropic/claude-opus-5-5 \
  --reasoning_effort low \
  --max_tokens 128000

# A model on Inference Providers
uv run evaluation/evaluate.py --backend endpoint \
  --model hf-inference-providers/Qwen/Qwen3-14B:nscale \
  --chat_template_kwargs '{"enable_thinking": false}'

# An OpenAI-compatible endpoint such as a dedicated Inference Endpoint
CARBON_API_KEY=$(hf auth token) uv run evaluation/evaluate.py --backend endpoint \
  --model openai-api/carbon/HuggingFaceBio/Carbon-X-SFT \
  --model_base_url https://<endpoint>.endpoints.huggingface.cloud/v1 \
  --secrets CARBON_API_KEY \
  --chat_template_kwargs '{"enable_thinking": false}'
```

Run `uv run evaluation/evaluate.py --help` for all options, and pass `--dry_run` to print the Jobs without
submitting them.

### Results

Logs go to the private Storage Bucket `<hf_user>/Carbon-X-Evals`, or to the one passed with `--bucket`, using the
same layout as the post-training leaderboard:

```
<model>[-<reasoning_effort>]/<revision>/<benchmark>/<timestamp>/
├── <timestamp>_<task>_<id>.eval    # Inspect log
└── summary.json                    # primary metric, all metrics and sample counts
```

`primary_metric` in `summary.json` holds the benchmark score: balanced accuracy for both BioReason tasks, the mean
of the subtask accuracies for GenomeQA, per-base accuracy for NextBase and accuracy for FrontierVEP. Browse the
logs with:

```sh
inspect view --log-dir hf://buckets/<hf_user>/Carbon-X-Evals/HuggingFaceBio/Carbon-X-SFT/main
```

## Run locally

Against a model served with vLLM, e.g.:

```sh
inspect eval evaluation/tasks/vepqa.py@vepqa_single_1k \
  --model vllm/HuggingFaceBio/Carbon-X \
  -M trust_remote_code=true
```
