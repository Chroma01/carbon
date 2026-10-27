# Carbon-X

Carbon-X is an open 14B model for DNA understanding and generation. It starts
from [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B), is midtrained on 300B
tokens of DNA and text, and is then post-trained with SFT and RL to answer
questions about sequences. See the
[release blog post](https://huggingface.co/spaces/HuggingFaceBio/Carbon-X-Release)
for details.

This directory contains the training recipe and evals for
[`HuggingFaceBio/Carbon-X`](https://huggingface.co/HuggingFaceBio/Carbon-X).

## Pipeline

| Stage | Framework | Starts from | Data | Recipe |
|---|---|---|---|---|
| Midtraining | Megatron-LM | `Qwen/Qwen3-14B` | 300B tokens of DNA and text | [`training/mid/`](training/mid) |
| SFT | TRL | `HuggingFaceBio/Carbon-X-Base` | `HuggingFaceBio/FineDNA-SFT` | [`training/sft/`](training/sft) |
| RL | TRL | `HuggingFaceBio/Carbon-X-SFT` | `HuggingFaceBio/FineDNA-RL` | [`training/rl/`](training/rl) |

## Layout

```
carbon-x/
├── src/carbon_x/         # code shared by the SFT and RL scripts
│   ├── configs.py        # script and trainer arguments
│   ├── callbacks.py      # push checkpoints to Hub revisions
│   ├── hub.py
│   ├── model_utils.py
│   ├── rewards.py        # verifiable rewards for RL
│   ├── vepqa_claims.py   # groundedness checks against the input sequence
│   └── dna_skills/       # verifiers for DNA skill tasks
├── training/
│   ├── mid/              # Megatron-LM
│   ├── sft/              # TRL
│   └── rl/               # TRL
└── evaluation/           # Inspect tasks for the five DNA benchmarks
```
