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
| Midtraining | Megatron-LM | `Qwen/Qwen3-14B` | `HuggingFaceBio/FineDNA` (`mid_*`) | [`training/midtraining/`](training/midtraining) |
| SFT | TRL | `HuggingFaceBio/Carbon-X-Base` | `HuggingFaceBio/FineDNA` (`sft_mix`) | [`training/posttraining/`](training/posttraining) |
| RL | TRL | `HuggingFaceBio/Carbon-X-SFT` | `HuggingFaceBio/FineDNA` (`rl_mix`) | [`training/posttraining/`](training/posttraining) |

## Installation

Run commands from `carbon-x/`. Install dependencies with:

```sh
uv sync
```

See [`training/posttraining/`](training/posttraining) for training launch commands.

## Layout

```
carbon-x/
├── src/carbon_x/             # code shared by the SFT and RL scripts
│   ├── configs.py            # script and trainer arguments
│   ├── callbacks.py          # log training samples to Trackio
│   ├── hub.py
│   ├── logging.py            # Trackio setup
│   ├── model_utils.py
│   ├── rewards.py            # verifiable rewards for RL
│   ├── vepqa_claims.py       # groundedness checks against the input sequence
│   └── dna_skills/           # verifiers for DNA skill tasks
├── training/
│   ├── midtraining/          # Megatron-LM
│   └── posttraining/         # TRL: sft.py, rl.py, configs and Slurm launchers
└── evaluation/               # Inspect tasks for the five DNA benchmarks and an HF Jobs launcher
```
