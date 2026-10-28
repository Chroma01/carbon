# Carbon-X post-training

Post-training of the midtrained
[`HuggingFaceBio/Carbon-X-Base`](https://huggingface.co/HuggingFaceBio/Carbon-X-Base)
checkpoint with TRL, in two stages:

1. **SFT** on [`HuggingFaceBio/FineDNA-SFT`](https://huggingface.co/datasets/HuggingFaceBio/FineDNA-SFT), producing
   [`HuggingFaceBio/Carbon-X-SFT`](https://huggingface.co/HuggingFaceBio/Carbon-X-SFT).
2. **RL** with verifiable rewards on
   [`HuggingFaceBio/FineDNA-RL`](https://huggingface.co/datasets/HuggingFaceBio/FineDNA-RL), starting from
   the SFT checkpoint. Rollouts are generated asynchronously by a vLLM server,
   and the config is sized for one generation node and one training node. The
   rewards listed under `reward_funcs` in [`rl.yaml`](rl.yaml) cover
   mutation-effect prediction, DNA question answering, sequence-position tasks
   and instruction following, and live in
   [`src/carbon_x/rewards.py`](../../src/carbon_x/rewards.py).

| File | Contents |
|---|---|
| [`sft.py`](sft.py) / [`rl.py`](rl.py) | Training scripts |
| [`sft.yaml`](sft.yaml) / [`rl.yaml`](rl.yaml) | Training configs |
| [`sft.slurm`](sft.slurm) / [`rl.slurm`](rl.slurm) | Slurm launchers; for RL, the last node serves vLLM and the rest train |
| [`accelerate_configs/`](accelerate_configs) | DeepSpeed ZeRO-3 and FSDP2 configs |

Run commands from the `carbon-x/` directory. Install dependencies with:

```sh
uv sync
```

### SFT

The production recipe uses BF16 model weights, assistant-only causal language
model loss, a maximum length of 9,984 tokens, and each example's
`chat_template_kwargs`, including `tokenizer_kwargs.auto_dna_tags`. It uses the
model's bundled tokenizer and training chat template, with `<|im_end|>` as EOS.
`warmup_steps: 0.03` means a 3% warmup.

The production config enables Trackio and Hub checkpoint publishing:

```sh
# SFT: 2 nodes, global batch size 32, 3 epochs
sbatch --nodes=2 training/posttraining/sft.slurm --config training/posttraining/sft.yaml --accelerator deepspeed_zero3
```

Checkpoints stay in the configured output directory on shared storage.
Adjust the Slurm partition and CUDA module for your cluster.

Pass training overrides after `--`:

```sh
sbatch --nodes=1 training/posttraining/sft.slurm \
  --config training/posttraining/sft.yaml --accelerator deepspeed_zero3 -- \
  --output_dir scratch/my-sft --push_to_hub false --report_to none \
  --resume_from_checkpoint scratch/my-sft/checkpoint-100
```

Without an explicit resume argument, training resumes the latest checkpoint in
`output_dir`. Use a new output directory for a fresh run. Local outputs are
retained after publishing.

### RL

```sh
# RL
sbatch --nodes=2 training/posttraining/rl.slurm --config training/posttraining/rl.yaml --accelerator fsdp2 \
  --vllm_args "data_parallel_size=8 tensor_parallel_size=1 dtype=bfloat16 gpu_memory_utilization=0.85"
```
