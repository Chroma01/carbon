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

See [`sft.yaml`](sft.yaml) for training hyperparameters. Run on a single node with eight GPUs:

```sh
uv run accelerate launch \
  --config_file training/posttraining/accelerate_configs/deepspeed_zero3.yaml \
  --num_machines 1 --num_processes 8 \
  training/posttraining/sft.py --config training/posttraining/sft.yaml
```

Or launch with Slurm:

```sh
sbatch --nodes=2 training/posttraining/sft.slurm \
  --config training/posttraining/sft.yaml --accelerator deepspeed_zero3
```

The Slurm scripts are optimized for the Hugging Face Compute Cluster. Adjust their resource requests and environment
setup for your cluster.

### RL

```sh
# RL
sbatch --nodes=2 training/posttraining/rl.slurm --config training/posttraining/rl.yaml --accelerator fsdp2 \
  --vllm_args "data_parallel_size=8 tensor_parallel_size=1 dtype=bfloat16 gpu_memory_utilization=0.85"
```
