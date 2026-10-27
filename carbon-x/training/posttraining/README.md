# Carbon-X post-training

Post-training of the midtrained `HuggingFaceBio/Carbon-X-Base` checkpoint with
TRL, in two stages:

1. **SFT** on `HuggingFaceBio/FineDNA-SFT`, producing `HuggingFaceBio/Carbon-X-SFT`.
2. **RL** with verifiable rewards on `HuggingFaceBio/FineDNA-RL`, starting from
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

The released model and dataset require a Hugging Face login with access to
`HuggingFaceBio/Carbon-X-Base` and `HuggingFaceBio/FineDNA-SFT`. The launcher
preserves `HF_TOKEN` or the original token-file location when moving caches
to local storage on each node.

### SFT

The production recipe uses BF16 model weights, assistant-only causal language
model loss, a maximum length of 9,984 tokens, and each example's
`chat_template_kwargs`, including `tokenizer_kwargs.auto_dna_tags`. It uses the
model's bundled training chat template and `<|im_end|>` as EOS. It does not use
the separate Carbon FNS trainer. `warmup_steps: 0.03` means a 3% warmup with the
pinned Transformers version.

The tokenizer loads directly from the model repository. This recipe requires
the upstream [EOS persistence fix](https://huggingface.co/HuggingFaceBio/Carbon-X-Base/discussions/2)
so saved checkpoints retain `<|im_end|>` when reloaded with
`AutoTokenizer.from_pretrained(..., trust_remote_code=True)`.

The production config enables Trackio and Hub checkpoint publishing:

```sh
# SFT: 2 nodes, global batch size 32, 3 epochs
sbatch --nodes=2 training/posttraining/sft.slurm --config training/posttraining/sft.yaml --accelerator deepspeed_zero3
```

Checkpoints stay in the configured output directory on shared storage, including
all nodes' optimizer shards. Each node creates its own local caches. The
launcher determines the machine rank inside each Slurm task. Adjust the Slurm
partition and `CARBON_CUDA_MODULE` for your cluster; the default is H100s with
`cuda/13.0`.

Set `CARBON_STAGE_VENV=1` to copy installed packages to each node's local disk
before startup when imports from shared storage are slow. Smoke jobs enable
this by default and reuse that local copy when resuming within the allocation.
If metadata access is also slow, prepare a single archive on the login node and
export its absolute path as `CARBON_VENV_ARCHIVE` before submitting:

```sh
mkdir -p .cache
tar -C .venv -cf .cache/sft-venv.tar .
export CARBON_VENV_ARCHIVE="$PWD/.cache/sft-venv.tar"
```

Rebuild the archive after changing dependencies. Repository source remains
editable and is loaded from the shared checkout.

Pass training overrides after `--`; the older `--args "..."` form also works:

```sh
sbatch --nodes=1 training/posttraining/sft.slurm \
  --config training/posttraining/sft.yaml --accelerator deepspeed_zero3 -- \
  --output_dir scratch/my-sft --push_to_hub false --report_to none \
  --resume_from_checkpoint scratch/my-sft/checkpoint-100
```

Without an explicit resume argument, training resumes the latest checkpoint in
`output_dir`. Use a new output directory for a fresh run. Local outputs are
retained after publishing; `--cleanup_after_push true` opts into deleting them
after a successful push.

Dataset split selectors such as `--dataset_train_split 'train[:32]'` work.
Evaluation loads `dataset_test_split` only when `eval_strategy` or `do_eval`
requests it. `--do_eval true` also evaluates and saves metrics after training.
Automatic HF Jobs benchmark evaluation is deferred.

### SFT smoke test

```sh
# One allocation, eight H100s, at most one hour; no Hub publishing or Trackio.
sbatch training/posttraining/sft_smoke.slurm
# Alternative on two B300s, using PyTorch SDPA instead of the H100-specific kernel:
sbatch --partition=hopper-b300 --gres=gpu:2 --cpus-per-task=16 --mem=512G \
  training/posttraining/sft_smoke.slurm --attn_implementation sdpa
```

This runs the actual 14B model on 32 FineDNA-SFT examples with the production
sequence limit, batch size 1 per GPU, and gradient accumulation 1. It trains
three steps, resumes checkpoint 2 to step 4, checks finite losses and gradient
norms and saved optimizer shards, then reloads the final model/tokenizer and
generates two tokens. Results and checkpoints remain in
`scratch/sft-smoke-JOB_ID`; Slurm logs are in `logs/`.

To repeat only resume/reload validation against a completed checkpoint 2, set
`CARBON_SMOKE_OUTPUT_DIR` to that existing run's absolute output path and
`CARBON_SMOKE_RESUME_ONLY=1` before submitting the smoke job. Each job snapshots
its launcher so edits to the checkout cannot change a running shell script.

### RL

RL validation is a separate follow-up.

```sh
# RL
sbatch --nodes=2 training/posttraining/rl.slurm --config training/posttraining/rl.yaml --accelerator fsdp2 \
  --vllm_args "data_parallel_size=8 tensor_parallel_size=1 dtype=bfloat16 gpu_memory_utilization=0.85"
```
