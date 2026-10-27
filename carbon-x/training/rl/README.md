# Carbon-X RL

Reinforcement learning with verifiable rewards, starting from
`HuggingFaceBio/Carbon-X-SFT` and training on `HuggingFaceBio/FineDNA-RL`.
Rollouts are generated asynchronously by a vLLM server, and the config is sized
for one generation node and one training node.

The rewards listed under `reward_funcs` in [`config.yaml`](config.yaml) cover
mutation-effect prediction, DNA question answering, sequence-position tasks and
instruction following. They live in
[`src/carbon_x/rewards.py`](../../src/carbon_x).

| File | Contents |
|---|---|
| [`config.yaml`](config.yaml) | Training config |
| `rl.py` | Training script |
| `rl.slurm` | Slurm launcher |
