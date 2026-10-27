# Carbon-X SFT

Supervised fine-tuning of the midtrained `HuggingFaceBio/Carbon-X-Base`
checkpoint on `HuggingFaceBio/FineDNA-SFT`.

| File | Contents |
|---|---|
| [`config.yaml`](config.yaml) | Training config: 2 nodes, global batch size 32, 3 epochs |
| `sft.py` | Training script |
| `sft.slurm` | Slurm launcher |
