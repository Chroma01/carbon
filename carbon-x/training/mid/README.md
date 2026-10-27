# Carbon-X midtraining

Carbon-X was midtrained from Qwen3-14B on 300B tokens of DNA and text with
Megatron-LM, using a cosine learning-rate schedule. The tokenizer extends
Qwen3's vocabulary with 4,096 DNA 6-mer tokens, plus tokens for sequence
boundaries and partial sequences.

| Directory | Contents |
|---|---|
| [`configs/`](configs) | Megatron training configs |
| [`conversion/`](conversion) | Checkpoint conversion between Hugging Face and Megatron formats |
| [`slurm/`](slurm) | Slurm launchers |
