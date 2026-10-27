# Carbon-X midtraining

Carbon-X was midtrained from [`Qwen/Qwen3-14B`](https://huggingface.co/Qwen/Qwen3-14B), the instruction-tuned
release, on 300B tokens of DNA and text with Megatron-LM, using a cosine learning-rate schedule. The tokenizer adds
5,515 rows to Qwen3's vocabulary (`dna_vocab_size` in `dna_config.json`): the 4,096 full 6-mers, the shorter padded
k-mers of the `pad_n` tail, and `<dna>`, `</dna>`, `<oov>`. The vocabulary goes from 151,669 to 157,184, and
`dna_start_id` 151669 is the index of the first DNA row.

Midtraining produces [`HuggingFaceBio/Carbon-X-Base`](https://huggingface.co/HuggingFaceBio/Carbon-X-Base).
That is step one of the adaptation, not the released model: it is where DNA ability is learned, and post-training
([`../posttraining`](../posttraining)) is what makes it usable in conversation.

| Directory | Contents |
|---|---|
| [`tokenization/`](tokenization) | Tokenize text to Megatron format `.bin`/`.idx` |
| [`configs/`](configs) | The 300B data mixture |
| [`conversion/`](conversion) | Checkpoint conversion between Hugging Face and Megatron formats |
| [`slurm/`](slurm) | Slurm launchers for the two training stages |

Megatron is configured through CLI flags, so the launchers in `slurm/` are also the training config.

## Requirements

Two different environments:

**Steps 1-2 (tokenization, vocabulary extension) and all of `../../evaluation/midtraining`** run in the repo's
own Python 3.12 environment: `uv sync` from `carbon-x/`. No GPU is needed for steps 1-2; the evaluations need one.

Be aware that `uv sync` installs the whole project, because `[tool.uv] default-groups` includes the post-training
and evaluation groups: roughly 286 packages and ~9 GB, including vLLM, DeepSpeed, flashinfer CUDA cubins and two
git dependencies. Measured here: 37 minutes on a 28-core node, and still unfinished at 40 minutes on a 9-core one,
so give it cores and run it in a batch job rather than interactively.

If you only want to tokenize, `uv sync --no-default-groups` installs just the base set (`datatrove`, `torch`,
`transformers`, `datasets`), which is all steps 1-2 need and is far quicker.

**Steps 3-6 (import, freeze, midtrain, export)** need an environment you provide, with
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM), [Megatron-Bridge](https://github.com/NVIDIA/Megatron-Bridge)
and Transformer Engine. These are not in `pyproject.toml` because they depend on your CUDA and driver versions, so
pinning them here would be wrong for most people; install them per their own instructions. We ran these steps
inside NVIDIA's NeMo 26.02 container, which already contains all three, but that is what we used, not a
requirement. The only file you add to that environment is `pretrain_gpt_freeze_body_untied.py`, copied into the
Megatron checkout for step 4.

The conversion scripts patch Megatron-Bridge's source in place and default to its path inside the NeMo container
(`/opt/Megatron-Bridge/src`). Outside the container, set `BRIDGE_SRC=/path/to/Megatron-Bridge/src`, or
`APPLY_BRIDGE_PATCHES=0` if your version no longer needs the patches.

## The pipeline

The hybrid tokenizer adds DNA rows to Qwen3's vocabulary, and those rows start out randomly initialised, so they are
trained on their own before anything else is unfrozen.

| step | script | what it does |
|---|---|---|
| 1. tokenize | `tokenization/tokenize_finedna.py` | FineDNA sources to Megatron `.bin`/`.idx`, one run per source |
| 2. extend | `extend_vocab.py` | add the DNA rows to Qwen3's embedding and lm_head, randomly initialised |
| 3. import | `conversion/hf_to_megatron.sh` | convert the extended HF checkpoint to Megatron format |
| 4. freeze | `slurm/freeze_dna_embeddings.slurm` | train the DNA rows only, body and English embeddings held still (~10B tokens of DNA); see the weight-decay caveat below |
| 5. midtrain | `slurm/midtrain_300b.slurm` | train everything on the full mixture (300B tokens) |
| 6. export | `conversion/megatron_to_hf.sh` | convert back to HF for evaluation or release |

```bash
cd training/midtraining
mkdir -p logs          # the tokenization scripts write their datatrove logs here

# 1. tokenize, once per FineDNA source. See the mapping table below for the output path and name each
#    source has to land under, and set --n_tasks near the file count of that source.
python tokenization/tokenize_finedna.py \
    --input hf://datasets/HuggingFaceBio/FineDNA/data/dna_eukaryote_fungi \
    --output $DNA_DIR/eukaryote/gener_metadata_fungi_qwen3h_npad \
    --save_name gener_metadata_fungi_qwen3h_npad --n_tasks 16

# 2-3. extend the vocabulary, then import into Megatron
python extend_vocab.py --base_model Qwen/Qwen3-14B \
    --tokenizer HuggingFaceBio/Carbon-X --output_dir ./qwen3-14b-hybrid-npad-random-init
bash conversion/hf_to_megatron.sh ./qwen3-14b-hybrid-npad-random-init ./megatron/qwen3-14b-hybrid-npad

export MEGATRON_PATH=/path/to/Megatron-LM
cp pretrain_gpt_freeze_body_untied.py "$MEGATRON_PATH"/     # step 4 runs it from inside the Megatron tree

# 4. train the DNA rows only, starting from the imported scaffold
export MEG_DATA=/path/to/tokenized/eukaryote
export PRETRAINED_MEGATRON_CKPT=./megatron/qwen3-14b-hybrid-npad
export SAVE_PATH=./checkpoints/freeze-body-dna-10b
sbatch slurm/freeze_dna_embeddings.slurm

# 5. midtrain, starting from step 4's OUTPUT, not from the scaffold
export PRETRAINED_MEGATRON_CKPT=./checkpoints/freeze-body-dna-10b
export DNA_DIR=... BIO_DIR=... ENGLISH_DIR=...
export MIXTURE=$PWD/configs/mixture_300b.sh          # sbatch runs from the spool, so pass this explicitly
bash slurm/midtrain_300b.slurm --dry-run             # check the blend first, no GPUs
sbatch slurm/midtrain_300b.slurm

# 6. export for evaluation
export MEGATRON_CKPT_DIR=./checkpoints/carbon-x-midtrain-300b-14b RUN_NAME=carbon-x-midtrain-300b
export HF_MODEL=./qwen3-14b-hybrid-npad-random-init HF_EXPORT_BASE=./hf/carbon-x-midtrain-300b
bash conversion/megatron_to_hf.sh
```

Steps 3 and 6 run inside the NeMo 26.02 container and patch Megatron-Bridge source in place to get past two
assertions and a `trust_remote_code` that is not forwarded. Check whether your Bridge version still needs those
patches before applying them; step 3 takes `APPLY_BRIDGE_PATCHES=0` to skip. The conversion itself is six lines of
`AutoBridge`.

## Data

Everything is in one dataset, [`HuggingFaceBio/FineDNA`](https://huggingface.co/datasets/HuggingFaceBio/FineDNA),
under `data/<source>/`. All 23 sources share the schema `text`, `id`, `source`, `metadata`, so one script
tokenizes all of them.

The DNA text arrives already rendered: eukaryote rows carry their metadata tag and `<dna>` wrapper
(`<fungi_species><dna>CTCC...`), mRNA and prokaryote are `<dna>`-wrapped sequence. Bio and English are plain
prose, which the tokenizer tags itself. FineDNA also ships the filtering and tagging code that produced this under
`pipeline/`, if you want to see how the text was built.

| share | FineDNA source | write it to |
|---|---|---|
| 38.5% | `dna_eukaryote_{fungi,invertebrate,plant,protozoa,vertebrate_mammalian,vertebrate_other}` | `$DNA_DIR/eukaryote/gener_metadata_<subset>_qwen3h_npad` |
| 8.8% | `dna_mrna` | `$DNA_DIR/mrna_pure_qwen3h_npad` |
| 2.2% | `dna_mrna_splice` | `$DNA_DIR/mrna_splice_promoter_qwen3h_npad` |
| 5.5% | `dna_prokaryote` | `$DNA_DIR/prokaryote_qwen3h_npad` |
| 7% | `bio_p90_{biorxiv,biostars,finepdfs,fineweb,pubmed}` | `$BIO_DIR/p90_train/<name>_qwen3h_npad` |
| 7% | `bio_rest` | `$BIO_DIR/rest_train/bio_rest_qwen3h_npad` |
| 1% | `bio_gemini_fim`, `bio_gemini_nonfim` | `$BIO_DIR/gemini_{fim,nonfim}_train/gemini_{fim,nonfim}_qwen3h_npad` |
| 8% | `english_fineweb_edu` | `$ENGLISH_DIR/fwedu_100B_megatron_qwen3h` |
| 8% | `english_dclm` | `$ENGLISH_DIR/dclm_100B_megatron_qwen3h` |
| 4% | `english_finepdfs_edu` | `$ENGLISH_DIR/finepdfs_edu_100B_megatron_qwen3h` |
| 10% | `instruct_nemotron_{chat_v3,if_v2,if_v3}` | `$ENGLISH_DIR/sft_nemotron_ifchat_v3_qwen3h_npad` (all three into one directory) |

`slurm/midtrain_300b.slurm` globs these paths, so a shard in the wrong place is simply not found and the mixture
aborts with "no usable shards". The names are historical and do not match their sources (`dna_mrna` becomes
`mrna_pure_*`, the English ones have no `_npad`); copy them exactly, and treat
[`configs/mixture_300b.sh`](configs/mixture_300b.sh) as the source of truth.

**Step 4 uses a smaller eukaryote set than step 5** and depends on the shard count, because
`slurm/freeze_dna_embeddings.slurm` lists each shard index explicitly: tokenize the eukaryote sources again with
`--save_name gener_metadata_100k_<subset>_qwen3h_npad` and `--n_tasks` of fungi 10, invertebrate 10, plant 10,
protozoa 4, vertebrate_mammalian 20, vertebrate_other 30. A different count leaves a shard it expects missing, and
Megatron dies at startup looking like a hang.

### Running the tokenizer

It **submits Slurm jobs by default**, with this cluster's partitions baked in; pass `--local` to run in the current
process instead. `--limit N` caps documents **per task**, so `--n_tasks 2 --limit 200` reads 400. `--output` and
`--log_dir` default to relative paths, i.e. into this checkout if you run from here. The full mixture is multiple
terabytes, so tokenize a subset with `--limit` and check the shards before committing to it.

## Checking the pipeline before you trust it

Conversion and freezing both fail quietly: a wrong flag gives you a model that loads, trains and evaluates, just
worse. Three checks catch almost everything.

1. **Round-trip conversion, no training.** HF to Megatron and back, then run the English evals on the
   round-tripped model and confirm they match the original HF model. Any gap is a conversion bug.
2. **Freeze weight-diff.** After step 4, diff the init scaffold against the trained freeze checkpoint:
   `python verify_freeze.py <scaffold_iter_dir> <freeze_iter_dir> 151669`. It checks per vector:
   every body parameter and every original-vocab embedding row must be bitwise identical, and it reports how many
   DNA rows actually moved (the lm_head should be 100%).
3. **English preserved after freeze.** Convert the freeze checkpoint to HF and re-run the English evals; they
   should still match the base model.

These two statements look contradictory, so to be explicit about which is which:

- **What step 4 enforces** is zero *gradient* on the original-vocab rows. That is the mechanism, and check B is
  what confirms it held.
- **Whether those rows end up bitwise identical** is a separate question, because weight decay can move a
  parameter with no gradient at all (see the caveat under Notes). Check B is also what answers it.

So read check B as a measurement, not an assumption: if it reports 0 moved rows, both properties hold and the
weight-decay caveat is moot for your run. If it reports a non-zero count with tiny deltas, the freeze worked and
weight decay moved the rows; rerun the stage with `--weight-decay 0.0` if you need them untouched. A large count
means the freeze itself is broken.

## Step 5 configuration

| | |
|---|---|
| init | the step-4 freeze checkpoint |
| tokenizer | [`HuggingFaceBio/Carbon-X`](https://huggingface.co/HuggingFaceBio/Carbon-X): Qwen3 BPE plus a 6-mer DNA tail, vocabulary 157,184 |
| tokens | 300B, 72,000 iterations |
| schedule | cosine, LR 5e-5 to 5e-6, warmup 2,000 |
| batch | GBS 512, sequence length 8,192 |
| parallelism | TP 4, PP 1, 32 nodes x 8 GPUs |
| arch | 40 layers, hidden 5,120, FFN 17,408, 40 heads, KV 128, GQA 8, untied embeddings |

| share | source |
|---:|---|
| 55% | Carbon DNA: eukaryote 70 / mRNA 16 / mRNA-splice 4 / prokaryote 10 |
| 15% | bio text: p90 7 / rest 7 / synthetic 1 |
| 20% | English: FineWeb-Edu 8 / DCLM 8 / FinePDFs-Edu 4 |
| 10% | Nemotron ifchat-v3 instruction SFT, reasoning traces preserved |

Eukaryote weights follow the natural species distribution of the source corpus (vertebrate-other 42%,
vertebrate-mammalian 27%, invertebrate 20%, plant 8%, fungi 3%, protozoa 0.5%). The exact per-source weights are in
[`configs/mixture_300b.sh`](configs/mixture_300b.sh).

## Megatron

[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) works. The hybrid tokenizer loads through the standard
`--tokenizer-type HuggingFaceTokenizer --tokenizer-model <repo> --trust-remote-code` path, and neither stage uses a
custom loss. We ran on a fork pinned at upstream `0dba6a00`, but the only changes on it are logging (training ETA,
tokens/s, wandb run names), so they do not affect results.

The one file you do need is `pretrain_gpt_freeze_body_untied.py`, shipped here, which step 4 uses as its entrypoint.
Step 5 uses Megatron's own `pretrain_gpt.py`.

## Notes
- Add **`--untie-embeddings-and-output-weights`** in step 5. Qwen3-14B has untied embeddings; without this flag
  Megatron ties them and the `lm_head` is dropped at conversion, with no error.
- Use **`ORIGINAL_VOCAB_SIZE=151669`** in step 4. This is the tokenizer's `dna_start_id`, the first DNA row. A larger
  value such as Qwen3's 151,936 would freeze part of the DNA vocabulary you are trying to train.
- **"Frozen" in step 4 means zero gradient, not a constant tensor.** The entrypoint masks gradients on the
  original-vocab rows, but the launcher runs `--weight-decay 0.1` and Megatron applies decoupled weight decay to
  every 2-D parameter, so those rows still shrink by `lr * wd * p` each step. Small at this LR and token budget,
  but not zero. **This is unverified either way**: the scaffold and freeze checkpoints from the released run are no
  longer on disk, so `verify_freeze.py` could not be re-run against them. If its check B (original-vocab rows
  bitwise identical) passes on your own run, weight decay is not moving them in practice and this caveat is moot.
  If it fails, use `--weight-decay 0.0` for this stage, or give the embedding its own param group through
  `no_weight_decay_cond`. The released checkpoint was trained with weight decay on.
- `conversion/patch_exported_config.py` is not optional on export: Megatron-Bridge writes the reference model's
  `vocab_size` rather than the padded one, and can leave `rope_theta` null, which makes transformers fall back to
  10,000 and silently break Qwen3's positional encoding (the correct value is 1,000,000).
