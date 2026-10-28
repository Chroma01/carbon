"""Tokenize a FineDNA source into Megatron .bin/.idx with the Carbon-X hybrid tokenizer.

Every source in HuggingFaceBio/FineDNA has the same schema (`text`, `id`, `source`, `metadata`), so one script
covers all 23 of them: DNA, bio text, English and instruction data alike. Run it once per source.

Feed the text in as it comes. Nothing needs tagging beforehand: the tokenizer wraps A/C/G/T runs in
<dna>...</dna> itself and is idempotent on runs that are already wrapped.

Set --n_tasks close to the number of input files: datatrove writes one shard per task, so asking for more tasks
than there are files leaves 0-byte shards that the training blend then has to skip.

    python tokenize_finedna.py --input hf://datasets/HuggingFaceBio/FineDNA/data/dna_eukaryote_fungi \
        --output <out>/eukaryote/gener_metadata_fungi_qwen3h_npad \
        --save_name gener_metadata_fungi_qwen3h_npad --n_tasks 16
    # --local runs in this process instead of submitting Slurm jobs; --limit N caps documents PER TASK
"""
import argparse
import os
import re
from datatrove.executor import SlurmPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.tokens.megatron_tokenizer import MegatronDocumentTokenizer
from datatrove.data import DocumentsPipeline

# Carbon-X ships the same hybrid tokenizer as the standalone npad repo this was built with: identical
# vocabulary (157,184) and identical token ids on tagged, untagged and plain text. One repo is simpler.
TOKENIZER = "HuggingFaceBio/Carbon-X"

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--output", required=True)
ap.add_argument("--save_name", required=True)
ap.add_argument("--glob", default="train-*.parquet",
                help="seven bio_* subsets also ship validation-*.parquet, the docs held out to track "
                     "loss, so the default reads only the train shards")
ap.add_argument("--n_tasks", type=int, default=8)
ap.add_argument("--tokenizer", default=TOKENIZER)
ap.add_argument("--log_dir", default="./logs/tokenization")
ap.add_argument("--partition", default="hopper-cpu")
ap.add_argument("--limit", type=int, default=-1, help="stop after N documents (-1 = all); for smoke tests")
ap.add_argument("--overwrite", action="store_true", help="replace existing shards of the same save_name")
ap.add_argument("--local", action="store_true", help="run locally (LocalPipelineExecutor) instead of submitting Slurm jobs")
args = ap.parse_args()


class HybridMegatronTokenizer(MegatronDocumentTokenizer):
    name = "Hybrid Megatron Tokenizer"
    _hf_tokenizer = None

    @property
    def tokenizer(self):
        if self._hf_tokenizer is None:
            from transformers import AutoTokenizer
            self._hf_tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name_or_path, trust_remote_code=True)
        return self._hf_tokenizer

    @property
    def token_size(self) -> int:
        return 4  # npad vocab ~157k -> int32

    @property
    def eos_token_id(self) -> int:
        if self.eos_token:
            return self.tokenizer.convert_tokens_to_ids(self.eos_token)
        return self.tokenizer.eos_token_id

    def write_tokens(self, data: DocumentsPipeline, filename: str):
        from datatrove.pipeline.tokens.megatron_tokenizer import MegatronTokenizedFile
        from datatrove.utils.batching import batched
        unshuff = MegatronTokenizedFile(
            self.output_folder, filename,
            upload_block_size=self.upload_block_size, token_size=self.token_size)
        for batch in batched(data, self.batch_size):
            with self.track_time(unit="batch"):
                encoded = self.tokenizer(
                    [doc.text for doc in batch],
                    add_special_tokens=False, padding=False,
                    truncation=False, return_attention_mask=False)
                for tokens in encoded["input_ids"]:
                    if self.eos_token_id is not None:
                        tokens = list(tokens) + [self.eos_token_id]
                    unshuff.write(tokens)
                    self.stat_update("tokens", value=len(tokens))
        unshuff.close()
        return unshuff


if __name__ == "__main__":
    # logging_dir MUST be unique per output (train vs val share save_name!) or datatrove's
    # executor-state cache collides and both jobs run the same (last-submitted) config.
    # datatrove keys its completion markers on logging_dir, which is derived from --output below. Two runs with the
    # same --output therefore share state: the second finds the first's markers, skips, exits 0 and writes nothing.
    # Reading several sources into one directory has to be a single run with a glob that spans them, not a loop.
    import glob as _glob
    existing = _glob.glob(os.path.join(args.output, f"{args.save_name}_*_tokens.bin"))
    if existing and not args.overwrite:
        raise SystemExit(
            f"ERROR: {args.output} already holds {len(existing)} shard(s) named {args.save_name}_*.\n"
            f"  A second run with the same --output and --save_name would silently skip (shared datatrove state)\n"
            f"  or overwrite those shards. To read several sources into one directory, use ONE run with a glob that\n"
            f"  spans them, e.g. --input .../data --glob 'instruct_nemotron_*/*.parquet'.\n"
            f"  Pass --overwrite if you really mean to replace them."
        )
    # Derive the state dir from the INPUT as well as the output. Keyed on the output alone, two runs reading
    # different sources into one directory share datatrove state and the second silently skips. With the input in
    # the key they stay independent, so writing several sources into one directory works as long as each run uses
    # its own --save_name (which the check above enforces).
    _in = re.sub(r"[^A-Za-z0-9]+", "_", args.input.rstrip("/")).strip("_")[-40:]
    tag = "_".join(args.output.rstrip("/").split("/")[-2:]) + "__" + _in
    print(f"INPUT={args.input}\nOUTPUT={args.output}\nsave_name={args.save_name} n_tasks={args.n_tasks} tag={tag} local={args.local}")
    pipeline=[
        ParquetReader(args.input, glob_pattern=args.glob, text_key="text",
                      limit=args.limit if args.limit > 0 else -1),
        HybridMegatronTokenizer(
            output_folder=args.output,
            tokenizer_name_or_path=args.tokenizer,
            eos_token="<|endoftext|>",
            batch_size=100,
            save_filename=args.save_name),
    ]
    if args.local:
        from datatrove.executor import LocalPipelineExecutor
        ex = LocalPipelineExecutor(pipeline=pipeline, tasks=args.n_tasks, workers=args.n_tasks,
                                   logging_dir=f"{args.log_dir}/tokenize_{tag}")
    else:
        ex = SlurmPipelineExecutor(
            job_name=f"meg-tok-{tag}", pipeline=pipeline,
            tasks=args.n_tasks, time="6:00:00", partition=args.partition,
            cpus_per_task=24, qos="normal", mem_per_cpu_gb=1,
            logging_dir=f"{args.log_dir}/tokenize_{tag}",
        )
    ex.run()
