# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Remove exact whole-span DNA or eligible word n-gram matches with evaluation examples.

Evaluation examples are the five HuggingFaceBio/DNAIntelligenceIndex benchmarks, at a revision pinned in EVAL_REVISION.
DNA detection uses the HuggingFaceBio/Carbon-X tokenizer.
Complete DNA-token spans must match exactly and contain at least six DNA tokens, excluding boundary markers.
Tail and <oov> tokens count toward span length; all-<oov> spans are excluded.
Word matching normalizes Unicode and case and ignores punctuation.
The <dna> placeholder remains a distinct word token, separate from the ordinary word DNA.
Eval records without a qualifying DNA span use word matching, with DNA replaced by a placeholder.
DNA-bearing eval records require a shared complete normalized DNA sequence for word matching.
Shared word n-grams must cover >50% of one masked eval prompt to trigger removal.
For DNA-free evals, n-grams in more than one eval prompt or more than ten training prompts are template text and ignored,
unless shared n-grams cover the whole eval prompt.
Coverage is measured against the first training user prompt; lower-coverage overlaps remain review candidates.
For message lists, only the first user message is checked. Options/choices contribute DNA only.
Every run prints training-set and match tables and saves JSON reports and a native Inspect decontamination.eval log.
Reports default to outputs/decontamination/<dataset>/<config>/<split>. Use --output_dir to choose another directory.
Reports contain flagged rows with original training and evaluation prompts, IDs, choices, and match reasons.
At most ten eval matches are stored per training row; decisions and counts use every match.
Use --push_to_hub to publish the cleaned split under <config>_decontaminated_ngram in the source repository.
A config already containing "decontaminated" gets only the _ngram suffix, so the three scripts chain into one name.
Publishing also updates a Decontamination section in the dataset card, replacing results for the same config/split.

Usage:

uv run data/decontaminate_ngram.py \
    --dataset HuggingFaceBio/FineDNA \
    --config sft_mix \
    --split train \
    --prompt_column messages
"""

import argparse
import json
import logging
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from datasets import Features, List, Value, disable_progress_bars, enable_progress_bars, load_dataset
from huggingface_hub import DatasetCard, HfApi, hf_hub_download
from huggingface_hub.utils import disable_progress_bars as disable_hub_progress_bars
from huggingface_hub.utils import enable_progress_bars as enable_hub_progress_bars
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.utils import logging as transformers_logging


logger = logging.getLogger(__name__)
EVAL_DATASET = "HuggingFaceBio/DNAIntelligenceIndex"
EVAL_REVISION = "2e156b12ef70c0d2295394f90c885ec450c5c79f"
# Subsets of EVAL_DATASET and the column holding their prompts. Each has a single train split of evaluation examples.
EVALS = {
    "bioreason_coding": "question",
    "bioreason_non_snv": "question",
    "frontier_vep": "prompt",
    "genomeqa": "question",
    "nextbase": "prompt",
}
TOKENIZER = "HuggingFaceBio/Carbon-X"
TOKENIZER_REVISION = "17de89165e25c49e7de4f845d0e5d464469de359"
MIN_DNA_TOKENS = 6
NGRAM_COVERAGE_THRESHOLD = 0.5
TEMPLATE_MAX_TRAINING_ROWS = 10
MAX_REPORT_MATCHES = 10


def record_prompt(record: dict, prompt_column: str) -> str:
    """
    Extract original text from a plain prompt or the first user message.

    Args:
        record (`dict`):
            A dataset record.
        prompt_column (`str`):
            Text or message-list column to check.

    Returns:
        `str`: The original prompt text, before normalization.
    """
    prompt = record[prompt_column]
    if isinstance(prompt, list):
        for message in prompt:
            if message["role"] == "user":
                return message["content"]
        raise ValueError(f"Prompt column {prompt_column!r} contains no user message.")
    return prompt


def text_signatures(text: str, tokenizer) -> tuple[set[tuple[int, ...]], set[str], str]:
    """
    Extract qualifying DNA-token spans, normalized DNA sequences, and normalized masked text.

    Args:
        text (`str`):
            Text containing DNA sequences, with or without <dna> tags.
        tokenizer (`HybridDNATokenizer`):
            The hybrid tokenizer, which owns DNA detection.

    Returns:
        `tuple`: The distinct qualifying DNA-token spans, normalized DNA sequences, and normalized masked text.
    """
    text = unicodedata.normalize("NFKC", text)
    text = tokenizer._add_auto_dna_tags(text)
    dna_spans, dna_sequences, masked_text = set(), set(), []
    for segment, is_dna in tokenizer._split_by_dna_tags(text):
        if not is_dna:
            masked_text.append(segment)
            continue
        sequence = re.sub(r"</?dna>|\s+", "", segment).upper()
        if sequence:
            dna_sequences.add(sequence)
        masked_text.append(" <dna> ")
        token_ids = tokenizer.encode(segment, add_special_tokens=False, auto_dna_tags=False)
        span = tuple(
            token for token in token_ids if token not in (tokenizer.dna_begin_token_id, tokenizer.dna_end_token_id)
        )
        # Compare the entire span; short motifs fall through to masked word matching.
        if len(span) >= MIN_DNA_TOKENS and any(token != tokenizer.oov_token_id for token in span):
            dna_spans.add(span)
    return dna_spans, dna_sequences, " ".join("".join(masked_text).casefold().split())


def record_signatures(
    record: dict, prompt_column: str, tokenizer, ngram_size: int = 8
) -> tuple[set[tuple[int, ...]], int, dict[str, set[int]], set[str]]:
    """
    Extract qualifying whole DNA-token spans and DNA-masked word n-grams.

    Args:
        record (`dict`):
            A dataset record. Assistant messages and reference answers are excluded.
        prompt_column (`str`):
            Text or message-list column to check; message lists use the first user message.
        tokenizer (`HybridDNATokenizer`):
            The hybrid tokenizer. Automatic DNA tagging is always enabled.
        ngram_size (`int`, *optional*, defaults to `8`):
            Number of words in each n-gram. Choices contribute DNA sequences and spans only.

    Returns:
        `tuple`: DNA spans, prompt word count, n-gram positions, and normalized DNA sequences.
    """
    dna_spans, dna_sequences, text = text_signatures(record_prompt(record, prompt_column), tokenizer)
    choices = set()
    for column in ("options", "choices"):
        if column in record:
            choices.update(record[column])
    for choice in choices:
        choice_dna, choice_sequences, _ = text_signatures(choice, tokenizer)
        dna_spans.update(choice_dna)
        dna_sequences.update(choice_sequences)
    # Every training prompt can match eval text; evals with a qualifying span use DNA matching instead.
    # Keep the DNA mask distinct from the ordinary word "dna".
    words = re.findall(r"<dna>|\w+", text)
    ngrams = {}
    for index in range(len(words) - ngram_size + 1):
        ngram = " ".join(words[index : index + ngram_size])
        ngrams.setdefault(ngram, set()).update(range(index, index + ngram_size))
    return dna_spans, len(words), ngrams, dna_sequences


def update_decontamination_card(card: DatasetCard, block_id: str, block: str):
    """
    Insert or replace a config/split's results, rejecting an incomplete existing block.

    Args:
        card (`DatasetCard`):
            Dataset card to update in place, preserving its metadata and other text.
        block_id (`str`):
            Published config/split identifying the generated subsection.
        block (`str`):
            Markdown results enclosed in the config/split's HTML comment markers.
    """
    start_marker = f"<!-- decontamination:start:{block_id} -->"
    end_marker = f"<!-- decontamination:end:{block_id} -->"
    if start_marker in card.text:
        start = card.text.index(start_marker)
        if end_marker not in card.text[start:]:
            raise ValueError(f"Dataset card is missing the decontamination end marker for {block_id}.")
        end = card.text.index(end_marker, start) + len(end_marker)
        card.text = card.text[:start] + block + card.text[end:]
    else:
        heading = re.search(r"^# Decontamination[ \t]*$", card.text, flags=re.MULTILINE)
        if heading:
            card.text = card.text[: heading.end()] + "\n\n" + block + "\n\n" + card.text[heading.end() :]
        else:
            card.text = card.text.rstrip() + "\n\n# Decontamination\n\n" + block + "\n"


def write_match_log(rows: list[dict], summary: dict, path: Path, started_at: str):
    """
    Write flagged training rows and their evaluation matches as a native Inspect log.

    Args:
        rows (`list[dict]`):
            Removal and review-candidate records with original prompts and match details.
        summary (`dict`):
            Statistics and source dataset metadata for the complete input split.
        path (`Path`):
            Destination .eval file.
        started_at (`str`):
            Time the decontamination run started, in UTC ISO format.
    """
    from inspect_ai.log import (
        EvalConfig,
        EvalDataset,
        EvalLog,
        EvalMetric,
        EvalResults,
        EvalSample,
        EvalScore,
        EvalSpec,
        EvalStats,
        write_eval_log,
    )
    from inspect_ai.model import ChatMessageUser
    from inspect_ai.scorer import Score

    samples = []
    for row in rows:
        messages = [
            ChatMessageUser(
                id=f"training-{row['row_index']}",
                source="input",
                content=f"## Training prompt\n\n{row['prompt']}",
            )
        ]
        choices = list(dict.fromkeys(choice for column in ("options", "choices") for choice in row.get(column, [])))
        if choices:
            messages.append(
                ChatMessageUser(
                    id=f"training-{row['row_index']}-choices",
                    source="input",
                    content="## Training choices\n\n"
                    + "\n".join(f"{index + 1}. {choice}" for index, choice in enumerate(choices)),
                )
            )
        targets = []
        explanations = [
            f"Reporting {sum(len(matches) for matches in row['matches'].values())} of "
            f"{sum(row['match_counts'].values())} eval matches ({row['omitted_matches']} omitted)."
        ]
        for eval_name, matches in row["matches"].items():
            for match in matches:
                label = f"{eval_name} / row {match['row_index']} / id {match['id']}"
                target = match["prompt"]
                for column in ("options", "choices"):
                    if column in match:
                        target += f"\n\n{column}:\n" + "\n".join(
                            f"{index + 1}. {choice}" for index, choice in enumerate(match[column])
                        )
                targets.append(target)
                explanation = ", ".join(match["reasons"])
                if "ngrams" in match:
                    decision = "Triggers removal" if match["ngram_removal"] else "Review candidate only"
                    if match["eval_copy"]:
                        decision += " as a full eval-prompt copy"
                    explanation += (
                        f"\n{decision}: eval-word coverage {match['ngram_coverage']:.2%}; "
                        f"removal requires >{NGRAM_COVERAGE_THRESHOLD:.0%}.\n"
                        f"{match['covered_eval_words']}/{match['eval_words']} eval words.\nShared n-grams:\n"
                        + "\n".join(match["ngrams"])
                    )
                explanations.append(f"{label}: {explanation}")
                messages.append(
                    ChatMessageUser(
                        id=f"evaluation-{row['row_index']}-{eval_name}-{match['row_index']}",
                        source="input",
                        content="## Evaluation match\n\n"
                        "| benchmark | row | id |\n| --- | ---: | --- |\n"
                        f"| {eval_name} | {match['row_index']} | {match['id']} |\n\n"
                        f"{target}\n\n### Why it matched\n\n{explanation}",
                    )
                )
        samples.append(
            EvalSample(
                id=row["row_index"],
                epoch=1,
                input=row["prompt"],
                choices=choices or None,
                target=targets,
                messages=messages,
                scores={
                    "removed": Score(
                        value=int(row["removed"]),
                        answer="removed" if row["removed"] else "review_candidate",
                        explanation="\n\n".join(explanations),
                    )
                },
                metadata=row,
            )
        )
    log = EvalLog(
        status="success",
        eval=EvalSpec(
            created=started_at,
            task="decontamination",
            task_display_name="Decontamination review",
            model="none",
            dataset=EvalDataset(name=summary["dataset"], samples=len(samples)),
            config=EvalConfig(),
            metadata=summary,
        ),
        results=EvalResults(
            total_samples=len(samples),
            completed_samples=len(samples),
            scores=[
                EvalScore(
                    name="removed",
                    scorer="decontamination",
                    scored_samples=len(samples),
                    metrics={
                        name: EvalMetric(name=name, value=summary[name])
                        for name in ("removed_rows", "ngram_removals", "review_candidate_rows")
                    },
                )
            ],
        ),
        stats=EvalStats(started_at=started_at, completed_at=datetime.now(timezone.utc).isoformat()),
        samples=samples,
    )
    write_eval_log(log, path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    transformers_logging.set_verbosity_warning()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to check for contamination.")
    parser.add_argument("--config", type=str, default="default", help="Name of the dataset config to load.")
    parser.add_argument("--split", type=str, default="train", help="Name of the input split to decontaminate.")
    parser.add_argument(
        "--prompt_column", type=str, default="problem", help="Name of the column containing the prompt."
    )
    parser.add_argument(
        "--ngram_size",
        type=int,
        default=8,
        help="Number of words per n-gram for evals without qualifying DNA spans, defaults to 8.",
    )
    parser.add_argument("--num_proc", type=int, default=8, help="Number of processes to use.")
    parser.add_argument(
        "--num_shards", type=int, default=None, help="Number of shards to use when pushing to the Hub."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Publish the cleaned split to the Hub after displaying the preview. Disabled by default.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Directory for review reports. Defaults to outputs/decontamination/<dataset>/<config>/<split>.",
    )
    args = parser.parse_args()
    if args.ngram_size < 1:
        parser.error("--ngram_size must be positive")
    if args.output_dir is None:
        args.output_dir = (
            Path(__file__).resolve().parents[1]
            / "outputs"
            / "decontamination"
            / args.dataset
            / args.config
            / args.split
        )
    started_at = datetime.now(timezone.utc).isoformat()

    # Suppress download bars and dataset-builder notices until all evaluation datasets are loaded.
    disable_progress_bars()
    disable_hub_progress_bars()
    builder_logger = logging.getLogger("datasets.builder")
    builder_log_level = builder_logger.level
    builder_logger.setLevel(logging.ERROR)
    ds = load_dataset(args.dataset, name=args.config, split=args.split, num_proc=args.num_proc)
    initial_size = len(ds)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, revision=TOKENIZER_REVISION, trust_remote_code=True)

    eval_datasets = {
        eval_name: load_dataset(EVAL_DATASET, name=eval_name, split="train", revision=EVAL_REVISION)
        for eval_name in EVALS
    }
    enable_progress_bars()
    enable_hub_progress_bars()
    builder_logger.setLevel(builder_log_level)
    eval_signatures = {}
    eval_dna_rows = {}
    for eval_name, eval_dataset in eval_datasets.items():
        dna_lookup, ngram_lookup, word_counts, word_dna = {}, {}, {}, {}
        dna_rows = 0
        for row_index, row in enumerate(tqdm(eval_dataset, desc=f"Tokenizing {eval_name}")):
            row_dna, word_count, row_ngrams, dna_sequences = record_signatures(
                row, EVALS[eval_name], tokenizer, args.ngram_size
            )
            dna_rows += bool(dna_sequences)
            for signature in row_dna:
                dna_lookup.setdefault(signature, []).append(row_index)
            # Short DNA uses text matching with a shared-sequence requirement; DNA-free evals use text alone.
            if not row_dna:
                word_counts[row_index] = word_count
                word_dna[row_index] = dna_sequences
                for ngram, word_positions in row_ngrams.items():
                    ngram_lookup.setdefault(ngram, {})[row_index] = word_positions
        eval_signatures[eval_name] = (dna_lookup, ngram_lookup, word_counts, word_dna)
        eval_dna_rows[eval_name] = dna_rows

    # Count the training prompts holding each eval n-gram; wording repeated across many prompts is a template.
    eval_ngrams = set().union(*(ngram_lookup.keys() for _, ngram_lookup, _, _ in eval_signatures.values()))
    training_ngrams = ds.map(
        lambda row: {
            "ngrams": sorted(
                record_signatures(row, args.prompt_column, tokenizer, args.ngram_size)[2].keys() & eval_ngrams
            )
        },
        remove_columns=ds.column_names,
        # Declare the type: a shard whose first rows share no n-grams would otherwise infer list<null>.
        features=Features({"ngrams": List(Value("string"))}),
        num_proc=args.num_proc,
        desc="Counting shared n-grams",
    )["ngrams"]
    training_rows = Counter(ngram for ngrams in training_ngrams for ngram in ngrams)

    def record_matches(row):
        dna_spans, _, ngrams, dna_sequences = record_signatures(row, args.prompt_column, tokenizer, args.ngram_size)
        matches = {
            "_decontamination_dna_matches": [],
            "_decontamination_ngram_matches": [],
            "_decontamination_ngram_removals": [],
            "_decontamination_has_dna": bool(dna_sequences),
        }
        details = {}
        for eval_name, (dna_lookup, ngram_lookup, word_counts, word_dna) in eval_signatures.items():
            dna_hits = set()
            for signature in dna_spans:
                eval_rows = dna_lookup.get(signature)
                if eval_rows:
                    dna_hits.update(eval_rows)
            matches["_decontamination_dna_matches"].append(bool(dna_hits))
            hits = {row_index: {"row_index": row_index, "reasons": ["dna"]} for row_index in sorted(dna_hits)}
            word_hits = {}
            # Template overlap counts only when DNA-bearing evals share a complete normalized DNA sequence.
            eligible_word_rows = {
                row_index for row_index, sequences in word_dna.items() if not sequences or sequences & dna_sequences
            }
            if eligible_word_rows:
                for ngram in sorted(ngrams.keys() & ngram_lookup.keys()):
                    template = len(ngram_lookup[ngram]) > 1 or training_rows[ngram] > TEMPLATE_MAX_TRAINING_ROWS
                    for row_index, positions in ngram_lookup[ngram].items():
                        if row_index not in eligible_word_rows:
                            continue
                        word_hit = word_hits.setdefault(
                            row_index, {"positions": set(), "all_positions": set(), "ngrams": []}
                        )
                        word_hit["all_positions"].update(positions)
                        # DNA-bearing evals already require a shared sequence, so their templates still count.
                        if template and not word_dna[row_index]:
                            continue
                        word_hit["positions"].update(positions)
                        word_hit["ngrams"].append(ngram)
            for row_index, word_hit in word_hits.items():
                # Template n-grams still count when they cover every eval word: the training prompt holds a full copy.
                eval_copy = len(word_hit["all_positions"]) == word_counts[row_index]
                if not word_hit["positions"] and not eval_copy:
                    continue
                hit = hits.setdefault(row_index, {"row_index": row_index, "reasons": []})
                hit["reasons"].append("ngram")
                covered_words = len(word_hit["positions"])
                # Measure one eval prompt against the selected training prompt; never pool eval rows.
                coverage = covered_words / word_counts[row_index]
                hit.update(
                    {
                        "covered_eval_words": covered_words,
                        "eval_words": word_counts[row_index],
                        "ngram_coverage": coverage,
                        "ngram_removal": coverage > NGRAM_COVERAGE_THRESHOLD or eval_copy,
                        "eval_copy": eval_copy,
                        "ngrams": word_hit["ngrams"],
                    }
                )
            matches["_decontamination_ngram_matches"].append(any("ngram" in hit["reasons"] for hit in hits.values()))
            matches["_decontamination_ngram_removals"].append(
                any(hit.get("ngram_removal", False) for hit in hits.values())
            )
            if hits:
                details[eval_name] = [hits[index] for index in sorted(hits)]
        matches["_decontamination_match_counts"] = json.dumps(
            {eval_name: len(hits) for eval_name, hits in details.items()}
        )
        # Compute every decision before capping report examples, prioritizing removal evidence.
        report_hits = sorted(
            ((eval_name, hit) for eval_name, hits in details.items() for hit in hits),
            key=lambda item: (
                "dna" not in item[1]["reasons"],
                not item[1].get("ngram_removal", False),
                item[0],
                item[1]["row_index"],
            ),
        )[:MAX_REPORT_MATCHES]
        details = {}
        for eval_name, hit in report_hits:
            details.setdefault(eval_name, []).append(hit)
        matches["_decontamination_match_details"] = json.dumps(details, ensure_ascii=False)
        return matches

    # Tokenize each source row once, retaining per-eval matches for reporting.
    ds = ds.map(record_matches, num_proc=args.num_proc, desc="Checking DNA and word n-gram matches")
    dna_matches = ds["_decontamination_dna_matches"] if initial_size else []
    ngram_matches = ds["_decontamination_ngram_matches"] if initial_size else []
    ngram_removals = ds["_decontamination_ngram_removals"] if initial_size else []
    removed_flags = [any(dna) or any(ngram) for dna, ngram in zip(dna_matches, ngram_removals, strict=True)]
    benchmark_stats = {}
    for index, eval_name in enumerate(eval_signatures):
        dna_count = sum(matches[index] for matches in dna_matches)
        removed_count = sum(dna[index] or ngram[index] for dna, ngram in zip(dna_matches, ngram_removals, strict=True))
        ngram_count = sum(matches[index] for matches in ngram_matches)
        benchmark_stats[eval_name] = {
            "revision": EVAL_REVISION,
            "eval_rows": len(eval_datasets[eval_name]),
            "eval_dna_rows": eval_dna_rows[eval_name],
            "eval_dna_free_rows": len(eval_datasets[eval_name]) - eval_dna_rows[eval_name],
            "eval_word_matching_rows": len(eval_signatures[eval_name][2]),
            "dna_matches": dna_count,
            "ngram_matches": ngram_count,
            "ngram_removals": sum(matches[index] for matches in ngram_removals),
            "removed_rows": removed_count,
            "removed_fraction": removed_count / initial_size if initial_size else 0.0,
        }

    dna_count = sum(any(matches) for matches in dna_matches)
    removed_count = sum(removed_flags)
    dna_rows = sum(ds["_decontamination_has_dna"]) if initial_size else 0
    summary = {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "prompt_column": args.prompt_column,
        "id_column": "id" if "id" in ds.column_names else None,
        "eval_dataset": EVAL_DATASET,
        "tokenizer": {"name": TOKENIZER, "revision": TOKENIZER_REVISION},
        "dna_matching": "whole_span",
        "min_dna_tokens": MIN_DNA_TOKENS,
        "push_to_hub": args.push_to_hub,
        "ngram_size": args.ngram_size,
        "ngram_word_tokenization": r"<dna>|\w+",
        "ngram_coverage_threshold": NGRAM_COVERAGE_THRESHOLD,
        "ngram_coverage_unit": "eval_word_positions",
        "ngram_eval_eligibility": "no_qualifying_dna_span",
        "ngram_dna_masking": True,
        "ngram_requires_shared_dna": True,
        "ngram_template_max_training_rows": TEMPLATE_MAX_TRAINING_ROWS,
        "max_report_matches": MAX_REPORT_MATCHES,
        "initial_rows": initial_size,
        "dna_rows": dna_rows,
        "dna_free_rows": initial_size - dna_rows,
        "retained_rows": initial_size - removed_count,
        "removed_rows": removed_count,
        "removed_fraction": removed_count / initial_size if initial_size else 0.0,
        "dna_matches": dna_count,
        "ngram_matches": sum(any(matches) for matches in ngram_matches),
        "ngram_removals": sum(any(matches) for matches in ngram_removals),
        "review_candidate_rows": sum(
            any(matches) and not removed for matches, removed in zip(ngram_matches, removed_flags, strict=True)
        ),
        "benchmarks": benchmark_stats,
    }
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report_rows = []
    with (output_dir / "matches.jsonl").open("w", encoding="utf-8") as output_file:
        for row_index, removed in enumerate(removed_flags):
            if not removed and not any(ngram_matches[row_index]):
                continue
            source = ds[row_index]
            source_info = {
                "row_index": row_index,
                "id": source.get("id"),
                "prompt": record_prompt(source, args.prompt_column),
                **{column: source[column] for column in ("options", "choices") if column in source},
            }
            details = json.loads(source["_decontamination_match_details"])
            matches = {}
            for eval_name, hits in details.items():
                for hit in hits:
                    eval_row = eval_datasets[eval_name][hit["row_index"]]
                    match = {
                        "row_index": hit["row_index"],
                        "id": eval_row.get("id"),
                        "prompt": eval_row[EVALS[eval_name]],
                        **{column: eval_row[column] for column in ("options", "choices") if column in eval_row},
                        "reasons": hit["reasons"],
                    }
                    if "ngrams" in hit:
                        match.update(
                            {
                                key: hit[key]
                                for key in (
                                    "ngrams",
                                    "covered_eval_words",
                                    "eval_words",
                                    "ngram_coverage",
                                    "ngram_removal",
                                    "eval_copy",
                                )
                            }
                        )
                    matches.setdefault(eval_name, []).append(match)
            match_counts = json.loads(source["_decontamination_match_counts"])
            row = {
                **source_info,
                "removed": removed,
                "matches": matches,
                "match_counts": match_counts,
                "omitted_matches": sum(match_counts.values()) - sum(len(hits) for hits in matches.values()),
            }
            report_rows.append(row)
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_match_log(report_rows, summary, output_dir / "decontamination.eval", started_at)

    info_rows = [
        ["Training set", "Value"],
        ["Dataset", args.dataset],
        ["Config", args.config],
        ["Split", args.split],
        ["Prompt column", args.prompt_column],
        ["Rows", f"{initial_size:,}"],
        ["Flagged rows", f"{len(report_rows):,}"],
        ["Review candidates", f"{summary['review_candidate_rows']:,}"],
        ["Evaluation dataset", f"{EVAL_DATASET}@{EVAL_REVISION}"],
        ["DNA removal rule", f"Exact complete DNA span with >={MIN_DNA_TOKENS} DNA tokens"],
        ["Word tokenization", r"<dna>|\w+ (Unicode; DNA mask preserved; punctuation ignored)"],
        ["Word DNA requirement", "DNA-bearing evals must share a complete normalized DNA sequence"],
        ["N-gram removal rule", f">{NGRAM_COVERAGE_THRESHOLD:.0%} masked eval-word coverage per training prompt"],
        [
            "Word template rule",
            f"DNA-free evals ignore n-grams in >1 eval prompt or >{TEMPLATE_MAX_TRAINING_ROWS} training prompts, "
            "except in full eval-prompt copies",
        ],
        ["Report match limit", f"{MAX_REPORT_MATCHES} examples per training row; complete counts retained"],
        ["Summary", str((output_dir / "summary.json").resolve())],
        ["Inspect log", str((output_dir / "decontamination.eval").resolve())],
        ["Push to Hub", "Enabled" if args.push_to_hub else "Disabled (use --push_to_hub)"],
    ]
    match_rows = [["Benchmark", "DNA", "N-gram", "Removed", "Removed %"]]
    for name, stats in [*benchmark_stats.items(), ("All benchmarks", summary)]:
        match_rows.append(
            [
                name,
                f"{stats['dna_matches']:,}",
                f"{stats['ngram_removals']:,}",
                f"{stats['removed_rows']:,}",
                f"{stats['removed_fraction']:.2%}",
            ]
        )
    for rows, numeric in ((info_rows, False), (match_rows, True)):
        widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
        borders = ["─" * (width + 2) for width in widths]
        print("\n┌" + "┬".join(borders) + "┐")
        for index, row in enumerate(rows):
            cells = [
                row[0].ljust(widths[0]),
                *(
                    cell.rjust(width) if numeric else cell.ljust(width)
                    for cell, width in zip(row[1:], widths[1:], strict=True)
                ),
            ]
            print("│ " + " │ ".join(cells) + " │")
            if index == 0 or (numeric and index == len(rows) - 2):
                print("├" + "┼".join(borders) + "┤")
        print("└" + "┴".join(borders) + "┘")

    if args.push_to_hub:
        if initial_size:
            ds = ds.filter(
                lambda row: not any(row["_decontamination_dna_matches"])
                and not any(row["_decontamination_ngram_removals"]),
                num_proc=args.num_proc,
                desc="Removing matched rows",
            ).remove_columns(
                [
                    "_decontamination_dna_matches",
                    "_decontamination_ngram_matches",
                    "_decontamination_ngram_removals",
                    "_decontamination_has_dna",
                    "_decontamination_match_counts",
                    "_decontamination_match_details",
                ]
            )

        # Chain onto earlier steps, e.g. all -> all_decontaminated_ngram_dna_homology.
        prefix = args.config if "decontaminated" in args.config else f"{args.config}_decontaminated"
        config_name = f"{prefix}_ngram"
        card_info_rows = [row for row in info_rows if row[0] not in ("Summary", "Inspect log")]
        tables = []
        for rows in (card_info_rows, match_rows):
            table_rows = [rows[0], ["---"] * len(rows[0]), *rows[1:]]
            # Escape pipes and angle brackets so cells like "<dna>|\w+" render intact in Markdown tables.
            tables.append(
                "\n".join(
                    "| "
                    + " | ".join(cell.replace("|", "\\|").replace("<", "&lt;").replace(">", "&gt;") for cell in row)
                    + " |"
                    for row in table_rows
                )
            )

        block_id = f"{config_name}/{args.split}"
        start_marker = f"<!-- decontamination:start:{block_id} -->"
        end_marker = f"<!-- decontamination:end:{block_id} -->"
        block = f"{start_marker}\n## {block_id}\n\n" + "\n\n".join(tables) + f"\n\n{end_marker}"
        # Read the card at a fixed commit; reject a stale write when publishing.
        api = HfApi()
        parent_commit = api.dataset_info(args.dataset).sha
        card = DatasetCard.load(
            hf_hub_download(args.dataset, "README.md", repo_type="dataset", revision=parent_commit)
        )
        update_decontamination_card(card, block_id, block)

        # Check markers and validate the generated card before publishing any data.
        card.validate()
        url = ds.push_to_hub(args.dataset, config_name=config_name, split=args.split, num_shards=args.num_shards)
        logger.info("Decontaminated dataset: %s", url)

        # Read the updated config metadata at a fixed commit; reject a stale card write.
        parent_commit = api.dataset_info(args.dataset).sha
        card = DatasetCard.load(
            hf_hub_download(args.dataset, "README.md", repo_type="dataset", revision=parent_commit)
        )
        update_decontamination_card(card, block_id, block)
        url = card.push_to_hub(
            args.dataset, commit_message=f"Document decontamination of {block_id}", parent_commit=parent_commit
        )
        logger.info("Decontamination dataset card: %s", url)
