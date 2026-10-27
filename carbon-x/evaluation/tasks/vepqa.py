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

import math
import re
from collections import defaultdict
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.log import HeadlineMetric
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Metric,
    SampleScore,
    Score,
    Scorer,
    Target,
    accuracy,
    metric,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState, generate


DATASET_PATH = "HuggingFaceBio/vepqa"
DATASET_REVISION = "1dbf17df3f8edf7c28f9f65ea2740bca86fb9dc6"
MULTI_DATASET_REVISION = "bcaf10dbd8399c56f75d43d6ed363bf6caa3eea9"
GROUP_DATASET_REVISION = "9cee4208ab2912f9faecdc8e93e6a4b85bbc9948"
ALL_DATASET_REVISION = "bcaf10dbd8399c56f75d43d6ed363bf6caa3eea9"
SINGLE_1K_DATASET_REVISION = "a94c567c8d96717f367867010513691bf6eba9f4"
FAMILIES = {
    "coding_only": {"nonsense", "missense_variant", "synonymous_variant", "initiator_codon_variant", "stop_lost"},
    "canonical_splice_only": {"splice_acceptor_variant", "splice_donor_variant"},
    "noncoding_only": {
        "intron_variant",
        "5_prime_UTR_variant",
        "3_prime_UTR_variant",
        "non-coding_transcript_variant",
        "genic_upstream_transcript_variant",
        "genic_downstream_transcript_variant",
    },
}
CODING_SUBGROUPS = {
    "nonsense": "nonsense_only",
    "missense_variant": "missense_only",
    "synonymous_variant": "synonymous_only",
    "initiator_codon_variant": "initiator_codon_only",
    "stop_lost": "stop_lost_only",
}
SUBGROUPS = (
    "nonsense_only",
    "missense_only",
    "synonymous_only",
    "multiple_coding_effects",
    "noncoding_only",
    "canonical_splice_only",
    "mixed_annotation_families",
    "unannotated",
)
MACRO_GROUPS = (
    "missense_only",
    "synonymous_only",
    "noncoding_only",
    "canonical_splice_only",
    "mixed_annotation_families",
    "unannotated",
)
COT_SUFFIX = "Let's think step by step. End with <answer>...</answer> containing only the final answer."
_FINAL_ANSWER_PATTERN = re.compile(r"(<answer>[^<]*</answer>)\s*\Z", re.DOTALL)


def record_to_sample(record: dict[str, Any], config: str, suffix: str | None = None) -> Sample:
    return Sample(
        input=record["question"] + "\n\n" + (record["prompt_suffix"] if suffix is None else suffix),
        target=record["answer_text"],
        id=record["id"],
        metadata={
            "tag": "binary",
            "config": config,
            "label_binary": record["label_binary"],
            "group": record["region_group"],
            "subtype": record["region_subtype"],
            "region_group": record["region_group"],
            "region_subtype": record["region_subtype"],
        },
    )


def _extract_final_answer(completion: str) -> str | None:
    match = _FINAL_ANSWER_PATTERN.search(completion)
    return match.group(1) if match else None


def classify_consequence_terms(raw_terms: list[str], sample_id: str) -> tuple[str, str]:
    """Assign the mutually exclusive annotation family and coding subgroup from all consequence terms."""
    terms = set(raw_terms)
    unknown = terms - set().union(*FAMILIES.values(), {"no_sequence_alteration"})
    if unknown:
        raise ValueError(f"Sample {sample_id}: unmapped consequence terms {sorted(unknown)}")
    if not terms:
        return "unannotated", "unannotated"
    effects = terms - {"no_sequence_alteration"}
    if not effects:
        raise ValueError(f"Sample {sample_id}: no_sequence_alteration alone has no annotation family")
    families = [name for name, members in FAMILIES.items() if effects & members]
    if len(families) > 1:
        return "mixed_annotation_families", "mixed_annotation_families"
    group = families[0]
    if group == "coding_only":
        return group, CODING_SUBGROUPS[next(iter(effects))] if len(effects) == 1 else "multiple_coding_effects"
    return group, group


def all_record_to_sample(record: dict[str, Any], config: str = "binary_v3_random") -> Sample:
    sample = record_to_sample(record, config)
    _, subgroup = classify_consequence_terms(record["consequence_terms"], sample.id)
    sample.metadata["subgroup"] = subgroup
    return sample


@metric(scores="unreduced")
def vepqa_all_metrics() -> Metric:
    """Report overall and per-subgroup confusion counts, rates, and rate standard errors."""

    def summarize(scores: list[SampleScore]) -> dict[str, float | int]:
        def agresti_coull_stderr(successes: int, total: int) -> float:
            # Adding two successes and two failures keeps the stderr nonzero for small cells at 0% or 100%.
            adjusted = (successes + 2) / (total + 4)
            return math.sqrt(adjusted * (1 - adjusted) / (total + 4)) if total else math.nan

        def counts(rows: list[SampleScore]) -> dict[str, float | int]:
            tn = fp = fn = tp = invalid_benign = invalid_pathogenic = 0
            for row in rows:
                label = row.sample_metadata["label_binary"]
                prediction = row.score.metadata["predicted_label"]
                if label == "benign":
                    tn += prediction == "benign"
                    fp += prediction == "pathogenic"
                    invalid_benign += prediction is None
                else:
                    fn += prediction == "benign"
                    tp += prediction == "pathogenic"
                    invalid_pathogenic += prediction is None
            benign_n = tn + fp + invalid_benign
            pathogenic_n = fn + tp + invalid_pathogenic
            n = benign_n + pathogenic_n
            invalid_n = invalid_benign + invalid_pathogenic
            benign_recall = tn / benign_n if benign_n else math.nan
            pathogenic_recall = tp / pathogenic_n if pathogenic_n else math.nan
            benign_recall_stderr = agresti_coull_stderr(tn, benign_n)
            pathogenic_recall_stderr = agresti_coull_stderr(tp, pathogenic_n)
            mcc_n = tn + fp + fn + tp
            denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            mcc = math.nan
            if tn + fp and tp + fn:
                mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
            return {
                "n": n,
                "benign_n": benign_n,
                "pathogenic_n": pathogenic_n,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "invalid_benign_n": invalid_benign,
                "invalid_pathogenic_n": invalid_pathogenic,
                "invalid_n": invalid_n,
                "invalid_rate": invalid_n / n if n else math.nan,
                "invalid_rate_stderr": agresti_coull_stderr(invalid_n, n),
                "accuracy": (tn + tp) / n if n else math.nan,
                "accuracy_stderr": agresti_coull_stderr(tn + tp, n),
                "benign_recall": benign_recall,
                "benign_recall_stderr": benign_recall_stderr,
                "pathogenic_recall": pathogenic_recall,
                "pathogenic_recall_stderr": pathogenic_recall_stderr,
                "balanced_accuracy": (benign_recall + pathogenic_recall) / 2,
                "balanced_accuracy_stderr": math.sqrt(benign_recall_stderr**2 + pathogenic_recall_stderr**2) / 2,
                "mcc_valid_answers": mcc,
                "mcc_n": mcc_n,
            }

        result = counts(scores)
        groups = defaultdict(list)
        for row in scores:
            groups[row.sample_metadata["subgroup"]].append(row)
        for group in SUBGROUPS:
            result.update({f"{group}_{name}": value for name, value in counts(groups[group]).items()})
        result["mean_subgroup_balanced_accuracy"] = sum(
            result[f"{group}_balanced_accuracy"] for group in MACRO_GROUPS
        ) / len(MACRO_GROUPS)
        result["mean_subgroup_balanced_accuracy_stderr"] = math.sqrt(
            sum(result[f"{group}_balanced_accuracy_stderr"] ** 2 for group in MACRO_GROUPS)
        ) / len(MACRO_GROUPS)
        return result

    return summarize


@scorer(metrics=[vepqa_all_metrics()])
def vepqa_all_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        answer = _extract_final_answer(state.output.completion)
        normalized_answer = (
            answer.removeprefix("<answer>").removesuffix("</answer>").strip().casefold()
            if answer is not None
            else None
        )
        predicted_label = normalized_answer if normalized_answer in {"benign", "pathogenic"} else None
        normalized_target = target.text.strip().removeprefix("<answer>").removesuffix("</answer>").strip().casefold()
        return Score(
            value=CORRECT if predicted_label == normalized_target else INCORRECT,
            answer=answer,
            metadata={"predicted_label": predicted_label},
        )

    return score


@metric
def vepqa_class_metrics() -> Metric:
    """Report overall and per-group class confusion counts and rates."""

    def summarize(scores: list[SampleScore]) -> dict[str, float | int]:
        def counts(rows: list[SampleScore]) -> dict[str, float | int]:
            result: dict[str, float | int] = {"n": len(rows)}
            for label in ("benign", "pathogenic"):
                true_count = sum(row.sample_metadata["label_binary"] == label for row in rows)
                predicted_count = sum(row.score.metadata["predicted_label"] == label for row in rows)
                true_positive = sum(
                    row.sample_metadata["label_binary"] == label and row.score.metadata["predicted_label"] == label
                    for row in rows
                )
                result[f"{label}_true"] = true_count
                result[f"{label}_predicted"] = predicted_count
                result[f"{label}_tp"] = true_positive
                result[f"{label}_fp"] = predicted_count - true_positive
                result[f"{label}_fn"] = true_count - true_positive
                result[f"{label}_recall"] = true_positive / true_count if true_count else 0.0
                result[f"{label}_precision"] = true_positive / predicted_count if predicted_count else 0.0
            invalid_count = sum(row.score.metadata["predicted_label"] is None for row in rows)
            result["invalid_count"] = invalid_count
            result["invalid_rate"] = invalid_count / len(rows) if rows else 0.0
            return result

        result = counts(scores)
        groups = defaultdict(list)
        for row in scores:
            groups[row.sample_metadata["group"]].append(row)
        for group, rows in sorted(groups.items()):
            result[f"{group}_accuracy"] = sum(
                row.sample_metadata["label_binary"] == row.score.metadata["predicted_label"] for row in rows
            ) / len(rows)
            result.update({f"{group}_{name}": value for name, value in counts(rows).items()})
        return result

    return summarize


@scorer(metrics=[accuracy(), stderr(), vepqa_class_metrics()])
def vepqa_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        answer = _extract_final_answer(state.output.completion)
        normalized_answer = (
            answer.removeprefix("<answer>").removesuffix("</answer>").strip().casefold()
            if answer is not None
            else None
        )
        normalized_target = target.text.strip().removeprefix("<answer>").removesuffix("</answer>").strip().casefold()
        predicted_label = normalized_answer if normalized_answer in {"benign", "pathogenic"} else None
        return Score(
            value=CORRECT if normalized_answer == normalized_target else INCORRECT,
            answer=answer,
            metadata={"predicted_label": predicted_label},
        )

    return score


@scorer(metrics=[accuracy(), stderr(), vepqa_class_metrics()])
def vepqa_group_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        answer = _extract_final_answer(state.output.completion)
        normalized_answer = (
            answer.removeprefix("<answer>").removesuffix("</answer>").strip().casefold()
            if answer is not None
            else None
        )
        predicted_label = normalized_answer if normalized_answer in {"benign", "pathogenic"} else None
        normalized_target = target.text.strip().removeprefix("<answer>").removesuffix("</answer>").strip().casefold()
        return Score(
            value=CORRECT if predicted_label == normalized_target else INCORRECT,
            answer=answer,
            metadata={"predicted_label": predicted_label},
        )

    return score


def group_record_to_sample(record: dict[str, Any], config: str, group_key: str, multi: bool) -> Sample:
    subtype_key = group_key.removesuffix("_group") + "_subtype"
    metadata = {
        "tag": "binary",
        "config": config,
        "label_binary": record["label_binary"],
        "group": record[group_key],
        "subtype": record[subtype_key],
        group_key: record[group_key],
        subtype_key: record[subtype_key],
        "source_split": record["source_split"],
    }
    if multi:
        metadata.update({key: record[key] for key in ("source_id", "view", "layout", "wording")})
    return Sample(
        input=record["prompt"] if multi else record["question"] + "\n\n" + record["prompt_suffix"],
        target=record["answer_text"],
        id=record["id"],
        metadata=metadata,
    )


def _vepqa_group_task(config: str, group_key: str, multi: bool) -> Task:
    dataset = hf_dataset(
        path=DATASET_PATH,
        name=config,
        split="test",
        revision=GROUP_DATASET_REVISION,
        sample_fields=lambda record: group_record_to_sample(record, config, group_key, multi),
    )
    return Task(dataset=dataset, solver=generate(), scorer=vepqa_group_scorer())


def _vepqa_task(config: str, suffix: str | None = None) -> Task:
    dataset = hf_dataset(
        path=DATASET_PATH,
        name=config,
        split="test",
        revision=DATASET_REVISION,
        sample_fields=lambda record: record_to_sample(record, config, suffix),
    )
    return Task(
        dataset=dataset,
        solver=generate(),
        scorer=vepqa_scorer(),
    )


@task
def vepqa_single() -> Task:
    return _vepqa_task("binary_v3_random")


@task
def vepqa_single_all() -> Task:
    dataset = hf_dataset(
        path=DATASET_PATH,
        name="binary_v3_random",
        split="train+test",
        revision=ALL_DATASET_REVISION,
        sample_fields=all_record_to_sample,
    )
    return Task(
        dataset=dataset,
        solver=generate(),
        scorer=vepqa_all_scorer(),
        headline_metric=HeadlineMetric(scorer="vepqa_all_scorer", metric="accuracy"),
    )


@task
def vepqa_single_1k() -> Task:
    dataset = hf_dataset(
        path=DATASET_PATH,
        name="binary_single_1k",
        split="test",
        revision=SINGLE_1K_DATASET_REVISION,
        sample_fields=lambda record: all_record_to_sample(record, "binary_single_1k"),
    )
    return Task(
        dataset=dataset,
        solver=generate(),
        scorer=vepqa_all_scorer(),
        headline_metric=HeadlineMetric(scorer="vepqa_all_scorer", metric="accuracy"),
    )


@task
def vepqa_double() -> Task:
    return _vepqa_task("binary_v5_random")


@task
def vepqa_multi() -> Task:
    dataset = hf_dataset(
        path=DATASET_PATH,
        name="binary_multi",
        split="test",
        revision=MULTI_DATASET_REVISION,
        sample_fields=lambda record: Sample(
            input=record["prompt"],
            target=record["answer_text"],
            id=record["id"],
            metadata={
                "tag": "binary",
                "config": "binary_multi",
                "label_binary": record["label_binary"],
                "group": record["region_group"],
                "subtype": record["region_subtype"],
                "region_group": record["region_group"],
                "region_subtype": record["region_subtype"],
                "source_id": record["source_id"],
                "view": record["view"],
                "layout": record["layout"],
                "wording": record["wording"],
            },
        ),
    )
    return Task(dataset=dataset, solver=generate(), scorer=vepqa_scorer())


@task
def vepqa_single_cot() -> Task:
    return _vepqa_task("binary_v3_random", COT_SUFFIX)


@task
def vepqa_double_cot() -> Task:
    return _vepqa_task("binary_v5_random", COT_SUFFIX)


@task
def vepqa_single_consequence() -> Task:
    return _vepqa_group_task("binary_single_consequence", "consequence_group", multi=False)


@task
def vepqa_single_region() -> Task:
    return _vepqa_group_task("binary_single_region", "region_group", multi=False)


@task
def vepqa_multi_consequence() -> Task:
    return _vepqa_group_task("binary_multi_consequence", "consequence_group", multi=True)


@task
def vepqa_multi_region() -> Task:
    return _vepqa_group_task("binary_multi_region", "region_group", multi=True)
