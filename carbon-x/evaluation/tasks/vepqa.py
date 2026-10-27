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


DATASET_PATH = "HuggingFaceBio/DNAIntelligenceIndex"
DATASET_REVISION = "aebffe19836c2fdeaabb73627793bf3d2c089dc8"
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
_FINAL_ANSWER_PATTERN = re.compile(r"(<answer>[^<]*</answer>)\s*\Z", re.DOTALL)


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


def record_to_sample(record: dict[str, Any]) -> Sample:
    _, subgroup = classify_consequence_terms(record["consequence_terms"], record["id"])
    return Sample(
        input=record["prompt"],
        target=record["answer_text"],
        id=record["id"],
        metadata={
            "label_binary": record["answer_text"].removeprefix("<answer>").removesuffix("</answer>"),
            "subgroup": subgroup,
        },
    )


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


@task
def vepqa_single_1k() -> Task:
    dataset = hf_dataset(
        path=DATASET_PATH,
        name="frontiervep",
        split="train",
        revision=DATASET_REVISION,
        sample_fields=record_to_sample,
    )
    return Task(
        dataset=dataset,
        solver=generate(),
        scorer=vepqa_all_scorer(),
        headline_metric=HeadlineMetric(scorer="vepqa_all_scorer", metric="accuracy"),
    )
