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
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.log import HeadlineMetric
from inspect_ai.scorer import Metric, SampleScore, accuracy, metric, stderr
from inspect_ai.solver import generate

from vepqa import vepqa_class_metrics, vepqa_scorer


# Built from the BioReason test splits of wanglab/variant_effect_coding and wanglab/variant_effect_non_snv.
# bioreason_non_snv keeps every source row. bioreason_coding asks every coding variant one fixed question and keeps
# one row per sequence pair, because the source coding questions name the gene only for pathogenic variants and use
# disjoint templates per label, so their wording gives the label away.
DATASET_PATH = "HuggingFaceBio/DNAIntelligenceIndex"
DATASET_REVISION = "aebffe19836c2fdeaabb73627793bf3d2c089dc8"


def record_to_sample(record: dict[str, Any], group: str) -> Sample:
    return Sample(
        input=record["question"],
        target=record["answer_text"],
        id=str(record["id"]),
        metadata={
            "label_binary": record["answer_text"].removeprefix("<answer>").removesuffix("</answer>"),
            "group": group,
        },
    )


@metric
def balanced_accuracy() -> Metric:
    """Report the mean of benign and pathogenic recall and its standard error, counting invalid answers as misses."""

    def summarize(scores: list[SampleScore]) -> dict[str, float]:
        def agresti_coull_stderr(successes: int, total: int) -> float:
            # Adding two successes and two failures keeps the stderr nonzero for small cells at 0% or 100%.
            adjusted = (successes + 2) / (total + 4)
            return math.sqrt(adjusted * (1 - adjusted) / (total + 4)) if total else math.nan

        recalls = []
        recall_stderrs = []
        for label in ("benign", "pathogenic"):
            total = sum(row.sample_metadata["label_binary"] == label for row in scores)
            correct = sum(
                row.sample_metadata["label_binary"] == label and row.score.metadata["predicted_label"] == label
                for row in scores
            )
            recalls.append(correct / total if total else math.nan)
            recall_stderrs.append(agresti_coull_stderr(correct, total))
        return {
            "balanced_accuracy": sum(recalls) / 2,
            "balanced_accuracy_stderr": math.sqrt(sum(value**2 for value in recall_stderrs)) / 2,
        }

    return summarize


def _class_f1(scores: list[SampleScore], label: str) -> float:
    true_count = sum(row.sample_metadata["label_binary"] == label for row in scores)
    predicted_count = sum(row.score.metadata["predicted_label"] == label for row in scores)
    true_positive = sum(
        row.sample_metadata["label_binary"] == label and row.score.metadata["predicted_label"] == label
        for row in scores
    )
    false_positive = predicted_count - true_positive
    false_negative = true_count - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else 0.0


@metric
def pathogenic_f1() -> Metric:
    """Report pathogenic-class F1 over all scored samples, counting invalid pathogenic answers as false negatives."""

    def summarize(scores: list[SampleScore]) -> float:
        return _class_f1(scores, "pathogenic")

    return summarize


@metric
def macro_f1() -> Metric:
    """Report the unweighted mean of benign and pathogenic F1, counting invalid answers as false negatives."""

    def summarize(scores: list[SampleScore]) -> float:
        return (_class_f1(scores, "benign") + _class_f1(scores, "pathogenic")) / 2

    return summarize


def _bioreason_task(config: str, group: str) -> Task:
    dataset = hf_dataset(
        path=DATASET_PATH,
        name=config,
        split="train",
        revision=DATASET_REVISION,
        sample_fields=lambda record: record_to_sample(record, group),
    )
    return Task(
        dataset=dataset,
        solver=generate(),
        scorer=vepqa_scorer(),
        metrics=[balanced_accuracy(), accuracy(), stderr(), vepqa_class_metrics(), pathogenic_f1(), macro_f1()],
        headline_metric=HeadlineMetric(scorer="vepqa_scorer", metric="balanced_accuracy"),
    )


@task
def bioreason_coding_sequence_only() -> Task:
    return _bioreason_task("bioreason_coding", "coding_sequence_only")


@task
def bioreason_non_snv() -> Task:
    return _bioreason_task("bioreason_non_snv", "variant_effect_non_snv")
