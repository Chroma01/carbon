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

import json
import re
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Scorer, Target, accuracy, grouped, scorer, stderr
from inspect_ai.solver import TaskState, generate

from metrics import equal_weight_group_stderr


DATASET_PATH = "HuggingFaceBio/sequence-recovery-qa"
DATASET_REVISION = "98c477afaeef68be134929337f543980d9606f98"
VARLEN_DATASET_REVISION = "15e4bc594304bcfd97f814671bf66e7d4ee87f8c"
REWRITE_DATASET_REVISION = "fe58119f35ba52b5c1af3c4e4638681640616ef5"
POSITION_DATASET_REVISION = "2e35c20b3a6a70825cdaae97dd8029c4374b694a"
# The last <answer>...</answer> must end the completion; the group cannot span an earlier <answer> tag.
_POSITION_ANSWER_PATTERN = re.compile(r"<answer>((?:(?!<answer>).)*?)</answer>\s*\Z", re.DOTALL)
_DNA_TAG_PATTERN = re.compile(r"</?dna>", re.IGNORECASE)
_DIFF_ANSWER_PATTERN = re.compile(r"(\d+)\W*([ACGT])")


def record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        input=record["prompt"],
        target=record["answer_text"],
    )


def position_record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        input=record["prompt"][0]["content"],
        target=record["answer_text"],
        id=record["id"],
        metadata={
            **json.loads(record["gold_answer_json"]),
            "species_type": record["species_type"],
            "template_type": record["template_type"],
            "type_length": f"{record['task_type']}_{record['length_bin']}",
        },
    )


def _extract_position_answer(completion: str, task_type: str) -> str | None:
    match = _POSITION_ANSWER_PATTERN.search(completion)
    if match is None:
        return None
    answer = "".join(_DNA_TAG_PATTERN.sub("", match.group(1)).split()).upper()
    if task_type == "diff":
        diff = _DIFF_ANSWER_PATTERN.fullmatch(answer)
        return f"{int(diff.group(1))} {diff.group(2)}" if diff else None
    return answer or None


def _position_shift(answer: str, metadata: dict[str, Any]) -> int | None:
    # Signed distance from the reference to the nearest matching span in the stored window: in 6-mer blocks for
    # `block` and `vepqa` (aligned blocks only), in bases for `span12`.
    window, start = metadata["window"], metadata["window_start"] - 1
    if metadata["task_type"] == "span12":
        shifts = [start + i - (metadata["position"] - 1) for i in range(len(window)) if window.startswith(answer, i)]
    else:
        shifts = [
            (start + i) // 6 - metadata["token_index"] for i in range(0, len(window), 6) if window[i : i + 6] == answer
        ]
    return min(shifts, key=abs, default=None)


def _extract_final_sequence(completion: str, expected_length: int | None = None) -> str | None:
    length_pattern = f"{{{expected_length}}}" if expected_length is not None else "+"
    bare_start = r"(?<!\w)" if expected_length is not None else r"(?:\A|\n)[ \t]*"
    pattern = re.compile(
        rf"(?:<dna>\s*(?P<tagged>[ACGT]{length_pattern})\s*</dna>|"
        rf"{bare_start}(?P<bare>[ACGT]{length_pattern}))[\W_]*\Z",
        re.IGNORECASE,
    )
    match = pattern.search(completion)
    return (match.group("tagged") or match.group("bare")).upper() if match else None


@scorer(metrics=[accuracy(), stderr()])
def nucleotide_accuracy_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        reference = target[0].strip().upper()
        answer = _extract_final_sequence(state.output.completion, len(reference))
        matching_nucleotides = (
            sum(candidate == expected for candidate, expected in zip(answer, reference, strict=True)) if answer else 0
        )
        return Score(
            value=matching_nucleotides / len(reference),
            answer=answer,
            metadata={"matching_nucleotides": matching_nucleotides, "total_nucleotides": len(reference)},
        )

    return score


@scorer(metrics={"accuracy_nt": [accuracy(), stderr()], "accuracy_len": [accuracy(), stderr()]})
def variable_length_accuracy_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        reference = target[0].strip().upper()
        answer = _extract_final_sequence(state.output.completion)
        matching_nucleotides = (
            sum(candidate == expected for candidate, expected in zip(answer, reference, strict=False)) if answer else 0
        )
        return Score(
            value={
                "accuracy_nt": matching_nucleotides / len(reference),
                "accuracy_len": int(answer is not None and len(answer) == len(reference)),
            },
            answer=answer,
            metadata={
                "generated_length": len(answer) if answer is not None else 0,
                "reference_length": len(reference),
                "matching_nucleotides": matching_nucleotides,
            },
        )

    return score


@scorer(metrics=[accuracy(), stderr()])
def rewrite_accuracy_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        answer = state.output.completion
        return Score(value=CORRECT if answer.strip() == target[0].strip() else INCORRECT, answer=answer)

    return score


@scorer(metrics=[accuracy(), stderr()])
def position_accuracy_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        task_type = state.metadata["task_type"]
        answer = _extract_position_answer(state.output.completion, task_type)
        shift = (
            _position_shift(answer, state.metadata) if answer and task_type in ("block", "vepqa", "span12") else None
        )
        return Score(value=CORRECT if answer == target.text else INCORRECT, answer=answer, metadata={"shift": shift})

    return score


@task
def seqrecqa_1nt() -> Task:
    dataset = hf_dataset(
        path=DATASET_PATH,
        name="1_nt",
        split="test",
        revision=DATASET_REVISION,
        sample_fields=record_to_sample,
    )
    return Task(
        dataset=dataset,
        solver=generate(),
        scorer=nucleotide_accuracy_scorer(),
    )


@task
def seqrecqa_6nt() -> Task:
    dataset = hf_dataset(
        path=DATASET_PATH,
        name="6_nt",
        split="test",
        revision=DATASET_REVISION,
        sample_fields=record_to_sample,
    )
    return Task(
        dataset=dataset,
        solver=generate(),
        scorer=nucleotide_accuracy_scorer(),
    )


@task
def seqrecqa_varlen() -> Task:
    dataset = hf_dataset(
        path=DATASET_PATH,
        name="varlen",
        split="test",
        revision=VARLEN_DATASET_REVISION,
        sample_fields=record_to_sample,
    )
    return Task(
        dataset=dataset,
        solver=generate(),
        scorer=variable_length_accuracy_scorer(),
    )


@task
def seqrecqa_rewrite() -> Task:
    dataset = hf_dataset(
        path=DATASET_PATH,
        name="rewrite",
        split="test",
        revision=REWRITE_DATASET_REVISION,
        sample_fields=record_to_sample,
    )
    return Task(
        dataset=dataset,
        solver=generate(),
        scorer=rewrite_accuracy_scorer(),
    )


@task
def seqrecqa_position() -> Task:
    dataset = hf_dataset(
        path=DATASET_PATH,
        name="position",
        split="test",
        revision=POSITION_DATASET_REVISION,
        sample_fields=position_record_to_sample,
    )
    return Task(
        dataset=dataset,
        solver=generate(),
        scorer=position_accuracy_scorer(),
        metrics=[
            grouped(accuracy(), "task_type", all="groups", name_template="{group_name}_acc"),
            equal_weight_group_stderr("task_type"),
            grouped(accuracy(), "length_bin", all=False, name_template="len_{group_name}_acc"),
            grouped(accuracy(), "type_length", all=False, name_template="{group_name}_acc"),
        ],
    )
