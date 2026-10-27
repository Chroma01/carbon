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

import re
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState, generate


DATASET_PATH = "HuggingFaceBio/DNAIntelligenceIndex"
DATASET_REVISION = "afbcf9022f581d92ab697ce5c8dfe65359e4ae5f"
# A DNA sequence, optionally in <dna> tags, that ends the completion
_FINAL_SEQUENCE_PATTERN = re.compile(
    r"(?:<dna>\s*(?P<tagged>[ACGT]+)\s*</dna>|(?:\A|\n)[ \t]*(?P<bare>[ACGT]+))[\W_]*\Z", re.IGNORECASE
)


def record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        input=record["prompt"],
        target=record["answer_text"],
    )


def _extract_final_sequence(completion: str) -> str | None:
    match = _FINAL_SEQUENCE_PATTERN.search(completion)
    return (match.group("tagged") or match.group("bare")).upper() if match else None


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


@task
def seqrecqa_varlen() -> Task:
    dataset = hf_dataset(
        path=DATASET_PATH,
        name="nextbase",
        split="train",
        revision=DATASET_REVISION,
        sample_fields=record_to_sample,
    )
    return Task(
        dataset=dataset,
        solver=generate(),
        scorer=variable_length_accuracy_scorer(),
    )
