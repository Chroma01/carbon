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

from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import accuracy, choice, grouped
from inspect_ai.solver import multiple_choice

from metrics import equal_weight_group_stderr


DATASET_PATH = "HuggingFaceBio/DNAIntelligenceIndex"
DATASET_REVISION = "872f12bf811f1945949e628c89e06821f804c9ae"


def record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        input=record["question"],
        choices=record["options"],
        target="ABCD"[record["answer_index"]],
        id=f"{record['subtask']}-{record['id']}",
        metadata={"subtask": record["subtask"]},
    )


@task
def genomeqa_mcq_carbon() -> Task:
    dataset = hf_dataset(
        path=DATASET_PATH,
        name="genomeqa",
        split="train",
        revision=DATASET_REVISION,
        sample_fields=record_to_sample,
    )
    return Task(
        dataset=dataset,
        solver=multiple_choice(cot=True),
        scorer=choice(),
        metrics=[
            grouped(accuracy(), "subtask", all="groups", name_template="{group_name}_acc"),
            equal_weight_group_stderr("subtask"),
        ],
    )
