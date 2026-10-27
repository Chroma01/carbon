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
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample, hf_dataset
from inspect_ai.scorer import accuracy, choice, grouped
from inspect_ai.solver import multiple_choice, system_message

from metrics import equal_weight_group_stderr


DATASET_PATH = "HuggingFaceBio/genomeqa"
DATASET_REVISION = "f504767e8eba9d46b1390a642b207519828d1468"

SYSTEM_PROMPT = "\n".join(
    [
        "You are an expert in Computational Biology, Regulatory Genomics, and Bioinformatics. Your goal is to infer the correct answer by analyzing nucleotide sequences with probabilistic reasoning and biological domain knowledge, not by exploiting superficial text patterns or option positions.",
        "",
        "Questions implicitly belong to one of six domains (you must infer which):",
        "1. Regulatory identity: enhancer vs promoter in human.",
        "2. Splice-site analysis: acceptor vs donor vs both, and distinguishing real sites from dinucleotide-preserved shuffled controls.",
        "3. Taxonomy: classifying sequences as eukaryote, prokaryote, or virus.",
        "4. Epigenetics: histone mark identity and chromatin state (open/accessible vs closed/repressive) in human K562 cells.",
        "5. TFBS prediction: locating binding sites of specific transcription factors within longer human sequences.",
        "6. Motif identification: recognizing short (6-20 bp) consensus patterns.",
        "",
        "Please adhere to the following Reasoning Protocol:",
        "",
        "1. Infer task & question type  ",
        "   From the wording, infer the biological domain and whether the question is: binary verification, multi-class label selection, sequence selection (“which sequence has X?”), odd-one-out, or real-vs-shuffled/background discrimination.",
        "",
        "2. Analyze sequence patterns  ",
        "   Assume all sequences are in functional 5'->3' orientation unless stated otherwise. Do NOT perform reverse-complement analysis unless explicitly asked.  ",
        "   Focus on whatever patterns are informative for the task, including but not limited to:",
        "   - Short motifs and motif-like patterns, their approximate positions and local context.",
        "   - Base composition, GC content, k-mer and codon-like usage, periodicity, and sequence complexity.",
        "   - Differences between options (for selection / odd-one-out), such as one sequence having a stronger or more coherent pattern than the others.  ",
        "   Treat any examples you know (e.g., canonical splice motifs, promoter features, TF consensus-like sites, species-specific composition) as *hints*, not as a closed list: you are encouraged to use any additional recurrent or statistically distinctive patterns you detect in the sequences, even if they are not classic textbook motifs.",
        "",
        "3. Make a probabilistic decision  ",
        "   Do NOT require perfect consensus matches; biological signals are degenerate and noisy.  ",
        "   - Interpret strict verbs (“is this X?”, “confirm”, “verify”) as asking which label is *more likely* given the sequence.  ",
        "   - For real vs dinucleotide-preserved shuffled/background comparisons, remember that low-level composition is similar; rely more on motif organization, local structure, and plausibly functional subpatterns.  ",
        "   - If multiple features are present, answering “Yes” to a queried property is valid as long as that property is clearly supported; treat exclusivity strictly only when words like “only” or “exclusively” appear.  ",
        "   - Do not hedge in your analysis (e.g., “it depends”, “unknown”, “cannot determine”); always choose the single most biologically plausible option.",
        "",
        "Output rules:",
        "",
        "- Base your decision strictly on nucleotide patterns and biological meaning of the labels, not on dataset biases or text-only artifacts.",
        "- Do NOT copy or restate any full DNA sequence in your analysis; refer only to features (motifs, composition, relative differences, etc.).",
        "- Binary questions typically use “Yes” and “No” as option texts; you must still answer with the corresponding option letter (e.g., A or B), not the words themselves.",
        "- For all questions, there is exactly one best answer among the options. Always pick a single option letter.",
        "- On the final line, replace $LETTER below with one capital option letter. Write exactly `ANSWER: X`, where X is your chosen letter.",
        "- Never repeat the option text after the letter. For example, `ANSWER: B` is valid; `ANSWER: B) No` and `ANSWER: B - No` are invalid.",
        "",
        "Output format (strictly follow):",
        "",
        "### Analysis",
        "1-3 concise sentences explaining (i) the inferred task/domain and (ii) the key sequence-based or comparative reasoning that makes your chosen option most probable.",
        "",
        "### Answer",
        "ANSWER: $LETTER",
    ]
)

QUESTION_TEMPLATE = "Question:\n{question}\n\nOptions:\n{choices}\n\nPlease select the correct option."


def record_to_sample(record: dict[str, Any], organism_group: str | None = None) -> Sample:
    metadata = {**json.loads(record["metadata"]), "subtask": record["subtask"]}
    if organism_group is not None:
        metadata["organism_group"] = organism_group
    return Sample(
        input=record["question"],
        choices=record["options"],
        target="ABCD"[record["answer_index"]],
        id=f"{record['subtask']}-{record['id']}",
        metadata=metadata,
    )


def _genomeqa_task(config: str, group_by_organism: bool = False) -> Task:
    if group_by_organism:
        eukaryote_dataset = hf_dataset(
            path=DATASET_PATH,
            name=f"{config}_eukaryote",
            split="train",
            revision=DATASET_REVISION,
            sample_fields=lambda record: record_to_sample(record, "eukaryote"),
        )
        prokaryote_dataset = hf_dataset(
            path=DATASET_PATH,
            name=f"{config}_prokaryote",
            split="train",
            revision=DATASET_REVISION,
            sample_fields=lambda record: record_to_sample(record, "prokaryote"),
        )
        samples = []
        for index in range(max(len(eukaryote_dataset), len(prokaryote_dataset))):
            if index < len(eukaryote_dataset):
                samples.append(eukaryote_dataset[index])
            if index < len(prokaryote_dataset):
                samples.append(prokaryote_dataset[index])
        dataset = MemoryDataset(samples)
        group_key = "organism_group"
    else:
        dataset = hf_dataset(
            path=DATASET_PATH,
            name=config,
            split="train",
            revision=DATASET_REVISION,
            sample_fields=record_to_sample,
        )
        group_key = "subtask"
    if config in {"mcq_carbon", "mcq_carbon_600"}:
        solver = multiple_choice(cot=True)
    else:
        solver = [system_message(SYSTEM_PROMPT), multiple_choice(template=QUESTION_TEMPLATE)]
    return Task(
        dataset=dataset,
        solver=solver,
        scorer=choice(),
        metrics=[
            grouped(accuracy(), group_key, all="groups", name_template="{group_name}_acc"),
            equal_weight_group_stderr(group_key),
        ],
    )


@task
def genomeqa_binary() -> Task:
    return _genomeqa_task("binary")


@task
def genomeqa_binary_600() -> Task:
    return _genomeqa_task("binary_600")


@task
def genomeqa_binary_organism() -> Task:
    return _genomeqa_task("binary", group_by_organism=True)


@task
def genomeqa_mcq() -> Task:
    return _genomeqa_task("mcq")


@task
def genomeqa_mcq_carbon() -> Task:
    return _genomeqa_task("mcq_carbon")


@task
def genomeqa_mcq_carbon_600() -> Task:
    return _genomeqa_task("mcq_carbon_600")


@task
def genomeqa_mcq_eukaryote() -> Task:
    return _genomeqa_task("eukaryote_cls")


@task
def genomeqa_mcq_600() -> Task:
    return _genomeqa_task("mcq_600")


@task
def genomeqa_mcq_organism() -> Task:
    return _genomeqa_task("mcq", group_by_organism=True)
