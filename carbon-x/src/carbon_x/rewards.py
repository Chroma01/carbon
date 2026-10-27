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

"""Reward functions for RL training."""

import functools
import re
from collections.abc import Callable

from .dna_skills.verify import verify as verify_dna_skills
from .vepqa_claims import verify as verify_vepqa_claims


def vepqa_accuracy_reward(completions, answer_text, **kwargs) -> list[float]:
    """Score VEPQA completions against the benign/pathogenic answer, requiring reasoning before the tag.

    Uses the same exact-match contract as the Inspect benchmark, but a completion that is only the terminal
    `<answer>...</answer>` tag scores 0.0 even when correct, so the policy cannot collapse onto a bare answer.
    """
    pattern = re.compile(r"(<answer>[^<]*</answer>)\s*\Z", re.DOTALL)
    rewards = []
    for completion, target in zip(completions, answer_text, strict=True):
        content = completion[0]["content"]
        match = pattern.search(content)
        if match is None or not content[: match.start()].strip():
            rewards.append(0.0)
            continue
        answer = match.group(1).removeprefix("<answer>").removesuffix("</answer>").strip().casefold()
        expected = target.removeprefix("<answer>").removesuffix("</answer>").strip().casefold()
        rewards.append(float(answer == expected))
    return rewards


def vepqa_groundedness_reward(completions, prompts, **kwargs) -> list[float]:
    """Score 1.0 when every statement of a templated VEPQA rationale is grounded in its prompt, else 0.0.

    Uses [`~carbon_x.vepqa_claims.verify`] on the user turn of each conversational prompt: the substitution, its
    type and the 6-base block are required, stated optional facts must be true, and wording outside the template's
    grammar fails closed. A likelihood statement is optional, and its level is checked only for consistency with the
    other likelihood statements and concessions, never against the data: the whole-window likelihood depends on the
    alternate sequence, which a single-view prompt never shows. It does not check the final answer.
    """
    rewards = []
    for completion, prompt in zip(completions, prompts, strict=True):
        user_text = "\n\n".join(message["content"] for message in prompt if message["role"] == "user")
        rewards.append(float(verify_vepqa_claims(user_text, completion[0]["content"])["grounded"]))
    return rewards


def vepqa_groundedness_penalty_reward(completions, prompts, **kwargs) -> list[float]:
    """Score -0.25 when a templated VEPQA rationale is not grounded in its prompt, else 0.0.

    Groundedness is [`vepqa_groundedness_reward`]. Listed with [`vepqa_accuracy_reward`], whose score async GRPO adds
    to this one, a correct grounded answer scores 1.0, a correct ungrounded one 0.75, a wrong grounded one 0.0 and a
    wrong ungrounded one -0.25. Unlike a product of the two, the penalty never zeroes a correct answer, so accuracy
    keeps its learning signal while ungrounded claims still cost reward.
    """
    return [0.0 if grounded else -0.25 for grounded in vepqa_groundedness_reward(completions, prompts)]


# Mirrors how the SeqRecQA position benchmark extracts answers: the last <answer>...</answer> must end the
# completion, its <dna> tags and whitespace are dropped, and `diff` answers are normalised to "POSITION BASE".
SEQRECQA_POSITION_ANSWER_PATTERN = re.compile(r"<answer>((?:(?!<answer>).)*?)</answer>\s*\Z", re.DOTALL)
SEQRECQA_POSITION_DNA_TAG_PATTERN = re.compile(r"</?dna>", re.IGNORECASE)
SEQRECQA_POSITION_DIFF_PATTERN = re.compile(r"(\d+)\W*([ACGT])")


def _seqrecqa_position_answer(content: str, task_type: str) -> str | None:
    match = SEQRECQA_POSITION_ANSWER_PATTERN.search(content)
    if match is None:
        return None
    answer = "".join(SEQRECQA_POSITION_DNA_TAG_PATTERN.sub("", match.group(1)).split()).upper()
    if task_type == "diff":
        diff = SEQRECQA_POSITION_DIFF_PATTERN.fullmatch(answer)
        return f"{int(diff.group(1))} {diff.group(2)}" if diff else None
    return answer or None


def seqrecqa_position_reward(completions, answer_text, task_type, **kwargs) -> list[float]:
    """Exact match of the final `<answer>` against the reference, as the SeqRecQA position benchmark scores it.

    The benchmark accepts reasoning before the answer but does not require it, and so does this reward. A completion
    with no extractable answer scores 0.0.
    """
    return [
        float(_seqrecqa_position_answer(completion[0]["content"], kind) == target)
        for completion, target, kind in zip(completions, answer_text, task_type, strict=True)
    ]


# Basic-DNA responses end on an `Answer: <gold>` line, as the SFT data does.
BASIC_DNA_ANSWER_PATTERN = re.compile(r"(?im)^[ \t]*answer[ \t]*:[ \t]*(.+?)\s*\Z")
BASIC_DNA_SEQUENCE_TASKS = {
    "aligned_complement",
    "bisulfite_conversion",
    "mirna_seed_extraction",
    "protospacer_extraction",
    "reverse_complement",
    "transcription_coding",
    "transcription_template",
    "trna_anticodon_3to5",
}


def _basic_dna_key(answer: str, task_type: str) -> str:
    # Carbon's tokenizer wraps DNA runs in <dna> tags, which the reference answers do not have
    answer = re.sub(r"</?dna>", "", answer, flags=re.IGNORECASE).strip().rstrip(".").strip()
    if task_type in BASIC_DNA_SEQUENCE_TASKS:
        answer = re.sub(r"\s|-|5'|3'", "", answer).upper()
        # The reference mixes RNA and DNA spellings of the anticodon; T and U stay distinct for transcription
        return answer.replace("U", "T") if task_type == "trna_anticodon_3to5" else answer
    answer = re.sub(r"\s+", " ", answer).casefold()
    if task_type == "restriction_enzyme_identity":
        return ",".join(sorted(name.strip() for name in re.split(r",|\band\b", answer) if name.strip()))
    if task_type == "nr_spacing":
        # The question asks for the receptor pair, which follows from the direct-repeat spacing
        pair, spacing = re.search(r"\w+/rxr", answer), re.findall(r"\bdr\s*(\d+)", answer)
        return f"{pair.group() if pair else ''}@dr{spacing[-1] if spacing else ''}"
    if task_type == "mutation_effect":
        # The rest of the sentence restates the consensus, whose IUPAC N the DNA tokenizer cannot write
        if re.match(r"binding (?:likely )?retained", answer):
            return "retained"
        position = re.match(r"reduced\b.*?\bposition (\d+)", answer)
        return f"reduced@{position.group(1)}" if position else answer
    return re.sub(r"(?<=\d)\s*(?:°?\s*c|amino acids?)$", "", answer)


def basic_dna_accuracy_reward(completions, answer_text, task_type, **kwargs) -> list[float]:
    """Score basic-DNA completions whose final `Answer:` line matches the deterministic reference answer.

    Answers are compared after stripping `<dna>` tags and casing, as exact sequences for sequence tasks, as enzyme sets
    for `restriction_enzyme_identity`, and within 0.05 when both sides are numbers (Tm, GC%). `nr_spacing` compares the
    receptor pair and spacing, and `mutation_effect` compares retained, or reduced at a position. As in
    [`vepqa_accuracy_reward`], a completion that is only the answer line scores 0.0.
    """
    rewards = []
    for completion, target, task in zip(completions, answer_text, task_type, strict=True):
        content = completion[0]["content"]
        match = BASIC_DNA_ANSWER_PATTERN.search(content)
        if match is None or not content[: match.start()].strip():
            rewards.append(0.0)
            continue
        answer, expected = _basic_dna_key(match.group(1), task), _basic_dna_key(target, task)
        try:
            correct = abs(float(answer.removesuffix("%")) - float(expected.removesuffix("%"))) <= 0.05
        except ValueError:
            correct = answer == expected
        rewards.append(float(correct))
    return rewards


# The biology_mcq prompt asks for a final `ANSWER: LETTER` line; Inspect's `multiple_choice` reads the last such line.
MCQ_ANSWER_PATTERN = re.compile(r"(?im)^answer[ \t]*:[ \t]*([a-z])[ \t]*\.?\s*\Z")


def mcq_accuracy_reward(completions, answer_text, **kwargs) -> list[float]:
    """Score multiple-choice completions whose final `ANSWER: <letter>` line names the gold option letter.

    Accepts only answers that the `biology_mcq` benchmark's parser reads as the same letter: one letter in either case,
    with an optional trailing period, on the last line. Wrappers the benchmark strips (`**B**`, `$B$`, `(B)`) and text
    after the answer line score 0.0, so the policy learns the requested format. As in [`vepqa_accuracy_reward`], a
    completion that is only the answer line scores 0.0.
    """
    rewards = []
    for completion, target in zip(completions, answer_text, strict=True):
        content = completion[0]["content"]
        match = MCQ_ANSWER_PATTERN.search(content)
        if match is None or not content[: match.start()].strip():
            rewards.append(0.0)
            continue
        rewards.append(float(match.group(1).upper() == target.strip().upper()))
    return rewards


def if_accuracy_reward(completions, verifier_metadata, **kwargs) -> list[float]:
    """Score instruction-following completions 1.0 when they satisfy every constraint of their prompt, else 0.0.

    Uses the checkers of NeMo Gym's instruction-following verifier, whose inputs the `verifier_metadata` column of
    `HuggingFaceBio/Nemotron-RL-IF` stores: each constraint of `instruction_id_list` is built from its `kwargs` with
    null values dropped, and a constraint whose checker raises counts as unmet. Every `count:count_increment_word`
    constraint does, because the dataset stores its keywords as lists. As in the `ifeval` benchmark's strict scoring,
    which uses the same checkers for the IFEval constraint types, an empty completion meets no constraint.
    `count:count_unique` is checked on words only, as in open-instruct's fixed checker, because NeMo's counts repeated
    punctuation as repeated words.
    """
    import nltk
    from instruction_following_eval.evaluation import ensure_nltk_resource
    from verifiable_instructions.instructions_registry import INSTRUCTION_DICT

    # The sentence-counting checkers load NLTK's Punkt tokenizer
    ensure_nltk_resource()
    rewards = []
    for completion, metadata in zip(completions, verifier_metadata, strict=True):
        content = completion[0]["content"]
        followed = []
        for instruction_id, instruction_kwargs in zip(
            metadata["instruction_id_list"], metadata["kwargs"], strict=True
        ):
            instruction = INSTRUCTION_DICT[instruction_id](instruction_id)
            try:
                instruction.build_description(**{k: v for k, v in (instruction_kwargs or {}).items() if v is not None})
                if instruction_id == "count:count_unique":
                    # NeMo's checker fails any answer that repeats "." or ",", so only one-sentence answers can pass.
                    # Drop pure-punctuation tokens first: https://github.com/allenai/open-instruct/pull/1718
                    words = [token for token in nltk.word_tokenize(content) if any(char.isalnum() for char in token)]
                    passed = len(words) == len(set(words))
                else:
                    passed = instruction.check_following(content)
                followed.append(bool(content.strip()) and passed)
            except Exception:
                followed.append(False)
        rewards.append(float(all(followed)))
    return rewards


def dna_skills_reward(completions, gold_answer_json, **kwargs) -> list[float]:
    """Score `HuggingFaceBio/carbon-dna-skills` (SeqLocate) completions with the strict DNA-skills verifier.

    Uses [`~carbon_x.dna_skills.verify.verify`] on `gold_answer_json`: the single final answer, an `<answer>` block
    or else an `Answer:` line, must meet the task's grammar and gold, and every claim the reasoning writes in the SFT
    targets' shapes (intervals, block ordinals, codons, counts, lengths, ...) must be true. Base lookups also need a
    true interval claim covering the queried position, so a bare guess scores 0.0. `<dna>` tags are removed first,
    since the verifier reads plain sequences.
    """
    return [
        verify_dna_skills(re.sub(r"</?dna>", "", completion[0]["content"], flags=re.IGNORECASE), gold)
        for completion, gold in zip(completions, gold_answer_json, strict=True)
    ]


def _scoped(name, inner, task_group, completions, columns) -> list[float | None]:
    """Apply `inner` only to the rows whose `task_group` is `name`, leaving the rest `None`.

    The mixed-task dataset shares one `answer_text` column across five tasks, so an unscoped reward would score another
    task's rows instead of abstaining, and those values would be summed into the reward. Returning `None` excludes the
    row from that reward, which is the trainer's documented multi-task contract.
    """
    rewards: list[float | None] = [None] * len(completions)
    indices = [i for i, group in enumerate(task_group) if group == name]
    if indices:
        scored = inner([completions[i] for i in indices], **{k: [v[i] for i in indices] for k, v in columns.items()})
        for index, reward in zip(indices, scored, strict=True):
            rewards[index] = reward
    return rewards


def vepqa_task_reward(completions, task_group, answer_text, **kwargs) -> list[float | None]:
    """VEPQA rows of a mixed-task run, scored by [`vepqa_accuracy_reward`]."""
    return _scoped("vepqa", vepqa_accuracy_reward, task_group, completions, {"answer_text": answer_text})


def vepqa_groundedness_penalty_task_reward(completions, task_group, prompts, **kwargs) -> list[float | None]:
    """VEPQA rows of a mixed-task run, penalised by [`vepqa_groundedness_penalty_reward`].

    Scoped because the claims verifier only parses VEPQA prompts: it would raise on a BioReason prompt, which shows
    reference and variant sequences without the position and allele sentence.
    """
    return _scoped("vepqa", vepqa_groundedness_penalty_reward, task_group, completions, {"prompts": prompts})


def bioreason_task_reward(completions, task_group, answer_text, **kwargs) -> list[float | None]:
    """BioReason rows of a mixed-task run, scored by [`vepqa_accuracy_reward`]."""
    return _scoped("bioreason", vepqa_accuracy_reward, task_group, completions, {"answer_text": answer_text})


def basic_dna_task_reward(completions, task_group, answer_text, task_type, **kwargs) -> list[float | None]:
    """Basic-DNA rows of a mixed-task run. Scoped because `task_type` is null for every other task."""
    return _scoped(
        "basic_dna",
        basic_dna_accuracy_reward,
        task_group,
        completions,
        {"answer_text": answer_text, "task_type": task_type},
    )


def bio_mcq_task_reward(completions, task_group, answer_text, **kwargs) -> list[float | None]:
    """Biology MCQ rows of a mixed-task run, scored by [`mcq_accuracy_reward`]."""
    return _scoped("bio_mcq", mcq_accuracy_reward, task_group, completions, {"answer_text": answer_text})


def genomeqa_task_reward(completions, task_group, answer_text, **kwargs) -> list[float | None]:
    """GenomeQA MCQ rows of a mixed-task run, scored by [`mcq_accuracy_reward`]."""
    return _scoped("genomeqa", mcq_accuracy_reward, task_group, completions, {"answer_text": answer_text})


def if_task_reward(
    completions, task_group, verifier_metadata, completion_ids, max_completion_length, **kwargs
) -> list[float | None]:
    """Instruction-following rows of a mixed-task run. Scoped because `verifier_metadata` is null for every other task.

    A completion that reaches `max_completion_length` tokens was cut off before it ended, so it scores 0.0 even when
    the cut-off text meets every constraint.
    """
    rewards = _scoped("if", if_accuracy_reward, task_group, completions, {"verifier_metadata": verifier_metadata})
    return [
        0.0 if reward is not None and len(ids) >= max_completion_length else reward
        for reward, ids in zip(rewards, completion_ids, strict=True)
    ]


def dna_skills_task_reward(completions, task_group, gold_answer_json, **kwargs) -> list[float | None]:
    """DNA-skills rows of a mixed-task run. Scoped because `gold_answer_json` is null for every other task."""
    return _scoped("dna_skills", dna_skills_reward, task_group, completions, {"gold_answer_json": gold_answer_json})


def seqrecqa_position_task_reward(completions, task_group, answer_text, task_type, **kwargs) -> list[float | None]:
    """SeqRecQA position rows of a mixed-task run, scored on exact match of the final answer."""
    return _scoped(
        "seqrecqa_position",
        seqrecqa_position_reward,
        task_group,
        completions,
        {"answer_text": answer_text, "task_type": task_type},
    )


def get_reward_funcs(script_args, max_completion_length) -> list[Callable]:
    REWARD_FUNCS_REGISTRY = {
        "vepqa_task": vepqa_task_reward,
        "vepqa_groundedness_penalty_task": vepqa_groundedness_penalty_task_reward,
        "bioreason_task": bioreason_task_reward,
        "basic_dna_task": basic_dna_task_reward,
        "bio_mcq_task": bio_mcq_task_reward,
        "genomeqa_task": genomeqa_task_reward,
        "if_task": functools.partial(if_task_reward, max_completion_length=max_completion_length),
        "dna_skills_task": dna_skills_task_reward,
        "seqrecqa_position_task": seqrecqa_position_task_reward,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    return reward_funcs
