# DNA skills verifier

`verify.py` grades replies to the DNA skills (SeqLocate) prompts of `HuggingFaceBio/FineDNA-RL`, the rows whose
`task_group` is `dna_skills`. The `dna_skills_task` reward in `src/carbon_x/rewards.py` calls its `verify` after
removing the `<dna>` tags that Carbon's tokenizer adds around DNA.

In strict mode:

- Length claims after a cue such as "before the edit" are checked against the sequence it names, since an indel
  changes the length. Without a cue before the claim in its clause, any of the task's sequences still matches.
- Sequence cues match whole words, so "unedited" does not select the edited sequence.
