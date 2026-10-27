# DNA skills verifier

`verify.py` grades replies to [`HuggingFaceBio/carbon-dna-skills`](https://huggingface.co/datasets/HuggingFaceBio/carbon-dna-skills)
(SeqLocate) prompts. It started as a copy of the dataset's `scripts/verify.py` at revision
`51f94d6a0d49539adb452571e3254055d40b4923` (verifier v0.4) and is maintained here; the Hub copy is not kept in sync.
The `dna_skills_task` reward in `src/carbon_x/rewards.py` calls
its `verify` after removing the `<dna>` tags that Carbon's tokenizer adds around DNA.

Changes since that revision, all in strict mode:

- Length claims after a cue such as "before the edit" are checked against the sequence it names, since an indel
  changes the length. Without a cue before the claim in its clause, any of the task's sequences still matches.
- Sequence cues match whole words, so "unedited" no longer selects the edited sequence.

The dataset's builder (`scripts/build_locate.py`) and pass-rate script (`scripts/passrate.py`) still use the Hub copy.
After changing this module, run the dataset's `scripts/test_verify.py` against it on the `train` and `test` splits
exported as JSONL: every reference reply must score 1.0 and every mutation 0.0.

The module is formatted with Ruff like the rest of the repository, so it no longer diffs cleanly against the Hub copy.
