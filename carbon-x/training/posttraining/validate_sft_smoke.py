"""Validate a completed real-model SFT smoke run, including inference reload."""

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir
    state = json.loads((output / "trainer_state.json").read_text())
    assert state["global_step"] == 4, state["global_step"]
    steps = set()
    for entry in state["log_history"]:
        if "loss" in entry:
            assert math.isfinite(entry["loss"]), entry
            assert math.isfinite(entry["grad_norm"]), entry
            steps.add(entry["step"])
    assert {1, 2, 3, 4}.issubset(steps), steps
    # Retention may prune checkpoint 2 after it has been successfully resumed.
    checkpoint = output / "checkpoint-4"
    assert (checkpoint / "trainer_state.json").is_file(), checkpoint
    optimizer_shards = list(checkpoint.glob("global_step*/*optim_states.pt"))
    assert optimizer_shards, checkpoint
    for shard in optimizer_shards:
        # These are this job's own checkpoints. Memory mapping avoids loading
        # tens of GB of moment tensors just to inspect the optimizer counters.
        saved = torch.load(shard, map_location="cpu", mmap=True, weights_only=False)
        optimizer_states = saved["optimizer_state_dict"]["optimizer_state_dict"]["state"]
        assert optimizer_states and all(float(value["step"]) == 4 for value in optimizer_states.values()), shard

    tokenizer = AutoTokenizer.from_pretrained(output, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(output, dtype=torch.bfloat16).to("cuda:0")
    assert model.config.use_cache
    assert model.generation_config.eos_token_id == tokenizer.eos_token_id
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What are the four bases in DNA?"}],
        add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        result = model.generate(**inputs, max_new_tokens=2, do_sample=False)
    assert result.shape[1] > inputs["input_ids"].shape[1]
    print(json.dumps({"status": "passed", "global_step": state["global_step"], "output_dir": str(output)}))


if __name__ == "__main__":
    main()
