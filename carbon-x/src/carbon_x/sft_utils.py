"""Dataset and checkpoint selection for the Carbon-X SFT entrypoint."""

from pathlib import Path

from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint


def load_sft_datasets(script_args, training_args):
    """Load only requested splits, including selectors such as ``train[:32]``."""
    kwargs = {
        "path": script_args.dataset_name,
        "name": script_args.dataset_config,
        "streaming": script_args.dataset_streaming,
    }
    train_dataset = load_dataset(**kwargs, split=script_args.dataset_train_split)
    eval_dataset = None
    if training_args.eval_strategy != "no" or training_args.do_eval:
        eval_dataset = load_dataset(**kwargs, split=script_args.dataset_test_split)
    return train_dataset, eval_dataset


def resolve_checkpoint(training_args):
    """Explicit resume choices take precedence over automatic checkpoint discovery."""
    if training_args.resume_from_checkpoint is not None:
        return training_args.resume_from_checkpoint
    if getattr(training_args, "overwrite_output_dir", False):
        return None
    if Path(training_args.output_dir).is_dir():
        return get_last_checkpoint(training_args.output_dir)
    return None
