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

"""
RL training script for Carbon-X, using TRL's asynchronous GRPO trainer.

Usage:

Start a vLLM generation server (a plain `vllm serve`, not `trl vllm-serve`):

VLLM_SERVER_DEV_MODE=1 vllm serve HuggingFaceBio/Carbon-X-SFT \
    --logprobs-mode processed_logprobs \
    --weight-transfer-config '{"backend":"nccl"}' \
    --trust-remote-code

Then launch training:

accelerate launch --config_file training/posttraining/accelerate_configs/fsdp2.yaml training/posttraining/rl.py \
    --config training/posttraining/rl.yaml
"""

import logging
import os
import sys

import datasets
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import TrlParser
from trl.experimental.async_grpo import AsyncGRPOTrainer

from carbon_x.configs import RLConfig, RLScriptArguments
from carbon_x.logging import init_trackio_training
from carbon_x.rewards import get_reward_funcs


logger = logging.getLogger(__name__)


def main(script_args, training_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "trackio" in training_args.report_to:
        init_trackio_training(training_args)

    ################
    # Load datasets
    ################
    dataset = load_dataset(
        script_args.dataset_name, script_args.dataset_config, num_proc=training_args.dataset_num_proc
    )
    train_dataset = dataset[script_args.dataset_train_split]
    if script_args.dataset_prompt_column != "prompt":
        train_dataset = train_dataset.rename_column(script_args.dataset_prompt_column, "prompt")

    ################
    # Model init kwargs
    ################
    model_init_kwargs = training_args.model_init_kwargs or {}
    model_init_kwargs.setdefault("revision", script_args.model_revision)
    training_args.model_init_kwargs = model_init_kwargs

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        revision=script_args.model_revision,
        trust_remote_code=training_args.trust_remote_code,
    )

    ################
    # Training
    ################
    trainer = AsyncGRPOTrainer(
        model=script_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=get_reward_funcs(script_args, training_args.max_completion_length),
        processing_class=tokenizer,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # No final `save_model` or `push_to_hub`: the trainer already saved and pushed the last step as a regular checkpoint,
    # whether training stopped at `num_train_epochs` or `max_steps`. With FSDP2 the extra save also hung every run:
    # ranks other than 0 only queue the state-dict all-gathers, so they return and exit while rank 0 still needs them
    # for the gather


if __name__ == "__main__":
    parser = TrlParser((RLScriptArguments, RLConfig))
    script_args, training_args = parser.parse_args_and_config()
    main(script_args, training_args)
