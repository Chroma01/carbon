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
Supervised fine-tuning script for Carbon-X.

Usage:

accelerate launch --config_file training/posttraining/accelerate_configs/deepspeed_zero3.yaml training/posttraining/sft.py \
    --config training/posttraining/sft.yaml
"""

import logging
import sys
from pathlib import Path

import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import ModelConfig, ScriptArguments, SFTTrainer, TrlParser, get_peft_config

from carbon_x.callbacks import LogTrainingSamplesCallback
from carbon_x.configs import SFTConfig
from carbon_x.logging import init_trackio_training
from carbon_x.model_utils import get_model, get_tokenizer


logger = logging.getLogger(__name__)


def main(script_args, training_args, model_args):
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

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    checkpoint = training_args.resume_from_checkpoint
    if (
        checkpoint is None
        and not getattr(training_args, "overwrite_output_dir", False)
        and Path(training_args.output_dir).is_dir()
    ):
        checkpoint = get_last_checkpoint(training_args.output_dir)
    if checkpoint:
        logger.info("Resuming training from %s", checkpoint)

    if "trackio" in training_args.report_to:
        init_trackio_training(training_args)

    ################
    # Load datasets
    ################
    dataset_kwargs = {
        "path": script_args.dataset_name,
        "name": script_args.dataset_config,
        "streaming": script_args.dataset_streaming,
    }
    train_dataset = datasets.load_dataset(**dataset_kwargs, split=script_args.dataset_train_split)
    eval_dataset = None
    if training_args.eval_strategy != "no" or training_args.do_eval:
        eval_dataset = datasets.load_dataset(**dataset_kwargs, split=script_args.dataset_test_split)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)

    ###################
    # Load model
    ###################
    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)

    ############################
    # Initialize the SFT Trainer
    ############################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # Add LogTrainingSamplesCallback to log training data samples to Trackio
    if "trackio" in training_args.report_to:
        log_samples_callback = LogTrainingSamplesCallback(trainer=trainer, n_samples=50, decode_columns=["input_ids"])
        trainer.add_callback(log_samples_callback)

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    if not script_args.dataset_streaming:
        metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if training_args.do_eval:
        metrics = trainer.evaluate()
        if not script_args.dataset_streaming:
            metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    # Align the model's generation config with the tokenizer's eos token
    # to avoid unbounded generation in the transformers `pipeline()` function
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.model.generation_config.pad_token_id = tokenizer.pad_token_id
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["carbon-x"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        tokenizer.save_pretrained(training_args.output_dir)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
