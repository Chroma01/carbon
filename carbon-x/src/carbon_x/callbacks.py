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

import itertools
import logging

import pandas as pd
from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


logger = logging.getLogger(__name__)


class LogTrainingSamplesCallback(TrainerCallback):
    r"""
    A [`~transformers.TrainerCallback`] that logs formatted training samples.

    This callback logs the first N processed training samples to a Trackio Table at the beginning of training.

    Usage:

    ```python
    trainer = SFTTrainer(...)
    log_samples_callback = LogTrainingSamplesCallback(trainer=trainer, n_samples=10, decode_columns=["input_ids"])
    trainer.add_callback(log_samples_callback)
    ```

    Args:
        trainer (`Trainer`):
            Trainer to which the callback will be attached. The trainer's training dataset will be used for logging
            samples.
        n_samples (`int`, *optional*, defaults to 10):
            The number of training samples to log.
        decode_columns (`list[str]`, *optional*, defaults to `["input_ids"]`):
            List of column names containing token IDs to decode and display in the table. Each decoded column will be
            prefixed with "decoded_" in the output table.
    """

    def __init__(
        self,
        trainer,
        n_samples: int = 10,
        decode_columns: list[str] | None = None,
    ):
        self.trainer = trainer
        self.n_samples = n_samples
        self.decode_columns = decode_columns or ["input_ids"]

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,  # noqa: ARG002
        **kwargs,
    ):
        if state.is_world_process_zero and "trackio" in (args.report_to or []):
            self._log_samples_to_trackio(**kwargs)

    def _log_samples_to_trackio(self, **kwargs):
        """Extract samples from trainer.train_dataset and log to a Trackio Table"""

        import trackio

        if self.trainer.train_dataset is None:
            logger.warning("No training dataset available, skipping training samples logging")
            return

        # Collect sample data
        samples_data = []
        tokenizer = kwargs.get("processing_class")

        try:
            for i, sample in enumerate(itertools.islice(self.trainer.train_dataset, self.n_samples)):
                sample_id = i + 1
                row_data: list[int | str] = [sample_id]
                columns = ["sample_id"]

                # Process each decode column
                for column_name in self.decode_columns:
                    token_ids = sample.get(column_name, [])

                    # Decode text if possible
                    decoded_text = "[Unable to decode]"
                    if tokenizer is not None and token_ids:
                        try:
                            decoded_text = tokenizer.decode(token_ids, skip_special_tokens=False)
                        except Exception as e:
                            decoded_text = f"[Decode error: {str(e)}]"

                    # Add decoded text column with "decoded_" prefix
                    row_data.append(decoded_text)
                    columns.append(f"decoded_{column_name}")

                samples_data.append(row_data)

            # Create DataFrame and log to Trackio
            df = pd.DataFrame(samples_data, columns=columns)

            trackio.log({"training_samples": trackio.Table(dataframe=df)})
            logger.info(f"Logged {len(samples_data)} training samples")

        except Exception as e:
            logger.warning(f"Failed to log training samples: {str(e)}")
