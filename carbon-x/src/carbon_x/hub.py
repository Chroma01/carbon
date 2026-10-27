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

import logging
import shutil

from accelerate.utils import gather_object
from transformers import TrainingArguments


logger = logging.getLogger(__name__)


def cleanup_output_dir(training_args: TrainingArguments, push_succeeded: bool = True):
    """Deletes the local training output directory if all processes completed a successful Hub push."""
    if training_args.hub_model_id is None:
        return

    if not all(gather_object([push_succeeded])):
        return

    training_args.distributed_state.wait_for_everyone()
    if training_args.should_save:
        try:
            shutil.rmtree(training_args.output_dir)
        except OSError:
            logger.warning(
                "Failed to delete local training output directory %s", training_args.output_dir, exc_info=True
            )
