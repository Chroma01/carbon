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


from dataclasses import dataclass, field

import trl
from trl.experimental.async_grpo import AsyncGRPOConfig


@dataclass
class ConfigMixin:
    """Mixin class containing shared configuration fields for training configs."""

    chat_template: str | None = field(default=None, metadata={"help": "The chat template to use."})
    project: str | None = field(
        default=None,
        metadata={"help": "The Trackio project to log to."},
    )
    trackio_space_id: str | None = field(
        default=None,
        metadata={"help": "The Trackio space ID to log to."},
    )


@dataclass
class SFTConfig(ConfigMixin, trl.SFTConfig):
    """
    SFT config that adds chat template and Trackio fields on top of TRL's SFT config.
    """

    pass


@dataclass
class RLScriptArguments(trl.ScriptArguments):
    """
    Script arguments for the RL training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions, by their names in `carbon_x.rewards.get_reward_funcs`.
        dataset_prompt_column (`str`):
            Column to use as prompts for training.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "List of reward functions, by their names in `carbon_x.rewards.get_reward_funcs`."},
    )
    dataset_prompt_column: str = field(
        default="prompt",
        metadata={"help": "Column to use as prompts for training."},
    )


@dataclass
class RLModelConfig(trl.ModelConfig):
    """Model arguments for RL; AsyncGRPOConfig owns dtype."""

    dtype: str | None = field(default=None, init=False)


@dataclass
class RLConfig(ConfigMixin, AsyncGRPOConfig):
    """
    RL config that adds chat template and Trackio fields on top of the experimental async GRPO config.
    """

    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "Number of processes to use for loading and preparing the dataset."},
    )
