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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from trl import ModelConfig, get_quantization_config

from .configs import SFTConfig
from .tokenizer import HybridDNATokenizer


def get_tokenizer(model_args: ModelConfig, training_args: SFTConfig) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=training_args.trust_remote_code,
    )

    if type(tokenizer).__name__ == "HybridDNATokenizer":
        # The released wrapper does not persist SFT's EOS override. Use the
        # compatible local implementation, which is copied into saved checkpoints.
        tokenizer = HybridDNATokenizer.from_pretrained(
            model_args.model_name_or_path,
            revision=model_args.model_revision,
            trust_remote_code=training_args.trust_remote_code,
        )

    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template

    return tokenizer


def get_model(model_args: ModelConfig, training_args: SFTConfig) -> AutoModelForCausalLM:
    """Get the model"""
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=training_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )

    return model
