"""Megatron GPT pretraining with everything frozen except the new DNA embedding rows.

A thin wrapper around Megatron-LM's own entrypoint
(https://github.com/NVIDIA/Megatron-LM/blob/main/pretrain_gpt.py). It changes nothing in Megatron core and reuses
that file's model, dataset and loss setup. The one addition is a gradient mask, registered after the model is
built, that zeroes the gradient of every embedding row below ORIGINAL_VOCAB_SIZE.

Why: the hybrid tokenizer appends DNA rows to Qwen3's vocabulary, and those rows start out randomly initialised.
Training the whole model while they are still noise would drag the pretrained weights around to compensate. So
stage one trains only the new rows (NatureLM-style), on DNA alone, and everything else is held still. Stage two
(`slurm/midtrain_300b.slurm`) unfreezes the model and runs the full mixture, using Megatron's stock pretrain_gpt.py.

Qwen3 has UNTIED embeddings, so `output_layer.weight` is a tensor of its own rather than a view of the input
embedding. The mask is therefore registered on both: mask only the input embedding and the English rows of the
lm_head still drift during the DNA-only stage.

Set ORIGINAL_VOCAB_SIZE to the tokenizer's `dna_start_id`, the index of the first DNA row -- 151669 for
HuggingFaceBio/Carbon-X. Do NOT use Qwen3's own vocab size of 151936: that is larger than dna_start_id, so it would
freeze the first 267 DNA rows you are trying to train. Leaving it unset trains all rows, i.e. no freezing at all.

TP-aware (each rank masks its own vocab shard). PP=1 is assumed, so the embedding and the output layer live on the
same rank.

CAVEAT -- "frozen" here means zero GRADIENT, not a constant tensor. The hook runs on the gradient; Megatron's
optimizer then applies decoupled weight decay (`--weight-decay 0.1` in the launcher) to every 2-D parameter,
embeddings included, as `p -= lr * wd * p`. That term does not depend on the gradient, so the original-vocab rows
still shrink slowly over the stage. The effect is small at this LR and token budget, but it is not zero, and it is
unverified: run `verify_freeze.py <scaffold> <freeze_ckpt>` and look at check B, which compares the original-vocab
rows bitwise. If they come back identical, weight decay is not moving them and this caveat does not apply. If you need the original rows bit-identical, run this stage with
`--weight-decay 0.0`, or give the embedding its own param group with `no_weight_decay_cond`. The released
checkpoint was trained as-is, with weight decay on.
"""

import time
_PROGRAM_START_TIME = time.time()

import os
import warnings
rank = int(os.environ.get('RANK', 0))
if rank != 0:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

from functools import partial
from typing import Union

import torch

from megatron.core.models.gpt import GPTModel
from megatron.core.models.mamba import MambaModel
from megatron.core.enums import ModelType
from megatron.training import (
    get_args,
    inprocess_restart,
    pretrain,
    print_rank_0,
    set_startup_timestamps,
)

from gpt_builders import gpt_builder
from model_provider import model_provider as _original_model_provider
from pretrain_gpt import (
    forward_step,
    train_valid_test_datasets_provider,
    get_embedding_ranks,
)

try:
    from megatron.post_training.arguments import add_modelopt_args
    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

import megatron.legacy.model  # noqa: F401


def _freeze_non_embedding_params(model):
    """Freeze all params except embedding.word_embeddings and output_layer.weight (untied lm_head)."""
    frozen_count = trainable_count = 0
    for name, param in model.named_parameters():
        if 'embedding.word_embeddings' in name or 'output_layer.weight' in name:
            param.requires_grad = True; trainable_count += param.numel()
            print_rank_0(f"  TRAINABLE: {name} ({param.numel():,})")
        else:
            param.requires_grad = False; frozen_count += param.numel()
    total = frozen_count + trainable_count
    print_rank_0(f"  Freeze summary: {trainable_count:,} trainable "
                 f"({100*trainable_count/total:.2f}%), {frozen_count:,} frozen")


def _mask_hook_factory(num_local, freeze_end):
    """Backward hook zeroing grad rows [0:freeze_end] (original-vocab rows)."""
    _cache = {}
    def hook(grad):
        key = (grad.device, grad.dtype)
        if key not in _cache:
            m = torch.ones(num_local, 1, dtype=grad.dtype, device=grad.device)
            m[:freeze_end] = 0.0
            _cache[key] = m
        return grad * _cache[key]
    return hook


def _register_embedding_gradient_mask(model, original_vocab_size):
    """Zero grads for original-vocab rows of the INPUT embedding (VocabParallelEmbedding, TP-aware)."""
    emb = model.embedding.word_embeddings
    vocab_start = emb.vocab_start_index
    num_local = emb.num_embeddings_per_partition
    freeze_end = min(max(original_vocab_size - vocab_start, 0), num_local)
    if freeze_end == 0:
        print_rank_0(f"  [emb] all {num_local} local rows trainable (vocab_start={vocab_start})"); return
    print_rank_0(f"  [emb] freeze {freeze_end} + train {num_local-freeze_end} rows (vocab_start={vocab_start})")
    emb.weight.register_hook(_mask_hook_factory(num_local, freeze_end))


def _register_output_layer_gradient_mask(model, original_vocab_size):
    """UNTIED only: zero grads for original-vocab rows of output_layer.weight (lm_head).

    Skips if tied (shares storage with the input embedding — already masked) or absent (PP split)."""
    ol = getattr(model, 'output_layer', None)
    if ol is None or getattr(ol, 'weight', None) is None:
        print_rank_0("  [lm_head] no output_layer on this rank; skip"); return
    if ol.weight.data_ptr() == model.embedding.word_embeddings.weight.data_ptr():
        print_rank_0("  [lm_head] tied to input embedding; already masked"); return
    from megatron.core import parallel_state
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    num_local = ol.weight.shape[0]                 # vocab is column-parallel -> contiguous chunk per rank
    vocab_start = tp_rank * num_local
    freeze_end = min(max(original_vocab_size - vocab_start, 0), num_local)
    if freeze_end == 0:
        print_rank_0(f"  [lm_head] all {num_local} local rows trainable (vocab_start={vocab_start})"); return
    print_rank_0(f"  [lm_head] freeze {freeze_end} + train {num_local-freeze_end} rows (vocab_start={vocab_start})")
    ol.weight.register_hook(_mask_hook_factory(num_local, freeze_end))


def model_provider_freeze_body(model_builder, pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None):
    model = _original_model_provider(model_builder, pre_process, post_process, vp_stage, config=config, pg_collection=pg_collection)
    print_rank_0("> Freezing non-embedding params (NatureLM-style, UNTIED)...")
    _freeze_non_embedding_params(model)
    original_vocab_size = int(os.environ.get('ORIGINAL_VOCAB_SIZE', '0'))
    if original_vocab_size > 0:
        if pre_process and hasattr(model, 'embedding'):
            _register_embedding_gradient_mask(model, original_vocab_size)
        if post_process:
            _register_output_layer_gradient_mask(model, original_vocab_size)
        print_rank_0(f"  Selective freeze: only rows >= {original_vocab_size} trainable (emb + lm_head)")
    else:
        print_rank_0("  ORIGINAL_VOCAB_SIZE not set: all embedding rows trainable")
    return model


if __name__ == "__main__":
    _MAIN_ENTRY_TIME = time.time()
    set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=_MAIN_ENTRY_TIME)
    train_valid_test_datasets_provider.is_distributed = True
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)
    pretrain(
        train_valid_test_datasets_provider,
        partial(model_provider_freeze_body, gpt_builder),
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
        store=store,
        get_embedding_ranks=get_embedding_ranks,
    )
