"""Score (context, continuation) pairs with vLLM: the one primitive both MCF/CF scorers are built on.

vLLM has no scoring-only API, so each pair is sent as a single prompt (context + continuation) with
`prompt_logprobs=0`, which returns the logprob of every prompt token under the model. Summing over the
continuation's positions gives exactly the teacher-forced NLL that a `forward()` pass would give, with no
generation and no sampling.

The options of one question share their whole context, so prefix caching can in principle prefill the long DNA
prefix once instead of once per option. It is left on (vLLM's own default) and `--no-prefix-caching` turns it off.
The reason to care is correctness, not speed: scoring SeqQA both ways gave identical NLLs, confirming that caching
does not perturb `prompt_logprobs`, and it made no measurable difference to runtime on these prompt lengths.
"""
import time

PAD = 8  # headroom so a prompt truncated to exactly max_len still leaves room for the 1 sampled token


def build_llm(model, max_len, tensor_parallel_size=1, gpu_memory_utilization=0.90, prefix_caching=True):
    from vllm import LLM
    return LLM(model=model, trust_remote_code=True, max_model_len=max_len + PAD,
               tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=gpu_memory_utilization,
               enable_prefix_caching=prefix_caching, dtype="bfloat16", seed=0)


def shutdown(llm):
    """Best-effort release of the engine, called only after the results are already on disk.

    vLLM's engine process can raise "terminate called without an active exception" (SIGABRT) while the interpreter
    tears down, long after scoring is finished. Dropping the handle and collecting first usually avoids it. Callers
    must not depend on this: run_evals.sh treats the result file, not the exit status, as the record of success.
    """
    try:
        del llm
        import gc
        gc.collect()
    except Exception:
        pass


def score(llm, pairs, max_len, reserve=0):
    """pairs: list of (context_ids, continuation_ids). Returns the summed NLL of each continuation, in nats.

    The context is left-truncated, never the continuation, so the scored tokens are always the full continuation.
    `reserve` is how many extra token slots the transformers implementation being reproduced left free: its
    mcq scorer keeps `max_length - len(cont) - 1` context tokens (reserve=1) and its GenomeQA scorer truncates the
    concatenation to `maxlen` with nothing reserved (reserve=0). The two differ by a single context token on the rows
    that actually overflow, so each scorer passes the value its reference used.
    """
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    prompts, n_cont = [], []
    for ctx, cont in pairs:
        keep = max(1, max_len - len(cont) - reserve)
        prompts.append(TokensPrompt(prompt_token_ids=ctx[-keep:] + cont))
        n_cont.append(len(cont))

    t0 = time.time()
    print(f"[vllm] scoring {len(prompts)} pairs, longest {max(len(p['prompt_token_ids']) for p in prompts)} tokens",
          flush=True)
    outs = llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0))
    print(f"[vllm] done in {time.time() - t0:.1f}s", flush=True)

    nll = []
    for out, n in zip(outs, n_cont):
        plp, ids = out.prompt_logprobs, out.prompt_token_ids
        total = 0.0
        for pos in range(len(plp) - n, len(plp)):
            entry = plp[pos]
            if not entry or ids[pos] not in entry:
                # position 0 has no logprob; a continuation token can never land there because keep >= 1.
                raise RuntimeError(f"no prompt logprob at position {pos} of {len(plp)} -- "
                                   f"the backend returned an incomplete prompt_logprobs")
            total += entry[ids[pos]].logprob
        nll.append(-total)
    return nll
