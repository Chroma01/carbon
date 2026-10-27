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
Evaluate a model on the five Carbon-X DNA benchmarks with Hugging Face Jobs, one Job per benchmark.

`--model` is an Inspect model string. Hub models (`vllm/<repo_id>`) are served with vLLM on a GPU; any other provider,
e.g. `anthropic/claude-opus-5-5` or `hf-inference-providers/<repo_id>:<provider>`, is called from a CPU Job. Inspect
logs and a `summary.json` per benchmark are written to a Storage Bucket.

Usage:

uv run evaluation/evaluate.py \
    --model vllm/HuggingFaceBio/Carbon-X-SFT
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from huggingface_hub import HfApi, Volume, get_token, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from huggingface_hub.utils import disable_progress_bars


CARBON_X_ROOT = Path(__file__).resolve().parents[1]
JOB_ROOT = "/workspace/carbon-x"
# Keep in sync with the vllm pin in pyproject.toml
VLLM_IMAGE = "vllm/vllm-openai:v0.25.1"
CPU_IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm"
DEFAULT_BUCKET_NAME = "Carbon-X-Evals"
DEFAULT_GPU_FLAVOR = "rtx-pro-6000"
DEFAULT_CPU_FLAVOR = "cpu-basic"
MAX_CONTEXT_LENGTH = 131_072
# Benchmark -> (Inspect task, prompt tokens reserved when deriving max_tokens from the context length)
BENCHMARKS = {
    "bioreason_coding_sequence_only": ("evaluation/tasks/bioreason.py@bioreason_coding_sequence_only", 5_120),
    "bioreason_non_snv": ("evaluation/tasks/bioreason.py@bioreason_non_snv", 5_120),
    # Leaves room for the longest prompt, 17,225 tokens
    "genomeqa_mcq_carbon": ("evaluation/tasks/genomeqa.py@genomeqa_mcq_carbon", 17_408),
    "seqrecqa_varlen": ("evaluation/tasks/seqrecqa.py@seqrecqa_varlen", 5_120),
    "vepqa_single_1k": ("evaluation/tasks/vepqa.py@vepqa_single_1k", 5_120),
}
# Reasoning endpoints at high effort can take much longer per sample than a local vLLM server
INSPECT_LIMITS = {
    "vllm": {"time_limit": 1800, "timeout": 2100, "attempt_timeout": 2100, "max_retries": 3},
    "endpoint": {"time_limit": 6600, "timeout": 6000, "attempt_timeout": 6600, "max_retries": 10},
}
PROVIDER_API_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY")
JOB_ENV = {
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "HF_HUB_DISABLE_EXPERIMENTAL_WARNING": "1",
    "TOKENIZERS_PARALLELISM": "false",
}
VLLM_JOB_ENV = {
    "TORCH_COMPILE_DISABLE": "1",
    # Works around a race in vLLM's AOT compile cache: https://github.com/vllm-project/vllm/issues/33804
    "VLLM_USE_AOT_COMPILE": "0",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
}


def _key_value(value: str) -> tuple[str, str]:
    key, separator, argument_value = value.partition("=")
    if not key or not separator:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {value!r}")
    return key, argument_value


def _model_arg(value: str) -> tuple[str, object]:
    key, argument_value = _key_value(value)
    # Parse numbers and booleans as Inspect's -M flag does, e.g. client_timeout=600
    try:
        return key, json.loads(argument_value)
    except json.JSONDecodeError:
        return key, argument_value


def _json_object(value: str) -> dict:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(f"expected a JSON object, got {value!r}")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model",
        required=True,
        help="Inspect model string: vllm/<repo_id> for a Hub model, otherwise an endpoint such as anthropic/<model>",
    )
    parser.add_argument("--revision", help="Hub model revision (default: main)")
    parser.add_argument("--benchmarks", nargs="+", choices=list(BENCHMARKS), default=list(BENCHMARKS))
    parser.add_argument("--bucket", help=f"Storage Bucket for the logs (default: <hf_user>/{DEFAULT_BUCKET_NAME})")
    parser.add_argument(
        "--flavor", help=f"Job hardware (default: {DEFAULT_GPU_FLAVOR} for Hub models, {DEFAULT_CPU_FLAVOR} otherwise)"
    )
    parser.add_argument("--timeout", default="1d", help="Maximum Job duration, e.g. 6h or 1d")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N samples of each benchmark")
    parser.add_argument("--max_connections", type=int, help="Concurrent requests (default: 128 for vLLM, else 50)")
    parser.add_argument("--max_tokens", type=int, help="Default for Hub models: context length minus a prompt margin")
    parser.add_argument("--temperature", type=float, help="Default for Hub models: generation_config.json")
    parser.add_argument("--top_p", type=float, help="Default for Hub models: generation_config.json")
    parser.add_argument("--top_k", type=int, help="Default for Hub models: generation_config.json")
    parser.add_argument("--reasoning_effort", help="Forwarded to Inspect and appended to the model's log path")
    parser.add_argument(
        "--chat_template_kwargs",
        type=_json_object,
        help='JSON object for extra_body.chat_template_kwargs (default for Hub models: {"enable_thinking": false})',
    )
    parser.add_argument("--model_base_url", help="Base URL for OpenAI-compatible endpoints")
    parser.add_argument(
        "--model_args", nargs="+", type=_model_arg, default=[], metavar="KEY=VALUE", help="Inspect model args (-M)"
    )
    parser.add_argument(
        "--vllm_args", nargs="+", type=_key_value, default=[], metavar="KEY=VALUE", help="Overrides for `vllm serve`"
    )
    parser.add_argument(
        "--secrets", nargs="+", default=[], metavar="NAME", help="Extra local environment variables to forward"
    )
    parser.add_argument("--dry_run", action="store_true", help="Print the Jobs without uploading or submitting")
    return parser.parse_args()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def log_model_id(model: str, reasoning_effort: str | None) -> str:
    """Returns the model's folder in the bucket, matching the post-training leaderboard's layout."""
    model_id = model.removeprefix("vllm/").removeprefix("hf-inference-providers/").split(":", 1)[0]
    return f"{model_id}-{reasoning_effort}" if reasoning_effort else model_id


def _hub_json(repo_id: str, filename: str, revision: str) -> dict:
    try:
        return json.loads(Path(hf_hub_download(repo_id, filename, revision=revision)).read_text())
    except EntryNotFoundError:
        return {}


def context_length(repo_id: str, revision: str) -> int:
    config = _hub_json(repo_id, "config.json", revision)
    for config_object in (config, config.get("text_config") or {}):
        if max_position_embeddings := config_object.get("max_position_embeddings"):
            return min(int(max_position_embeddings), MAX_CONTEXT_LENGTH)
    raise ValueError(f"Could not find the context length of {repo_id}; pass --vllm_args max_model_len=...")


def gpu_count(api: HfApi, flavor: str) -> int:
    hardware = next((hardware for hardware in api.list_jobs_hardware() if hardware.name == flavor), None)
    if hardware is None:
        raise ValueError(f"Unknown HF Jobs flavor {flavor!r}")
    if hardware.accelerator is None:
        raise ValueError(f"Hub models are served with vLLM and need a GPU flavor, got {flavor!r}")
    return int(hardware.accelerator.quantity)


def snapshot_volume(api: HfApi) -> Volume:
    """Uploads the files the Job needs to a content-addressed folder of the private jobs-artifacts bucket."""
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--"]
        + ["README.md", "pyproject.toml", "uv.lock", "evaluation"],
        cwd=CARBON_X_ROOT,
        check=True,
        capture_output=True,
    )
    paths = sorted(path for path in completed.stdout.decode().split("\0") if (CARBON_X_ROOT / path).is_file())
    digest = sha256()
    for path in paths:
        digest.update(path.encode())
        digest.update((CARBON_X_ROOT / path).read_bytes())
    with tempfile.TemporaryDirectory() as staging_dir:
        for path in paths:
            destination = Path(staging_dir) / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(CARBON_X_ROOT / path, destination)
        return api.sync_job_volume(staging_dir, JOB_ROOT, remote_name=f"carbon-x/{digest.hexdigest()[:16]}")


def main():
    args = parse_args()
    token = get_token()
    if token is None:
        raise RuntimeError("HF Jobs need a Hugging Face token; run `hf auth login` first")
    api = HfApi(token=token)
    bucket = args.bucket or f"{api.whoami()['name']}/{DEFAULT_BUCKET_NAME}"

    is_vllm = args.model.startswith("vllm/")
    backend = "vllm" if is_vllm else "endpoint"
    if not is_vllm and (args.revision or args.vllm_args):
        raise ValueError("--revision and --vllm_args only apply to Hub models served with vLLM (vllm/<repo_id>)")
    flavor = args.flavor or (DEFAULT_GPU_FLAVOR if is_vllm else DEFAULT_CPU_FLAVOR)
    revision = args.revision or "main"
    chat_template_kwargs = args.chat_template_kwargs
    sampling = {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k}
    max_model_len = None
    vllm_args = {}
    model_args = dict(args.model_args)
    if is_vllm:
        repo_id = args.model.removeprefix("vllm/")
        user_vllm_args = dict(args.vllm_args)
        if "max_model_len" in user_vllm_args:
            max_model_len = int(user_vllm_args["max_model_len"])
        else:
            max_model_len = context_length(repo_id, revision)
        tensor_parallel_size = int(user_vllm_args.get("tensor_parallel_size", 1))
        vllm_args = {
            "revision": revision,
            "max_model_len": max_model_len,
            "data_parallel_size": gpu_count(api, flavor) // tensor_parallel_size,
            "reasoning_parser": "qwen3",
            "trust_remote_code": "true",
            "disable_uvicorn_access_log": "true",
            **user_vllm_args,
        }
        generation_config = _hub_json(repo_id, "generation_config.json", revision)
        sampling = {key: value if value is not None else generation_config.get(key) for key, value in sampling.items()}
        if chat_template_kwargs is None:
            chat_template_kwargs = {"enable_thinking": False}
        model_args.setdefault("client_timeout", INSPECT_LIMITS["vllm"]["timeout"])
    max_connections = args.max_connections or (128 if is_vllm else 50)

    model_log_id = log_model_id(args.model, args.reasoning_effort)
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    secrets = {"HF_TOKEN": token}
    for name in args.secrets:
        if name not in os.environ:
            raise ValueError(f"--secrets {name} is not set in the local environment")
    for name in (*PROVIDER_API_KEYS, *args.secrets):
        if name in os.environ:
            secrets[name] = os.environ[name]

    jobs = []
    for benchmark in args.benchmarks:
        task, prompt_margin = BENCHMARKS[benchmark]
        max_tokens = args.max_tokens
        if max_tokens is None and max_model_len is not None:
            max_tokens = max_model_len - prompt_margin
        metadata = {
            "model_id": args.model.removeprefix("vllm/"),
            "revision": revision if is_vllm else None,
            "backend": backend,
            "reasoning_effort": args.reasoning_effort,
            **sampling,
            "context_length": max_model_len,
            "max_tokens": max_tokens,
            "prompt_config": {"system_message": None, "chat_template_kwargs": chat_template_kwargs},
        }
        eval_kwargs = {
            "tasks": task,
            "limit": args.limit,
            "max_connections": max_connections,
            "max_tokens": max_tokens,
            **sampling,
            "reasoning_effort": args.reasoning_effort,
            "extra_body": {"chat_template_kwargs": chat_template_kwargs} if chat_template_kwargs else None,
            "model_base_url": args.model_base_url,
            "model_args": model_args or None,
            **INSPECT_LIMITS[backend],
            # Score samples that still error after a retry as incorrect so metrics cover the whole dataset
            "retry_on_error": 1,
            "fail_on_error": False,
            "score_on_error": True,
            "metadata": metadata,
        }
        eval_kwargs = {key: value for key, value in eval_kwargs.items() if value is not None}
        output_path = f"{model_log_id}/{revision}/{benchmark}/{timestamp}"
        command = [
            "python3",
            f"{JOB_ROOT}/evaluation/job.py",
            "--model",
            args.model,
            "--output_dir",
            f"hf://buckets/{bucket}/{output_path}",
            "--eval_kwargs",
            json.dumps(eval_kwargs),
        ]
        if vllm_args:
            command += ["--vllm_args", *(f"{key}={value}" for key, value in vllm_args.items())]
        jobs.append(
            {
                "name": _safe_name(f"{benchmark}-{model_log_id}-{revision}"),
                "image": VLLM_IMAGE if is_vllm else CPU_IMAGE,
                "command": command,
                "env": JOB_ENV | (VLLM_JOB_ENV if is_vllm else {}),
                "flavor": flavor,
                "timeout": args.timeout,
                "output_url": f"https://huggingface.co/buckets/{bucket}/tree/{output_path}",
            }
        )

    if args.dry_run:
        print(json.dumps(jobs, indent=2))
        return

    api.create_bucket(bucket, private=True, exist_ok=True)
    disable_progress_bars()
    volume = snapshot_volume(api)
    for job in jobs:
        output_url = job.pop("output_url")
        job_info = api.run_job(**job, secrets=secrets, volumes=[volume])
        print(f"{job['name']}\n  Job: {job_info.url}\n  Logs: {output_url}")


if __name__ == "__main__":
    main()
