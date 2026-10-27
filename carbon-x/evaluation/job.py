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
Run one benchmark inside a Hugging Face Job launched by `evaluate.py`.

Installs the evaluation dependencies, serves the model with vLLM for `--backend vllm`, runs the Inspect task, writes
`summary.json` beside the log and uploads both to the output bucket.
"""

import argparse
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


MOUNTED_ROOT = Path("/workspace/carbon-x")
RUNTIME_ROOT = Path("/tmp/carbon-x")
RUNTIME_ENV_VAR = "CARBON_X_EVAL_RUNTIME"
LOG_DIR = Path("/tmp/inspect-logs")
VLLM_PORT = 8000
VLLM_STARTUP_TIMEOUT = 600
WEIGHT_DOWNLOAD_ATTEMPTS = 5
# Task -> (score, metric) reported as the benchmark's score
PRIMARY_METRICS = {
    "bioreason_coding_sequence_only": ("vepqa_scorer", "balanced_accuracy"),
    "bioreason_non_snv": ("vepqa_scorer", "balanced_accuracy"),
    "genomeqa_mcq_carbon": ("choice", "all"),
    "seqrecqa_varlen": ("accuracy_nt", "accuracy"),
    "vepqa_single_1k": ("vepqa_all_scorer", "accuracy"),
}
_NO_STDERR = object()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hub repo ID for vllm, or an Inspect model string for endpoint")
    parser.add_argument("--backend", choices=["vllm", "endpoint"], required=True)
    parser.add_argument("--output_dir", required=True, help="hf://buckets/... folder for the log and summary")
    parser.add_argument("--eval_kwargs", type=json.loads, required=True, help="JSON keyword arguments for eval()")
    parser.add_argument("--vllm_args", nargs="+", default=[], metavar="KEY=VALUE", help="`vllm serve` arguments")
    return parser.parse_args()


def bootstrap() -> None:
    """Installs the evaluation dependency group into the image's Python and re-executes this script with it."""
    # Copy out of the read-only mount so uv can write to the project directory
    shutil.copytree(MOUNTED_ROOT, RUNTIME_ROOT)
    env = os.environ.copy()
    env[RUNTIME_ENV_VAR] = "1"
    env.pop("VIRTUAL_ENV", None)
    # --inexact keeps packages that are not in the group, such as vLLM, PyTorch and CUDA from the vLLM image
    subprocess.run(
        ["uv", "sync", "--frozen", "--inexact", "--only-group", "evaluation"],
        cwd=RUNTIME_ROOT,
        env=env | {"UV_PROJECT_ENVIRONMENT": sys.prefix},
        check=True,
    )
    script = RUNTIME_ROOT / "evaluation" / "job.py"
    os.execve(sys.executable, [sys.executable, str(script), *sys.argv[1:]], env)


def download_weights(repo_id: str, revision: str | None) -> None:
    from huggingface_hub import snapshot_download

    # Jobs launched together download the same checkpoint at once and can exceed the Xet CDN limit of 10,000 requests
    # per 60 s per IP. Back off with jitter so concurrent Jobs spread out; completed shards stay cached.
    for attempt in range(1, WEIGHT_DOWNLOAD_ATTEMPTS + 1):
        try:
            snapshot_download(repo_id, revision=revision, allow_patterns="*.safetensors")
            return
        # hf_xet raises CDN failures such as 429 Too Many Requests as RuntimeError
        except RuntimeError as error:
            if attempt == WEIGHT_DOWNLOAD_ATTEMPTS:
                raise
            delay = 60 * 2 ** (attempt - 1) * random.uniform(0.5, 1.5)
            print(f"Weight download attempt {attempt} failed: {error}\nRetrying in {delay:.0f} s", file=sys.stderr)
            time.sleep(delay)


def start_vllm(repo_id: str, vllm_args: list[str]) -> subprocess.Popen:
    command = ["vllm", "serve", repo_id]
    for vllm_arg in vllm_args:
        key, value = vllm_arg.split("=", 1)
        flag = key.replace("_", "-")
        if value == "true":
            command.append(f"--{flag}")
        elif value == "false":
            command.append(f"--no-{flag}")
        else:
            command.extend([f"--{flag}", value])
    command.extend(["--host", "127.0.0.1", "--port", str(VLLM_PORT)])
    server = subprocess.Popen(command, start_new_session=True)

    deadline = time.monotonic() + VLLM_STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"vLLM exited before becoming healthy with status {server.returncode}")
        try:
            with urlopen(f"http://127.0.0.1:{VLLM_PORT}/health", timeout=2) as response:
                if response.status == 200:
                    return server
        except (TimeoutError, URLError):
            pass
        time.sleep(5)
    stop_vllm(server)
    raise TimeoutError(f"vLLM did not become healthy within {VLLM_STARTUP_TIMEOUT} s")


def stop_vllm(server: subprocess.Popen) -> None:
    if server.poll() is not None:
        return
    os.killpg(server.pid, signal.SIGTERM)
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait()


def _json_value(value: Any) -> Any:
    if isinstance(value, int | float) and not math.isfinite(value):
        return None
    return value


def _error_type(message: str) -> str:
    # Inspect reports exhausted retries as "RetryError(<Future ... raised AttemptTimeoutError>)"; count the cause
    match = re.search(r"raised (\w+)>", message) or re.match(r"(\w+)\(", message)
    return match.group(1) if match else "unknown"


def _stderr_value(score: Any, metric: Any) -> Any:
    # Inspect flattens VEPQA's accuracy, stderr and class metrics into one score; stderr applies only to accuracy and
    # balanced accuracy
    if score.scorer == "vepqa_scorer" and metric.name not in {"accuracy", "balanced_accuracy"}:
        return _NO_STDERR
    if metric.name.endswith("_acc"):
        stderr_name = f"{metric.name.removesuffix('_acc')}_stderr"
    elif score.scorer == "vepqa_all_scorer" or metric.name == "balanced_accuracy":
        stderr_name = f"{metric.name}_stderr"
    else:
        stderr_name = "stderr"
    for score_metric in score.metrics.values():
        if score_metric.name == stderr_name:
            return _json_value(score_metric.value)
    return _NO_STDERR


def _metric_summary(score: Any, metric: Any) -> dict[str, Any]:
    summary = {
        "score_name": score.name,
        "scorer": score.scorer,
        "reducer": score.reducer,
        "name": metric.name,
        "value": _json_value(metric.value),
        "params": dict(metric.params),
    }
    if (stderr := _stderr_value(score, metric)) is not _NO_STDERR:
        summary["stderr"] = stderr
    return summary


def build_summary(log: Any) -> dict[str, Any]:
    """Summarises an Inspect log in the `summary.json` format read by the post-training leaderboard."""
    samples = log.samples or []
    completed = [sample for sample in samples if sample.error is None and sample.output.choices]
    stop_reasons = Counter(sample.output.stop_reason for sample in completed)
    # Inspect's completed count excludes errored samples even when score_on_error scores them
    model_length_samples = stop_reasons["model_length"]
    completed_samples = log.results.completed_samples - model_length_samples
    metadata = dict(log.eval.metadata or {})
    metadata.update(
        {
            "task": log.eval.task,
            "task_args": dict(log.eval.task_args),
            "task_args_passed": dict(log.eval.task_args_passed),
            "started_at": log.stats.started_at,
            "completed_at": log.stats.completed_at,
            "status": log.status,
            "total_samples": log.results.total_samples,
            "scored_samples": sum(bool(sample.scores) for sample in samples),
            "completed_samples": completed_samples,
            "model_length_samples": model_length_samples,
            "model_length_fraction": model_length_samples / log.results.total_samples,
            "truncated_samples": stop_reasons["max_tokens"],
            "truncated_fraction": stop_reasons["max_tokens"] / completed_samples if completed_samples else None,
            "stop_reasons": dict(stop_reasons.most_common()),
            "error_types": dict(
                Counter(_error_type(sample.error.message) for sample in samples if sample.error).most_common()
            ),
            "retried_samples": sum(bool(sample.error_retries) for sample in samples),
        }
    )
    score_name, metric_name = PRIMARY_METRICS[log.eval.task]
    primary_score = next(score for score in log.results.scores if score.name == score_name)
    metrics = [
        _metric_summary(score, metric)
        for score in log.results.scores
        for metric in score.metrics.values()
        if metric.name not in {"stderr", "std"} and not metric.name.endswith("_stderr")
    ]
    return {
        "metadata": metadata,
        "primary_metric": _metric_summary(primary_score, primary_score.metrics[metric_name]),
        "metrics": metrics,
    }


def main():
    if os.environ.get(RUNTIME_ENV_VAR) != "1":
        bootstrap()

    from huggingface_hub import HfApi
    from inspect_ai import eval as inspect_eval
    from inspect_ai.log import read_eval_log

    args = parse_args()
    os.chdir(RUNTIME_ROOT)
    server = None
    status = None
    try:
        model = args.model
        if args.backend == "vllm":
            # Download before starting vLLM so the transfer does not count against its startup timeout
            download_weights(args.model, dict(arg.split("=", 1) for arg in args.vllm_args).get("revision"))
            server = start_vllm(args.model, args.vllm_args)
            # Inspect's vLLM provider connects to the server through VLLM_BASE_URL
            os.environ["VLLM_BASE_URL"] = f"http://127.0.0.1:{VLLM_PORT}/v1"
            model = f"vllm/{args.model}"
        log = inspect_eval(model=model, log_dir=str(LOG_DIR), display="plain", **args.eval_kwargs)[0]
        if server is not None and server.poll() is not None:
            raise RuntimeError(f"vLLM exited during the evaluation with status {server.returncode}")
        status = log.status
        if status == "success":
            summary = build_summary(read_eval_log(log.location))
            (Path(log.location).parent / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            primary_metric = summary["primary_metric"]
            print(f"{log.eval.task}: {primary_metric['name']} = {primary_metric['value']:.4f}")
    finally:
        if server is not None:
            stop_vllm(server)
        if LOG_DIR.is_dir() and any(LOG_DIR.iterdir()):
            HfApi().sync_bucket(str(LOG_DIR), args.output_dir)
            print(f"Uploaded the Inspect log to {args.output_dir}")
    if status != "success":
        sys.exit(f"Inspect evaluation finished with status {status!r}")


if __name__ == "__main__":
    main()
