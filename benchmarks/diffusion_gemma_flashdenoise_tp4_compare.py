# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Repeatable TP serving harness for DiffusionGemma sampler comparisons.

This script intentionally does not bake in benchmark claims. It starts a fresh
OpenAI server for each requested mode/shape/repeat, runs the serving benchmark,
captures the benchmark JSON, and writes computed medians to summary.json.
"""

from __future__ import print_function

import argparse
import datetime
import io
import json
import os
import re
import signal
import subprocess
import sys
import time

try:
    import urllib.request as urlrequest
except ImportError:  # pragma: no cover - Python 2 py_compile compatibility.
    import urllib2 as urlrequest

try:
    from shlex import quote as shell_quote
except ImportError:  # pragma: no cover - Python 2 py_compile compatibility.
    from pipes import quote as shell_quote


LOCAL_VOCAB_ENV = "VLLM_DIFFUSION_GEMMA_LOCAL_VOCAB_SAMPLER"
NATIVE_ENV = "VLLM_DIFFUSION_GEMMA_FLASHDENOISE_NATIVE"
NATIVE_TP_STATE_ENV = "VLLM_DIFFUSION_GEMMA_FLASHDENOISE_NATIVE_TP_STATE"
NATIVE_MODE_FLAGS_ENV = "VLLM_DIFFUSION_GEMMA_FLASHDENOISE_NATIVE_MODE_FLAGS"

DEFAULT_MODE_ORDER = [
    "default_dense",
    "local_vocab_pytorch",
    "local_vocab_fused",
    "native_tp_state",
]
OPTIONAL_MODE_ORDER = [
    "pr_style_soft_embed_equivalent",
]
ALL_MODE_ORDER = DEFAULT_MODE_ORDER + OPTIONAL_MODE_ORDER

MODE_NOTES = {
    "default_dense": "Dense baseline with DiffusionGemma local/native flags off.",
    "pr_style_soft_embed_equivalent": (
        "Current dense soft-embedding path used as the PR-style equivalent. "
        "For a literal upstream PR comparison, run this harness from that "
        "worktree and record its commit in summary.json."
    ),
    "local_vocab_pytorch": (
        "TP vocab-sharded local sampler path implemented in PyTorch."
    ),
    "local_vocab_fused": (
        "Local-vocab sampler with the current fused FlashDenoise native flag "
        "enabled. On revisions where the fused path is still TP=1-only, vLLM "
        "may fall back; the env provenance records that explicitly."
    ),
    "native_tp_state": (
        "Native TP-state placeholder/implementation path. Uses "
        "%s=1 as the integration flag." % NATIVE_TP_STATE_ENV
    ),
}

NATIVE_TP_STATE_LOG_MARKER = (
    "Using native DiffusionGemma FlashDenoise TP-state pre-logit path."
)

METRIC_KEYS = {
    "request_throughput": [
        "request_throughput",
        "request_throughput_req_per_s",
        "requests_per_second",
    ],
    "output_throughput": [
        "output_throughput",
        "output_token_throughput",
        "output_token_throughput_tok_per_s",
    ],
    "mean_ttft_ms": [
        "mean_ttft_ms",
        "ttft_mean_ms",
    ],
}


def utc_now():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def to_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return value.decode("utf-8", "replace")
    except AttributeError:
        return str(value)


def ensure_dir(path):
    if not path:
        return
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def write_json(path, payload):
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with io.open(path, "w", encoding="utf-8") as outfile:
        json.dump(payload, outfile, indent=2, sort_keys=True)
        outfile.write(u"\n")


def read_json(path):
    with io.open(path, "r", encoding="utf-8") as infile:
        text = infile.read().strip()
    if not text:
        raise ValueError("empty JSON result file: %s" % path)
    try:
        return json.loads(text)
    except ValueError:
        # append-result style files are newline-delimited JSON. Use the last row.
        lines = [line for line in text.splitlines() if line.strip()]
        return json.loads(lines[-1])


def inspect_server_log(path):
    inspection = {
        "native_tp_state_confirmed": False,
        "native_tp_state_fallback_warnings": [],
    }
    if not os.path.exists(path):
        return inspection
    with io.open(path, "r", encoding="utf-8", errors="ignore") as infile:
        text = infile.read()
    inspection["native_tp_state_confirmed"] = NATIVE_TP_STATE_LOG_MARKER in text
    fallback_lines = []
    for line in text.splitlines():
        if NATIVE_TP_STATE_ENV not in line:
            continue
        lowered = line.lower()
        if (
            "falling back" in lowered
            or "requires" in lowered
            or "expected" in lowered
            or "estimated" in lowered
        ):
            fallback_lines.append(line[-500:])
    inspection["native_tp_state_fallback_warnings"] = fallback_lines[:8]
    return inspection


def run_text(cmd, cwd):
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, _ = proc.communicate()
    if proc.returncode != 0:
        return None
    return to_text(stdout).strip()


def git_provenance(cwd, target_file):
    return {
        "commit": run_text(["git", "rev-parse", "HEAD"], cwd),
        "commit_short": run_text(["git", "rev-parse", "--short", "HEAD"], cwd),
        "branch": run_text(["git", "branch", "--show-current"], cwd),
        "target_file_status": run_text(
            ["git", "status", "--short", "--", target_file], cwd
        ),
    }


def parse_multi_value(raw_values):
    if raw_values is None:
        return []
    if isinstance(raw_values, str):
        raw_values = [raw_values]
    values = []
    for raw in raw_values:
        for part in str(raw).split(","):
            part = part.strip()
            if part:
                values.append(part)
    return values


def parse_int_list(raw_values, flag_name):
    values = parse_multi_value(raw_values)
    if not values:
        raise ValueError("%s must contain at least one integer" % flag_name)
    parsed = []
    for value in values:
        try:
            parsed.append(int(value))
        except ValueError:
            raise ValueError("%s contains a non-integer value: %s" % (flag_name, value))
    return parsed


def parse_optional_int_list(raw_values, flag_name):
    values = parse_multi_value(raw_values)
    if not values:
        return [None]
    parsed = []
    for value in values:
        if value.lower() in ("none", "null", "-"):
            parsed.append(None)
        else:
            try:
                parsed.append(int(value))
            except ValueError:
                raise ValueError(
                    "%s contains a non-integer value: %s" % (flag_name, value)
                )
    return parsed


def parse_modes(raw_values):
    modes = parse_multi_value(raw_values)
    if not modes:
        return list(DEFAULT_MODE_ORDER)
    unknown = [mode for mode in modes if mode not in ALL_MODE_ORDER]
    if unknown:
        raise ValueError(
            "unknown mode(s): %s. Available modes: %s"
            % (", ".join(unknown), ", ".join(ALL_MODE_ORDER))
        )
    return modes


def safe_part(value):
    text = "none" if value is None else str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text)
    return text.strip("-") or "none"


def command_to_string(cmd):
    return " ".join(shell_quote(str(part)) for part in cmd)


def env_command_to_string(env_vars, cmd):
    prefix = " ".join(
        "%s=%s" % (key, shell_quote(str(env_vars[key])))
        for key in sorted(env_vars)
    )
    rendered = command_to_string(cmd)
    return ("%s %s" % (prefix, rendered)).strip()


def default_python_executable():
    if sys.version_info[0] >= 3:
        return sys.executable
    return os.environ.get("VLLM_BENCH_PYTHON", "python3")


def build_mode_envs(native_mode_flags):
    native_flags = str(native_mode_flags)
    return {
        "default_dense": {
            LOCAL_VOCAB_ENV: "0",
            NATIVE_ENV: "0",
            NATIVE_TP_STATE_ENV: "0",
        },
        "pr_style_soft_embed_equivalent": {
            LOCAL_VOCAB_ENV: "0",
            NATIVE_ENV: "0",
            NATIVE_TP_STATE_ENV: "0",
        },
        "local_vocab_pytorch": {
            LOCAL_VOCAB_ENV: "1",
            NATIVE_ENV: "0",
            NATIVE_TP_STATE_ENV: "0",
        },
        "local_vocab_fused": {
            LOCAL_VOCAB_ENV: "1",
            NATIVE_ENV: "1",
            NATIVE_TP_STATE_ENV: "0",
            NATIVE_MODE_FLAGS_ENV: native_flags,
        },
        "native_tp_state": {
            LOCAL_VOCAB_ENV: "1",
            NATIVE_ENV: "1",
            NATIVE_TP_STATE_ENV: "1",
            NATIVE_MODE_FLAGS_ENV: native_flags,
        },
    }


def benchmark_script_is_usable(cwd):
    script = os.path.join(cwd, "benchmarks", "benchmark_serving.py")
    if not os.path.exists(script):
        return False
    with io.open(script, "r", encoding="utf-8", errors="ignore") as infile:
        head = infile.read(4096)
    return "DEPRECATED" not in head


def select_benchmark_runner(cwd, requested):
    if requested == "cli":
        return "cli"
    if requested == "script":
        return "script"
    if benchmark_script_is_usable(cwd):
        return "script"
    return "cli"


def health_url(host, port):
    health_host = host
    if health_host in ("0.0.0.0", "::"):
        health_host = "127.0.0.1"
    if ":" in health_host and not health_host.startswith("["):
        health_host = "[%s]" % health_host
    return "http://%s:%d/health" % (health_host, port)


def is_healthy(url, timeout):
    try:
        response = urlrequest.urlopen(url, timeout=timeout)
        getcode = getattr(response, "getcode", None)
        status = getcode() if getcode is not None else getattr(response, "code", None)
        return status == 200
    except Exception:
        return False


def wait_for_health(proc, url, timeout_s, probe_timeout_s):
    deadline = time.time() + timeout_s
    last_state = "not ready"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                "server exited before /health became ready (returncode=%s)"
                % proc.returncode
            )
        if is_healthy(url, probe_timeout_s):
            return
        time.sleep(1.0)
    raise RuntimeError("timed out waiting for %s (%s)" % (url, last_state))


def wait_process(proc, timeout_s):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        ret = proc.poll()
        if ret is not None:
            return ret
        time.sleep(0.2)
    return None


def terminate_process_tree(proc, timeout_s):
    if proc is None or proc.poll() is not None:
        return
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
    except OSError:
        return

    if wait_process(proc, timeout_s) is not None:
        return

    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:
            proc.kill()
    except OSError:
        return
    wait_process(proc, timeout_s)


def launch_server(cmd, env, cwd, log_path):
    log_file = io.open(log_path, "w", encoding="utf-8")
    popen_kwargs = {
        "cwd": cwd,
        "env": env,
        "stdout": log_file,
        "stderr": subprocess.STDOUT,
    }
    if hasattr(os, "setsid"):
        popen_kwargs["preexec_fn"] = os.setsid
    try:
        proc = subprocess.Popen(cmd, **popen_kwargs)
    except Exception:
        log_file.close()
        raise
    return proc, log_file


def run_logged_command(cmd, env, cwd, log_path):
    started_at = utc_now()
    with io.open(log_path, "w", encoding="utf-8") as logfile:
        logfile.write(u"$ %s\n\n" % command_to_string(cmd))
        logfile.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=logfile,
            stderr=subprocess.STDOUT,
        )
        returncode = proc.wait()
    return {
        "started_at": started_at,
        "ended_at": utc_now(),
        "returncode": returncode,
    }


def build_server_command(args, shape):
    cmd = [
        args.python_executable,
        "-m",
        args.vllm_cli_module,
        "serve",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--served-model-name",
        args.served_model_name,
        "--max-num-seqs",
        str(shape["max_num_seqs"]),
    ]
    if shape["max_num_batched_tokens"] is not None:
        cmd.extend(["--max-num-batched-tokens", str(shape["max_num_batched_tokens"])])
    if args.max_model_len is not None:
        cmd.extend(["--max-model-len", str(args.max_model_len)])
    if args.gpu_memory_utilization is not None:
        cmd.extend(["--gpu-memory-utilization", str(args.gpu_memory_utilization)])
    if args.dtype:
        cmd.extend(["--dtype", args.dtype])
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    cmd.extend(args.server_arg or [])
    return cmd


def build_benchmark_command(args, shape, runner, result_dir, result_filename):
    if runner == "script":
        cmd = [
            args.python_executable,
            os.path.join("benchmarks", "benchmark_serving.py"),
        ]
        save_flag = "--save-results"
    else:
        cmd = [
            args.python_executable,
            "-m",
            args.vllm_cli_module,
            "bench",
            "serve",
        ]
        save_flag = "--save-result"

    cmd.extend(
        [
            "--backend",
            args.backend,
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--endpoint",
            args.endpoint,
            "--model",
            args.model,
            "--served-model-name",
            args.served_model_name,
            "--dataset-name",
            "random",
            "--num-prompts",
            str(args.num_prompts),
            "--input-len",
            str(args.input_len),
            "--output-len",
            str(args.output_len),
            "--request-rate",
            args.request_rate,
            "--max-concurrency",
            str(shape["concurrency"]),
            "--percentile-metrics",
            "ttft,tpot,itl",
            "--metric-percentiles",
            "50,90,99",
            "--disable-tqdm",
            save_flag,
            "--result-dir",
            result_dir,
            "--result-filename",
            result_filename,
        ]
    )
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    cmd.extend(args.benchmark_arg or [])
    return cmd


def extract_number(data, keys):
    if not isinstance(data, dict):
        return None
    for key in keys:
        value = data.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def extract_metrics(result_json):
    metrics = {}
    for name, keys in METRIC_KEYS.items():
        metrics[name] = extract_number(result_json, keys)

    if metrics["mean_ttft_ms"] is None:
        ttft_description = result_json.get("ttft_description")
        if isinstance(ttft_description, dict):
            value = ttft_description.get("mean")
            if isinstance(value, (int, float)):
                # Historical benchmark_serving.py stored ttft values in seconds.
                metrics["mean_ttft_ms"] = float(value) * 1000.0
    return metrics


def median(values):
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return None
    mid = len(clean) // 2
    if len(clean) % 2:
        return clean[mid]
    return (clean[mid - 1] + clean[mid]) / 2.0


def pct_delta(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return (numerator / denominator - 1.0) * 100.0


def shape_key(shape):
    return (
        "concurrency=%s|max_num_seqs=%s|max_num_batched_tokens=%s"
        % (
            shape["concurrency"],
            shape["max_num_seqs"],
            shape["max_num_batched_tokens"],
        )
    )


def shape_slug(shape):
    return "c%s_mns%s_mnbt%s" % (
        safe_part(shape["concurrency"]),
        safe_part(shape["max_num_seqs"]),
        safe_part(shape["max_num_batched_tokens"]),
    )


def build_shapes(concurrency_values, max_num_seqs_values, max_batched_values):
    shapes = []
    for concurrency in concurrency_values:
        for max_num_seqs in max_num_seqs_values:
            for max_num_batched_tokens in max_batched_values:
                shapes.append(
                    {
                        "concurrency": concurrency,
                        "max_num_seqs": max_num_seqs,
                        "max_num_batched_tokens": max_num_batched_tokens,
                    }
                )
    return shapes


def build_summary(args, mode_envs, shapes, runs, started_at, completed_at):
    shape_summaries = []
    for shape in shapes:
        per_mode = {}
        for mode in args.modes:
            matching_runs = [
                run
                for run in runs
                if run.get("mode") == mode
                and run.get("shape_key") == shape_key(shape)
                and run.get("metrics")
                and not run.get("error")
            ]
            metric_summary = {
                "successful_repeats": len(matching_runs),
                "failed_repeats": len(
                    [
                        run
                        for run in runs
                        if run.get("mode") == mode
                        and run.get("shape_key") == shape_key(shape)
                        and run.get("error")
                    ]
                ),
            }
            for metric in ("request_throughput", "output_throughput", "mean_ttft_ms"):
                metric_summary["%s_median" % metric] = median(
                    [run.get("metrics", {}).get(metric) for run in matching_runs]
                )
            per_mode[mode] = metric_summary

        native_req = per_mode.get("native_tp_state", {}).get(
            "request_throughput_median"
        )
        default_req = per_mode.get("default_dense", {}).get(
            "request_throughput_median"
        )
        pr_style_req = per_mode.get("pr_style_soft_embed_equivalent", {}).get(
            "request_throughput_median"
        )

        native_vs_pr_style = None
        if (
            args.enable_pr_style_equivalent_comparison
            and "pr_style_soft_embed_equivalent" in per_mode
        ):
            native_vs_pr_style = pct_delta(native_req, pr_style_req)

        shape_summary = {
            "shape": dict(shape),
            "shape_key": shape_key(shape),
            "modes": per_mode,
            "native_vs_default_pct_median": pct_delta(native_req, default_req),
            "native_vs_pr_style_pct_median": native_vs_pr_style,
            "native_vs_pr_style_note": (
                "Disabled unless --enable-pr-style-equivalent-comparison is "
                "set. The in-tree pr_style_soft_embed_equivalent mode is only "
                "a label; run this harness from a literal PR-style worktree or "
                "import an external PR baseline before using this delta."
            ),
        }
        shape_summaries.append(shape_summary)

    summary = {
        "started_at": started_at,
        "completed_at": completed_at,
        "provenance": {
            "git": git_provenance(
                args.repo_root,
                os.path.join(
                    "benchmarks", "diffusion_gemma_flashdenoise_tp4_compare.py"
                ),
            ),
            "cwd": args.repo_root,
            "python_executable": args.python_executable,
            "vllm_cli_module": args.vllm_cli_module,
            "benchmark_runner": args.benchmark_runner_selected,
        },
        "config": {
            "model": args.model,
            "served_model_name": args.served_model_name,
            "tensor_parallel_size": args.tensor_parallel_size,
            "host": args.host,
            "port": args.port,
            "concurrency": args.concurrency_values,
            "max_num_seqs": args.max_num_seqs_values,
            "max_num_batched_tokens": args.max_num_batched_tokens_values,
            "num_prompts": args.num_prompts,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "repeats": args.repeats,
            "consumer_state_trace_dir": args.consumer_state_trace_dir,
            "request_rate": args.request_rate,
            "out_dir": args.out_dir,
            "modes": args.modes,
            "enable_pr_style_equivalent_comparison": (
                args.enable_pr_style_equivalent_comparison
            ),
        },
        "mode_definitions": {
            mode: {
                "env": mode_envs[mode],
                "note": MODE_NOTES[mode],
            }
            for mode in args.modes
        },
        "shape_summaries": shape_summaries,
        "runs": runs,
    }

    if len(shape_summaries) == 1:
        summary["native_vs_default_pct_median"] = shape_summaries[0][
            "native_vs_default_pct_median"
        ]
        summary["native_vs_pr_style_pct_median"] = shape_summaries[0][
            "native_vs_pr_style_pct_median"
        ]
    return summary


def execute_run(args, mode, mode_env, shape, repeat_index):
    run_slug = "%s/%s/repeat_%02d" % (safe_part(mode), shape_slug(shape), repeat_index)
    run_dir = os.path.join(args.out_dir, run_slug)
    result_filename = "benchmark.json"
    result_path = os.path.join(run_dir, result_filename)
    server_log_path = os.path.join(run_dir, "server.log")
    benchmark_log_path = os.path.join(run_dir, "benchmark.log")

    server_cmd = build_server_command(args, shape)
    benchmark_cmd = build_benchmark_command(
        args,
        shape,
        args.benchmark_runner_selected,
        run_dir,
        result_filename,
    )

    env = os.environ.copy()
    env.update(mode_env)
    trace_jsonl = None
    if args.consumer_state_trace_dir:
        trace_jsonl = os.path.join(args.consumer_state_trace_dir, run_slug + ".jsonl")
        ensure_dir(os.path.dirname(trace_jsonl))
        env["VLLM_CONSUMER_STATE_TRACE_JSONL"] = trace_jsonl
    cache_base = os.environ.get("VLLM_FLASHDENOISE_BENCH_CACHE_ROOT")
    if cache_base:
        cache_base = os.path.join(cache_base, safe_part(run_slug))
    else:
        cache_base = run_dir
    cache_env = {
        "TRITON_CACHE_DIR": os.path.join(cache_base, "triton_cache"),
        "TORCHINDUCTOR_CACHE_DIR": os.path.join(cache_base, "torchinductor_cache"),
        "CUDA_CACHE_PATH": os.path.join(cache_base, "cuda_cache"),
    }
    env.update(cache_env)

    run = {
        "mode": mode,
        "mode_env": dict(mode_env),
        "cache_env": dict(cache_env),
        "shape": dict(shape),
        "shape_key": shape_key(shape),
        "repeat": repeat_index,
        "run_dir": run_dir,
        "server_command": server_cmd,
        "benchmark_command": benchmark_cmd,
        "server_command_text": env_command_to_string(mode_env, server_cmd),
        "benchmark_command_text": env_command_to_string(mode_env, benchmark_cmd),
        "server_log": server_log_path,
        "benchmark_log": benchmark_log_path,
        "result_json": result_path,
        "consumer_state_trace_jsonl": trace_jsonl,
        "started_at": utc_now(),
    }

    if args.dry_run:
        print("[dry-run] %s" % run_slug)
        print("  server:    %s" % run["server_command_text"])
        print("  benchmark: %s" % run["benchmark_command_text"])
        run["ended_at"] = utc_now()
        run["dry_run"] = True
        return run

    ensure_dir(run_dir)
    for cache_dir in cache_env.values():
        ensure_dir(cache_dir)
    url = health_url(args.host, args.port)
    server_proc = None
    server_log = None
    try:
        if is_healthy(url, args.health_probe_timeout):
            raise RuntimeError(
                "%s is already healthy before launch; refusing to benchmark "
                "against an existing server" % url
            )

        server_proc, server_log = launch_server(
            server_cmd,
            env,
            args.repo_root,
            server_log_path,
        )
        run["server_pid"] = server_proc.pid
        run["server_started_at"] = utc_now()
        wait_for_health(
            server_proc,
            url,
            args.health_timeout,
            args.health_probe_timeout,
        )
        run["health_ready_at"] = utc_now()

        bench_result = run_logged_command(
            benchmark_cmd,
            env,
            args.repo_root,
            benchmark_log_path,
        )
        run["benchmark_started_at"] = bench_result["started_at"]
        run["benchmark_ended_at"] = bench_result["ended_at"]
        run["benchmark_returncode"] = bench_result["returncode"]
        if bench_result["returncode"] != 0:
            raise RuntimeError(
                "benchmark command failed with returncode=%s; see %s"
                % (bench_result["returncode"], benchmark_log_path)
            )
        if not os.path.exists(result_path):
            raise RuntimeError("benchmark did not write JSON result: %s" % result_path)

        result_json = read_json(result_path)
        run["metrics"] = extract_metrics(result_json)
        run["benchmark_json_keys"] = sorted(result_json.keys())
        if server_log is not None:
            server_log.flush()
        run.update(inspect_server_log(server_log_path))
        if mode == "native_tp_state":
            if run["native_tp_state_fallback_warnings"]:
                raise RuntimeError(
                    "native_tp_state server logged fallback warning(s); see %s"
                    % server_log_path
                )
            if not run["native_tp_state_confirmed"]:
                raise RuntimeError(
                    "native_tp_state did not log the native TP-state path; see %s"
                    % server_log_path
                )
    except Exception as exc:
        run["error"] = str(exc)
    finally:
        if os.path.exists(server_log_path):
            for key, value in inspect_server_log(server_log_path).items():
                run.setdefault(key, value)
        if server_log is not None:
            server_log.flush()
            server_log.close()
        if server_proc is not None:
            terminate_process_tree(server_proc, args.shutdown_timeout)
            run["server_returncode_after_cleanup"] = server_proc.poll()
        run["ended_at"] = utc_now()
    return run


def create_argument_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compare DiffusionGemma FlashDenoise sampler modes with repeated "
            "TP serving benchmarks."
        )
    )
    parser.add_argument("--model", required=True, help="Model path or HF model id.")
    parser.add_argument(
        "--served-model-name",
        default="diffusion-gemma-flashdenoise-tp4",
        help="Stable model name exposed by the temporary OpenAI server.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=4,
        help="vLLM tensor parallel size for the server.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--concurrency",
        "--concurrency-list",
        dest="concurrency",
        nargs="+",
        default=["32"],
        help="Comma-separated and/or space-separated max concurrency values.",
    )
    parser.add_argument(
        "--max-num-seqs",
        "--max-num-seqs-list",
        dest="max_num_seqs",
        nargs="+",
        default=["16"],
        help="Comma-separated and/or space-separated server max_num_seqs values.",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        "--max-num-batched-tokens-list",
        dest="max_num_batched_tokens",
        nargs="+",
        default=None,
        help=(
            "Optional comma-separated and/or space-separated "
            "max_num_batched_tokens values. Use 'none' to omit the server flag."
        ),
    )
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--input-len", type=int, default=32)
    parser.add_argument("--output-len", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--consumer-state-trace-dir",
        default=None,
        help=(
            "Optional directory for per-run VLLM_CONSUMER_STATE_TRACE_JSONL "
            "runtime vocab-state traces."
        ),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[",".join(DEFAULT_MODE_ORDER)],
        help=(
            "Comma-separated and/or space-separated modes to run. "
            "Default omits pr_style_soft_embed_equivalent because it is only "
            "valid when explicitly run from a literal PR-style worktree."
        ),
    )
    parser.add_argument(
        "--native-mode-flags",
        type=int,
        default=16,
        help="Value for %s in native/fused modes." % NATIVE_MODE_FLAGS_ENV,
    )
    parser.add_argument(
        "--request-rate",
        default="inf",
        help="Request rate passed to the serving benchmark.",
    )
    parser.add_argument(
        "--enable-pr-style-equivalent-comparison",
        action="store_true",
        help=(
            "Compute native_vs_pr_style_pct_median. Use only when this harness "
            "is run from a literal PR-style worktree or an equivalent external "
            "baseline has been substituted."
        ),
    )
    parser.add_argument(
        "--backend",
        default="openai",
        help="Benchmark backend passed to vLLM serving benchmark.",
    )
    parser.add_argument("--endpoint", default="/v1/completions")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--benchmark-command",
        choices=["auto", "script", "cli"],
        default="auto",
        help=(
            "auto prefers benchmarks/benchmark_serving.py only when it is not "
            "the deprecated wrapper, otherwise uses vllm bench serve."
        ),
    )
    parser.add_argument(
        "--python-executable",
        default=default_python_executable(),
        help="Python executable used to launch vLLM server and benchmark commands.",
    )
    parser.add_argument(
        "--vllm-cli-module",
        default="vllm.entrypoints.cli.main",
        help="Module used for `python -m ... serve` and `python -m ... bench serve`.",
    )
    parser.add_argument(
        "--server-arg",
        action="append",
        default=[],
        help="Extra single argument appended to the vLLM serve command.",
    )
    parser.add_argument(
        "--benchmark-arg",
        action="append",
        default=[],
        help="Extra single argument appended to the benchmark command.",
    )
    parser.add_argument("--health-timeout", type=float, default=900.0)
    parser.add_argument("--health-probe-timeout", type=float, default=2.0)
    parser.add_argument("--shutdown-timeout", type=float, default=30.0)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record failed runs and continue with the remaining matrix.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands for the full matrix without launching subprocesses.",
    )
    return parser


def normalize_args(args):
    args.repo_root = repo_root()
    args.modes = parse_modes(args.modes)
    args.concurrency_values = parse_int_list(args.concurrency, "--concurrency")
    args.max_num_seqs_values = parse_int_list(args.max_num_seqs, "--max-num-seqs")
    args.max_num_batched_tokens_values = parse_optional_int_list(
        args.max_num_batched_tokens,
        "--max-num-batched-tokens",
    )
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.tensor_parallel_size < 1:
        raise ValueError("--tensor-parallel-size must be >= 1")
    args.benchmark_runner_selected = select_benchmark_runner(
        args.repo_root,
        args.benchmark_command,
    )
    return args


def main():
    parser = create_argument_parser()
    args = normalize_args(parser.parse_args())
    mode_envs = build_mode_envs(args.native_mode_flags)
    shapes = build_shapes(
        args.concurrency_values,
        args.max_num_seqs_values,
        args.max_num_batched_tokens_values,
    )

    started_at = utc_now()
    runs = []

    if args.dry_run:
        print(
            "Dry run: %d mode(s) x %d shape(s) x %d repeat(s); "
            "benchmark_runner=%s"
            % (
                len(args.modes),
                len(shapes),
                args.repeats,
                args.benchmark_runner_selected,
            )
        )

    for mode in args.modes:
        for shape in shapes:
            for repeat_index in range(1, args.repeats + 1):
                run = execute_run(args, mode, mode_envs[mode], shape, repeat_index)
                runs.append(run)

                if args.dry_run:
                    continue

                summary = build_summary(
                    args,
                    mode_envs,
                    shapes,
                    runs,
                    started_at,
                    utc_now(),
                )
                write_json(os.path.join(args.out_dir, "summary.json"), summary)

                if run.get("error") and not args.continue_on_error:
                    print("ERROR: %s" % run["error"], file=sys.stderr)
                    print(
                        "Partial summary written to %s"
                        % os.path.join(args.out_dir, "summary.json"),
                        file=sys.stderr,
                    )
                    return 1

    if args.dry_run:
        return 0

    completed_at = utc_now()
    summary = build_summary(args, mode_envs, shapes, runs, started_at, completed_at)
    write_json(os.path.join(args.out_dir, "summary.json"), summary)
    print("summary: %s" % os.path.join(args.out_dir, "summary.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
