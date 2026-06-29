#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

import vllm._custom_ops as ops


REQUIRED_OPS = [
    "diffusion_gemma_flashdenoise",
    "diffusion_gemma_flashdenoise_scaled",
    "diffusion_gemma_flashdenoise_local_state_scaled",
    "diffusion_gemma_flashdenoise_pack_local_state",
]


def op_status() -> list[dict[str, object]]:
    rows = []
    for name in REQUIRED_OPS:
        rows.append(
            {
                "name": name,
                "python_wrapper": hasattr(ops, name),
                "native_registered": hasattr(torch.ops, "_C")
                and hasattr(torch.ops._C, name),
            }
        )
    return rows


def run_pytest(pytest_args: list[str]) -> dict[str, object]:
    command = [sys.executable, "-m", "pytest", *pytest_args]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def pytest_reported_skip(pytest_result: dict[str, object]) -> bool:
    output = "\n".join(
        str(pytest_result.get(key, "")) for key in ("stdout", "stderr")
    ).lower()
    return "skipped" in output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--run-native-tests",
        action="store_true",
        help="Run native dense-reference pytest after checking op registration.",
    )
    parser.add_argument(
        "--native-test-path",
        default="tests/models/test_diffusion_gemma_flashdenoise_native_tp.py",
        help="Path to the native dense-reference pytest file.",
    )
    args = parser.parse_args()

    rows = op_status()
    all_registered = all(row["native_registered"] for row in rows)
    payload: dict[str, object] = {
        "required_ops": rows,
        "all_native_registered": all_registered,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if args.run_native_tests:
        if torch.cuda.is_available():
            payload["pytest"] = run_pytest(["-q", "-rs", args.native_test_path])
        else:
            payload["native_tests_error"] = (
                "CUDA is unavailable; native CUDA correctness tests did not run"
            )

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if not all_registered:
        print(json.dumps(payload, indent=2))
        return 2
    if args.run_native_tests:
        if not torch.cuda.is_available():
            print(json.dumps(payload, indent=2))
            return 3
        pytest_result = payload["pytest"]
        assert isinstance(pytest_result, dict)
        pytest_returncode = int(pytest_result["returncode"])
        if pytest_returncode != 0:
            return pytest_returncode
        if pytest_reported_skip(pytest_result):
            print(json.dumps(payload, indent=2))
            return 4
        return 0
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
