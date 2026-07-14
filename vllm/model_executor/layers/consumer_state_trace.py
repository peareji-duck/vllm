# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in output-phase memory tracing for consumer-state experiments."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, Protocol, runtime_checkable

import torch

import vllm.envs as envs
from vllm.logger import init_logger

_TRACE_LOCK = threading.Lock()
logger = init_logger(__name__)
_CONSUMER_SOURCE_FIELDS = (
    "sample",
    "clean_argmax",
    "entropy_remask",
    "soft_self_conditioning",
)
_CONTRACT_METADATA_FIELDS = (
    "effective_variant",
    "consumer_contract",
    "pareto_role",
    "pareto_gate_eligible",
)
_BATCH_PHASES = (
    "prefill_only",
    "consumer_output_only",
    "mixed_prefill_consumer_output",
)


@runtime_checkable
class ConsumerStateBatchPhaseProvider(Protocol):
    """Sampler capability required by opt-in consumer-state tracing."""

    def trace_batch_phase(self, input_batch: Any) -> str: ...


def _require_batch_phase(batch_phase: object) -> str:
    if type(batch_phase) is not str or batch_phase not in _BATCH_PHASES:
        allowed = ", ".join(_BATCH_PHASES)
        raise RuntimeError(
            "Consumer-state trace requires batch_phase to be one of: " + allowed
        )
    return batch_phase


def peak_memory_trace_path() -> str:
    return envs.VLLM_CONSUMER_STATE_PEAK_MEMORY_TRACE_JSONL


def resolve_rank_trace_path(configured_path: str, global_rank: int) -> str:
    """Resolve one deterministic output file for each global rank."""
    has_rank_placeholder = any(
        placeholder in configured_path for placeholder in ("{rank}", "{global_rank}")
    )
    path = configured_path.replace("{global_rank}", str(global_rank)).replace(
        "{rank}", str(global_rank)
    )
    if has_rank_placeholder:
        return path
    root, extension = os.path.splitext(path)
    if extension:
        return f"{root}.rank{global_rank}{extension}"
    return f"{path}.rank{global_rank}.jsonl"


def _trace_process_metadata() -> dict[str, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        global_rank = torch.distributed.get_rank()
    else:
        global_rank = int(os.getenv("RANK", "0"))

    local_rank_value = os.getenv("LOCAL_RANK")
    local_rank = (
        int(local_rank_value)
        if local_rank_value is not None
        else torch.cuda.current_device()
    )
    from vllm.distributed.parallel_state import (
        get_tensor_model_parallel_rank,
        model_parallel_is_initialized,
    )

    if model_parallel_is_initialized():
        tp_rank = get_tensor_model_parallel_rank()
    else:
        tp_rank = int(os.getenv("TP_RANK", str(global_rank)))

    return {
        "global_rank": global_rank,
        "local_rank": local_rank,
        "tp_rank": tp_rank,
        "pid": os.getpid(),
        "cuda_device": torch.cuda.current_device(),
    }


def _consumer_state_attribution_issue(
    consumer_sources: Mapping[str, str] | None,
    contract_metadata: Mapping[str, Any] | None,
    *,
    expected_variant: str | None = None,
) -> tuple[str, str] | None:
    missing_sources = [
        field
        for field in _CONSUMER_SOURCE_FIELDS
        if consumer_sources is None or field not in consumer_sources
    ]
    missing_metadata = [
        field
        for field in _CONTRACT_METADATA_FIELDS
        if contract_metadata is None or field not in contract_metadata
    ]
    if missing_sources or missing_metadata:
        details = []
        if missing_sources:
            details.append(f"sources={','.join(missing_sources)}")
        if missing_metadata:
            details.append(f"metadata={','.join(missing_metadata)}")
        return "missing", "; ".join(details)

    assert consumer_sources is not None
    assert contract_metadata is not None
    invalid_sources = [
        field
        for field in _CONSUMER_SOURCE_FIELDS
        if not isinstance(consumer_sources[field], str) or not consumer_sources[field]
    ]
    invalid_metadata = [
        field
        for field in ("effective_variant", "consumer_contract", "pareto_role")
        if not isinstance(contract_metadata[field], str) or not contract_metadata[field]
    ]
    if type(contract_metadata["pareto_gate_eligible"]) is not bool:
        invalid_metadata.append("pareto_gate_eligible")
    if invalid_sources or invalid_metadata:
        details = []
        if invalid_sources:
            details.append(f"sources={','.join(invalid_sources)}")
        if invalid_metadata:
            details.append(f"metadata={','.join(invalid_metadata)}")
        return "invalid", "; ".join(details)

    pareto_role = contract_metadata["pareto_role"]
    pareto_gate_eligible = contract_metadata["pareto_gate_eligible"]
    if (pareto_role == "dominator_candidate") != pareto_gate_eligible:
        return "invalid", "pareto_role and pareto_gate_eligible disagree"
    effective_variant = contract_metadata["effective_variant"]
    if expected_variant is not None and effective_variant != expected_variant:
        return (
            "invalid",
            f"effective_variant={effective_variant!r}, expected={expected_variant!r}",
        )
    return None


def require_consumer_state_attribution(
    sampler: Any,
    seam_name: str,
    *seam_args: Any,
    expected_variant: str | None = None,
) -> tuple[dict[str, str], dict[str, str | bool]]:
    """Resolve complete attribution or invalidate the benchmark run."""
    seam = getattr(sampler, seam_name, None)
    if not callable(seam):
        raise RuntimeError(
            "Consumer-state benchmark attribution requires callable "
            f"sampler.{seam_name}(); benchmark run is invalid"
        )
    try:
        spec = seam(*seam_args)
        consumer_sources = dict(spec.as_source_tags())
        contract_metadata = dict(spec.as_contract_metadata())
    except Exception as exc:
        raise RuntimeError(
            f"Consumer-state attribution seam sampler.{seam_name}() failed; "
            "benchmark run is invalid"
        ) from exc

    issue = _consumer_state_attribution_issue(
        consumer_sources,
        contract_metadata,
        expected_variant=expected_variant,
    )
    if issue is not None:
        status, detail = issue
        raise RuntimeError(
            f"sampler.{seam_name}() returned incomplete consumer-state "
            f"attribution ({status}: {detail}); benchmark run is invalid"
        )
    return consumer_sources, contract_metadata


def require_consumer_state_batch_phase(sampler: object, input_batch: Any) -> str:
    """Resolve a causal batch phase or invalidate the benchmark run."""
    if not isinstance(sampler, ConsumerStateBatchPhaseProvider):
        raise RuntimeError(
            "Consumer-state tracing requires callable "
            "sampler.trace_batch_phase(); benchmark run is invalid"
        )
    try:
        batch_phase = sampler.trace_batch_phase(input_batch)
    except Exception as exc:
        raise RuntimeError(
            "Consumer-state batch phase seam sampler.trace_batch_phase() failed; "
            "benchmark run is invalid"
        ) from exc
    try:
        return _require_batch_phase(batch_phase)
    except RuntimeError as exc:
        raise RuntimeError(
            "sampler.trace_batch_phase() returned invalid batch_phase; "
            "benchmark run is invalid"
        ) from exc


def build_output_phase_peak_record(
    *,
    component: str,
    variant: str,
    batch_phase: str | None = None,
    global_rank: int,
    local_rank: int,
    tp_rank: int,
    pid: int,
    cuda_device: int,
    start_allocated: int,
    start_reserved: int,
    peak_allocated: int,
    peak_reserved: int,
    component_elapsed_seconds: float,
    consumer_sources: Mapping[str, str] | None = None,
    contract_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    batch_phase = _require_batch_phase(batch_phase)
    attribution_issue = _consumer_state_attribution_issue(
        consumer_sources, contract_metadata
    )
    attribution_status = (
        "complete" if attribution_issue is None else attribution_issue[0]
    )
    effective_variant = (
        str(contract_metadata["effective_variant"])
        if attribution_issue is None and contract_metadata is not None
        else "unattributed"
    )
    record: dict[str, Any] = {
        "schema_version": 4,
        "timestamp_unix_s": time.time(),
        "framework": "vllm",
        "component": component,
        "batch_phase": batch_phase,
        "variant": effective_variant,
        "configured_variant": variant,
        "attribution_status": attribution_status,
        "global_rank": int(global_rank),
        "local_rank": int(local_rank),
        "tp_rank": int(tp_rank),
        "pid": int(pid),
        "cuda_device": int(cuda_device),
        "path": "output_phase_peak_hbm",
        "start_allocated_bytes": int(start_allocated),
        "start_reserved_bytes": int(start_reserved),
        "peak_allocated_bytes": int(peak_allocated),
        "peak_reserved_bytes": int(peak_reserved),
        "peak_allocated_delta_bytes": max(
            0, int(peak_allocated) - int(start_allocated)
        ),
        "component_elapsed_ms": max(0.0, float(component_elapsed_seconds) * 1000.0),
    }
    if attribution_issue is None and consumer_sources is not None:
        record["consumer_sources"] = dict(consumer_sources)
    if attribution_issue is None and contract_metadata is not None:
        if "consumer_contract" in contract_metadata:
            record["consumer_contract"] = str(contract_metadata["consumer_contract"])
        if "pareto_role" in contract_metadata:
            record["pareto_role"] = str(contract_metadata["pareto_role"])
        if "pareto_gate_eligible" in contract_metadata:
            record["pareto_gate_eligible"] = bool(
                contract_metadata["pareto_gate_eligible"]
            )
    return record


def _append_jsonl(path: str, record: Mapping[str, Any]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    line = json.dumps(record, sort_keys=True, separators=(",", ":"))
    with _TRACE_LOCK, open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


@contextmanager
def trace_output_phase_peak(
    component: str,
    *,
    batch_phase: str | None = None,
    consumer_sources: Mapping[str, str] | None = None,
    contract_metadata: Mapping[str, Any] | None = None,
    variant: str | None = None,
) -> Iterator[None]:
    """Trace output-head peak HBM only when a profile path is configured."""
    configured_path = peak_memory_trace_path()
    if not configured_path:
        yield
        return

    batch_phase = _require_batch_phase(batch_phase)
    metadata = _trace_process_metadata()
    path = resolve_rank_trace_path(configured_path, metadata["global_rank"])
    if variant is None:
        variant = envs.VLLM_DIFFUSION_GEMMA_VALIDATION_VARIANT
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_allocated = torch.cuda.memory_allocated()
    start_reserved = torch.cuda.memory_reserved()
    component_start = time.perf_counter()
    body_error: BaseException | None = None
    try:
        yield
    except BaseException as exc:
        body_error = exc
        raise
    finally:
        try:
            torch.cuda.synchronize()
            component_elapsed_seconds = time.perf_counter() - component_start
            record = build_output_phase_peak_record(
                component=component,
                variant=variant,
                batch_phase=batch_phase,
                start_allocated=start_allocated,
                start_reserved=start_reserved,
                peak_allocated=torch.cuda.max_memory_allocated(),
                peak_reserved=torch.cuda.max_memory_reserved(),
                component_elapsed_seconds=component_elapsed_seconds,
                consumer_sources=consumer_sources,
                contract_metadata=contract_metadata,
                **metadata,
            )
            _append_jsonl(path, record)
        except BaseException:
            if body_error is None:
                raise
            logger.exception(
                "Output-phase trace cleanup failed while preserving the "
                "sampling exception"
            )
