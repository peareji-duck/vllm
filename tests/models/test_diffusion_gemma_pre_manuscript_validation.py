# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

import vllm.model_executor.layers.consumer_state_trace as consumer_state_trace
import vllm.model_executor.models.diffusion_gemma as diffusion_gemma
import vllm.v1.worker.gpu.model_runner as model_runner
from vllm.model_executor.layers.consumer_state_trace import (
    build_output_phase_peak_record,
    peak_memory_trace_path,
    trace_output_phase_peak,
)
from vllm.model_executor.models.diffusion_gemma import (
    DiffusionGemmaRequestStates,
    DiffusionSampler,
    get_token_microbatch_rows,
    get_validation_variant,
    validation_fallback_reason,
    validation_variant_spec,
)
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.output import SamplerOutput


def _dense_consumer_oracle(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    embed_weight: torch.Tensor,
    gumbel: torch.Tensor,
):
    logits = F.linear(hidden_states, lm_head_weight).float()
    log_probs = logits.log_softmax(dim=-1)
    probs = log_probs.exp()
    return {
        "clean_tokens": logits.argmax(dim=-1),
        "sample_tokens": (logits + gumbel).argmax(dim=-1),
        "entropy": -(probs * log_probs).sum(dim=-1),
        "soft_embeds": probs @ embed_weight.float(),
    }


@dataclass
class _DenseConsumerOutputs:
    clean_tokens: torch.Tensor
    sample_tokens: torch.Tensor
    entropy: torch.Tensor
    soft_embeds: torch.Tensor


def _token_microbatch_consumer_outputs(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    embed_weight: torch.Tensor,
    *,
    gumbel_for_rows,
    chunk_rows: int,
) -> _DenseConsumerOutputs:
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")

    rows = hidden_states.shape[0]
    clean = torch.empty(rows, dtype=torch.int64, device=hidden_states.device)
    sampled = torch.empty_like(clean)
    entropy = torch.empty(rows, dtype=torch.float32, device=hidden_states.device)
    soft = torch.empty(
        rows,
        embed_weight.shape[1],
        dtype=torch.float32,
        device=hidden_states.device,
    )
    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        logits = F.linear(hidden_states[start:stop], lm_head_weight).float()
        log_probs = logits.log_softmax(dim=-1)
        probs = log_probs.exp()
        clean[start:stop] = logits.argmax(dim=-1)
        sampled[start:stop] = (
            logits + gumbel_for_rows(start, stop).to(logits.dtype)
        ).argmax(dim=-1)
        entropy[start:stop] = -(probs * log_probs).sum(dim=-1)
        soft[start:stop] = probs @ embed_weight.float()
    return _DenseConsumerOutputs(clean, sampled, entropy, soft)


def test_validation_controls_are_default_off(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("VLLM_DIFFUSION_GEMMA_VALIDATION_VARIANT", raising=False)
    monkeypatch.delenv("VLLM_DIFFUSION_GEMMA_TOKEN_MICROBATCH_ROWS", raising=False)
    monkeypatch.delenv("VLLM_CONSUMER_STATE_PEAK_MEMORY_TRACE_JSONL", raising=False)

    assert get_validation_variant() == "off"
    assert get_token_microbatch_rows() == 0
    assert peak_memory_trace_path() == ""


@pytest.mark.parametrize(
    "variant",
    [
        "off",
        "token_axis_logit_microbatch",
        "sample_only_dense_consumers",
        "sample_entropy_state_dense_soft_embed",
    ],
)
def test_validation_variant_accepts_only_declared_choices(
    monkeypatch: pytest.MonkeyPatch, variant: str
):
    monkeypatch.setenv("VLLM_DIFFUSION_GEMMA_VALIDATION_VARIANT", variant)
    assert get_validation_variant() == variant

    monkeypatch.setenv("VLLM_DIFFUSION_GEMMA_VALIDATION_VARIANT", "not-a-variant")
    with pytest.raises(ValueError, match="VLLM_DIFFUSION_GEMMA_VALIDATION_VARIANT"):
        get_validation_variant()


def test_token_microbatch_matches_dense_fixed_gumbel():
    generator = torch.Generator(device="cpu").manual_seed(7)
    hidden = torch.randn(96, 23, generator=generator)
    head = torch.randn(257, 23, generator=generator)
    embed = torch.randn(257, 19, generator=generator)
    gumbel = torch.randn(96, 257, generator=generator)

    dense = _dense_consumer_oracle(hidden, head, embed, gumbel)
    chunked = _token_microbatch_consumer_outputs(
        hidden,
        head,
        embed,
        gumbel_for_rows=lambda start, stop: gumbel[start:stop],
        chunk_rows=11,
    )

    assert torch.equal(chunked.clean_tokens, dense["clean_tokens"])
    assert torch.equal(chunked.sample_tokens, dense["sample_tokens"])
    torch.testing.assert_close(chunked.entropy, dense["entropy"], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        chunked.soft_embeds, dense["soft_embeds"], rtol=1e-5, atol=1e-6
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_token_microbatch_matches_dense_fixed_gumbel_cuda_bfloat16():
    torch.manual_seed(7)
    hidden = torch.randn(96, 23, device="cuda", dtype=torch.bfloat16)
    head = torch.randn(257, 23, device="cuda", dtype=torch.bfloat16)
    embed = torch.randn(257, 19, device="cuda", dtype=torch.bfloat16)
    gumbel = torch.randn(96, 257, device="cuda", dtype=torch.float32)

    dense = _dense_consumer_oracle(hidden, head, embed, gumbel)
    chunked = _token_microbatch_consumer_outputs(
        hidden,
        head,
        embed,
        gumbel_for_rows=lambda start, stop: gumbel[start:stop],
        chunk_rows=11,
    )

    assert torch.equal(chunked.clean_tokens, dense["clean_tokens"])
    assert torch.equal(chunked.sample_tokens, dense["sample_tokens"])
    torch.testing.assert_close(chunked.entropy, dense["entropy"], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        chunked.soft_embeds, dense["soft_embeds"], rtol=1e-5, atol=1e-6
    )


def test_token_microbatch_rejects_nonpositive_chunk_rows():
    hidden = torch.zeros(2, 3)
    head = torch.zeros(5, 3)
    embed = torch.zeros(5, 7)

    with pytest.raises(ValueError, match="chunk_rows must be positive"):
        _token_microbatch_consumer_outputs(
            hidden,
            head,
            embed,
            gumbel_for_rows=lambda start, stop: torch.zeros(stop - start, 5),
            chunk_rows=0,
        )


def test_fixed_gumbel_oracle_is_not_a_production_surface():
    assert not hasattr(diffusion_gemma, "DenseConsumerOutputs")
    assert not hasattr(diffusion_gemma, "token_microbatch_consumer_outputs")


def test_intermediate_variant_keeps_only_soft_embed_dense():
    spec = validation_variant_spec("sample_entropy_state_dense_soft_embed")

    assert spec.sample == "local_state"
    assert spec.clean_argmax == "local_state"
    assert spec.entropy_remask == "local_state"
    assert spec.soft_self_conditioning == "dense_logits"


def test_sample_only_variant_keeps_only_sample_local_and_is_not_pareto_eligible():
    spec = validation_variant_spec("sample_only_dense_consumers")

    assert spec.sample == "local_state"
    assert spec.clean_argmax == "dense_logits"
    assert spec.entropy_remask == "dense_logits"
    assert spec.soft_self_conditioning == "dense_logits"
    assert spec.consumer_contract == "vllm_diffusiongemma"
    assert spec.pareto_role == "ablation_only"
    assert spec.pareto_gate_eligible is False
    assert (
        validation_variant_spec("token_axis_logit_microbatch").pareto_gate_eligible
        is True
    )


@pytest.mark.parametrize(
    ("variant", "role", "eligible"),
    [
        ("off", "baseline_context", False),
        ("full_consumer_state", "reference", False),
        ("token_axis_logit_microbatch", "dominator_candidate", True),
        ("sample_only_dense_consumers", "ablation_only", False),
        ("sample_entropy_state_dense_soft_embed", "ablation_only", False),
    ],
)
def test_variant_specs_define_unambiguous_pareto_roles(
    variant: str, role: str, eligible: bool
):
    spec = validation_variant_spec(variant)

    assert spec.effective_variant == variant
    assert spec.consumer_contract == "vllm_diffusiongemma"
    assert spec.pareto_role == role
    assert spec.pareto_gate_eligible is eligible
    assert (role == "dominator_candidate") is eligible


def test_variant_specs_tag_every_consumer():
    for variant in (
        "off",
        "token_axis_logit_microbatch",
        "sample_only_dense_consumers",
        "sample_entropy_state_dense_soft_embed",
    ):
        spec = validation_variant_spec(variant)
        assert all(
            getattr(spec, field)
            for field in (
                "sample",
                "clean_argmax",
                "entropy_remask",
                "soft_self_conditioning",
            )
        )


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"chunk_rows": 0}, "nonpositive_chunk_rows"),
        ({"requires_full_logprobs": True}, "full_logprobs"),
        ({"has_grammar": True}, "grammar"),
        ({"has_rejection_sampler": True}, "rejection_sampling"),
        ({"actual_rows": 63}, "malformed_rows"),
    ],
)
def test_token_microbatch_unsupported_inputs_select_dense_fallback(
    overrides: dict[str, object], reason: str
):
    inputs = {
        "chunk_rows": 32,
        "requires_full_logprobs": False,
        "has_grammar": False,
        "has_rejection_sampler": False,
        "expected_rows": 64,
        "actual_rows": 64,
    }
    inputs.update(overrides)

    assert validation_fallback_reason(**inputs) == reason


def test_token_microbatch_supported_inputs_do_not_fallback():
    assert (
        validation_fallback_reason(
            chunk_rows=32,
            requires_full_logprobs=False,
            has_grammar=False,
            has_rejection_sampler=False,
            expected_rows=64,
            actual_rows=64,
        )
        is None
    )


@pytest.mark.parametrize(
    ("per_request_logits", "num_draft_tokens", "expected_phase"),
    [
        ([1, 1], 0, "prefill_only"),
        ([3, 2], 5, "consumer_output_only"),
        ([0, 3, 0, 2], 5, "mixed_prefill_consumer_output"),
    ],
)
def test_diffusion_sampler_trace_batch_phase_classifies_logits_layout(
    per_request_logits: list[int], num_draft_tokens: int, expected_phase: str
):
    sampler = DiffusionSampler.__new__(DiffusionSampler)
    cumulative_logits = np.concatenate(
        ([0], np.cumsum(per_request_logits, dtype=np.int64))
    )
    input_batch = SimpleNamespace(
        num_reqs=len(per_request_logits),
        num_draft_tokens=num_draft_tokens,
        cu_num_logits_np=cumulative_logits,
    )

    assert sampler.trace_batch_phase(input_batch) == expected_phase


def test_trace_batch_phase_groups_commit_and_denoise_as_consumer_output():
    sampler = DiffusionSampler.__new__(DiffusionSampler)

    class ForbiddenStateRead:
        def __getattr__(self, name: str):
            raise AssertionError(f"batch phase must not synchronize GPU state: {name}")

    sampler.diffusion_states = ForbiddenStateRead()
    input_batch = SimpleNamespace(
        num_reqs=2,
        num_draft_tokens=5,
        cu_num_logits_np=np.array([0, 3, 5], dtype=np.int32),
    )

    # Positive-logit consumers include both denoise and commit steps. The
    # structural trace category deliberately covers both without a GPU sync.
    assert sampler.trace_batch_phase(input_batch) == "consumer_output_only"


@pytest.mark.parametrize(
    ("num_reqs", "num_draft_tokens", "cumulative_logits", "error"),
    [
        (0, 0, np.array([0], dtype=np.int32), "positive num_reqs"),
        (1, 0, np.array([], dtype=np.int32), r"num_reqs \+ 1"),
        (1, 0, np.array([[0, 1]], dtype=np.int32), "one-dimensional"),
        (1, 0, np.array([1, 2], dtype=np.int32), "start at zero"),
        (2, 1, np.array([0, 2, 1], dtype=np.int32), "negative"),
        (2, 1, np.array([0, 127, -128], dtype=np.int8), "negative"),
        (1, 0, np.array([0.0, 1.0]), "integer"),
        (1, -1, np.array([0, 1], dtype=np.int32), "non-negative"),
        (1, 0, np.array([0, 0], dtype=np.int32), "no-draft"),
        (1, 0, np.array([0, 2], dtype=np.int32), "no-draft"),
        (
            1,
            0,
            np.array([0, 1, 99], dtype=np.int32),
            r"exactly num_reqs \+ 1",
        ),
        (2, 1, np.array([0, 1, 2], dtype=np.int32), "must equal"),
    ],
)
def test_diffusion_sampler_trace_batch_phase_rejects_invalid_logits_layout(
    num_reqs: int,
    num_draft_tokens: int,
    cumulative_logits: np.ndarray,
    error: str,
):
    sampler = DiffusionSampler.__new__(DiffusionSampler)
    input_batch = SimpleNamespace(
        num_reqs=num_reqs,
        num_draft_tokens=num_draft_tokens,
        cu_num_logits_np=cumulative_logits,
    )

    with pytest.raises(ValueError, match=error):
        sampler.trace_batch_phase(input_batch)


def test_memory_trace_record_emits_phase_delta():
    spec = validation_variant_spec("full_consumer_state")
    record = build_output_phase_peak_record(
        component="diffusion_gemma",
        variant="off",
        batch_phase="mixed_prefill_consumer_output",
        global_rank=3,
        local_rank=1,
        tp_rank=1,
        pid=1234,
        cuda_device=1,
        start_allocated=100,
        start_reserved=120,
        peak_allocated=180,
        peak_reserved=240,
        component_elapsed_seconds=0.0125,
        consumer_sources=spec.as_source_tags(),
        contract_metadata={
            "effective_variant": "full_consumer_state",
            "consumer_contract": "vllm_diffusiongemma",
            "pareto_role": "reference",
            "pareto_gate_eligible": False,
        },
    )

    assert record["schema_version"] == 4
    assert record["batch_phase"] == "mixed_prefill_consumer_output"
    assert record["component"] == "diffusion_gemma"
    assert record["path"] == "output_phase_peak_hbm"
    assert record["peak_allocated_delta_bytes"] == 80
    assert record["peak_reserved_bytes"] == 240
    assert record["attribution_status"] == "complete"
    assert record["variant"] == "full_consumer_state"
    assert record["configured_variant"] == "off"
    assert record["global_rank"] == 3
    assert record["local_rank"] == 1
    assert record["tp_rank"] == 1
    assert record["pid"] == 1234
    assert record["cuda_device"] == 1
    assert record["component_elapsed_ms"] == 12.5
    assert record["consumer_contract"] == "vllm_diffusiongemma"
    assert record["pareto_role"] == "reference"
    assert record["pareto_gate_eligible"] is False


def test_memory_trace_does_not_treat_configured_variant_as_executed():
    record = build_output_phase_peak_record(
        component="diffusion_gemma",
        variant="sample_only_dense_consumers",
        batch_phase="prefill_only",
        global_rank=0,
        local_rank=0,
        tp_rank=0,
        pid=1234,
        cuda_device=0,
        start_allocated=100,
        start_reserved=120,
        peak_allocated=180,
        peak_reserved=240,
        component_elapsed_seconds=0.01,
        consumer_sources={},
        contract_metadata={},
    )

    assert record["configured_variant"] == "sample_only_dense_consumers"
    assert record["variant"] == "unattributed"
    assert record["attribution_status"] == "missing"


def test_memory_trace_record_rejects_missing_batch_phase():
    with pytest.raises(RuntimeError, match="batch_phase"):
        build_output_phase_peak_record(
            component="diffusion_gemma",
            variant="off",
            global_rank=0,
            local_rank=0,
            tp_rank=0,
            pid=1234,
            cuda_device=0,
            start_allocated=100,
            start_reserved=120,
            peak_allocated=180,
            peak_reserved=240,
            component_elapsed_seconds=0.01,
        )


@pytest.mark.parametrize("batch_phase", ["", "prefill", "decode", 1])
def test_memory_trace_record_rejects_invalid_batch_phase(batch_phase: object):
    with pytest.raises(RuntimeError, match="batch_phase"):
        build_output_phase_peak_record(
            component="diffusion_gemma",
            variant="off",
            batch_phase=batch_phase,
            global_rank=0,
            local_rank=0,
            tp_rank=0,
            pid=1234,
            cuda_device=0,
            start_allocated=100,
            start_reserved=120,
            peak_allocated=180,
            peak_reserved=240,
            component_elapsed_seconds=0.01,
        )


@pytest.mark.parametrize(
    ("configured", "rank", "expected"),
    [
        ("/tmp/peak.jsonl", 0, "/tmp/peak.rank0.jsonl"),
        ("/tmp/peak.jsonl", 7, "/tmp/peak.rank7.jsonl"),
        ("/tmp/peak-{rank}.jsonl", 3, "/tmp/peak-3.jsonl"),
        ("/tmp/peak-{global_rank}.jsonl", 4, "/tmp/peak-4.jsonl"),
    ],
)
def test_memory_trace_path_is_deterministic_per_global_rank(
    configured: str, rank: int, expected: str
):
    assert consumer_state_trace.resolve_rank_trace_path(configured, rank) == expected


def test_disabled_memory_trace_does_not_touch_cuda(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("VLLM_CONSUMER_STATE_PEAK_MEMORY_TRACE_JSONL", raising=False)

    def fail(*args, **kwargs):
        raise AssertionError("disabled trace must not call CUDA memory APIs")

    monkeypatch.setattr(torch.cuda, "synchronize", fail)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", fail)
    monkeypatch.setattr(torch.cuda, "memory_allocated", fail)
    monkeypatch.setattr(torch.cuda, "memory_reserved", fail)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", fail)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", fail)
    monkeypatch.setattr(consumer_state_trace.time, "perf_counter", fail)

    with trace_output_phase_peak("diffusion_gemma"):
        pass


@pytest.mark.parametrize("batch_phase", [None, "decode"])
def test_enabled_memory_trace_rejects_missing_or_invalid_batch_phase_before_cuda(
    monkeypatch: pytest.MonkeyPatch, tmp_path, batch_phase: str | None
):
    monkeypatch.setenv(
        "VLLM_CONSUMER_STATE_PEAK_MEMORY_TRACE_JSONL", str(tmp_path / "peak.jsonl")
    )

    def fail(*args, **kwargs):
        raise AssertionError("invalid batch_phase must fail before CUDA tracing")

    monkeypatch.setattr(torch.cuda, "synchronize", fail)
    with (
        pytest.raises(RuntimeError, match="batch_phase"),
        trace_output_phase_peak(
            "diffusion_gemma",
            batch_phase=batch_phase,
        ),
    ):
        pass


def test_enabled_memory_trace_writes_jsonl(monkeypatch: pytest.MonkeyPatch, tmp_path):
    output_path = tmp_path / "peak.jsonl"
    monkeypatch.setenv("VLLM_CONSUMER_STATE_PEAK_MEMORY_TRACE_JSONL", str(output_path))
    monkeypatch.setenv(
        "VLLM_DIFFUSION_GEMMA_VALIDATION_VARIANT",
        "sample_entropy_state_dense_soft_embed",
    )
    calls: list[str] = []

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("sync"))
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", lambda: calls.append("reset")
    )
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 100)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 120)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 180)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 240)
    perf_counter_values = iter([10.0, 10.125])
    monkeypatch.setattr(
        consumer_state_trace.time,
        "perf_counter",
        lambda: next(perf_counter_values),
    )
    monkeypatch.setattr(
        consumer_state_trace,
        "_trace_process_metadata",
        lambda: {
            "global_rank": 3,
            "local_rank": 1,
            "tp_rank": 1,
            "pid": 1234,
            "cuda_device": 1,
        },
    )

    consumer_sources: dict[str, str] = {}
    contract_metadata: dict[str, object] = {}
    with trace_output_phase_peak(
        "diffusion_gemma",
        batch_phase="consumer_output_only",
        consumer_sources=consumer_sources,
        contract_metadata=contract_metadata,
    ):
        calls.append("body")
        consumer_sources.update(
            sample="local_state",
            clean_argmax="local_state",
            entropy_remask="local_state",
            soft_self_conditioning="dense_logits",
        )
        contract_metadata.update(
            effective_variant="sample_entropy_state_dense_soft_embed",
            consumer_contract="vllm_diffusiongemma",
            pareto_role="ablation_only",
            pareto_gate_eligible=False,
        )

    assert calls == ["sync", "reset", "body", "sync"]
    rank_path = tmp_path / "peak.rank3.jsonl"
    assert not output_path.exists()
    record = json.loads(rank_path.read_text())
    assert record["schema_version"] == 4
    assert record["batch_phase"] == "consumer_output_only"
    assert record["component"] == "diffusion_gemma"
    assert record["peak_allocated_delta_bytes"] == 80
    assert record["consumer_sources"] == {
        "sample": "local_state",
        "clean_argmax": "local_state",
        "entropy_remask": "local_state",
        "soft_self_conditioning": "dense_logits",
    }
    assert record["attribution_status"] == "complete"
    assert record["variant"] == "sample_entropy_state_dense_soft_embed"
    assert record["configured_variant"] == "sample_entropy_state_dense_soft_embed"
    assert record["global_rank"] == 3
    assert record["local_rank"] == 1
    assert record["tp_rank"] == 1
    assert record["pid"] == 1234
    assert record["cuda_device"] == 1
    assert record["component_elapsed_ms"] == 125.0
    assert record["consumer_contract"] == "vllm_diffusiongemma"
    assert record["pareto_role"] == "ablation_only"
    assert record["pareto_gate_eligible"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_enabled_memory_trace_records_actual_cuda_elapsed(tmp_path, monkeypatch):
    output_path = tmp_path / "actual-cuda.jsonl"
    monkeypatch.setenv("VLLM_CONSUMER_STATE_PEAK_MEMORY_TRACE_JSONL", str(output_path))
    spec = validation_variant_spec("token_axis_logit_microbatch")

    with trace_output_phase_peak(
        "diffusion_gemma",
        batch_phase="consumer_output_only",
        consumer_sources=spec.as_source_tags(),
        contract_metadata=spec.as_contract_metadata(),
        variant="token_axis_logit_microbatch",
    ):
        value = torch.randn(256, 256, device="cuda")
        torch.mm(value, value)

    record = json.loads((tmp_path / "actual-cuda.rank0.jsonl").read_text())
    assert record["attribution_status"] == "complete"
    assert isinstance(record["component_elapsed_ms"], float)
    assert record["component_elapsed_ms"] >= 0.0


@pytest.mark.parametrize("failure_site", ["final_sync", "append"])
def test_trace_cleanup_failure_does_not_replace_body_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path, failure_site: str
):
    monkeypatch.setenv(
        "VLLM_CONSUMER_STATE_PEAK_MEMORY_TRACE_JSONL", str(tmp_path / "peak.jsonl")
    )
    monkeypatch.setattr(
        consumer_state_trace,
        "_trace_process_metadata",
        lambda: {
            "global_rank": 0,
            "local_rank": 0,
            "tp_rank": 0,
            "pid": 1234,
            "cuda_device": 0,
        },
    )
    sync_calls = 0

    def synchronize() -> None:
        nonlocal sync_calls
        sync_calls += 1
        if failure_site == "final_sync" and sync_calls == 2:
            raise RuntimeError("trace final sync failed")

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 100)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 120)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 180)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 240)
    if failure_site == "append":
        monkeypatch.setattr(
            consumer_state_trace,
            "_append_jsonl",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                OSError("trace append failed")
            ),
        )

    with (
        pytest.raises(ValueError, match="sampling failed"),
        trace_output_phase_peak(
            "diffusion_gemma", batch_phase="mixed_prefill_consumer_output"
        ),
    ):
        raise ValueError("sampling failed")


@pytest.mark.parametrize("failure_site", ["final_sync", "append"])
def test_trace_cleanup_failure_propagates_after_successful_body(
    monkeypatch: pytest.MonkeyPatch, tmp_path, failure_site: str
):
    monkeypatch.setenv(
        "VLLM_CONSUMER_STATE_PEAK_MEMORY_TRACE_JSONL", str(tmp_path / "peak.jsonl")
    )
    monkeypatch.setattr(
        consumer_state_trace,
        "_trace_process_metadata",
        lambda: {
            "global_rank": 0,
            "local_rank": 0,
            "tp_rank": 0,
            "pid": 1234,
            "cuda_device": 0,
        },
    )
    sync_calls = 0

    def synchronize() -> None:
        nonlocal sync_calls
        sync_calls += 1
        if failure_site == "final_sync" and sync_calls == 2:
            raise RuntimeError("trace final sync failed")

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 100)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 120)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 180)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 240)
    if failure_site == "append":
        monkeypatch.setattr(
            consumer_state_trace,
            "_append_jsonl",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                OSError("trace append failed")
            ),
        )

    error = RuntimeError if failure_site == "final_sync" else OSError
    with (
        pytest.raises(error, match="trace .* failed"),
        trace_output_phase_peak("diffusion_gemma", batch_phase="prefill_only"),
    ):
        pass


def _sampler_output(token: int = 1) -> SamplerOutput:
    return SamplerOutput(
        sampled_token_ids=torch.tensor([[token]], dtype=torch.int64),
        logprobs_tensors=None,
        num_nans=None,
        num_sampled=torch.tensor([1], dtype=torch.int32),
        num_rejected=torch.tensor([0], dtype=torch.int32),
    )


class _RunnerModel:
    def __init__(self, events: list[str], logits_width: int = 5):
        self.events = events
        self.logits_width = logits_width

    @contextmanager
    def enable_local_vocab_logits(self):
        self.events.append("local_context_enter")
        yield
        self.events.append("local_context_exit")

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.events.append("compute_logits")
        return torch.zeros(hidden_states.shape[0], self.logits_width)


class _RunnerSampler:
    def __init__(
        self,
        events: list[str],
        validation_output: SamplerOutput | None,
    ):
        self.events = events
        self.validation_output = validation_output

    def sample_from_hidden_states(self, *args) -> SamplerOutput | None:
        self.events.append("validation")
        return self.validation_output

    def __call__(self, *args) -> SamplerOutput:
        self.events.append("dense_sampler")
        return _sampler_output(9)

    def validation_consumer_spec(self):
        return validation_variant_spec(get_validation_variant())

    def trace_batch_phase(self, input_batch):
        self.events.append("batch_phase")
        return DiffusionSampler.trace_batch_phase(self, input_batch)

    def trace_consumer_spec_for_logits(self, logits, input_batch):
        return validation_variant_spec("off")


def _runner_inputs(
    *,
    num_draft_tokens: int = 1,
    per_request_logits: tuple[int, ...] = (1,),
) -> SimpleNamespace:
    cumulative_logits = np.concatenate(
        ([0], np.cumsum(per_request_logits, dtype=np.int32))
    )
    total_logits = int(cumulative_logits[-1])
    return SimpleNamespace(
        logits_indices=torch.arange(total_logits, dtype=torch.int64),
        num_draft_tokens=num_draft_tokens,
        num_reqs=len(per_request_logits),
        idx_mapping_np=np.arange(len(per_request_logits)),
        cu_num_logits_np=cumulative_logits,
        req_ids=[f"request-{index}" for index in range(len(per_request_logits))],
    )


def _set_runner_controls(
    monkeypatch: pytest.MonkeyPatch, *, variant: str, trace_enabled: bool
) -> None:
    monkeypatch.setattr(model_runner, "_DIFFUSION_GEMMA_VALIDATION_VARIANT", variant)
    monkeypatch.setattr(model_runner, "_CONSUMER_STATE_TRACE_ENABLED", trace_enabled)


def test_model_runner_disabled_controls_use_base_path_without_probes(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_DIFFUSION_GEMMA_VALIDATION_VARIANT", "off")
    monkeypatch.delenv("VLLM_CONSUMER_STATE_PEAK_MEMORY_TRACE_JSONL", raising=False)
    _set_runner_controls(monkeypatch, variant="off", trace_enabled=False)
    events: list[str] = []

    def fail(*args, **kwargs):
        raise AssertionError("disabled path must not invoke validation or tracing")

    runner_sampler = _RunnerSampler(events, _sampler_output(7))
    runner_sampler.sample_from_hidden_states = fail
    runner_sampler.trace_batch_phase = fail
    monkeypatch.setattr(model_runner, "trace_output_phase_peak", fail)
    runner = SimpleNamespace(
        model=_RunnerModel(events),
        sampler=runner_sampler,
        rejection_sampler=None,
    )

    actual, num_sampled, num_rejected = GPUModelRunner.sample(
        runner,
        torch.ones(1, 3),
        _runner_inputs(),
        grammar_output=None,
    )

    assert actual.sampled_token_ids.item() == 9
    assert num_sampled is actual.num_sampled
    assert num_rejected is actual.num_rejected
    assert events == [
        "local_context_enter",
        "compute_logits",
        "local_context_exit",
        "dense_sampler",
    ]


def test_model_runner_validation_result_bypasses_dense_logits(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(
        "VLLM_DIFFUSION_GEMMA_VALIDATION_VARIANT", "token_axis_logit_microbatch"
    )
    monkeypatch.delenv("VLLM_CONSUMER_STATE_PEAK_MEMORY_TRACE_JSONL", raising=False)
    _set_runner_controls(
        monkeypatch, variant="token_axis_logit_microbatch", trace_enabled=False
    )
    events: list[str] = []
    expected = _sampler_output(7)
    runner_sampler = _RunnerSampler(events, expected)

    def fail(*args, **kwargs):
        raise AssertionError("validation-only path must not classify trace phase")

    runner_sampler.trace_batch_phase = fail
    runner = SimpleNamespace(
        model=_RunnerModel(events),
        sampler=runner_sampler,
        rejection_sampler=None,
    )

    actual, num_sampled, num_rejected = GPUModelRunner.sample(
        runner,
        torch.ones(1, 3),
        _runner_inputs(),
        grammar_output=None,
    )

    assert actual is expected
    assert num_sampled is expected.num_sampled
    assert num_rejected is expected.num_rejected
    assert events == ["validation"]


def test_model_runner_validation_success_requires_attribution_seam(
    monkeypatch: pytest.MonkeyPatch,
):
    variant = "sample_only_dense_consumers"
    monkeypatch.setenv("VLLM_DIFFUSION_GEMMA_VALIDATION_VARIANT", variant)
    monkeypatch.delenv("VLLM_CONSUMER_STATE_PEAK_MEMORY_TRACE_JSONL", raising=False)
    _set_runner_controls(monkeypatch, variant=variant, trace_enabled=False)
    events: list[str] = []

    class UnattributedValidationSampler:
        def sample_from_hidden_states(self, *args) -> SamplerOutput:
            events.append("validation")
            return _sampler_output(7)

        def __call__(self, *args) -> SamplerOutput:
            raise AssertionError("successful validation must bypass dense sampling")

    runner = SimpleNamespace(
        model=_RunnerModel(events),
        sampler=UnattributedValidationSampler(),
        rejection_sampler=None,
    )

    with pytest.raises(RuntimeError, match="validation_consumer_spec"):
        GPUModelRunner.sample(
            runner,
            torch.ones(1, 3),
            _runner_inputs(),
            grammar_output=None,
        )

    assert events == ["validation"]


@pytest.mark.parametrize(
    ("variant", "validation_falls_back", "trace_enabled"),
    [
        ("sample_only_dense_consumers", True, True),
        ("sample_only_dense_consumers", True, False),
        ("off", False, True),
    ],
    ids=[
        "explicit-validation-fallback-traced",
        "explicit-validation-fallback-untraced",
        "trace-only-off",
    ],
)
def test_model_runner_dense_path_requires_attribution_seam(
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    validation_falls_back: bool,
    trace_enabled: bool,
):
    monkeypatch.setenv("VLLM_DIFFUSION_GEMMA_VALIDATION_VARIANT", variant)
    monkeypatch.delenv("VLLM_CONSUMER_STATE_PEAK_MEMORY_TRACE_JSONL", raising=False)
    _set_runner_controls(monkeypatch, variant=variant, trace_enabled=trace_enabled)
    events: list[str] = []

    class UnattributedDenseSampler:
        def trace_batch_phase(self, input_batch) -> str:
            return DiffusionSampler.trace_batch_phase(self, input_batch)

        def sample_from_hidden_states(self, *args) -> None:
            assert validation_falls_back
            events.append("validation_fallback")
            return None

        def __call__(self, *args) -> SamplerOutput:
            events.append("dense_sampler")
            return _sampler_output(9)

    runner = SimpleNamespace(
        model=_RunnerModel(events),
        sampler=UnattributedDenseSampler(),
        rejection_sampler=None,
    )

    with pytest.raises(RuntimeError, match="trace_consumer_spec_for_logits"):
        GPUModelRunner.sample(
            runner,
            torch.ones(1, 3),
            _runner_inputs(),
            grammar_output=None,
        )

    expected_prefix = ["validation_fallback"] if validation_falls_back else []
    assert events == expected_prefix + [
        "local_context_enter",
        "compute_logits",
        "local_context_exit",
        "dense_sampler",
    ]


def test_model_runner_rejects_incomplete_consumer_attribution(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_DIFFUSION_GEMMA_VALIDATION_VARIANT", "off")
    monkeypatch.delenv("VLLM_CONSUMER_STATE_PEAK_MEMORY_TRACE_JSONL", raising=False)
    _set_runner_controls(monkeypatch, variant="off", trace_enabled=True)
    events: list[str] = []

    class IncompleteSpec:
        def as_source_tags(self) -> dict[str, str]:
            return {"sample": "dense_logits"}

        def as_contract_metadata(self) -> dict[str, object]:
            return {"effective_variant": "off"}

    class IncompletelyAttributedSampler:
        def trace_batch_phase(self, input_batch) -> str:
            return DiffusionSampler.trace_batch_phase(self, input_batch)

        def __call__(self, *args) -> SamplerOutput:
            events.append("dense_sampler")
            return _sampler_output(9)

        def trace_consumer_spec_for_logits(self, *args) -> IncompleteSpec:
            return IncompleteSpec()

    runner = SimpleNamespace(
        model=_RunnerModel(events),
        sampler=IncompletelyAttributedSampler(),
        rejection_sampler=None,
    )

    with pytest.raises(RuntimeError, match="incomplete"):
        GPUModelRunner.sample(
            runner,
            torch.ones(1, 3),
            _runner_inputs(),
            grammar_output=None,
        )

    assert events[-1] == "dense_sampler"


@pytest.mark.parametrize("body_path", ["validation", "dense"])
def test_model_runner_body_exception_precedes_missing_attribution(
    monkeypatch: pytest.MonkeyPatch,
    body_path: str,
):
    variant = "sample_only_dense_consumers" if body_path == "validation" else "off"
    monkeypatch.setenv("VLLM_DIFFUSION_GEMMA_VALIDATION_VARIANT", variant)
    monkeypatch.delenv("VLLM_CONSUMER_STATE_PEAK_MEMORY_TRACE_JSONL", raising=False)
    _set_runner_controls(monkeypatch, variant=variant, trace_enabled=True)

    class BodyFailure(RuntimeError):
        pass

    class FailingSampler:
        def trace_batch_phase(self, input_batch) -> str:
            return DiffusionSampler.trace_batch_phase(self, input_batch)

        def sample_from_hidden_states(self, *args) -> SamplerOutput:
            raise BodyFailure("sampler body failed")

        def __call__(self, *args) -> SamplerOutput:
            raise BodyFailure("sampler body failed")

    runner = SimpleNamespace(
        model=_RunnerModel([]),
        sampler=FailingSampler(),
        rejection_sampler=None,
    )

    with pytest.raises(BodyFailure, match="sampler body failed"):
        GPUModelRunner.sample(
            runner,
            torch.ones(1, 3),
            _runner_inputs(),
            grammar_output=None,
        )


@pytest.mark.parametrize(
    ("variant", "expected_sources", "pareto_role", "pareto_gate_eligible"),
    [
        (
            "token_axis_logit_microbatch",
            {
                "sample": "token_microbatch_dense_logits",
                "clean_argmax": "token_microbatch_dense_logits",
                "entropy_remask": "token_microbatch_dense_logits",
                "soft_self_conditioning": "token_microbatch_dense_logits",
            },
            "dominator_candidate",
            True,
        ),
        (
            "sample_only_dense_consumers",
            {
                "sample": "local_state",
                "clean_argmax": "dense_logits",
                "entropy_remask": "dense_logits",
                "soft_self_conditioning": "dense_logits",
            },
            "ablation_only",
            False,
        ),
        (
            "sample_entropy_state_dense_soft_embed",
            {
                "sample": "local_state",
                "clean_argmax": "local_state",
                "entropy_remask": "local_state",
                "soft_self_conditioning": "dense_logits",
            },
            "ablation_only",
            False,
        ),
    ],
)
def test_model_runner_trace_reports_executed_validation_contract(
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    expected_sources: dict[str, str],
    pareto_role: str,
    pareto_gate_eligible: bool,
):
    monkeypatch.setenv("VLLM_DIFFUSION_GEMMA_VALIDATION_VARIANT", variant)
    _set_runner_controls(monkeypatch, variant=variant, trace_enabled=True)
    captured: dict[str, object] = {}

    @contextmanager
    def capture_trace(
        component,
        *,
        batch_phase,
        consumer_sources,
        contract_metadata,
        variant,
    ):
        yield
        captured["component"] = component
        captured["batch_phase"] = batch_phase
        captured["sources"] = dict(consumer_sources)
        captured["metadata"] = dict(contract_metadata)
        captured["variant"] = variant

    monkeypatch.setattr(model_runner, "trace_output_phase_peak", capture_trace)
    events: list[str] = []
    runner = SimpleNamespace(
        model=_RunnerModel(events),
        sampler=_RunnerSampler(events, _sampler_output(7)),
        rejection_sampler=None,
    )

    GPUModelRunner.sample(
        runner,
        torch.ones(1, 3),
        _runner_inputs(),
        grammar_output=None,
    )

    assert events == ["batch_phase", "validation"]
    assert captured["batch_phase"] == "consumer_output_only"
    assert captured["sources"] == expected_sources
    assert captured["metadata"] == {
        "effective_variant": variant,
        "consumer_contract": "vllm_diffusiongemma",
        "pareto_role": pareto_role,
        "pareto_gate_eligible": pareto_gate_eligible,
    }
    assert captured["variant"] == variant


def test_model_runner_trace_prefill_preserves_dense_attribution(
    monkeypatch: pytest.MonkeyPatch,
):
    variant = "token_axis_logit_microbatch"
    monkeypatch.setenv("VLLM_DIFFUSION_GEMMA_VALIDATION_VARIANT", variant)
    _set_runner_controls(monkeypatch, variant=variant, trace_enabled=True)
    captured: dict[str, object] = {}

    @contextmanager
    def capture_trace(
        component,
        *,
        batch_phase,
        consumer_sources,
        contract_metadata,
        variant,
    ):
        yield
        captured["batch_phase"] = batch_phase
        captured["sources"] = dict(consumer_sources)
        captured["metadata"] = dict(contract_metadata)

    monkeypatch.setattr(model_runner, "trace_output_phase_peak", capture_trace)
    events: list[str] = []
    runner = SimpleNamespace(
        model=_RunnerModel(events),
        sampler=_RunnerSampler(events, None),
        rejection_sampler=None,
    )

    GPUModelRunner.sample(
        runner,
        torch.ones(1, 3),
        _runner_inputs(num_draft_tokens=0, per_request_logits=(1,)),
        grammar_output=None,
    )

    assert events == [
        "batch_phase",
        "validation",
        "local_context_enter",
        "compute_logits",
        "local_context_exit",
        "dense_sampler",
    ]
    assert captured["batch_phase"] == "prefill_only"
    assert captured["sources"] == {
        "sample": "dense_logits",
        "clean_argmax": "dense_logits",
        "entropy_remask": "dense_logits",
        "soft_self_conditioning": "dense_logits",
    }
    assert captured["metadata"] == {
        "effective_variant": "off",
        "consumer_contract": "vllm_diffusiongemma",
        "pareto_role": "baseline_context",
        "pareto_gate_eligible": False,
    }


def test_model_runner_trace_mixed_batch_preserves_selected_attribution(
    monkeypatch: pytest.MonkeyPatch,
):
    variant = "sample_only_dense_consumers"
    monkeypatch.setenv("VLLM_DIFFUSION_GEMMA_VALIDATION_VARIANT", variant)
    _set_runner_controls(monkeypatch, variant=variant, trace_enabled=True)
    captured: dict[str, object] = {}

    @contextmanager
    def capture_trace(
        component,
        *,
        batch_phase,
        consumer_sources,
        contract_metadata,
        variant,
    ):
        yield
        captured["batch_phase"] = batch_phase
        captured["metadata"] = dict(contract_metadata)

    monkeypatch.setattr(model_runner, "trace_output_phase_peak", capture_trace)
    events: list[str] = []
    runner = SimpleNamespace(
        model=_RunnerModel(events),
        sampler=_RunnerSampler(events, _sampler_output(7)),
        rejection_sampler=None,
    )

    GPUModelRunner.sample(
        runner,
        torch.ones(1, 3),
        _runner_inputs(per_request_logits=(0, 1)),
        grammar_output=None,
    )

    assert events == ["batch_phase", "validation"]
    assert captured["batch_phase"] == "mixed_prefill_consumer_output"
    assert captured["metadata"] == {
        "effective_variant": variant,
        "consumer_contract": "vllm_diffusiongemma",
        "pareto_role": "ablation_only",
        "pareto_gate_eligible": False,
    }


@pytest.mark.parametrize("phase_result", [None, "decode"])
def test_model_runner_trace_rejects_missing_or_invalid_phase_seam_result(
    monkeypatch: pytest.MonkeyPatch, phase_result: str | None
):
    monkeypatch.setenv("VLLM_DIFFUSION_GEMMA_VALIDATION_VARIANT", "off")
    _set_runner_controls(monkeypatch, variant="off", trace_enabled=True)

    class InvalidPhaseSampler(_RunnerSampler):
        def trace_batch_phase(self, input_batch):
            return phase_result

    runner = SimpleNamespace(
        model=_RunnerModel([]),
        sampler=InvalidPhaseSampler([], None),
        rejection_sampler=None,
    )

    with pytest.raises(RuntimeError, match="trace_batch_phase.*batch_phase"):
        GPUModelRunner.sample(
            runner,
            torch.ones(1, 3),
            _runner_inputs(),
            grammar_output=None,
        )


def test_model_runner_trace_requires_batch_phase_seam(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_DIFFUSION_GEMMA_VALIDATION_VARIANT", "off")
    _set_runner_controls(monkeypatch, variant="off", trace_enabled=True)

    class MissingPhaseSampler:
        def __call__(self, *args) -> SamplerOutput:
            raise AssertionError("missing phase seam must fail before sampling")

    runner = SimpleNamespace(
        model=_RunnerModel([]),
        sampler=MissingPhaseSampler(),
        rejection_sampler=None,
    )

    with pytest.raises(RuntimeError, match=r"callable sampler\.trace_batch_phase"):
        GPUModelRunner.sample(
            runner,
            torch.ones(1, 3),
            _runner_inputs(),
            grammar_output=None,
        )


def test_model_runner_grammar_forces_dense_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    _set_runner_controls(
        monkeypatch, variant="token_axis_logit_microbatch", trace_enabled=False
    )
    events: list[str] = []

    class StructuredOutputsWorker:
        def apply_grammar_bitmask(self, *args) -> None:
            events.append("grammar")

    runner = SimpleNamespace(
        model=_RunnerModel(events),
        sampler=_RunnerSampler(events, _sampler_output(7)),
        rejection_sampler=None,
        structured_outputs_worker=StructuredOutputsWorker(),
    )
    grammar_output = SimpleNamespace(
        structured_output_request_ids=["request"],
        grammar_bitmask=torch.ones(1, 5),
    )

    GPUModelRunner.sample(
        runner,
        torch.ones(1, 3),
        _runner_inputs(),
        grammar_output=grammar_output,
    )

    assert events == ["compute_logits", "grammar", "dense_sampler"]


def test_model_runner_rejection_sampling_forces_dense_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    _set_runner_controls(
        monkeypatch, variant="token_axis_logit_microbatch", trace_enabled=False
    )
    events: list[str] = []

    def rejection_sampler(*args) -> SamplerOutput:
        events.append("rejection_sampler")
        return _sampler_output(8)

    runner = SimpleNamespace(
        model=_RunnerModel(events),
        sampler=_RunnerSampler(events, _sampler_output(7)),
        rejection_sampler=rejection_sampler,
        speculator=SimpleNamespace(draft_logits=torch.zeros(1, 5)),
    )

    GPUModelRunner.sample(
        runner,
        torch.ones(1, 3),
        _runner_inputs(num_draft_tokens=1),
        grammar_output=None,
    )

    assert events == ["compute_logits", "rejection_sampler"]


def test_model_runner_peak_trace_wraps_dense_output_head(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    events: list[str] = []
    output_path = tmp_path / "peak.jsonl"
    monkeypatch.setenv("VLLM_DIFFUSION_GEMMA_VALIDATION_VARIANT", "off")
    monkeypatch.setenv("VLLM_CONSUMER_STATE_PEAK_MEMORY_TRACE_JSONL", str(output_path))
    _set_runner_controls(monkeypatch, variant="off", trace_enabled=True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("sync"))
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", lambda: events.append("reset")
    )
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 100)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 120)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 180)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 240)
    perf_counter_values = iter([20.0, 20.25])
    monkeypatch.setattr(
        consumer_state_trace.time,
        "perf_counter",
        lambda: next(perf_counter_values),
    )
    monkeypatch.setattr(
        consumer_state_trace,
        "_trace_process_metadata",
        lambda: {
            "global_rank": 0,
            "local_rank": 0,
            "tp_rank": 0,
            "pid": 1234,
            "cuda_device": 0,
        },
    )
    runner = SimpleNamespace(
        model=_RunnerModel(events),
        sampler=_RunnerSampler(events, None),
        rejection_sampler=None,
    )

    GPUModelRunner.sample(
        runner,
        torch.ones(1, 3),
        _runner_inputs(),
        grammar_output=None,
    )

    assert events == [
        "batch_phase",
        "sync",
        "reset",
        "local_context_enter",
        "compute_logits",
        "local_context_exit",
        "dense_sampler",
        "sync",
    ]
    record = json.loads((tmp_path / "peak.rank0.jsonl").read_text())
    assert record["batch_phase"] == "consumer_output_only"
    assert record["consumer_sources"] == {
        "sample": "dense_logits",
        "clean_argmax": "dense_logits",
        "entropy_remask": "dense_logits",
        "soft_self_conditioning": "dense_logits",
    }
    assert record["consumer_contract"] == "vllm_diffusiongemma"
    assert record["variant"] == "off"
    assert record["configured_variant"] == "off"
    assert record["pareto_role"] == "baseline_context"
    assert record["pareto_gate_eligible"] is False
    assert record["component_elapsed_ms"] == 250.0


@pytest.mark.parametrize(
    (
        "logits_width",
        "max_num_logprobs",
        "expected_source",
        "expected_variant",
        "expected_role",
    ),
    [
        (5, -1, "dense_logits", "off", "baseline_context"),
        (3, -1, "local_state", "full_consumer_state", "reference"),
        (3, 0, "dense_logits", "off", "baseline_context"),
    ],
    ids=["dense", "full-consumer-state", "full-logprobs-fallback"],
)
def test_model_runner_trace_attributes_actual_dense_or_full_state_path(
    monkeypatch: pytest.MonkeyPatch,
    logits_width: int,
    max_num_logprobs: int,
    expected_source: str,
    expected_variant: str,
    expected_role: str,
    tmp_path: Path,
):
    monkeypatch.setattr(diffusion_gemma, "_DIFFUSION_GEMMA_LOCAL_VOCAB_SAMPLER", True)
    output_path = tmp_path / "path.jsonl"
    monkeypatch.setenv("VLLM_CONSUMER_STATE_PEAK_MEMORY_TRACE_JSONL", str(output_path))
    _set_runner_controls(monkeypatch, variant="off", trace_enabled=True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 100)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 120)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 180)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 240)
    perf_counter_values = iter([30.0, 30.01])
    monkeypatch.setattr(
        consumer_state_trace.time,
        "perf_counter",
        lambda: next(perf_counter_values),
    )
    monkeypatch.setattr(
        consumer_state_trace,
        "_trace_process_metadata",
        lambda: {
            "global_rank": 0,
            "local_rank": 0,
            "tp_rank": 0,
            "pid": 1234,
            "cuda_device": 0,
        },
    )
    events: list[str] = []
    sampler = _RunnerSampler(events, None)
    sampler.vocab_size = 5
    sampler.sampling_states = SimpleNamespace(
        max_num_logprobs=lambda slots: max_num_logprobs
    )
    sampler.trace_consumer_spec_for_logits = (
        DiffusionSampler.trace_consumer_spec_for_logits.__get__(
            sampler, DiffusionSampler
        )
    )
    runner = SimpleNamespace(
        model=_RunnerModel(events, logits_width=logits_width),
        sampler=sampler,
        rejection_sampler=None,
    )

    GPUModelRunner.sample(
        runner,
        torch.ones(1, 3),
        _runner_inputs(),
        grammar_output=None,
    )

    record = json.loads((tmp_path / "path.rank0.jsonl").read_text())
    assert record["batch_phase"] == "consumer_output_only"
    assert record["configured_variant"] == "off"
    assert record["variant"] == expected_variant
    assert record["consumer_sources"] == {
        "sample": expected_source,
        "clean_argmax": expected_source,
        "entropy_remask": expected_source,
        "soft_self_conditioning": expected_source,
    }
    assert record["consumer_contract"] == "vllm_diffusiongemma"
    assert record["pareto_role"] == expected_role
    assert record["pareto_gate_eligible"] is False


@pytest.mark.parametrize(
    ("variant", "chunk_rows", "max_num_logprobs", "per_req_rows", "hidden_rows"),
    [
        ("token_axis_logit_microbatch", 2, 1, 3, 3),
        ("token_axis_logit_microbatch", 2, 0, 2, 2),
        ("token_axis_logit_microbatch", 0, 0, 3, 3),
        ("sample_only_dense_consumers", 2, 1, 3, 3),
        ("sample_only_dense_consumers", 2, 0, 2, 2),
        ("sample_entropy_state_dense_soft_embed", 2, 0, 2, 2),
    ],
    ids=[
        "full-logprobs",
        "malformed",
        "nonpositive-rows",
        "sample-only-full-logprobs",
        "sample-only-malformed",
        "malformed-intermediate",
    ],
)
def test_sampler_runtime_unsupported_inputs_fall_back_before_compute(
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    chunk_rows: int,
    max_num_logprobs: int,
    per_req_rows: int,
    hidden_rows: int,
):
    monkeypatch.setenv("VLLM_DIFFUSION_GEMMA_VALIDATION_VARIANT", variant)
    monkeypatch.setenv("VLLM_DIFFUSION_GEMMA_TOKEN_MICROBATCH_ROWS", str(chunk_rows))
    sampler = DiffusionSampler.__new__(DiffusionSampler)
    sampler.canvas_length = 3
    sampler.sampling_states = SimpleNamespace(
        max_num_logprobs=lambda slots: max_num_logprobs
    )
    input_batch = SimpleNamespace(
        num_draft_tokens=1,
        num_reqs=1,
        idx_mapping_np=np.array([0]),
        cu_num_logits_np=np.array([0, per_req_rows]),
        req_ids=["request"],
    )

    class FailModel:
        def compute_logits(self, hidden_states):
            raise AssertionError("fallback must precede output-head compute")

        def compute_local_logits(self, hidden_states):
            raise AssertionError("fallback must precede local output-head compute")

    assert (
        sampler.sample_from_hidden_states(
            torch.zeros(hidden_rows, 4), input_batch, FailModel()
        )
        is None
    )


def test_trace_consumer_spec_is_per_call_and_full_state_fallback_is_truthful(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(diffusion_gemma, "_DIFFUSION_GEMMA_LOCAL_VOCAB_SAMPLER", True)
    max_num_logprobs = {"value": -1}
    sampler = DiffusionSampler.__new__(DiffusionSampler)
    sampler.vocab_size = 6
    sampler.sampling_states = SimpleNamespace(
        max_num_logprobs=lambda slots: max_num_logprobs["value"]
    )
    sampler.trace_batch_phase = lambda _: (_ for _ in ()).throw(
        AssertionError("consumer attribution must reuse the runner's phase probe")
    )
    input_batch = SimpleNamespace(
        num_reqs=1,
        num_draft_tokens=3,
        idx_mapping_np=np.array([0]),
        cu_num_logits_np=np.array([0, 3]),
        req_ids=["request"],
    )

    full_state = sampler.trace_consumer_spec_for_logits(torch.zeros(3, 3), input_batch)
    dense = sampler.trace_consumer_spec_for_logits(torch.zeros(3, 6), input_batch)
    max_num_logprobs["value"] = 0
    fallback = sampler.trace_consumer_spec_for_logits(torch.zeros(3, 3), input_batch)
    max_num_logprobs["value"] = -1
    full_state_again = sampler.trace_consumer_spec_for_logits(
        torch.zeros(3, 3), input_batch
    )

    all_local = {
        "sample": "local_state",
        "clean_argmax": "local_state",
        "entropy_remask": "local_state",
        "soft_self_conditioning": "local_state",
    }
    all_dense = {
        "sample": "dense_logits",
        "clean_argmax": "dense_logits",
        "entropy_remask": "dense_logits",
        "soft_self_conditioning": "dense_logits",
    }
    assert full_state.as_source_tags() == all_local
    assert full_state_again.as_source_tags() == all_local
    assert dense.as_source_tags() == all_dense
    assert fallback.as_source_tags() == all_dense
    assert full_state.as_source_tags() is not full_state_again.as_source_tags()


def test_trace_consumer_spec_preserves_dense_attribution_for_real_prefill(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(diffusion_gemma, "_DIFFUSION_GEMMA_LOCAL_VOCAB_SAMPLER", True)
    sampler = DiffusionSampler.__new__(DiffusionSampler)
    sampler.vocab_size = 6
    sampler.sampling_states = SimpleNamespace(max_num_logprobs=lambda slots: -1)
    sampler.trace_batch_phase = lambda _: (_ for _ in ()).throw(
        AssertionError("consumer attribution must not reclassify prefill")
    )
    input_batch = SimpleNamespace(
        num_reqs=1,
        num_draft_tokens=0,
        idx_mapping_np=np.array([0]),
        cu_num_logits_np=np.array([0, 1]),
        req_ids=["request"],
    )

    spec = sampler.trace_consumer_spec_for_logits(torch.zeros(1, 3), input_batch)

    assert spec.effective_variant == "off"
    assert set(spec.as_source_tags().values()) == {"dense_logits"}


def test_sample_from_hidden_states_routes_sample_only_variant(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(
        "VLLM_DIFFUSION_GEMMA_VALIDATION_VARIANT", "sample_only_dense_consumers"
    )

    class FakeUvaBuffer:
        def __init__(self):
            self.np = np.zeros(1, dtype=np.int64)
            self.gpu = torch.zeros(1, dtype=torch.int64)

        def copy_to_uva(self) -> None:
            self.gpu.copy_(torch.from_numpy(self.np))

    sampler = DiffusionSampler.__new__(DiffusionSampler)
    sampler.canvas_length = 3
    sampler.sampling_states = SimpleNamespace(max_num_logprobs=lambda slots: -1)
    sampler._decode_slots = FakeUvaBuffer()
    sampler._decode_idx = FakeUvaBuffer()
    sampler._sampled = torch.zeros(1, 3, dtype=torch.int64)
    sampler._num_sampled = torch.zeros(1, dtype=torch.int32)
    sampler.diffusion_states = SimpleNamespace(
        is_encoder_phase=torch.zeros(1, dtype=torch.bool)
    )
    routed: list[dict[str, object]] = []
    expected = _sampler_output(7)
    sampler._sample_only_dense_consumers = lambda *args, **kwargs: routed.append(kwargs)
    sampler._build_output = lambda *args, **kwargs: expected
    monkeypatch.setattr(
        diffusion_gemma,
        "async_copy_to_gpu",
        lambda value, device: torch.from_numpy(value).to(device),
    )
    input_batch = SimpleNamespace(
        num_draft_tokens=1,
        num_reqs=1,
        idx_mapping_np=np.array([0]),
        idx_mapping=torch.tensor([0]),
        cu_num_logits_np=np.array([0, 3]),
        req_ids=["request"],
    )

    actual = sampler.sample_from_hidden_states(
        torch.zeros(3, 4), input_batch, SimpleNamespace()
    )

    assert actual is expected
    assert len(routed) == 1
    assert torch.equal(routed[0]["decode_slots"], torch.tensor([0]))
    assert torch.equal(routed[0]["valid_canvas_len"], torch.tensor([3]))


def _make_state_sampler(
    device: torch.device, *, embed_weight: torch.Tensor
) -> DiffusionSampler:
    sampler = DiffusionSampler.__new__(DiffusionSampler)
    num_reqs = 2
    canvas_length = 3
    hidden_size = embed_weight.shape[1]
    sampler.canvas_length = canvas_length
    sampler.vocab_size = embed_weight.shape[0]
    sampler.embed_weight = embed_weight
    sampler.normalizer = torch.tensor(1.0, device=device, dtype=embed_weight.dtype)
    sampler.sc_vocab_start = 0
    sampler.sc_vocab_end = sampler.vocab_size
    sampler.tp_size = 1
    sampler.t_min = 1.0
    sampler.t_max = 1.0
    sampler.confidence_threshold = 100.0
    sampler.entropy_bound = -1.0
    sampler.embed_vocab_start_index = 0
    sampler.diffusion_states = SimpleNamespace(
        canvas=torch.zeros(num_reqs, canvas_length, dtype=torch.int64, device=device),
        argmax_canvas=torch.zeros(
            num_reqs, canvas_length, dtype=torch.int64, device=device
        ),
        step=torch.zeros(num_reqs, dtype=torch.int32, device=device),
        is_encoder_phase=torch.zeros(num_reqs, dtype=torch.bool, device=device),
        confident=torch.zeros(num_reqs, dtype=torch.bool, device=device),
        self_conditioning_embeds=torch.zeros(
            num_reqs, canvas_length, hidden_size, dtype=torch.float32, device=device
        ),
        accepted_canvas_history=torch.zeros(
            num_reqs, 2, canvas_length, dtype=torch.int64, device=device
        ),
        accepted_canvas_history_len=torch.zeros(
            num_reqs, dtype=torch.int32, device=device
        ),
        max_denoising_steps=4,
        stability_threshold=2,
    )
    sampler.req_states = SimpleNamespace(
        draft_tokens=torch.zeros(
            num_reqs, canvas_length, dtype=torch.int64, device=device
        )
    )
    return sampler


def _state_snapshot(sampler: DiffusionSampler) -> dict[str, torch.Tensor]:
    states = sampler.diffusion_states
    return {
        name: getattr(states, name).clone()
        for name in (
            "canvas",
            "argmax_canvas",
            "step",
            "is_encoder_phase",
            "confident",
            "self_conditioning_embeds",
            "accepted_canvas_history",
            "accepted_canvas_history_len",
        )
    } | {"draft_tokens": sampler.req_states.draft_tokens.clone()}


def test_full_consumer_default_step_bypasses_validation_decomposition(
    monkeypatch: pytest.MonkeyPatch,
):
    device = torch.device("cpu")
    generator = torch.Generator(device=device).manual_seed(31)
    embed = torch.randn(7, 5, generator=generator)
    logits = torch.randn(6, 7, generator=generator)
    sampler = _make_state_sampler(device, embed_weight=embed)
    sampler.embed_vocab_start_index = 0
    sampler.embed_vocab_end_index = 7
    sampler.seed = 11
    sampler._local_vocab_generator = None

    def fail(*args, **kwargs):
        raise AssertionError("base full-consumer path used a validation helper")

    sampler._compute_local_consumer_outputs = fail
    sampler._apply_denoise_step_outputs = fail
    monkeypatch.setattr(
        diffusion_gemma,
        "_tp_argmax_reduce_multi",
        lambda values, indices: (values, indices),
    )
    monkeypatch.setattr(diffusion_gemma, "_tp_all_reduce_sum", lambda value: value)
    monkeypatch.setattr(diffusion_gemma, "_tp_rank_in_group", lambda: 0)
    monkeypatch.setattr(
        diffusion_gemma, "_tp_broadcast_from_rank0", lambda value: value
    )
    sampled = torch.zeros(2, 3, dtype=torch.int64)
    num_sampled = torch.zeros(2, dtype=torch.int32)

    scaled = sampler._sample_local_vocab_step(
        logits,
        decode_slots=torch.tensor([0, 1]),
        decode_idx=torch.tensor([0, 1]),
        all_slots=torch.tensor([0, 1]),
        valid_canvas_len=torch.tensor([3, 3]),
        is_committing=torch.tensor([False, False]),
        sampled=sampled,
        num_sampled=num_sampled,
    )

    assert scaled.shape == (2, 3, 7)
    assert torch.equal(sampler.diffusion_states.step, torch.ones(2, dtype=torch.int32))
    assert torch.equal(sampler.req_states.draft_tokens, sampler.diffusion_states.canvas)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_token_microbatch_endpoint_matches_dense_state_with_fixed_rng(
    monkeypatch: pytest.MonkeyPatch,
):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(19)
    hidden = torch.randn(6, 5, device=device, dtype=torch.bfloat16, generator=generator)
    head = torch.randn(7, 5, device=device, dtype=torch.bfloat16, generator=generator)
    embed = torch.randn(7, 5, device=device, dtype=torch.bfloat16, generator=generator)
    fixed_gumbel = torch.randn(6, 7, device=device, generator=generator)
    fixed_random_tokens = torch.tensor([[2, 3, 4], [5, 6, 1]], device=device)
    actual = _make_state_sampler(device, embed_weight=embed)
    expected = _make_state_sampler(device, embed_weight=embed.clone())
    monkeypatch.setenv("VLLM_DIFFUSION_GEMMA_TOKEN_MICROBATCH_ROWS", "2")
    monkeypatch.setattr(diffusion_gemma, "_tp_broadcast_from_rank0", lambda x: x)
    calls: list[tuple[int, int]] = []

    class SpyModel:
        def compute_logits(self, chunk: torch.Tensor) -> torch.Tensor:
            logits = F.linear(chunk, head)
            calls.append((chunk.shape[0], logits.shape[0]))
            return logits

    apply_calls = 0
    original_apply = actual._apply_denoise_step_outputs

    def counted_apply(**kwargs) -> None:
        nonlocal apply_calls
        apply_calls += 1
        original_apply(**kwargs)

    actual._apply_denoise_step_outputs = counted_apply
    dense = _dense_consumer_oracle(hidden, head, embed, fixed_gumbel)
    dense_logits = F.linear(hidden, head).float()
    dense_probs = dense_logits.log_softmax(dim=-1).exp()
    dense["soft_embeds"] = dense_probs.to(embed.dtype) @ embed
    common = {
        "decode_slots": torch.tensor([0, 1], device=device),
        "decode_idx": torch.tensor([0, 1], device=device),
        "all_slots": torch.tensor([0, 1], device=device),
        "valid_canvas_len": torch.tensor([3, 3], device=device),
        "is_committing": torch.tensor([False, False], device=device),
    }
    actual_sampled = torch.zeros(2, 3, dtype=torch.int64, device=device)
    actual_num_sampled = torch.zeros(2, dtype=torch.int32, device=device)

    with monkeypatch.context() as no_concat:
        no_concat.setattr(
            torch,
            "cat",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("full chunk outputs must not be concatenated")
            ),
        )
        actual._sample_token_axis_logit_microbatch(
            hidden,
            SpyModel(),
            sampled=actual_sampled,
            num_sampled=actual_num_sampled,
            gumbel_for_rows=lambda start, stop: fixed_gumbel[start:stop],
            random_tokens=fixed_random_tokens,
            **common,
        )

    expected_sampled = torch.zeros_like(actual_sampled)
    expected_num_sampled = torch.zeros_like(actual_num_sampled)
    expected._apply_denoise_step_outputs(
        sampled=expected_sampled,
        num_sampled=expected_num_sampled,
        new_tokens=dense["sample_tokens"].reshape(2, 3),
        argmax_tokens=dense["clean_tokens"].reshape(2, 3),
        token_entropy=dense["entropy"].reshape(2, 3),
        soft_embeds=dense["soft_embeds"].reshape(2, 3, -1),
        random_tokens=fixed_random_tokens,
        **common,
    )

    assert calls == [(2, 2), (2, 2), (2, 2)]
    assert apply_calls == 1
    torch.testing.assert_close(actual_sampled, expected_sampled, rtol=0, atol=0)
    torch.testing.assert_close(actual_num_sampled, expected_num_sampled, rtol=0, atol=0)
    actual_state = _state_snapshot(actual)
    expected_state = _state_snapshot(expected)
    assert actual_state.keys() == expected_state.keys()
    for name in actual_state:
        torch.testing.assert_close(
            actual_state[name], expected_state[name], rtol=1e-5, atol=1e-6
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_token_microbatch_cuda_rng_orders_random_tokens_after_all_gumbels(
    monkeypatch: pytest.MonkeyPatch,
):
    device = torch.device("cuda")
    embed = torch.randn(7, 5, device=device, dtype=torch.bfloat16)
    head = torch.randn(7, 5, device=device, dtype=torch.bfloat16)
    hidden = torch.randn(6, 5, device=device, dtype=torch.bfloat16)
    sampler = _make_state_sampler(device, embed_weight=embed)
    monkeypatch.setenv("VLLM_DIFFUSION_GEMMA_TOKEN_MICROBATCH_ROWS", "2")
    monkeypatch.setattr(diffusion_gemma, "_tp_broadcast_from_rank0", lambda x: x)
    events: list[str] = []
    random_shapes: list[tuple[int, ...]] = []
    original_rand_like = torch.rand_like
    original_randint = torch.randint

    def tracked_rand_like(*args, **kwargs):
        events.append("gumbel")
        random_shapes.append(tuple(args[0].shape))
        return original_rand_like(*args, **kwargs)

    def tracked_randint(*args, **kwargs):
        events.append("random_tokens")
        return original_randint(*args, **kwargs)

    monkeypatch.setattr(torch, "rand_like", tracked_rand_like)
    monkeypatch.setattr(torch, "randint", tracked_randint)
    sampler._sample_token_axis_logit_microbatch(
        hidden,
        SimpleNamespace(compute_logits=lambda chunk: F.linear(chunk, head)),
        decode_slots=torch.tensor([0, 1], device=device),
        decode_idx=torch.tensor([0, 1], device=device),
        all_slots=torch.tensor([0, 1], device=device),
        valid_canvas_len=torch.tensor([3, 3], device=device),
        is_committing=torch.tensor([False, False], device=device),
        sampled=torch.zeros(2, 3, dtype=torch.int64, device=device),
        num_sampled=torch.zeros(2, dtype=torch.int32, device=device),
    )

    assert events == ["gumbel", "gumbel", "gumbel", "random_tokens"]
    assert random_shapes == [(2, 7), (2, 7), (2, 7)]


def test_sample_only_endpoint_uses_local_sample_and_dense_consumers_once(
    monkeypatch: pytest.MonkeyPatch,
):
    local_embed = torch.arange(15).reshape(3, 5).float() / 10
    remote_embed = torch.arange(15, 30).reshape(3, 5).float() / 10
    sampler = _make_state_sampler(torch.device("cpu"), embed_weight=local_embed)
    sampler.tp_size = 2
    sampler.vocab_size = 6
    sampler.embed_vocab_start_index = 0
    sampler.embed_vocab_end_index = 3
    sampler.sc_vocab_start = 0
    sampler.sc_vocab_end = 3
    hidden = torch.zeros(6, 5)
    local_scaled = torch.arange(18).reshape(6, 3).float() / 10
    remote_scaled = torch.arange(18, 36).reshape(6, 3).float() / 10
    fixed_gumbel = torch.arange(18).reshape(2, 3, 3).float() / 100
    expected_sample = torch.full((2, 3), 4, dtype=torch.int64)
    fixed_random_tokens = torch.tensor([[0, 1, 2], [3, 4, 5]])
    events: list[str] = []

    class LocalModel:
        def compute_local_logits(self, value: torch.Tensor) -> torch.Tensor:
            events.append("local_logits")
            assert value is hidden
            return local_scaled

        def compute_logits(self, value: torch.Tensor) -> torch.Tensor:
            raise AssertionError("sample-only must reuse gathered local logits")

    def reduce_local_sample(values, indices):
        events.append("local_sample_reduce")
        assert values.shape == (2, 3, 1)
        assert indices.shape == (2, 3, 1)
        expected_local_values = (
            (local_scaled.reshape(2, 3, 3) + fixed_gumbel).max(dim=-1).values
        )
        torch.testing.assert_close(values[..., 0], expected_local_values)
        return values, expected_sample.unsqueeze(-1)

    def gather_dense(local_logits):
        events.append("gather_dense_consumers")
        assert local_logits is not local_scaled
        torch.testing.assert_close(local_logits, local_scaled.reshape(2, 3, 3))
        return torch.cat([local_logits, remote_scaled.reshape(2, 3, 3)], dim=-1)

    def all_reduce_soft_embed(local_soft):
        events.append("all_reduce_soft_embed")
        full_scaled = torch.cat([local_scaled, remote_scaled], dim=-1).reshape(2, 3, 6)
        full_probs = full_scaled.log_softmax(dim=-1).exp()
        return local_soft + full_probs[..., 3:].to(remote_embed.dtype) @ remote_embed

    captured: dict[str, object] = {}
    apply_calls = 0

    def capture_apply(**kwargs) -> None:
        nonlocal apply_calls
        apply_calls += 1
        events.append("apply")
        captured.update(kwargs)

    sampler._apply_denoise_step_outputs = capture_apply
    sampler._compute_local_consumer_outputs = lambda *args, **kwargs: (
        _ for _ in ()
    ).throw(
        AssertionError("sample-only must not compute clean/entropy from local state")
    )
    monkeypatch.setattr(diffusion_gemma, "_tp_argmax_reduce_multi", reduce_local_sample)
    monkeypatch.setattr(diffusion_gemma, "_tp_all_gather_vocab", gather_dense)
    monkeypatch.setattr(diffusion_gemma, "_tp_all_reduce_sum", all_reduce_soft_embed)

    sampler._sample_only_dense_consumers(
        hidden,
        LocalModel(),
        decode_slots=torch.tensor([0, 1]),
        decode_idx=torch.tensor([0, 1]),
        all_slots=torch.tensor([0, 1]),
        valid_canvas_len=torch.tensor([3, 3]),
        is_committing=torch.tensor([False, False]),
        sampled=torch.zeros(2, 3, dtype=torch.int64),
        num_sampled=torch.zeros(2, dtype=torch.int32),
        gumbel=fixed_gumbel,
        random_tokens=fixed_random_tokens,
    )

    assert apply_calls == 1
    assert events == [
        "local_logits",
        "local_sample_reduce",
        "gather_dense_consumers",
        "all_reduce_soft_embed",
        "apply",
    ]
    full_scaled = torch.cat([local_scaled, remote_scaled], dim=-1)
    full_log_probs = full_scaled.log_softmax(dim=-1)
    full_probs = full_log_probs.exp()
    assert torch.equal(captured["new_tokens"], expected_sample)
    assert torch.equal(
        captured["argmax_tokens"], full_scaled.argmax(dim=-1).reshape(2, 3)
    )
    torch.testing.assert_close(
        captured["token_entropy"],
        -(full_probs * full_log_probs).sum(dim=-1).reshape(2, 3),
    )
    expected_soft = (
        full_probs[:, :3].to(local_embed.dtype) @ local_embed
        + full_probs[:, 3:].to(remote_embed.dtype) @ remote_embed
    )
    torch.testing.assert_close(captured["soft_embeds"], expected_soft.reshape(2, 3, -1))
    assert captured["consumer_sources"].as_source_tags() == {
        "sample": "local_state",
        "clean_argmax": "dense_logits",
        "entropy_remask": "dense_logits",
        "soft_self_conditioning": "dense_logits",
    }
    assert captured["random_tokens"] is fixed_random_tokens


def test_intermediate_endpoint_executes_local_sources_and_dense_soft_embed(
    monkeypatch: pytest.MonkeyPatch,
):
    sampler = _make_state_sampler(
        torch.device("cpu"), embed_weight=torch.arange(15).reshape(3, 5).float()
    )
    sampler.tp_size = 2
    sampler.vocab_size = 6
    sampler.sc_vocab_start = 0
    sampler.sc_vocab_end = 3
    hidden = torch.zeros(6, 5)
    local_scaled = torch.arange(18).reshape(6, 3).float() / 10
    remote_scaled = local_scaled + 0.5
    local_new = torch.ones(2, 3, dtype=torch.int64)
    local_argmax = torch.full((2, 3), 2, dtype=torch.int64)
    local_entropy = torch.full((2, 3), 0.75)
    events: list[str] = []

    class LocalModel:
        def compute_local_logits(self, value: torch.Tensor) -> torch.Tensor:
            events.append("local_logits")
            assert value is hidden
            return local_scaled

    def local_consumers(local_logits, decode_slots, *, include_soft_embeds):
        events.append("local_consumers")
        assert local_logits is local_scaled
        assert include_soft_embeds is False
        return local_scaled, local_new, local_argmax, local_entropy, None

    def gather(local_logits):
        events.append("gather_dense_soft_embed")
        return torch.cat([local_logits, remote_scaled], dim=-1)

    remote_embed = torch.arange(15, 30).reshape(3, 5).float()

    def all_reduce_sum(local_soft):
        events.append("all_reduce_soft_embed")
        full_probs = torch.cat([local_scaled, remote_scaled], dim=-1).softmax(dim=-1)
        remote_soft = full_probs[:, 3:] @ remote_embed
        return local_soft + remote_soft

    captured: dict[str, object] = {}

    def capture_apply(**kwargs) -> None:
        events.append("apply")
        captured.update(kwargs)

    sampler._compute_local_consumer_outputs = local_consumers
    sampler._apply_denoise_step_outputs = capture_apply
    monkeypatch.setattr(diffusion_gemma, "_tp_all_gather_vocab", gather)
    monkeypatch.setattr(diffusion_gemma, "_tp_all_reduce_sum", all_reduce_sum)
    sampler._sample_entropy_state_dense_soft_embed(
        hidden,
        LocalModel(),
        decode_slots=torch.tensor([0, 1]),
        decode_idx=torch.tensor([0, 1]),
        all_slots=torch.tensor([0, 1]),
        valid_canvas_len=torch.tensor([3, 3]),
        is_committing=torch.tensor([False, False]),
        sampled=torch.zeros(2, 3, dtype=torch.int64),
        num_sampled=torch.zeros(2, dtype=torch.int32),
    )

    assert events == [
        "local_logits",
        "local_consumers",
        "gather_dense_soft_embed",
        "all_reduce_soft_embed",
        "apply",
    ]
    assert captured["new_tokens"] is local_new
    assert captured["argmax_tokens"] is local_argmax
    assert captured["token_entropy"] is local_entropy
    sources = captured["consumer_sources"]
    assert sources.as_source_tags() == {
        "sample": "local_state",
        "clean_argmax": "local_state",
        "entropy_remask": "local_state",
        "soft_self_conditioning": "dense_logits",
    }
    full_probs = torch.cat([local_scaled, remote_scaled], dim=-1).softmax(dim=-1)
    expected_soft = (
        full_probs[:, :3] @ sampler.embed_weight + full_probs[:, 3:] @ remote_embed
    )
    torch.testing.assert_close(captured["soft_embeds"], expected_soft)


@pytest.mark.parametrize(
    "variant",
    ["sample_only_dense_consumers", "sample_entropy_state_dense_soft_embed"],
)
def test_local_state_validation_variant_synchronizes_initial_canvas_before_forward(
    monkeypatch: pytest.MonkeyPatch, variant: str
):
    monkeypatch.setenv("VLLM_DIFFUSION_GEMMA_VALIDATION_VARIANT", variant)
    monkeypatch.setattr(
        diffusion_gemma,
        "_DIFFUSION_GEMMA_VALIDATION_VARIANT",
        variant,
    )
    states = DiffusionGemmaRequestStates(
        max_num_reqs=1,
        canvas_length=3,
        vocab_size=7,
        hidden_size=5,
        device=torch.device("cpu"),
        max_denoising_steps=4,
        stability_threshold=2,
    )
    broadcasts: list[torch.Tensor] = []

    def broadcast(value: torch.Tensor) -> torch.Tensor:
        broadcasts.append(value.clone())
        return torch.full_like(value, 6)

    monkeypatch.setattr(diffusion_gemma, "_tp_broadcast_from_rank0", broadcast)
    states.init_canvas(np.array([0]))

    assert len(broadcasts) == 1
    assert torch.equal(states.canvas[0], torch.full((3,), 6, dtype=torch.int64))


@pytest.mark.skipif(
    os.environ.get("VLLM_TEST_REAL_TP") != "1",
    reason="run explicitly with torchrun and two FA4 GPUs",
)
def test_real_tp2_validation_variants_preserve_rank_state_and_sources(
    monkeypatch: pytest.MonkeyPatch,
):
    assert torch.cuda.device_count() >= 2
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 2
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    created_process_group = not torch.distributed.is_initialized()
    if created_process_group:
        torch.distributed.init_process_group(backend="nccl")

    trace_config = "/tmp/diffusion-gemma-task2-real-tp-trace.jsonl"
    rank_trace_path = Path(
        consumer_state_trace.resolve_rank_trace_path(trace_config, rank)
    )
    rank_trace_path.unlink(missing_ok=True)
    torch.distributed.barrier(device_ids=[local_rank])
    monkeypatch.setenv("VLLM_CONSUMER_STATE_PEAK_MEMORY_TRACE_JSONL", trace_config)
    trace_spec = validation_variant_spec("token_axis_logit_microbatch")
    with trace_output_phase_peak(
        "DiffusionGemmaTP2Helper",
        batch_phase="consumer_output_only",
        consumer_sources=trace_spec.as_source_tags(),
        contract_metadata=trace_spec.as_contract_metadata(),
        variant="token_axis_logit_microbatch",
    ):
        torch.zeros(1, device=device)
    torch.distributed.barrier(device_ids=[local_rank])
    for expected_rank in range(world_size):
        expected_path = Path(
            consumer_state_trace.resolve_rank_trace_path(trace_config, expected_rank)
        )
        record = json.loads(expected_path.read_text().splitlines()[-1])
        assert record["global_rank"] == expected_rank
        assert record["local_rank"] == expected_rank
        assert record["tp_rank"] == expected_rank
        assert record["cuda_device"] == expected_rank
        assert record["pid"] > 0
        assert record["component"] == "DiffusionGemmaTP2Helper"
        assert record["attribution_status"] == "complete"
        assert record["variant"] == "token_axis_logit_microbatch"
        assert record["component_elapsed_ms"] >= 0.0

    def all_reduce_sum(value: torch.Tensor) -> torch.Tensor:
        result = value.clone()
        torch.distributed.all_reduce(result)
        return result

    def all_gather_vocab(value: torch.Tensor) -> torch.Tensor:
        gathered = [torch.empty_like(value) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, value)
        return torch.cat(gathered, dim=-1)

    def argmax_reduce_multi(
        values: torch.Tensor, indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gathered_values = [torch.empty_like(values) for _ in range(world_size)]
        gathered_indices = [torch.empty_like(indices) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_values, values)
        torch.distributed.all_gather(gathered_indices, indices)
        rank_values = torch.stack(gathered_values)
        rank_indices = torch.stack(gathered_indices)
        max_values = rank_values.max(dim=0).values
        candidates = torch.where(
            rank_values == max_values.unsqueeze(0),
            rank_indices,
            torch.iinfo(rank_indices.dtype).max,
        )
        return max_values, candidates.min(dim=0).values

    def broadcast_from_rank0(value: torch.Tensor) -> torch.Tensor:
        result = value.clone()
        torch.distributed.broadcast(result, src=0)
        return result

    def assert_rank_equal(value: torch.Tensor) -> None:
        gathered = [torch.empty_like(value) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, value)
        for other in gathered[1:]:
            torch.testing.assert_close(gathered[0], other, rtol=0, atol=0)

    monkeypatch.setattr(diffusion_gemma, "_tp_all_reduce_sum", all_reduce_sum)
    monkeypatch.setattr(diffusion_gemma, "_tp_all_gather_vocab", all_gather_vocab)
    monkeypatch.setattr(diffusion_gemma, "_tp_argmax_reduce_multi", argmax_reduce_multi)
    monkeypatch.setattr(
        diffusion_gemma, "_tp_broadcast_from_rank0", broadcast_from_rank0
    )
    monkeypatch.setattr(diffusion_gemma, "_tp_rank_in_group", lambda: rank)
    monkeypatch.setenv("VLLM_DIFFUSION_GEMMA_TOKEN_MICROBATCH_ROWS", "2")

    global_vocab = 5
    shard_width = 3
    hidden_size = 4
    shard_start = rank * shard_width
    shard_end = min(shard_start + shard_width, global_vocab)
    local_width = shard_end - shard_start
    full_head = (
        torch.arange(global_vocab * hidden_size, device=device, dtype=torch.float32)
        .reshape(global_vocab, hidden_size)
        .sub_(7)
        .div_(13)
    )
    full_embed = (
        torch.arange(global_vocab * hidden_size, device=device, dtype=torch.float32)
        .reshape(global_vocab, hidden_size)
        .flip(0)
        .div_(17)
    )
    local_head = torch.zeros(shard_width, hidden_size, device=device)
    local_embed = torch.zeros(shard_width, hidden_size, device=device)
    local_head[:local_width] = full_head[shard_start:shard_end]
    local_embed[:local_width] = full_embed[shard_start:shard_end]
    hidden = (
        torch.arange(6 * hidden_size, device=device, dtype=torch.float32)
        .reshape(6, hidden_size)
        .sub_(5)
        .div_(11)
    )

    def compute_local_logits(value: torch.Tensor) -> torch.Tensor:
        result = F.linear(value, local_head)
        if local_width < shard_width:
            result[:, local_width:] = 10_000
        return result

    class TPModel:
        def compute_local_logits(self, value: torch.Tensor) -> torch.Tensor:
            return compute_local_logits(value)

        def compute_logits(self, value: torch.Tensor) -> torch.Tensor:
            return all_gather_vocab(compute_local_logits(value))[..., :global_vocab]

    common = {
        "decode_slots": torch.tensor([0, 1], device=device),
        "decode_idx": torch.tensor([0, 1], device=device),
        "all_slots": torch.tensor([0, 1], device=device),
        "valid_canvas_len": torch.tensor([3, 3], device=device),
        "is_committing": torch.tensor([False, False], device=device),
    }

    def make_source_capture(source_tags, original_apply):
        def capture_sources(**kwargs) -> None:
            source_tags.append(kwargs["consumer_sources"].as_source_tags())
            original_apply(**kwargs)

        return capture_sources

    def make_gumbel_rows(fixed_gumbel):
        return lambda start, stop: fixed_gumbel[start:stop]

    try:
        for variant in (
            "token_axis_logit_microbatch",
            "sample_only_dense_consumers",
            "sample_entropy_state_dense_soft_embed",
        ):
            monkeypatch.setenv("VLLM_DIFFUSION_GEMMA_VALIDATION_VARIANT", variant)
            sampler = _make_state_sampler(device, embed_weight=local_embed)
            sampler.vocab_size = global_vocab
            sampler.tp_size = world_size
            sampler.embed_vocab_start_index = shard_start
            sampler.embed_vocab_end_index = shard_end
            sampler.sc_vocab_start = shard_start
            sampler.sc_vocab_end = shard_end
            sampler.seed = 23
            sampler._local_vocab_generator = None
            source_tags: list[dict[str, str]] = []
            original_apply = sampler._apply_denoise_step_outputs
            sampler._apply_denoise_step_outputs = make_source_capture(
                source_tags, original_apply
            )
            sampled = torch.zeros(2, 3, dtype=torch.int64, device=device)
            num_sampled = torch.zeros(2, dtype=torch.int32, device=device)
            if variant == "token_axis_logit_microbatch":
                fixed_gumbel = (
                    torch.arange(6 * global_vocab, device=device).reshape(
                        6, global_vocab
                    )
                    / 31
                )
                fixed_random_tokens = torch.tensor(
                    [[0, 1, 2], [3, 4, 0]], device=device
                )
                sampler._sample_token_axis_logit_microbatch(
                    hidden,
                    TPModel(),
                    sampled=sampled,
                    num_sampled=num_sampled,
                    gumbel_for_rows=make_gumbel_rows(fixed_gumbel),
                    random_tokens=fixed_random_tokens,
                    **common,
                )
                expected_sources = validation_variant_spec(variant).as_source_tags()
            elif variant == "sample_only_dense_consumers":
                full_gumbel = (
                    torch.arange(6 * global_vocab, device=device).reshape(
                        2, 3, global_vocab
                    )
                    / 31
                )
                local_gumbel = torch.zeros(
                    2, 3, shard_width, device=device, dtype=full_gumbel.dtype
                )
                local_gumbel[..., :local_width] = full_gumbel[
                    ..., shard_start:shard_end
                ]
                fixed_random_tokens = torch.tensor(
                    [[0, 1, 2], [3, 4, 0]], device=device
                )
                sampler._sample_only_dense_consumers(
                    hidden,
                    TPModel(),
                    sampled=sampled,
                    num_sampled=num_sampled,
                    gumbel=local_gumbel,
                    random_tokens=fixed_random_tokens,
                    **common,
                )
                expected_sources = validation_variant_spec(variant).as_source_tags()
            else:
                torch.manual_seed(29)
                sampler._sample_entropy_state_dense_soft_embed(
                    hidden,
                    TPModel(),
                    sampled=sampled,
                    num_sampled=num_sampled,
                    **common,
                )
                expected_sources = validation_variant_spec(variant).as_source_tags()

            assert source_tags == [expected_sources]
            assert sampler.diffusion_states.argmax_canvas.max().item() < global_vocab
            for value in _state_snapshot(sampler).values():
                assert_rank_equal(value)
            assert_rank_equal(sampled)
            assert_rank_equal(num_sampled)

            empty_sampler = DiffusionSampler.__new__(DiffusionSampler)
            empty_sampler.canvas_length = 3
            empty_batch = SimpleNamespace(
                num_draft_tokens=1,
                num_reqs=1,
                idx_mapping_np=np.array([0]),
                cu_num_logits_np=np.array([0, 0]),
            )
            assert (
                empty_sampler.sample_from_hidden_states(
                    torch.empty(0, hidden_size, device=device),
                    empty_batch,
                    TPModel(),
                )
                is None
            )
            torch.distributed.barrier(device_ids=[local_rank])
    finally:
        if created_process_group:
            torch.distributed.destroy_process_group()
