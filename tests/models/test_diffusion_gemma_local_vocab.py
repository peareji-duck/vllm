# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.models.diffusion_gemma import (
    _local_vocab_argmax_tokens,
    _local_vocab_generator_seed,
    _local_vocab_logprobs_dense_fallback,
    _local_vocab_requires_full_logprobs,
    _local_vocab_softmax_stats,
    _reduce_argmax_from_gathered_pairs,
)


def test_local_vocab_generator_seed_depends_on_engine_seed_and_rank():
    assert _local_vocab_generator_seed(17, 0) != _local_vocab_generator_seed(18, 0)
    assert _local_vocab_generator_seed(17, 0) != _local_vocab_generator_seed(17, 1)
    assert _local_vocab_generator_seed(17, 0) == _local_vocab_generator_seed(17, 0)


def _reduction_parts(
    scaled_logits: torch.Tensor,
    embed_weight: torch.Tensor,
    global_max: torch.Tensor,
    global_sum_exp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits_f = scaled_logits.float()
    local_exp = torch.exp(logits_f - global_max.unsqueeze(-1))
    local_sum_exp = local_exp.sum(dim=-1)
    weighted_logits = (local_exp * logits_f).sum(dim=-1)
    stats = torch.stack([local_sum_exp, weighted_logits], dim=-1)
    local_probs = (local_exp / global_sum_exp.unsqueeze(-1)).to(embed_weight.dtype)
    soft_part = torch.matmul(local_probs, embed_weight).float()
    return stats, soft_part


def _make_reducer(other_stats: torch.Tensor, other_soft: torch.Tensor):
    def reduce(value: torch.Tensor) -> torch.Tensor:
        return value + (other_stats if value.shape[-1] == 2 else other_soft)

    return reduce


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is required"
            ),
        ),
    ],
)
def test_local_vocab_softmax_stats_matches_dense_softmax_with_padding(device: str):
    torch.manual_seed(0)
    scaled = torch.randn(2, 3, 8, device=device)
    embed_weight = torch.randn(8, 5, device=device)
    normalizer = torch.tensor(1.7, device=device)

    # Rank 1 owns only three real vocab rows but has two padded rows in its
    # local tensor. The helper must ignore padded logits/weights.
    local_logits = torch.cat(
        [
            scaled[..., 5:],
            torch.full((*scaled.shape[:-1], 2), 123.0, device=device),
        ],
        dim=-1,
    )
    local_weight = torch.cat(
        [embed_weight[5:], torch.zeros(2, 5, device=device)], dim=0
    )
    global_max = scaled.float().max(dim=-1).values
    global_sum_exp = torch.exp(scaled.float() - global_max.unsqueeze(-1)).sum(dim=-1)
    rank0_stats, rank0_soft = _reduction_parts(
        scaled[..., :5], embed_weight[:5], global_max, global_sum_exp
    )

    entropy, soft_embeds = _local_vocab_softmax_stats(
        local_logits,
        local_weight,
        normalizer,
        local_vocab_width=3,
        global_max=global_max,
        all_reduce_sum_fn=_make_reducer(rank0_stats, rank0_soft),
    )

    dense_log_probs = scaled.log_softmax(dim=-1)
    dense_probs = dense_log_probs.exp()
    expected_entropy = -(dense_probs * dense_log_probs).sum(dim=-1)
    expected_soft = torch.matmul(dense_probs, embed_weight) * normalizer

    torch.testing.assert_close(entropy, expected_entropy)
    torch.testing.assert_close(soft_embeds, expected_soft)


@pytest.mark.parametrize("seed", [0, 1, 20260622])
def test_local_vocab_softmax_stats_random_shards_match_dense(seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows, canvas, vocab_size, hidden = 4, 5, 29, 11
    scaled = torch.randn(rows, canvas, vocab_size, generator=generator)
    embed_weight = torch.randn(vocab_size, hidden, generator=generator)
    normalizer = torch.tensor(1.25)
    shard_widths = [7, 3, 11, 8]

    global_max = scaled.float().max(dim=-1).values
    global_sum_exp = torch.exp(scaled.float() - global_max.unsqueeze(-1)).sum(dim=-1)
    parts = []
    start = 0
    for width in shard_widths:
        end = start + width
        parts.append(
            _reduction_parts(
                scaled[..., start:end],
                embed_weight[start:end],
                global_max,
                global_sum_exp,
            )
        )
        start = end

    rank = 2
    rank_start = sum(shard_widths[:rank])
    rank_width = shard_widths[rank]
    rank_logits = torch.cat(
        [
            scaled[..., rank_start : rank_start + rank_width],
            torch.full((rows, canvas, 2), 9999.0),
        ],
        dim=-1,
    )
    rank_weight = torch.cat(
        [
            embed_weight[rank_start : rank_start + rank_width],
            torch.zeros(2, hidden),
        ],
        dim=0,
    )
    other_stats = sum(part[0] for index, part in enumerate(parts) if index != rank)
    other_soft = sum(part[1] for index, part in enumerate(parts) if index != rank)

    entropy, soft_embeds = _local_vocab_softmax_stats(
        rank_logits,
        rank_weight,
        normalizer,
        local_vocab_width=rank_width,
        global_max=global_max,
        all_reduce_sum_fn=_make_reducer(other_stats, other_soft),
    )

    dense_log_probs = scaled.log_softmax(dim=-1)
    dense_probs = dense_log_probs.exp()
    expected_entropy = -(dense_probs * dense_log_probs).sum(dim=-1)
    expected_soft = torch.matmul(dense_probs, embed_weight) * normalizer

    torch.testing.assert_close(entropy, expected_entropy, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(soft_embeds, expected_soft, atol=1e-5, rtol=1e-5)


def test_local_vocab_soft_embed_matches_dense_bfloat16_quantization_order():
    generator = torch.Generator(device="cpu").manual_seed(17)
    scaled = torch.randn(2, 3, 32, generator=generator) * 3
    embed_weight = torch.randn(32, 16, generator=generator).to(torch.bfloat16)
    normalizer = torch.tensor(1.75, dtype=torch.bfloat16)
    global_max = scaled.max(dim=-1).values
    global_sum_exp = torch.exp(scaled - global_max.unsqueeze(-1)).sum(dim=-1)

    rank0_stats, rank0_soft = _reduction_parts(
        scaled[..., :16], embed_weight[:16], global_max, global_sum_exp
    )
    _, actual = _local_vocab_softmax_stats(
        scaled[..., 16:],
        embed_weight[16:],
        normalizer,
        global_max=global_max,
        all_reduce_sum_fn=_make_reducer(rank0_stats, rank0_soft),
    )

    dense_probs = scaled.log_softmax(dim=-1).exp().to(torch.bfloat16)
    expected = torch.matmul(dense_probs, embed_weight) * normalizer

    torch.testing.assert_close(actual, expected, rtol=0, atol=1 / 64)


def test_local_vocab_argmax_ignores_padded_vocab_columns():
    logits = torch.tensor([[[0.0, 3.0, 2.0, 99.0]]])

    values, indices = _local_vocab_argmax_tokens(
        logits,
        vocab_start_index=16,
        local_vocab_width=3,
    )

    torch.testing.assert_close(values, torch.tensor([[3.0]]))
    torch.testing.assert_close(indices, torch.tensor([[17]]))


def test_reduce_argmax_from_gathered_pairs_keeps_first_rank_on_tie():
    local_values = torch.empty(2)
    local_indices = torch.empty(2, dtype=torch.int64)
    gathered_pairs = torch.tensor(
        [
            [[4.0, 10.0], [4.0, 11.0], [3.0, 12.0]],
            [[1.0, 20.0], [5.0, 21.0], [2.0, 22.0]],
        ]
    )

    values, indices = _reduce_argmax_from_gathered_pairs(
        local_values, local_indices, gathered_pairs
    )

    torch.testing.assert_close(values, torch.tensor([4.0, 5.0]))
    torch.testing.assert_close(indices, torch.tensor([10, 21]))


def test_local_vocab_full_logprobs_gate_requires_sampled_logprob_for_real_requests():
    assert not _local_vocab_requires_full_logprobs(-1)
    assert _local_vocab_requires_full_logprobs(0, req_ids=["real_request"])
    assert not _local_vocab_requires_full_logprobs(
        0, req_ids=["_warmup_a", "_warmup_b"]
    )
    assert not _local_vocab_requires_full_logprobs(
        5, req_ids=["_warmup_a", "_warmup_b"]
    )
    assert _local_vocab_requires_full_logprobs(5, req_ids=["real_request"])


def test_local_vocab_logprobs_dense_fallback_gathers_and_truncates_logits():
    logits = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(6, 4)

    def gather_fn(sharded: torch.Tensor) -> torch.Tensor:
        assert sharded.shape == (2, 3, 4)
        return torch.cat([sharded, sharded + 100, sharded + 200], dim=-1)

    full_logits = _local_vocab_logprobs_dense_fallback(
        logits,
        num_decode=2,
        canvas_length=3,
        vocab_size=10,
        all_gather_fn=gather_fn,
    )

    expected = torch.cat(
        [
            logits.reshape(2, 3, 4),
            logits.reshape(2, 3, 4) + 100,
            logits.reshape(2, 3, 4) + 200,
        ],
        dim=-1,
    )[..., :10].reshape(6, 10)
    torch.testing.assert_close(full_logits, expected)
