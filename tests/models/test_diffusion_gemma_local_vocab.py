# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.models.diffusion_gemma import (
    _local_vocab_argmax_tokens,
    _local_vocab_logprobs_dense_fallback,
    _local_vocab_requires_full_logprobs,
    _local_vocab_softmax_stats,
    _reduce_argmax_from_gathered_pairs,
    _vocab_parallel_soft_embed,
)


def _packed_softmax_parts(
    scaled_logits: torch.Tensor,
    embed_weight: torch.Tensor,
    global_max: torch.Tensor,
) -> torch.Tensor:
    logits_f = scaled_logits.float()
    local_exp = torch.exp(logits_f - global_max.unsqueeze(-1))
    local_sum_exp = local_exp.sum(dim=-1)
    weighted_logits = (local_exp * logits_f).sum(dim=-1)
    soft_part = torch.matmul(local_exp.to(embed_weight.dtype), embed_weight).float()
    return torch.cat(
        [
            local_sum_exp.unsqueeze(-1),
            weighted_logits.unsqueeze(-1),
            soft_part,
        ],
        dim=-1,
    )


def test_local_vocab_softmax_stats_matches_dense_softmax_with_padding():
    torch.manual_seed(0)
    scaled = torch.randn(2, 3, 8)
    embed_weight = torch.randn(8, 5)
    normalizer = torch.tensor(1.7)

    # Rank 1 owns only three real vocab rows but has two padded rows in its
    # local tensor. The helper must ignore padded logits/weights.
    local_logits = torch.cat(
        [scaled[..., 5:], torch.full((*scaled.shape[:-1], 2), 123.0)], dim=-1
    )
    local_weight = torch.cat([embed_weight[5:], torch.zeros(2, 5)], dim=0)
    global_max = scaled.float().max(dim=-1).values
    rank0_pack = _packed_softmax_parts(scaled[..., :5], embed_weight[:5], global_max)

    entropy, soft_embeds = _local_vocab_softmax_stats(
        local_logits,
        local_weight,
        normalizer,
        local_vocab_width=3,
        global_max=global_max,
        all_reduce_sum_fn=lambda packed: packed + rank0_pack,
    )

    dense_log_probs = scaled.log_softmax(dim=-1)
    dense_probs = dense_log_probs.exp()
    expected_entropy = -(dense_probs * dense_log_probs).sum(dim=-1)
    expected_soft = torch.matmul(dense_probs, embed_weight) * normalizer

    torch.testing.assert_close(entropy, expected_entropy)
    torch.testing.assert_close(soft_embeds, expected_soft)


def test_local_vocab_argmax_ignores_padded_vocab_columns():
    logits = torch.tensor([[[0.0, 3.0, 2.0, 99.0]]])

    values, indices = _local_vocab_argmax_tokens(
        logits,
        vocab_start_index=16,
        local_vocab_width=3,
    )

    torch.testing.assert_close(values, torch.tensor([[3.0]]))
    torch.testing.assert_close(indices, torch.tensor([[17]]))


def test_vocab_parallel_soft_embed_matches_dense_with_all_reduce():
    torch.manual_seed(1)
    probs = torch.randn(2, 3, 7).softmax(dim=-1)
    embed_weight = torch.randn(7, 4)
    normalizer = torch.tensor(0.75)

    rank1_part = torch.matmul(probs[..., 4:], embed_weight[4:])
    soft_embeds = _vocab_parallel_soft_embed(
        probs,
        embed_weight[:4],
        normalizer,
        vocab_start_index=0,
        vocab_end_index=4,
        all_reduce_fn=lambda local: local + rank1_part,
    )

    expected = torch.matmul(probs, embed_weight) * normalizer
    torch.testing.assert_close(soft_embeds, expected)


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
