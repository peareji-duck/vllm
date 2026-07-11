import pytest
import torch

from vllm.model_executor.models.diffusion_gemma import (
    _dense_consumers_from_local_scaled_logits,
    _validate_consumer_state_ablation_mode,
)


def test_dense_consumers_from_local_logits_matches_dense_operation_order():
    full_logits = torch.tensor(
        [[[1.0, -0.5, 0.25, 2.0, -1.0, 0.75]]], dtype=torch.float32
    )
    embed_weight = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [-1.0, 0.5],
            [0.25, -0.5],
            [0.75, 1.25],
        ],
        dtype=torch.bfloat16,
    )
    probs = full_logits.log_softmax(dim=-1).exp()
    other_soft_part = probs[..., 3:].to(torch.bfloat16) @ embed_weight[3:]
    gathered = []

    def fake_gather(local_logits):
        gathered.append(local_logits.clone())
        return full_logits

    entropy, soft_embed = _dense_consumers_from_local_scaled_logits(
        full_logits[..., :3],
        embed_weight[:3],
        torch.tensor(1.75, dtype=torch.bfloat16),
        vocab_size=6,
        sc_vocab_start=0,
        sc_vocab_end=3,
        tp_size=2,
        all_gather_fn=fake_gather,
        all_reduce_sum_fn=lambda local: local + other_soft_part,
    )

    expected_log_probs = full_logits.log_softmax(dim=-1)
    expected_probs = expected_log_probs.exp()
    expected_entropy = -(expected_probs * expected_log_probs).sum(dim=-1)
    expected_soft_embed = (
        expected_probs[..., :3].to(torch.bfloat16) @ embed_weight[:3]
    ) + other_soft_part
    expected_soft_embed = expected_soft_embed * 1.75

    assert len(gathered) == 1
    torch.testing.assert_close(entropy, expected_entropy)
    torch.testing.assert_close(soft_embed, expected_soft_embed, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["full", "sample_only_dense_consumers"])
def test_consumer_state_ablation_mode_accepts_declared_modes(mode):
    assert _validate_consumer_state_ablation_mode(mode) == mode


def test_consumer_state_ablation_mode_rejects_unknown_mode():
    with pytest.raises(ValueError, match="consumer-state ablation mode"):
        _validate_consumer_state_ablation_mode("sample_only")
