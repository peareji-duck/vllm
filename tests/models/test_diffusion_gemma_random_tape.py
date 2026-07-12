from pathlib import Path

import torch

from vllm.model_executor.models.diffusion_gemma import (
    DiffusionGemmaRequestStates,
    _counter_based_random_tokens,
    _counter_based_uniform,
    _diffusion_tape_row_keys,
    _summarize_trajectory_tensor,
)


def test_counter_uniform_is_invariant_to_vocab_partitioning():
    slots = torch.tensor([2, 7], dtype=torch.int64)
    steps = torch.tensor([3, 11], dtype=torch.int64)
    row_keys = _diffusion_tape_row_keys(slots, steps, canvas_length=4)

    dense = _counter_based_uniform(
        row_keys,
        vocab_start=0,
        vocab_width=17,
        seed=123,
        dtype=torch.float32,
    )
    sharded = torch.cat(
        [
            _counter_based_uniform(
                row_keys,
                vocab_start=start,
                vocab_width=width,
                seed=123,
                dtype=torch.float32,
            )
            for start, width in ((0, 5), (5, 7), (12, 5))
        ],
        dim=-1,
    )

    torch.testing.assert_close(sharded, dense, rtol=0, atol=0)
    assert torch.all((dense > 0) & (dense < 1))


def test_tape_row_keys_change_with_slot_step_and_canvas_position():
    row_keys = _diffusion_tape_row_keys(
        torch.tensor([1, 2]),
        torch.tensor([3, 4]),
        canvas_length=3,
    )

    assert row_keys.shape == (2, 3)
    assert torch.unique(row_keys).numel() == row_keys.numel()
    assert not torch.equal(row_keys[0], row_keys[1])


def test_counter_random_tokens_are_deterministic_and_in_range():
    row_keys = _diffusion_tape_row_keys(
        torch.tensor([5, 9]),
        torch.tensor([0, 8]),
        canvas_length=6,
    )

    first = _counter_based_random_tokens(row_keys, seed=77, vocab_size=101)
    second = _counter_based_random_tokens(row_keys, seed=77, vocab_size=101)
    changed = _counter_based_random_tokens(row_keys, seed=78, vocab_size=101)

    assert torch.equal(first, second)
    assert not torch.equal(first, changed)
    assert first.dtype == torch.int64
    assert int(first.min()) >= 0
    assert int(first.max()) < 101


def test_fixed_random_tape_initializes_canvas_independent_of_global_rng():
    kwargs = dict(
        max_num_reqs=4,
        canvas_length=6,
        vocab_size=101,
        max_denoising_steps=16,
        device=torch.device("cpu"),
        hidden_size=8,
        stability_threshold=2,
        fixed_random_tape=True,
        seed=77,
    )
    first = DiffusionGemmaRequestStates(**kwargs)
    second = DiffusionGemmaRequestStates(**kwargs)

    torch.manual_seed(1)
    first.add_request(1, random_tape_request_key=12345)
    torch.manual_seed(999)
    second.add_request(3, random_tape_request_key=12345)

    assert torch.equal(first.canvas[1], second.canvas[3])
    assert int(first.canvas[1].min()) >= 0
    assert int(first.canvas[1].max()) < 101


def test_counter_gumbel_winner_is_partition_invariant():
    logits = torch.tensor(
        [[[0.1, 0.5, -0.2, 1.3, 0.7, -1.0, 0.9]]], dtype=torch.float32
    )
    keys = _diffusion_tape_row_keys(
        torch.tensor([4]), torch.tensor([6]), canvas_length=1
    )
    dense_u = _counter_based_uniform(
        keys, vocab_start=0, vocab_width=7, seed=9, dtype=torch.float32
    )
    dense_winner = (logits - torch.log(-torch.log(dense_u))).argmax(dim=-1)

    candidates = []
    for start, width in ((0, 2), (2, 3), (5, 2)):
        local_u = _counter_based_uniform(
            keys,
            vocab_start=start,
            vocab_width=width,
            seed=9,
            dtype=torch.float32,
        )
        local_scores = logits[..., start : start + width] - torch.log(
            -torch.log(local_u)
        )
        value, index = local_scores.max(dim=-1)
        candidates.append((value, index + start))
    values = torch.stack([candidate[0] for candidate in candidates], dim=-1)
    indices = torch.stack([candidate[1] for candidate in candidates], dim=-1)
    sharded_winner = indices.gather(-1, values.argmax(dim=-1, keepdim=True)).squeeze(-1)

    assert torch.equal(sharded_winner, dense_winner)


def test_fixed_random_tape_is_gated_and_used_by_both_sampler_paths():
    root = Path(__file__).resolve().parents[2]
    model_source = (root / "vllm/model_executor/models/diffusion_gemma.py").read_text(
        encoding="utf-8"
    )
    env_source = (root / "vllm/envs.py").read_text(encoding="utf-8")

    assert "VLLM_DIFFUSION_GEMMA_FIXED_RANDOM_TAPE" in env_source
    assert "self.fixed_random_tape" in model_source
    assert model_source.count("_counter_based_uniform(") >= 3
    assert model_source.count("_counter_based_random_tokens(") >= 3


def test_trajectory_tensor_summary_is_stable_and_finite():
    tensor = torch.tensor([[1.0, -2.0], [3.0, 4.0]], dtype=torch.float32)

    first = _summarize_trajectory_tensor(tensor)
    second = _summarize_trajectory_tensor(tensor.clone())

    assert first == second
    assert first["shape"] == [2, 2]
    assert first["sum"] == 6.0
    assert first["max_abs"] == 4.0
    assert len(first["sha256"]) == 64


def test_trajectory_recorder_is_debug_gated():
    root = Path(__file__).resolve().parents[2]
    model_source = (root / "vllm/model_executor/models/diffusion_gemma.py").read_text(
        encoding="utf-8"
    )
    env_source = (root / "vllm/envs.py").read_text(encoding="utf-8")

    assert "VLLM_DIFFUSION_GEMMA_TRAJECTORY_DIR" in env_source
    assert "_record_trajectory" in model_source
    assert "trajectory_dir" in model_source
