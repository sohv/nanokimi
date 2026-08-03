"""Data pipeline tests.

The scaling study rests on two properties: data order is reproducible from the seed,
and each model size trains on a strict prefix of the same tokenized slice. If either
breaks, differences between sizes stop being attributable to scale.
"""

import json

import numpy as np
import pytest
import torch

from nanokimi.data.loader import TOKEN_DTYPE, get_batch, load_meta, split_path, split_tokens
from nanokimi.utils.seeding import set_seed
from tests.conftest import REAL_DATA_DIR, VOCAB, requires_real_data


def test_meta_matches_file_sizes(toy_data_dir):
    meta = load_meta(toy_data_dir)
    for split in ("train", "val"):
        assert split_tokens(toy_data_dir, split) == meta[f"{split}_tokens"]


def test_missing_split_fails_loudly(toy_data_dir):
    with pytest.raises(FileNotFoundError):
        split_path(toy_data_dir, "test")


def test_missing_meta_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_meta(tmp_path)


def test_batch_shapes_and_dtype(toy_data_dir):
    x, y = get_batch(toy_data_dir, "train", batch_size=4, block_size=16, device="cpu")
    assert x.shape == (4, 16)
    assert y.shape == (4, 16)
    # int64 is what nn.Embedding requires; uint16 on disk must be widened on read
    assert x.dtype == torch.int64


def test_targets_are_inputs_shifted_by_one(toy_data_dir):
    x, y = get_batch(toy_data_dir, "train", batch_size=8, block_size=16, device="cpu")
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_token_ids_stay_in_vocab(toy_data_dir):
    x, y = get_batch(toy_data_dir, "train", batch_size=32, block_size=32, device="cpu")
    assert int(x.max()) < VOCAB and int(x.min()) >= 0
    assert int(y.max()) < VOCAB and int(y.min()) >= 0


def test_set_seed_makes_batches_reproducible(toy_data_dir):
    """Regression: a bare np.random.default_rng() default seeded from OS entropy and
    silently ignored set_seed, so training data order was not reproducible."""
    set_seed(42)
    a, _ = get_batch(toy_data_dir, "train", batch_size=4, block_size=16, device="cpu")
    set_seed(42)
    b, _ = get_batch(toy_data_dir, "train", batch_size=4, block_size=16, device="cpu")
    assert torch.equal(a, b)


def test_different_seeds_give_different_batches(toy_data_dir):
    set_seed(1)
    a, _ = get_batch(toy_data_dir, "train", batch_size=4, block_size=16, device="cpu")
    set_seed(2)
    b, _ = get_batch(toy_data_dir, "train", batch_size=4, block_size=16, device="cpu")
    assert not torch.equal(a, b)


def test_explicit_generator_is_reproducible_and_isolated(toy_data_dir):
    kwargs = dict(batch_size=4, block_size=16, device="cpu")
    a, _ = get_batch(toy_data_dir, "train", generator=np.random.default_rng(7), **kwargs)
    b, _ = get_batch(toy_data_dir, "train", generator=np.random.default_rng(7), **kwargs)
    assert torch.equal(a, b)

    # a caller's own generator must not be perturbed by the global seed
    gen = np.random.default_rng(7)
    set_seed(999)
    c, _ = get_batch(toy_data_dir, "train", generator=gen, **kwargs)
    assert torch.equal(a, c)


def test_max_tokens_restricts_sampling_to_a_prefix(toy_data_dir):
    """Each smaller model must read a strict prefix, not a fresh random sample."""
    data = np.memmap(split_path(toy_data_dir, "train"), dtype=TOKEN_DTYPE, mode="r")
    prefix_len = 5_000
    prefix = set(np.asarray(data[:prefix_len]).tolist())

    gen = np.random.default_rng(0)
    for _ in range(30):
        x, _ = get_batch(
            toy_data_dir, "train", batch_size=8, block_size=16, device="cpu",
            generator=gen, max_tokens=prefix_len,
        )
        assert set(x.flatten().tolist()) <= prefix


def test_prefixes_are_nested_across_budgets(toy_data_dir):
    """A smaller budget's sampling window must be contained in a larger one."""
    data = np.memmap(split_path(toy_data_dir, "train"), dtype=TOKEN_DTYPE, mode="r")
    small, large = np.asarray(data[:1000]), np.asarray(data[:5000])
    assert np.array_equal(small, large[:1000])


def test_budget_larger_than_split_fails_loudly(toy_data_dir):
    with pytest.raises(ValueError, match="usable tokens"):
        get_batch(toy_data_dir, "train", batch_size=2, block_size=64, device="cpu", max_tokens=32)


def test_block_size_boundary_is_respected(toy_data_dir):
    """block_size+1 tokens is the minimum usable window; one fewer must fail."""
    with pytest.raises(ValueError):
        get_batch(toy_data_dir, "train", batch_size=1, block_size=16, device="cpu", max_tokens=17)
    x, _ = get_batch(toy_data_dir, "train", batch_size=1, block_size=16, device="cpu", max_tokens=64)
    assert x.shape == (1, 16)


@requires_real_data()
def test_real_slice_is_internally_consistent():
    meta = load_meta(REAL_DATA_DIR)
    for split in ("train", "val"):
        assert split_tokens(REAL_DATA_DIR, split) == meta[f"{split}_tokens"]


@requires_real_data()
def test_real_slice_token_ids_fit_uint16():
    """uint16 silently wraps above 65535; the vocab must stay under it."""
    meta = load_meta(REAL_DATA_DIR)
    data = np.memmap(split_path(REAL_DATA_DIR, "train"), dtype=TOKEN_DTYPE, mode="r")
    idx = np.linspace(0, len(data) - 1, 2_000_000).astype(np.int64)
    sample = np.asarray(data[idx])
    assert int(sample.max()) < meta["vocab_size"]


@requires_real_data()
def test_real_slice_covers_every_planned_budget():
    """All four configs must fit as prefixes of the one tokenized slice."""
    meta = load_meta(REAL_DATA_DIR)
    budgets = {
        "25m": 500_000_000,
        "50m": 1_000_000_000,
        "125m": 2_460_000_000,
        "200m": 4_010_000_000,
    }
    for name, budget in budgets.items():
        assert budget <= meta["train_tokens"], f"{name} needs {budget:,}, slice has {meta['train_tokens']:,}"
