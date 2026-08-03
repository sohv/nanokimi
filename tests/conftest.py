"""Shared fixtures.

Everything here is tiny and synthetic so the suite stays fast. Tests that need the
real 4B-token slice are skipped when it is absent, so a fresh clone is not red.
"""

import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_DATA_DIR = REPO_ROOT / "data" / "processed" / "openwebtext"

VOCAB = 512

TOY_MODEL = dict(
    vocab_size=VOCAB,
    block_size=32,
    n_layer=2,
    n_head=4,
    n_embd=128,
    dropout=0.0,
    bias=True,
    use_moe=True,
    num_experts=4,
    top_k_experts=2,
    expert_capacity=8,
    apply_expert_capacity=False,
    load_balance_loss_coef=0.01,
    use_latent_attention=True,
    kv_lora_rank=128,
    q_lora_rank=384,
    qk_nope_head_dim=32,
    qk_rope_head_dim=16,
    v_head_dim=32,
    rope_theta=50000.0,
    attention_bias=False,
)


@pytest.fixture(scope="session")
def toy_data_dir(tmp_path_factory) -> Path:
    """A tiny dataset in the same on-disk format the real pipeline produces."""
    data_dir = tmp_path_factory.mktemp("toy_data")
    rng = np.random.default_rng(0)

    train = rng.integers(0, VOCAB, size=200_000, dtype=np.uint16)
    val = rng.integers(0, VOCAB, size=20_000, dtype=np.uint16)
    train.tofile(data_dir / "train.bin")
    val.tofile(data_dir / "val.bin")

    meta = {
        "dataset": "synthetic",
        "encoding": "gpt2",
        "vocab_size": VOCAB,
        "train_tokens": int(train.size),
        "val_tokens": int(val.size),
        "dtype": "uint16",
    }
    (data_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return data_dir


@pytest.fixture
def toy_model_config() -> dict:
    return dict(TOY_MODEL)


def requires_real_data():
    """Skip marker for tests that need the tokenized OpenWebText slice."""
    return pytest.mark.skipif(
        not (REAL_DATA_DIR / "train.bin").exists(),
        reason="run scripts/prepare_data.py --dataset openwebtext first",
    )
