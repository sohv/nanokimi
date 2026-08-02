"""Memmap batch loader over the flat .bin token files written by the prepare scripts.

The array is memory-mapped per call rather than held open, which is what nanoGPT does:
holding a memmap across a long run leaks memory as the OS page cache grows.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)

# uint16 is enough for GPT-2's 50257-token vocabulary and halves the file size.
TOKEN_DTYPE = np.uint16


def load_meta(data_dir: Path | str) -> dict[str, Any]:
    """Read the dataset metadata written alongside train.bin / val.bin."""
    data_dir = Path(data_dir)
    meta_json = data_dir / "meta.json"
    if meta_json.exists():
        return json.loads(meta_json.read_text())

    meta_pkl = data_dir / "meta.pkl"
    if meta_pkl.exists():
        with meta_pkl.open("rb") as handle:
            return pickle.load(handle)

    raise FileNotFoundError(f"no meta.json or meta.pkl in {data_dir}; run scripts/prepare_data.py first")


def split_path(data_dir: Path | str, split: str) -> Path:
    path = Path(data_dir) / f"{split}.bin"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/prepare_data.py first")
    return path


def split_tokens(data_dir: Path | str, split: str) -> int:
    """Number of tokens in a split, read from the file size."""
    return split_path(data_dir, split).stat().st_size // np.dtype(TOKEN_DTYPE).itemsize


def get_batch(
    data_dir: Path | str,
    split: str,
    batch_size: int,
    block_size: int,
    device: str = "cpu",
    generator: np.random.Generator | None = None,
    max_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of (inputs, targets) contiguous windows.

    `max_tokens` restricts sampling to a prefix of the split, which is how the smaller
    model sizes train on a strict prefix of the same tokenized slice as the larger ones.
    """
    data = np.memmap(split_path(data_dir, split), dtype=TOKEN_DTYPE, mode="r")
    limit = len(data) if max_tokens is None else min(len(data), max_tokens)
    if limit <= block_size + 1:
        raise ValueError(f"split '{split}' has {limit} usable tokens, need more than block_size+1={block_size + 1}")

    rng = generator if generator is not None else np.random.default_rng()
    starts = rng.integers(0, limit - block_size - 1, size=batch_size)

    x = np.stack([data[i : i + block_size].astype(np.int64) for i in starts])
    y = np.stack([data[i + 1 : i + 1 + block_size].astype(np.int64) for i in starts])

    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)
    if device.startswith("cuda"):
        # pin + non_blocking lets the copy overlap with compute
        return x_tensor.pin_memory().to(device, non_blocking=True), y_tensor.pin_memory().to(device, non_blocking=True)
    return x_tensor.to(device), y_tensor.to(device)
