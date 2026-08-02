"""Tokenize tiny-shakespeare into the same .bin format as OpenWebText.

Used for the cheap smoke run that validates the stack before committing GPU time.
"""

import json
import logging
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import tiktoken

LOGGER = logging.getLogger(__name__)

TOKEN_DTYPE = np.uint16
ENCODING = "gpt2"
SOURCE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def prepare(output_dir: Path | str, val_fraction: float = 0.1) -> dict:
    """Download, tokenize, and write train.bin / val.bin / meta.json."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / "shakespeare.txt"
    if not raw_path.exists():
        LOGGER.info("downloading %s", SOURCE_URL)
        urlretrieve(SOURCE_URL, raw_path)

    text = raw_path.read_text(encoding="utf-8")
    enc = tiktoken.get_encoding(ENCODING)
    ids = enc.encode_ordinary(text)
    LOGGER.info("%s characters -> %s tokens", f"{len(text):,}", f"{len(ids):,}")

    split_at = int(len(ids) * (1 - val_fraction))
    train_ids = np.array(ids[:split_at], dtype=TOKEN_DTYPE)
    val_ids = np.array(ids[split_at:], dtype=TOKEN_DTYPE)

    train_ids.tofile(output_dir / "train.bin")
    val_ids.tofile(output_dir / "val.bin")

    meta = {
        "dataset": "tiny-shakespeare",
        "encoding": ENCODING,
        "vocab_size": enc.n_vocab,
        "train_tokens": int(train_ids.size),
        "val_tokens": int(val_ids.size),
        "dtype": str(np.dtype(TOKEN_DTYPE)),
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    LOGGER.info("wrote train.bin (%s) and val.bin (%s)", f"{train_ids.size:,}", f"{val_ids.size:,}")
    return meta
