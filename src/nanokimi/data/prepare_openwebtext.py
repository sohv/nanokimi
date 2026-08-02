"""Tokenize a fixed slice of OpenWebText once into flat uint16 .bin files.

Tokenizing happens once and is reused by every model size. Each smaller run reads a
strict prefix of train.bin rather than a fresh random sample, so differences across
sizes cannot be confounded by different data.
"""

import json
import logging
from pathlib import Path

import numpy as np
import tiktoken
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

TOKEN_DTYPE = np.uint16
ENCODING = "gpt2"
DATASET = "Skylion007/openwebtext"

# uint16 holds 0..65535; GPT-2's vocabulary is 50257, so it fits and halves the file size.
assert tiktoken.get_encoding(ENCODING).n_vocab < np.iinfo(TOKEN_DTYPE).max


def _write_split(stream, handle, target_tokens: int, enc, desc: str) -> int:
    """Consume documents from the stream until target_tokens have been written."""
    written = 0
    with tqdm(total=target_tokens, desc=desc, unit="tok", unit_scale=True) as bar:
        while written < target_tokens:
            ids = enc.encode_ordinary(next(stream)["text"])
            ids.append(enc.eot_token)
            chunk = np.array(ids, dtype=TOKEN_DTYPE)
            handle.write(chunk.tobytes())
            written += len(chunk)
            bar.update(len(chunk))
    return written


def prepare(
    output_dir: Path | str,
    max_tokens: int = 4_000_000_000,
    val_tokens: int = 2_000_000,
    num_proc: int = 8,
    seed: int = 42,
) -> dict:
    """Stream OpenWebText, tokenize max_tokens of it, write train.bin / val.bin / meta.json."""
    from datasets import load_dataset

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    enc = tiktoken.get_encoding(ENCODING)
    LOGGER.info("streaming %s, target %s train tokens", DATASET, f"{max_tokens:,}")

    dataset = load_dataset(DATASET, split="train", streaming=True).shuffle(seed=seed, buffer_size=10_000)
    stream = iter(dataset)

    # The validation slice is drawn first off the same shuffled stream, so it matches
    # the training distribution but never appears in any model's training prefix.
    with (output_dir / "val.bin").open("wb") as handle:
        written_val = _write_split(stream, handle, val_tokens, enc, "val")

    with (output_dir / "train.bin").open("wb") as handle:
        written_train = _write_split(stream, handle, max_tokens, enc, "train")

    meta = {
        "dataset": DATASET,
        "encoding": ENCODING,
        "vocab_size": enc.n_vocab,
        "train_tokens": written_train,
        "val_tokens": written_val,
        "dtype": str(np.dtype(TOKEN_DTYPE)),
        "seed": seed,
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    LOGGER.info("wrote train.bin (%s tokens) and val.bin (%s tokens)", f"{written_train:,}", f"{written_val:,}")
    return meta
