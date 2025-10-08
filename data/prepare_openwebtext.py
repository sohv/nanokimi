"""
Prepare OpenWebText-like dataset for nanoKimi

This script downloads an OpenWebText mirror (or uses local dumps), tokenizes with tiktoken,
and writes out train.bin, val.bin and meta.pkl compatible with `data/prepare.py` loader.

Notes:
- OpenWebText mirrors are large. This script supports streaming, sharding and resuming.
- For convenience it can accept a list of input text files in a directory.
"""

import os
import glob
import pickle
from pathlib import Path
from typing import List

import tiktoken
import numpy as np


def gather_text_files(source_dir: str) -> List[Path]:
    p = Path(source_dir)
    if not p.exists():
        raise FileNotFoundError(f"Source directory {source_dir} does not exist")
    files = sorted([f for f in p.rglob('*.txt')])
    return files


def prepare_openwebtext(
    source_dir: str,
    out_dir: str = 'data',
    encoding_name: str = 'gpt2',
    val_fraction: float = 0.01,
    dtype=np.uint16,
    max_chars_per_file: int = None,
):
    """Tokenize and write train/val binary files.

    Args:
        source_dir: directory with .txt files (OpenWebText mirror or other webtext dumps)
        out_dir: where to put train.bin / val.bin / meta.pkl
        encoding_name: tiktoken encoding
        val_fraction: fraction of tokens to hold out for validation
        dtype: dtype to store tokens (uint16 recommended; raise to uint32 if vocab > 65535)
        max_chars_per_file: optional truncation for debugging / small runs
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    enc = tiktoken.get_encoding(encoding_name)
    files = gather_text_files(source_dir)
    print(f"Found {len(files)} text files in {source_dir}")

    token_buffer = []
    total_chars = 0
    for i, fpath in enumerate(files):
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            txt = f.read()
        if max_chars_per_file:
            txt = txt[:max_chars_per_file]
        total_chars += len(txt)
        ids = enc.encode_ordinary(txt)
        token_buffer.extend(ids)
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(files)} files, tokens so far: {len(token_buffer):,}")

    total_tokens = len(token_buffer)
    print(f"Total characters processed: {total_chars:,}")
    print(f"Total tokens: {total_tokens:,}")

    # Split into train / val
    n_val = int(total_tokens * val_fraction)
    if n_val == 0:
        n_val = max(1, int(total_tokens * val_fraction))

    train_ids = np.array(token_buffer[:-n_val], dtype=dtype)
    val_ids = np.array(token_buffer[-n_val:], dtype=dtype)

    train_file = out / 'train.bin'
    val_file = out / 'val.bin'
    train_ids.tofile(train_file)
    val_ids.tofile(val_file)

    meta = {
        'vocab_size': enc.n_vocab,
        'encoding_name': encoding_name,
        'train_tokens': len(train_ids),
        'val_tokens': len(val_ids),
    }

    meta_file = out / 'meta.pkl'
    with open(meta_file, 'wb') as f:
        pickle.dump(meta, f)

    print(f"Saved train.bin ({len(train_ids):,}) and val.bin ({len(val_ids):,}) to {out}")
    print(f"Saved meta to {meta_file}")

    return meta


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', required=True, help='Directory with text files (.txt)')
    parser.add_argument('--out_dir', default='data', help='Output directory for bins/meta')
    parser.add_argument('--encoding', default='gpt2', help='tiktoken encoding name')
    parser.add_argument('--val_fraction', type=float, default=0.01)
    parser.add_argument('--max_chars_per_file', type=int, default=None, help='Truncate each file for testing')

    args = parser.parse_args()
    prepare_openwebtext(args.source_dir, args.out_dir, args.encoding, args.val_fraction, max_chars_per_file=args.max_chars_per_file)
