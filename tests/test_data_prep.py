"""Data preparation tests.

These produce the artifact every training run depends on. Regenerating the slice with
a different budget must not silently produce a corrupt or short file.
"""

import json

import numpy as np
import pytest
import tiktoken

from nanokimi.data import prepare_openwebtext, prepare_shakespeare
from nanokimi.data.loader import TOKEN_DTYPE, load_meta, split_tokens


class FakeStream:
    """Stands in for the HuggingFace streaming iterator."""

    def __init__(self, texts):
        self._items = iter([{"text": t} for t in texts])

    def __iter__(self):
        return self._items

    def __next__(self):
        return next(self._items)


def test_uint16_can_hold_the_gpt2_vocabulary():
    """uint16 wraps silently above 65535; the assert in the module must hold."""
    assert tiktoken.get_encoding("gpt2").n_vocab < np.iinfo(TOKEN_DTYPE).max


def test_write_split_reaches_the_target_and_ends_on_a_document(tmp_path):
    enc = tiktoken.get_encoding("gpt2")
    texts = ["hello world " * 50] * 200
    path = tmp_path / "out.bin"
    with path.open("wb") as handle:
        written = prepare_openwebtext._write_split(FakeStream(texts), handle, 2_000, enc, "test")

    assert written >= 2_000
    on_disk = path.stat().st_size // np.dtype(TOKEN_DTYPE).itemsize
    assert on_disk == written

    data = np.fromfile(path, dtype=TOKEN_DTYPE)
    assert int(data.max()) < enc.n_vocab
    # every document is terminated, so the last token must be the separator
    assert int(data[-1]) == enc.eot_token


def test_write_split_inserts_one_separator_per_document(tmp_path):
    enc = tiktoken.get_encoding("gpt2")
    texts = ["one two three four five"] * 40
    path = tmp_path / "out.bin"
    with path.open("wb") as handle:
        prepare_openwebtext._write_split(FakeStream(texts), handle, 100, enc, "test", batch_docs=8)
    data = np.fromfile(path, dtype=TOKEN_DTYPE)
    n_docs = int((data == enc.eot_token).sum())
    assert n_docs == len(data[data == enc.eot_token])
    assert n_docs >= 1


def test_write_split_fails_loudly_when_the_corpus_runs_out(tmp_path):
    """Silently writing a short file would leave every budget check passing on bad data."""
    enc = tiktoken.get_encoding("gpt2")
    with (tmp_path / "out.bin").open("wb") as handle:
        with pytest.raises(RuntimeError, match="exhausted"):
            prepare_openwebtext._write_split(FakeStream(["tiny"] * 3), handle, 10_000, enc, "test")


def test_write_split_round_trips_the_original_text(tmp_path):
    enc = tiktoken.get_encoding("gpt2")
    text = "The quick brown fox jumps over the lazy dog. " * 20
    path = tmp_path / "out.bin"
    with path.open("wb") as handle:
        prepare_openwebtext._write_split(FakeStream([text] * 10), handle, 50, enc, "test", batch_docs=4)
    data = np.fromfile(path, dtype=TOKEN_DTYPE)
    decoded = enc.decode([t for t in data.tolist() if t != enc.eot_token])
    assert decoded.startswith("The quick brown fox")


def test_prepare_shakespeare_writes_a_loadable_dataset(tmp_path, monkeypatch):
    """Exercised without the network by seeding the raw file the downloader would fetch."""
    (tmp_path / "shakespeare.txt").write_text("To be, or not to be, that is the question. " * 500)

    def fail(*args, **kwargs):
        raise AssertionError("should not download when the raw file already exists")

    monkeypatch.setattr(prepare_shakespeare, "urlretrieve", fail)

    meta = prepare_shakespeare.prepare(tmp_path, val_fraction=0.1)

    assert meta["vocab_size"] == tiktoken.get_encoding("gpt2").n_vocab
    assert meta["train_tokens"] > 0 and meta["val_tokens"] > 0
    for split in ("train", "val"):
        assert split_tokens(tmp_path, split) == meta[f"{split}_tokens"]
    assert load_meta(tmp_path) == meta


def test_prepare_shakespeare_split_fraction_is_respected(tmp_path, monkeypatch):
    (tmp_path / "shakespeare.txt").write_text("word " * 5000)
    monkeypatch.setattr(prepare_shakespeare, "urlretrieve", lambda *a, **k: None)
    meta = prepare_shakespeare.prepare(tmp_path, val_fraction=0.2)
    total = meta["train_tokens"] + meta["val_tokens"]
    assert 0.18 < meta["val_tokens"] / total < 0.22


def test_prepare_shakespeare_writes_meta_json_not_pickle(tmp_path, monkeypatch):
    """meta.json is the current format; the loader still reads legacy meta.pkl."""
    (tmp_path / "shakespeare.txt").write_text("hello " * 2000)
    monkeypatch.setattr(prepare_shakespeare, "urlretrieve", lambda *a, **k: None)
    prepare_shakespeare.prepare(tmp_path)
    assert (tmp_path / "meta.json").exists()
    json.loads((tmp_path / "meta.json").read_text())


def test_prepare_orchestrates_val_then_train_without_overlap(tmp_path, monkeypatch):
    """The prepare() wrapper only needs load_dataset stubbed, not a live download."""

    class FakeDataset:
        def __init__(self, texts):
            self.texts = texts

        def shuffle(self, seed, buffer_size):
            assert seed == 7 and buffer_size > 0
            return self

        def __iter__(self):
            return iter([{"text": t} for t in self.texts])

    docs = [f"document number {i} " + "filler words here " * 30 for i in range(400)]
    captured = {}

    def fake_load_dataset(name, split, streaming):
        captured["name"], captured["split"], captured["streaming"] = name, split, streaming
        return FakeDataset(docs)

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

    meta = prepare_openwebtext.prepare(tmp_path, max_tokens=3_000, val_tokens=1_000, seed=7)

    assert captured == {"name": prepare_openwebtext.DATASET, "split": "train", "streaming": True}
    assert meta["train_tokens"] >= 3_000 and meta["val_tokens"] >= 1_000
    assert meta["vocab_size"] == tiktoken.get_encoding("gpt2").n_vocab
    assert meta["seed"] == 7

    for split in ("train", "val"):
        assert split_tokens(tmp_path, split) == meta[f"{split}_tokens"]
    assert load_meta(tmp_path) == meta

    # val is drawn first off the same stream, so the two splits must be disjoint
    train = np.fromfile(tmp_path / "train.bin", dtype=TOKEN_DTYPE)
    val = np.fromfile(tmp_path / "val.bin", dtype=TOKEN_DTYPE)
    assert val[:64].tobytes() not in train.tobytes()


def test_prepare_downloads_only_when_the_raw_file_is_absent(tmp_path, monkeypatch):
    calls = []

    def fake_urlretrieve(url, path):
        calls.append(url)
        open(path, "w").write("some shakespeare text " * 400)

    monkeypatch.setattr(prepare_shakespeare, "urlretrieve", fake_urlretrieve)

    prepare_shakespeare.prepare(tmp_path)
    assert len(calls) == 1 and calls[0] == prepare_shakespeare.SOURCE_URL

    prepare_shakespeare.prepare(tmp_path)
    assert len(calls) == 1, "second call must reuse the cached raw file"
