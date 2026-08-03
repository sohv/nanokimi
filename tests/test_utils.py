"""Utility tests: logging setup and seeding."""

import logging

import numpy as np
import torch

from nanokimi.utils.logging import NOISY_LOGGERS, setup_logging
from nanokimi.utils.seeding import set_seed


def test_setup_logging_writes_run_log(tmp_path):
    setup_logging(tmp_path)
    logging.getLogger("nanokimi.test").info("hello from the run")
    for handler in logging.getLogger().handlers:
        handler.flush()
    assert "hello from the run" in (tmp_path / "run.log").read_text()


def test_setup_logging_creates_the_directory(tmp_path):
    target = tmp_path / "nested" / "run"
    setup_logging(target)
    assert (target / "run.log").exists()


def test_noisy_libraries_are_pinned_to_warning(tmp_path):
    """httpx logging every HTTP range request produced 63 KB of log in a 90 s probe."""
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.INFO)

    setup_logging(tmp_path)

    for name in NOISY_LOGGERS:
        assert logging.getLogger(name).level == logging.WARNING, name
    assert not logging.getLogger("nanokimi").isEnabledFor(logging.DEBUG)
    assert logging.getLogger("nanokimi").isEnabledFor(logging.INFO)


def test_setup_logging_is_idempotent(tmp_path):
    """Calling twice must not double every line."""
    setup_logging(tmp_path)
    setup_logging(tmp_path)
    logging.getLogger("nanokimi.test").info("once")
    for handler in logging.getLogger().handlers:
        handler.flush()
    assert (tmp_path / "run.log").read_text().count("once") == 1


def test_set_seed_governs_every_rng():
    set_seed(11)
    a = (np.random.rand(4).tolist(), torch.randn(4).tolist())
    set_seed(11)
    b = (np.random.rand(4).tolist(), torch.randn(4).tolist())
    assert a == b


def test_set_seed_deterministic_flag(tmp_path):
    """The slow path used when a run must be bit-reproducible across machines."""
    set_seed(3, deterministic=True)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    # restore the fast default so later tests are unaffected
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
