"""Logging setup. Every run writes to output_dir/run.log as well as the console."""

import logging
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)

# These libraries log every HTTP request at INFO, which buries our own output and
# bloats run.log during long downloads. They stay at WARNING.
NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "filelock", "fsspec", "datasets", "huggingface_hub")


def setup_logging(output_dir: Path | str, level: int = logging.INFO) -> None:
    """Attach a console handler and a run.log file handler to the root logger."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    root.addHandler(console)

    log_file = logging.FileHandler(output_dir / "run.log")
    log_file.setFormatter(fmt)
    root.addHandler(log_file)

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
