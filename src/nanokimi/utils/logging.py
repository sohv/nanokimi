"""Logging setup. Every run writes to output_dir/run.log as well as the console."""

import logging
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)


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
