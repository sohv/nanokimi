"""Structured metrics output. Every number printed to the console also lands in a file."""

import json
import logging
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

FLOAT_PRECISION = 4


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, FLOAT_PRECISION)
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(v) for v in value]
    return value


class MetricsWriter:
    """Append-only JSONL writer for step-level training metrics.

    Appends incrementally so a crashed run keeps everything written up to that point.
    """

    def __init__(self, output_dir: Path | str, filename: str = "metrics.jsonl"):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.path = output_dir / filename
        self._handle = self.path.open("a", buffering=1)
        LOGGER.info("metrics -> %s", self.path)

    def log(self, **record: Any) -> None:
        self._handle.write(json.dumps(_round(record)) + "\n")

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "MetricsWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def write_summary(output_dir: Path | str, summary: dict[str, Any], filename: str = "summary.json") -> Path:
    """Write a run's final metrics as a single indented JSON object."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    path.write_text(json.dumps(_round(summary), indent=2) + "\n")
    LOGGER.info("wrote %s", path)
    return path
