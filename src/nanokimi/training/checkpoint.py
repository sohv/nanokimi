"""Checkpoint save and load.

Checkpoints are saved unwrapped: torch.compile prefixes every key with `_orig_mod.`,
and leaving that in the file is what made the first published model un-loadable.
"""

import logging
from pathlib import Path
from typing import Any

import torch

LOGGER = logging.getLogger(__name__)

WRAPPER_PREFIXES = ("_orig_mod.", "module.")


def strip_wrapper_prefixes(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Remove torch.compile / DDP key prefixes so a checkpoint loads into a plain model."""
    for prefix in WRAPPER_PREFIXES:
        if any(key.startswith(prefix) for key in state_dict):
            n = sum(1 for key in state_dict if key.startswith(prefix))
            state_dict = {key.removeprefix(prefix): value for key, value in state_dict.items()}
            LOGGER.info("stripped %r from %d keys", prefix, n)
    return state_dict


def save_checkpoint(
    path: Path | str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    tokens_seen: int,
    best_val_loss: float,
    model_args: dict[str, Any],
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": strip_wrapper_prefixes(model.state_dict()),
            "optimizer": optimizer.state_dict(),
            "model_args": model_args,
            "step": step,
            "tokens_seen": tokens_seen,
            "best_val_loss": best_val_loss,
        },
        path,
    )
    LOGGER.info("checkpoint -> %s (step %d)", path, step)
    return path


def load_checkpoint(
    path: Path | str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(strip_wrapper_prefixes(ckpt["model"]))
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt
