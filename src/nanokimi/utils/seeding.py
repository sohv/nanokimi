"""Seeding, so a run can be reproduced exactly."""

import logging
import random

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and torch. `deterministic` trades speed for exact reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    LOGGER.info("seed set to %d (deterministic=%s)", seed, deterministic)
