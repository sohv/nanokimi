"""Learning rate schedule and device/dtype helpers."""

import logging
import math
from contextlib import nullcontext
from typing import ContextManager

import torch

LOGGER = logging.getLogger(__name__)

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def get_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype(name: str) -> torch.dtype:
    if name not in DTYPES:
        raise ValueError(f"unknown dtype {name!r}, expected one of {sorted(DTYPES)}")
    return DTYPES[name]


def get_autocast_ctx(device: str, dtype: torch.dtype) -> ContextManager:
    """Autocast context for the device. CPU and MPS run in float32."""
    if device.startswith("cuda"):
        return torch.amp.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def get_lr(step: int, warmup_iters: int, learning_rate: float, lr_decay_iters: int, min_lr: float) -> float:
    """Linear warmup then cosine decay to min_lr."""
    if step < warmup_iters:
        return learning_rate * (step + 1) / (warmup_iters + 1)
    if step > lr_decay_iters:
        return min_lr
    decay_ratio = (step - warmup_iters) / max(1, lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Apply lr to every param group, including MuonClip's separate AdamW group."""
    for group in optimizer.param_groups:
        group["lr"] = lr
        if "adamw_lr" in group:
            group["adamw_lr"] = lr


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_active_parameters(model: torch.nn.Module) -> int:
    """Non-embedding parameters used per token, counting only top_k of each MoE layer's experts."""
    total = sum(p.numel() for p in model.parameters())
    embedding = model.transformer.wte.weight.numel()

    expert_params, active_expert_params = 0, 0
    for block in model.transformer.h:
        mlp = block.mlp
        if not hasattr(mlp, "experts"):
            continue
        params = sum(p.numel() for p in mlp.experts.parameters())
        expert_params += params
        active_expert_params += params * mlp.top_k / mlp.num_experts

    return int(total - embedding - expert_params + active_expert_params)
