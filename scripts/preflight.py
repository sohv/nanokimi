# runs a real config for a handful of steps and reports throughput, memory and health before a long run.
# uv run -m scripts.preflight --config_path configs/nanokimi_25m.yaml --output_dir results/raw/260803_preflight_25m

import json
import logging
import math
import time
from dataclasses import dataclass

import numpy as np
import simple_parsing
import torch

from nanokimi.data.loader import get_batch, load_meta, split_tokens
from nanokimi.model.transformer import KimiK2
from nanokimi.training.checkpoint import load_checkpoint, save_checkpoint
from nanokimi.training.optimizer import create_muon_optimizer
from nanokimi.training.schedule import (
    count_active_parameters,
    count_parameters,
    get_autocast_ctx,
    get_device,
    get_dtype,
)
from nanokimi.utils.config import RunConfig, write_config_json
from nanokimi.utils.logging import setup_logging
from nanokimi.utils.metrics import write_summary
from nanokimi.utils.seeding import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class PreflightConfig(RunConfig):
    steps: int = 20


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> None:
    config = simple_parsing.parse(PreflightConfig, add_config_path_arg=True)
    if not config.output_dir:
        raise ValueError("--output_dir is required")

    setup_logging(config.output_dir)
    set_seed(config.seed)
    write_config_json(config, config.output_dir)

    device = get_device(config.device)
    dtype = get_dtype(config.dtype)
    ctx = get_autocast_ctx(device, dtype)

    print(f"device {device}  dtype {config.dtype}  compile {config.compile}")
    if device.startswith("cuda"):
        props = torch.cuda.get_device_properties(0)
        print(f"gpu {props.name}  {props.total_memory / 1e9:.1f} GB  torch {torch.__version__}")

    meta = load_meta(config.data_dir)
    config.model.vocab_size = meta["vocab_size"]
    available = split_tokens(config.data_dir, "train")

    results: list[bool] = []
    results.append(
        check(
            "token budget fits the tokenized slice",
            config.train.max_tokens <= available,
            f"{config.train.max_tokens:,} needed, {available:,} available",
        )
    )

    model = KimiK2(config.model.as_dict()).to(device)
    total, active = count_parameters(model), count_active_parameters(model)
    print(f"params total {total:,}  active/token {active:,}  kv cache/token {config.model.kv_cache_per_token}")

    optimizer = create_muon_optimizer(model, vars(config.optimizer))
    if config.compile and device.startswith("cuda"):
        LOGGER.info("compiling")
        t0 = time.time()
        model = torch.compile(model)
        print(f"torch.compile wrapper built in {time.time() - t0:.1f}s (graph compiles on first step)")

    block = config.model.block_size
    rng = np.random.default_rng(config.seed)
    tokens_per_step = config.train.batch_size * block * config.train.gradient_accumulation_steps

    model.train()
    losses, step_times = [], []
    for step in range(config.steps):
        t0 = time.time()
        for _ in range(config.train.gradient_accumulation_steps):
            x, y = get_batch(
                config.data_dir, "train", config.train.batch_size, block, device,
                generator=rng, max_tokens=config.train.max_tokens,
            )
            with ctx:
                _, loss = model(x, y)
                loss = loss / config.train.gradient_accumulation_steps
            loss.backward()
        if config.optimizer.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        losses.append(loss.item() * config.train.gradient_accumulation_steps)
        step_times.append(time.time() - t0)
        if step % 5 == 0:
            print(f"  step {step:>3}  loss {losses[-1]:.4f}  {step_times[-1] * 1000:.0f} ms")

    results.append(check("all losses finite", all(math.isfinite(v) for v in losses)))
    results.append(
        check(
            "loss moved off the uniform baseline",
            losses[-1] < math.log(config.model.vocab_size),
            f"{losses[-1]:.4f} vs ln(vocab) = {math.log(config.model.vocab_size):.2f}",
        )
    )
    results.append(
        check("all weights finite", all(torch.isfinite(p).all().item() for p in model.parameters()))
    )

    collapsed = [
        name
        for name, p in model.named_parameters()
        if p.ndim >= 2 and p.numel() > 1000 and p.std().item() < 1e-5
    ]
    results.append(check("no weight matrix collapsed", not collapsed, f"{len(collapsed)} collapsed"))

    # steady-state throughput, ignoring the first steps that include compilation
    warm = step_times[3:] if len(step_times) > 4 else step_times
    per_step = sum(warm) / len(warm)
    tok_per_s = tokens_per_step / per_step
    eta_h = config.train.max_tokens / tok_per_s / 3600
    print(f"throughput {tok_per_s:,.0f} tokens/s  ->  {eta_h:.1f} h for {config.train.max_tokens:,} tokens")

    if device.startswith("cuda"):
        peak = torch.cuda.max_memory_allocated() / 1e9
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"peak gpu memory {peak:.1f} GB of {total_mem:.1f} GB ({100 * peak / total_mem:.0f}%)")
        results.append(check("fits in GPU memory with headroom", peak < total_mem * 0.9, f"{peak:.1f} GB"))

    base = model._orig_mod if hasattr(model, "_orig_mod") else model
    ckpt_path = save_checkpoint(
        f"{config.output_dir}/checkpoints/preflight.pt", base, optimizer, config.steps,
        config.steps * tokens_per_step, losses[-1], config.model.as_dict(),
    )
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
    results.append(
        check("checkpoint has no compile prefixes", not any(k.startswith(("_orig_mod.", "module.")) for k in saved))
    )

    restored = KimiK2(config.model.as_dict())
    load_checkpoint(ckpt_path, restored)
    results.append(
        check(
            "checkpoint reloads with identical weights",
            all(torch.equal(a.cpu(), b) for (_, a), (_, b) in zip(base.named_parameters(), restored.named_parameters())),
        )
    )

    write_summary(
        config.output_dir,
        {
            "device": device,
            "dtype": config.dtype,
            "total_params": total,
            "active_params": active,
            "steps": config.steps,
            "first_loss": losses[0],
            "last_loss": losses[-1],
            "tokens_per_second": tok_per_s,
            "projected_hours": eta_h,
            "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9 if device.startswith("cuda") else 0.0,
        },
        filename="preflight.json",
    )

    print(f"\n{sum(results)}/{len(results)} checks passed")
    if not all(results):
        raise SystemExit("preflight failed; do not start the long run")
    print(f"ready. projected {eta_h:.1f} h for the full budget")


if __name__ == "__main__":
    main()
