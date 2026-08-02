"""The training loop.

Budgets are expressed in tokens rather than iterations so the four model sizes are
directly comparable: each trains on a strict prefix of the same tokenized slice.
"""

import logging
import math
import time
from pathlib import Path

import torch

from nanokimi.data.loader import get_batch, load_meta, split_tokens
from nanokimi.model.transformer import KimiK2
from nanokimi.training.checkpoint import save_checkpoint
from nanokimi.training.optimizer import create_muon_optimizer
from nanokimi.training.schedule import (
    count_active_parameters,
    count_parameters,
    get_autocast_ctx,
    get_device,
    get_dtype,
    get_lr,
    set_lr,
)
from nanokimi.utils.config import RunConfig, write_config_json
from nanokimi.utils.metrics import MetricsWriter, write_summary

LOGGER = logging.getLogger(__name__)


@torch.no_grad()
def estimate_loss(model, data_dir, split, batch_size, block_size, device, ctx, eval_iters, max_tokens=None):
    """Mean loss over eval_iters batches. Returns a float."""
    model.eval()
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        x, y = get_batch(data_dir, split, batch_size, block_size, device, max_tokens=max_tokens)
        with ctx:
            _, loss = model(x, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


def train(config: RunConfig) -> dict:
    """Run one training job. Returns the summary dict that is also written to disk."""
    output_dir = Path(config.output_dir)
    write_config_json(config, output_dir)

    device = get_device(config.device)
    dtype = get_dtype(config.dtype)
    ctx = get_autocast_ctx(device, dtype)
    LOGGER.info("device=%s dtype=%s", device, config.dtype)

    meta = load_meta(config.data_dir)
    config.model.vocab_size = meta["vocab_size"]
    LOGGER.info("vocab_size=%d from %s", config.model.vocab_size, config.data_dir)

    model = KimiK2(config.model.as_dict())
    model.to(device)

    total_params = count_parameters(model)
    active_params = count_active_parameters(model)
    LOGGER.info(
        "params total=%s active/token=%s kv_cache/token=%d",
        f"{total_params:,}",
        f"{active_params:,}",
        config.model.kv_cache_per_token,
    )

    optimizer = create_muon_optimizer(model, vars(config.optimizer))

    if config.compile and device.startswith("cuda"):
        LOGGER.info("compiling model")
        model = torch.compile(model)

    block_size = config.model.block_size
    tokens_per_step = config.train.batch_size * block_size * config.train.gradient_accumulation_steps
    max_iters = max(1, config.train.max_tokens // tokens_per_step)
    train_budget = config.train.max_tokens
    available = split_tokens(config.data_dir, "train")
    if train_budget > available:
        raise ValueError(f"budget of {train_budget:,} tokens exceeds the {available:,} tokens in train.bin")
    LOGGER.info(
        "budget %s tokens = %d iters at %s tokens/step (train.bin holds %s)",
        f"{train_budget:,}",
        max_iters,
        f"{tokens_per_step:,}",
        f"{available:,}",
    )

    run = None
    if config.wandb_project:
        import wandb

        run = wandb.init(project=config.wandb_project, name=config.run_name or None, config=_wandb_config(config))

    metrics = MetricsWriter(output_dir)
    ckpt_dir = output_dir / "checkpoints"
    best_val_loss = math.inf
    tokens_seen = 0
    start = time.time()

    model.train()
    for step in range(max_iters):
        lr = get_lr(step, config.train.warmup_iters, config.optimizer.learning_rate, max_iters, config.optimizer.min_lr)
        set_lr(optimizer, lr)

        for _ in range(config.train.gradient_accumulation_steps):
            x, y = get_batch(
                config.data_dir, "train", config.train.batch_size, block_size, device, max_tokens=train_budget
            )
            with ctx:
                _, loss = model(x, y)
                loss = loss / config.train.gradient_accumulation_steps
            loss.backward()
            tokens_seen += x.numel()

        if config.optimizer.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        step_loss = loss.item() * config.train.gradient_accumulation_steps

        if step % config.train.log_interval == 0:
            record = {
                "step": step,
                "tokens_seen": tokens_seen,
                "train_loss": step_loss,
                "lr": lr,
                "max_attn_logit": optimizer.last_max_logit,
                "clipped_heads": optimizer.last_clipped_heads,
                "elapsed_s": time.time() - start,
            }
            metrics.log(**record)
            if run:
                run.log(record)
            LOGGER.info(
                "step %d/%d loss %.4f lr %.2e max_logit %.1f clipped %d",
                step,
                max_iters,
                step_loss,
                lr,
                optimizer.last_max_logit,
                optimizer.last_clipped_heads,
            )

        if step > 0 and step % config.train.eval_interval == 0:
            val_loss = estimate_loss(
                model, config.data_dir, "val", config.train.batch_size, block_size, device, ctx,
                config.train.eval_iters,
            )
            metrics.log(step=step, tokens_seen=tokens_seen, val_loss=val_loss)
            if run:
                run.log({"step": step, "val_loss": val_loss})
            LOGGER.info("step %d val_loss %.4f", step, val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    ckpt_dir / "ckpt.pt", model, optimizer, step, tokens_seen, best_val_loss,
                    config.model.as_dict(),
                )

    final_val = estimate_loss(
        model, config.data_dir, "val", config.train.batch_size, block_size, device, ctx, config.train.eval_iters
    )
    save_checkpoint(
        ckpt_dir / "ckpt_final.pt", model, optimizer, max_iters, tokens_seen, final_val, config.model.as_dict()
    )

    summary = {
        "total_params": total_params,
        "active_params": active_params,
        "kv_cache_per_token": config.model.kv_cache_per_token,
        "tokens_seen": tokens_seen,
        "iters": max_iters,
        "final_val_loss": final_val,
        "best_val_loss": min(best_val_loss, final_val),
        "final_val_ppl": math.exp(final_val),
        "wall_clock_s": time.time() - start,
    }
    write_summary(output_dir, summary)
    metrics.close()
    if run:
        for key, value in summary.items():
            run.summary[key] = value
        run.finish()

    return summary


def _wandb_config(config: RunConfig) -> dict:
    return {
        "seed": config.seed,
        "n_layer": config.model.n_layer,
        "n_embd": config.model.n_embd,
        "n_head": config.model.n_head,
        "num_experts": config.model.num_experts,
        "kv_lora_rank": config.model.kv_lora_rank,
        "max_tokens": config.train.max_tokens,
        "learning_rate": config.optimizer.learning_rate,
    }
