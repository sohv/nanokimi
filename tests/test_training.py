"""Training tests.

Covers the reproducibility spine (same seed, same run) and the failure modes that
silently ruined the first version of this project: weights decaying to zero, experts
dying, and checkpoints that cannot be reloaded.
"""

import json
import math

import numpy as np
import pytest
import torch

from nanokimi.model.transformer import KimiK2
from nanokimi.training.checkpoint import load_checkpoint, save_checkpoint, strip_wrapper_prefixes
from nanokimi.training.loop import train
from nanokimi.training.optimizer import create_muon_optimizer
from nanokimi.training.schedule import (
    count_active_parameters,
    count_parameters,
    get_device,
    get_dtype,
    get_lr,
    set_lr,
)
from nanokimi.utils.config import ModelConfig, OptimizerConfig, RunConfig, TrainConfig
from nanokimi.utils.metrics import MetricsWriter, write_summary
from nanokimi.utils.seeding import set_seed
from tests.conftest import TOY_MODEL


def make_run_config(output_dir, data_dir, **overrides) -> RunConfig:
    config = RunConfig(
        output_dir=str(output_dir),
        data_dir=str(data_dir),
        seed=overrides.pop("seed", 42),
        device="cpu",
        dtype="float32",
        compile=False,
        model=ModelConfig(**TOY_MODEL),
        optimizer=OptimizerConfig(learning_rate=1e-3, adamw_learning_rate=1e-3, min_lr=1e-4),
        train=TrainConfig(
            max_tokens=overrides.pop("max_tokens", 8 * 32 * 12),
            batch_size=8,
            gradient_accumulation_steps=1,
            warmup_iters=2,
            eval_interval=5,
            eval_iters=2,
            log_interval=2,
        ),
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


# --- learning rate schedule ------------------------------------------------


def test_lr_warms_up_then_decays():
    kwargs = dict(warmup_iters=10, learning_rate=1e-3, lr_decay_iters=100, min_lr=1e-4)
    warm = [get_lr(step, **kwargs) for step in range(10)]
    assert warm == sorted(warm), "warmup must be monotonically increasing"
    assert warm[0] < kwargs["learning_rate"]

    decay = [get_lr(step, **kwargs) for step in range(10, 100)]
    assert decay == sorted(decay, reverse=True), "cosine phase must be monotonically decreasing"


def test_lr_never_leaves_its_bounds():
    kwargs = dict(warmup_iters=10, learning_rate=1e-3, lr_decay_iters=100, min_lr=1e-4)
    for step in range(0, 200):
        lr = get_lr(step, **kwargs)
        assert 0 <= lr <= kwargs["learning_rate"] + 1e-12
    assert get_lr(500, **kwargs) == pytest.approx(kwargs["min_lr"])


def test_lr_is_never_zero_after_warmup():
    """A zero LR on the first step wastes it; warmup must start above zero."""
    assert get_lr(0, warmup_iters=10, learning_rate=1e-3, lr_decay_iters=100, min_lr=1e-4) > 0


def test_set_lr_reaches_the_adamw_group_too():
    """MuonClip carries a separate adamw_lr; missing it leaves embeddings undecayed."""
    set_seed(0)
    model = KimiK2(TOY_MODEL)
    optimizer = create_muon_optimizer(model, vars(OptimizerConfig()))
    set_lr(optimizer, 0.123)
    for group in optimizer.param_groups:
        assert group["lr"] == 0.123
        if "adamw_lr" in group:
            assert group["adamw_lr"] == 0.123


# --- checkpoints -----------------------------------------------------------


def test_strip_wrapper_prefixes_handles_compile_and_ddp():
    assert strip_wrapper_prefixes({"_orig_mod.a": 1, "_orig_mod.b": 2}) == {"a": 1, "b": 2}
    assert strip_wrapper_prefixes({"module.a": 1}) == {"a": 1}
    assert strip_wrapper_prefixes({"a": 1}) == {"a": 1}


def test_checkpoint_round_trips_exactly(tmp_path):
    set_seed(0)
    model = KimiK2(TOY_MODEL)
    optimizer = create_muon_optimizer(model, vars(OptimizerConfig()))

    ids = torch.randint(0, TOY_MODEL["vocab_size"], (2, 16))
    _, loss = model(ids, ids)
    loss.backward()
    optimizer.step()

    path = save_checkpoint(tmp_path / "ckpt.pt", model, optimizer, 5, 1234, 2.5, dict(TOY_MODEL))
    assert path.exists()

    restored = KimiK2(TOY_MODEL)
    ckpt = load_checkpoint(path, restored)
    assert ckpt["step"] == 5 and ckpt["tokens_seen"] == 1234

    for (name, a), (_, b) in zip(model.named_parameters(), restored.named_parameters()):
        assert torch.equal(a, b), name

    model.eval(), restored.eval()
    with torch.no_grad():
        assert torch.equal(model(ids)[0], restored(ids)[0])


def test_checkpoint_never_stores_compile_prefixes(tmp_path):
    """A published checkpoint carrying _orig_mod. is exactly what broke the first release."""
    set_seed(0)
    model = KimiK2(TOY_MODEL)
    optimizer = create_muon_optimizer(model, vars(OptimizerConfig()))
    path = save_checkpoint(tmp_path / "ckpt.pt", model, optimizer, 0, 0, 9.9, dict(TOY_MODEL))
    saved = torch.load(path, map_location="cpu", weights_only=False)["model"]
    assert not any(key.startswith(("_orig_mod.", "module.")) for key in saved)


def test_checkpoint_carries_model_args_needed_to_rebuild(tmp_path):
    """MLA has no wpe to read block_size from, so model_args must be complete."""
    set_seed(0)
    model = KimiK2(TOY_MODEL)
    optimizer = create_muon_optimizer(model, vars(OptimizerConfig()))
    path = save_checkpoint(tmp_path / "ckpt.pt", model, optimizer, 0, 0, 9.9, dict(TOY_MODEL))
    args = torch.load(path, map_location="cpu", weights_only=False)["model_args"]
    for key in ("n_layer", "n_head", "n_embd", "block_size", "kv_lora_rank", "qk_rope_head_dim"):
        assert key in args


# --- metrics ---------------------------------------------------------------


def test_metrics_are_jsonl_with_rounded_floats(tmp_path):
    with MetricsWriter(tmp_path) as writer:
        writer.log(step=1, loss=1.23456789, nested={"a": 2.3456789})
    lines = (tmp_path / "metrics.jsonl").read_text().strip().splitlines()
    record = json.loads(lines[0])
    assert record["loss"] == 1.2346
    assert record["nested"]["a"] == 2.3457
    assert record["step"] == 1


def test_metrics_append_rather_than_truncate(tmp_path):
    for i in range(3):
        with MetricsWriter(tmp_path) as writer:
            writer.log(step=i)
    assert len((tmp_path / "metrics.jsonl").read_text().strip().splitlines()) == 3


def test_summary_is_indented_json(tmp_path):
    path = write_summary(tmp_path, {"final_val_loss": 3.14159265})
    assert json.loads(path.read_text())["final_val_loss"] == 3.1416


# --- parameter accounting --------------------------------------------------


def test_active_params_are_below_total_for_moe():
    set_seed(0)
    model = KimiK2(TOY_MODEL)
    total, active = count_parameters(model), count_active_parameters(model)
    assert active < total
    # top_k of num_experts are used per token, so most expert weight is inactive
    assert active < total * 0.75


def test_active_equals_nonembedding_total_when_moe_is_off():
    config = dict(TOY_MODEL, use_moe=False)
    set_seed(0)
    model = KimiK2(config)
    expected = count_parameters(model) - model.transformer.wte.weight.numel()
    assert count_active_parameters(model) == expected


# --- device and dtype helpers ---------------------------------------------


def test_get_device_honours_an_explicit_request():
    assert get_device("cpu") == "cpu"
    assert get_device("cuda:3") == "cuda:3"
    assert get_device("auto") in {"cpu", "cuda", "mps"}


def test_unknown_dtype_fails_loudly():
    assert get_dtype("bfloat16") is torch.bfloat16
    with pytest.raises(ValueError, match="unknown dtype"):
        get_dtype("float8")


# --- end to end ------------------------------------------------------------


def test_train_writes_the_full_run_directory(tmp_path, toy_data_dir):
    config = make_run_config(tmp_path / "run", toy_data_dir)
    summary = train(config)

    run_dir = tmp_path / "run"
    for name in ("config.json", "metrics.jsonl", "summary.json"):
        assert (run_dir / name).exists(), name
    assert (run_dir / "checkpoints" / "ckpt_final.pt").exists()

    assert math.isfinite(summary["final_val_loss"])
    assert summary["tokens_seen"] > 0
    assert summary["active_params"] < summary["total_params"]


def test_two_runs_with_the_same_seed_are_identical(tmp_path, toy_data_dir):
    """The reproducibility spine: same seed must give the same loss curve."""
    losses = []
    for i in range(2):
        set_seed(42)
        config = make_run_config(tmp_path / f"run{i}", toy_data_dir, seed=42)
        train(config)
        records = [
            json.loads(line)
            for line in (tmp_path / f"run{i}" / "metrics.jsonl").read_text().strip().splitlines()
        ]
        losses.append([r["train_loss"] for r in records if "train_loss" in r])

    assert losses[0] == losses[1], f"diverged: {losses[0]} vs {losses[1]}"


def test_different_seeds_diverge(tmp_path, toy_data_dir):
    curves = []
    for i, seed in enumerate((1, 2)):
        set_seed(seed)
        config = make_run_config(tmp_path / f"seed{i}", toy_data_dir, seed=seed)
        train(config)
        records = [
            json.loads(line)
            for line in (tmp_path / f"seed{i}" / "metrics.jsonl").read_text().strip().splitlines()
        ]
        curves.append([r["train_loss"] for r in records if "train_loss" in r])
    assert curves[0] != curves[1]


def test_budget_beyond_the_dataset_fails_loudly(tmp_path, toy_data_dir):
    config = make_run_config(tmp_path / "run", toy_data_dir, max_tokens=10**12)
    with pytest.raises(ValueError, match="exceeds"):
        train(config)


def test_training_does_not_collapse_weights_or_kill_experts(tmp_path, toy_data_dir):
    """Guards the exact failure mode of the first published checkpoint."""
    set_seed(0)
    config = make_run_config(tmp_path / "run", toy_data_dir, max_tokens=8 * 32 * 60)
    train(config)

    ckpt = torch.load(tmp_path / "run" / "checkpoints" / "ckpt_final.pt", map_location="cpu", weights_only=False)
    state = ckpt["model"]

    for name, tensor in state.items():
        assert torch.isfinite(tensor).all(), f"{name} contains NaN or inf"

    for name in ("transformer.h.0.attn.q_b_proj.weight", "transformer.h.0.mlp.experts.0.fc1.weight"):
        assert state[name].std().item() > 1e-4, f"{name} collapsed to {state[name].std().item():.2e}"

    model = KimiK2(TOY_MODEL)
    model.load_state_dict(state)
    model.eval()

    ids = torch.randint(0, TOY_MODEL["vocab_size"], (2, 32))
    x = model.transformer.wte(ids)
    with torch.no_grad():
        for block in model.transformer.h:
            h = block.ln_2(x + block.attn(block.ln_1(x))).view(-1, TOY_MODEL["n_embd"])
            gate = torch.softmax(block.mlp.gate(h), dim=-1)
            _, idx = torch.topk(gate, TOY_MODEL["top_k_experts"], dim=-1)
            used = sum(int((idx == e).any().item()) for e in range(TOY_MODEL["num_experts"]))
            assert used >= 2, f"router collapsed to {used} experts"
            x, _ = block(x)


def test_load_checkpoint_restores_optimizer_state(tmp_path):
    """Resuming a run needs the momentum buffers, not just the weights."""
    set_seed(0)
    model = KimiK2(TOY_MODEL)
    optimizer = create_muon_optimizer(model, vars(OptimizerConfig()))
    ids = torch.randint(0, TOY_MODEL["vocab_size"], (2, 16))
    for _ in range(3):
        _, loss = model(ids, ids)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    path = save_checkpoint(tmp_path / "ckpt.pt", model, optimizer, 3, 99, 1.0, dict(TOY_MODEL))

    restored_model = KimiK2(TOY_MODEL)
    restored_opt = create_muon_optimizer(restored_model, vars(OptimizerConfig()))
    load_checkpoint(path, restored_model, restored_opt)

    assert len(restored_opt.state) == len(optimizer.state)
    buffers = [s["momentum_buffer"] for s in restored_opt.state.values() if "momentum_buffer" in s]
    assert buffers and any(b.abs().sum() > 0 for b in buffers)


def test_metrics_round_floats_inside_lists(tmp_path):
    with MetricsWriter(tmp_path) as writer:
        writer.log(step=0, values=[1.23456789, 2.3456789], pair=(0.111111, 0.222222))
    record = json.loads((tmp_path / "metrics.jsonl").read_text().strip())
    assert record["values"] == [1.2346, 2.3457]
    assert record["pair"] == [0.1111, 0.2222]


def test_git_hash_reports_unknown_outside_a_repo(tmp_path, monkeypatch):
    import subprocess

    from nanokimi.utils import config as config_module

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=128, stdout="", stderr="")

    monkeypatch.setattr(config_module.subprocess, "run", fake_run)
    assert config_module.git_hash() == "unknown"


def test_get_device_falls_back_to_cpu(monkeypatch):
    """The cpu branch is unreachable on a machine with MPS unless both are disabled."""
    import torch as torch_module

    from nanokimi.training import schedule

    monkeypatch.setattr(torch_module.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch_module.backends.mps, "is_available", lambda: False)
    assert schedule.get_device("auto") == "cpu"


def test_train_logs_to_wandb_when_a_project_is_set(tmp_path, toy_data_dir, monkeypatch):
    """The wandb branch never fires in tests otherwise, so a typo would reach a real run."""
    logged, summary_keys, finished = [], {}, []

    class FakeRun:
        summary = summary_keys

        def log(self, record):
            logged.append(record)

        def finish(self):
            finished.append(True)

    import wandb

    monkeypatch.setattr(wandb, "init", lambda **kwargs: FakeRun())

    config = make_run_config(tmp_path / "run", toy_data_dir)
    config.wandb_project = "nanokimi-test"
    config.run_name = "unit"
    summary = train(config)

    assert logged, "nothing was logged to wandb"
    assert any("train_loss" in record for record in logged)
    assert finished == [True]
    assert summary_keys["final_val_loss"] == summary["final_val_loss"]
