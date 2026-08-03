"""Baseline-path tests.

The dense variants (use_latent_attention=False, use_moe=False) are the comparison
points the scaling study measures MLA and MoE against. They are easy to break while
changing the MLA path and nothing else exercises them.
"""

import torch

from nanokimi.export.hf import export_checkpoint, infer_config, load_state_dict
from nanokimi.model.transformer import KimiK2
from nanokimi.training.checkpoint import save_checkpoint
from nanokimi.training.optimizer import create_muon_optimizer
from nanokimi.training.schedule import count_active_parameters, count_parameters
from nanokimi.utils.config import OptimizerConfig
from nanokimi.utils.seeding import set_seed
from tests.conftest import TOY_MODEL

VOCAB = TOY_MODEL["vocab_size"]
BLOCK = TOY_MODEL["block_size"]

DENSE_ATTN = dict(TOY_MODEL, use_latent_attention=False)
DENSE_FFN = dict(TOY_MODEL, use_moe=False)
FULLY_DENSE = dict(TOY_MODEL, use_latent_attention=False, use_moe=False)


def _train_a_few_steps(config, steps=5):
    set_seed(0)
    model = KimiK2(config)
    optimizer = create_muon_optimizer(model, vars(OptimizerConfig(learning_rate=1e-3)))
    ids = torch.randint(0, VOCAB, (2, BLOCK))
    losses = []
    for _ in range(steps):
        _, loss = model(ids, ids)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())
    return model, losses


def test_dense_attention_path_keeps_position_embeddings():
    """Without RoPE the model needs wpe; the MLA path drops it."""
    set_seed(0)
    dense = KimiK2(DENSE_ATTN)
    assert hasattr(dense.transformer, "wpe")
    assert not dense.use_rope

    mla = KimiK2(TOY_MODEL)
    assert not hasattr(mla.transformer, "wpe")
    assert mla.use_rope


def test_dense_attention_path_trains():
    model, losses = _train_a_few_steps(DENSE_ATTN)
    assert all(torch.isfinite(torch.tensor(v)) for v in losses)
    assert losses[-1] < losses[0]


def test_dense_ffn_path_trains():
    model, losses = _train_a_few_steps(DENSE_FFN)
    assert all(torch.isfinite(torch.tensor(v)) for v in losses)
    assert losses[-1] < losses[0]


def test_fully_dense_path_trains():
    model, losses = _train_a_few_steps(FULLY_DENSE)
    assert all(torch.isfinite(torch.tensor(v)) for v in losses)
    assert losses[-1] < losses[0]


def test_dense_ffn_has_no_expert_parameters():
    set_seed(0)
    model = KimiK2(DENSE_FFN)
    assert not any("experts" in name for name, _ in model.named_parameters())
    # with no MoE, active equals the non-embedding total
    expected = count_parameters(model) - model.transformer.wte.weight.numel()
    assert count_active_parameters(model) == expected


def test_mla_uses_a_smaller_kv_cache_than_dense_attention():
    set_seed(0)
    mla = KimiK2(TOY_MODEL).transformer.h[0].attn
    dense = KimiK2(DENSE_ATTN).transformer.h[0].attn
    assert mla.kv_cache_per_token() < dense.kv_cache_per_token()


def test_crop_block_size_on_both_paths():
    """Rewritten for RoPE; the dense path still crops wpe."""
    set_seed(0)
    mla = KimiK2(dict(TOY_MODEL))
    mla.crop_block_size(16)
    assert mla.config["block_size"] == 16
    with torch.no_grad():
        logits, _ = mla(torch.randint(0, VOCAB, (1, 16)))
    assert torch.isfinite(logits).all()

    dense = KimiK2(dict(DENSE_ATTN))
    dense.crop_block_size(16)
    assert dense.transformer.wpe.weight.shape[0] == 16
    with torch.no_grad():
        logits, _ = dense(torch.randint(0, VOCAB, (1, 16)))
    assert torch.isfinite(logits).all()


def test_model_does_not_mutate_the_config_it_was_given():
    """crop_block_size used to write through to the caller's dict, silently changing
    the config every other model was built from."""
    config = dict(TOY_MODEL)
    model = KimiK2(config)
    model.crop_block_size(8)
    assert config["block_size"] == TOY_MODEL["block_size"]
    assert model.config["block_size"] == 8

    sibling = KimiK2(config)
    assert sibling.config["block_size"] == TOY_MODEL["block_size"]


def test_qk_clip_works_on_the_dense_attention_path():
    set_seed(0)
    model = KimiK2(DENSE_ATTN)
    optimizer = create_muon_optimizer(model, vars(OptimizerConfig(qk_clip_tau=100.0)))
    assert len(optimizer.attention_modules) == DENSE_ATTN["n_layer"]

    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.qkv_proj.weight.mul_(30.0)
    model.train()
    ids = torch.randint(0, VOCAB, (2, BLOCK))
    for _ in range(4):
        _, loss = model(ids, ids)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    assert optimizer.last_max_logit < 10_000
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_export_round_trips_a_dense_checkpoint(tmp_path):
    """infer_config has a separate branch when there is no MLA to read dims from."""
    set_seed(0)
    model = KimiK2(DENSE_ATTN)
    optimizer = create_muon_optimizer(model, vars(OptimizerConfig()))
    ckpt = save_checkpoint(
        tmp_path / "ckpt.pt", model, optimizer, 0, 0, 9.9, dict(DENSE_ATTN)
    )

    state_dict, model_args = load_state_dict(ckpt)
    config = infer_config(state_dict, model_args)
    assert config["use_latent_attention"] is False
    assert config["block_size"] == DENSE_ATTN["block_size"]
    assert config["n_head"] == DENSE_ATTN["n_head"]


def test_export_checkpoint_writes_a_loadable_directory(tmp_path):
    """The top-level function scripts/export_hf.py calls."""
    set_seed(0)
    model = KimiK2(TOY_MODEL)
    optimizer = create_muon_optimizer(model, vars(OptimizerConfig()))
    ckpt = save_checkpoint(tmp_path / "ckpt.pt", model, optimizer, 0, 0, 9.9, dict(TOY_MODEL))

    out_dir = export_checkpoint(ckpt, tmp_path / "export")
    for name in ("model.safetensors", "config.json", "modeling_kimik2.py"):
        assert (out_dir / name).exists(), name

    from transformers import AutoModelForCausalLM

    reloaded = AutoModelForCausalLM.from_pretrained(out_dir, trust_remote_code=True).eval()
    model.eval()
    ids = torch.randint(0, VOCAB, (1, 16))
    with torch.no_grad():
        assert torch.allclose(model(ids)[0], reloaded(ids).logits[:, -1:, :], atol=1e-5)


def test_export_refuses_an_untied_lm_head(tmp_path):
    """Dropping lm_head is only safe because it is tied; silently losing it would be worse."""
    set_seed(0)
    model = KimiK2(TOY_MODEL)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    state["lm_head.weight"] = state["lm_head.weight"] + 1.0

    from nanokimi.export.hf import write_export

    config = infer_config(state, dict(TOY_MODEL))
    try:
        write_export(state, config, tmp_path / "bad")
    except ValueError as exc:
        assert "tie_word_embeddings" in str(exc)
    else:
        raise AssertionError("expected a ValueError for an untied lm_head")


def test_export_fails_loudly_on_a_missing_checkpoint(tmp_path):
    try:
        export_checkpoint(tmp_path / "nope.pt", tmp_path / "out")
    except FileNotFoundError as exc:
        assert "checkpoint not found" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_infer_config_requires_n_head_for_mla():
    """Head count is not recoverable from MLA shapes alone; a silent guess would be wrong."""
    set_seed(0)
    model = KimiK2(TOY_MODEL)
    try:
        infer_config(model.state_dict(), {})
    except ValueError as exc:
        assert "n_head missing" in str(exc)
    else:
        raise AssertionError("expected a ValueError when model_args lacks n_head")


def test_infer_config_falls_back_for_a_dense_ffn_checkpoint():
    """With no experts in the state dict, num_experts comes from model_args."""
    set_seed(0)
    model = KimiK2(DENSE_FFN)
    config = infer_config(model.state_dict(), dict(DENSE_FFN))
    assert config["use_moe"] is False
    assert config["num_experts"] == DENSE_FFN["num_experts"]


def test_export_pushes_to_the_hub_when_asked(tmp_path, monkeypatch):
    """The upload path is short but real; a typo here only shows up mid-release."""
    set_seed(0)
    model = KimiK2(TOY_MODEL)
    optimizer = create_muon_optimizer(model, vars(OptimizerConfig()))
    ckpt = save_checkpoint(tmp_path / "ckpt.pt", model, optimizer, 0, 0, 9.9, dict(TOY_MODEL))

    calls = {}

    class FakeApi:
        def create_repo(self, repo_id, private, exist_ok):
            calls["create"] = (repo_id, private, exist_ok)

        def upload_folder(self, folder_path, repo_id):
            calls["upload"] = (folder_path, repo_id)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)

    out_dir = export_checkpoint(ckpt, tmp_path / "export", push_to="someone/nanokimi-test", private=True)

    assert calls["create"] == ("someone/nanokimi-test", True, True)
    assert calls["upload"] == (str(out_dir), "someone/nanokimi-test")
