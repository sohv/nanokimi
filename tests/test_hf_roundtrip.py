#!/usr/bin/env python3
"""
Regression test for the HuggingFace export path.

Guards the three things that were actually broken in the first nanokimi-mini
upload, so they cannot silently break again:

  1. `_orig_mod.` / `module.` prefixes left on the keys by torch.compile / DDP
  2. config.json's vocab_size disagreeing with the shape of the weights
  3. `q_compress` being created lazily inside forward(), so a freshly built model
     does not have the parameter the checkpoint contains

Run with no arguments to exercise a fresh random model end to end:

    python test_hf_roundtrip.py

Run against an exported directory to validate a real checkpoint:

    python test_hf_roundtrip.py hf_export/nanokimi-25m
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

from nanokimi.export import hf as hf_export
from nanokimi.export.modeling_kimik2 import KimiK2Config, KimiK2ForCausalLM

TOY = dict(
    vocab_size=257, block_size=32, n_layer=2, n_head=4, n_embd=64, dropout=0.0,
    bias=True, use_moe=True, num_experts=4, expert_capacity=8, top_k_experts=2,
    use_latent_attention=True,
    # MLA dims, same DeepSeek-V3 ratios scaled to head_dim=16.
    kv_lora_rank=64, q_lora_rank=96, qk_nope_head_dim=16, qk_rope_head_dim=8,
    v_head_dim=16, rope_theta=50000.0, attention_bias=False,
)


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        raise AssertionError(name)


def _export_roundtrip(out_dir: Path):
    """A model exported by scripts/hf_export.py must reload with identical weights."""
    print("\n[1] export -> AutoModelForCausalLM round trip")

    cfg = KimiK2Config(**TOY)
    model = KimiK2ForCausalLM(cfg).eval()

    # Simulate what training produces: torch.compile prefixes and a tied lm_head.
    compiled_sd = {f"_orig_mod.{k}": v for k, v in model.state_dict().items()}
    ckpt = {"model": compiled_sd, "model_args": dict(TOY)}
    torch.save(ckpt, out_dir / "ckpt.pt")

    state_dict, model_args = hf_export.load_state_dict(out_dir / "ckpt.pt")
    check("compile prefix stripped", not any(k.startswith("_orig_mod.") for k in state_dict))

    config = hf_export.infer_config(state_dict, model_args)
    check(
        "config vocab_size matches weights",
        config["vocab_size"] == state_dict["transformer.wte.weight"].shape[0],
        f"{config['vocab_size']}",
    )
    check("config n_layer inferred", config["n_layer"] == TOY["n_layer"])
    check("config n_head inferred", config["n_head"] == TOY["n_head"])
    for key in ["kv_lora_rank", "q_lora_rank", "qk_nope_head_dim",
                "qk_rope_head_dim", "v_head_dim"]:
        check(f"config {key} inferred", config[key] == TOY[key],
              f"{config[key]} vs {TOY[key]}")

    export_dir = out_dir / "export"
    hf_export.write_export(state_dict, config, export_dir)

    reloaded = AutoModelForCausalLM.from_pretrained(export_dir, trust_remote_code=True).eval()

    orig_sd, new_sd = model.state_dict(), reloaded.state_dict()
    check("no missing keys", set(orig_sd) == set(new_sd),
          f"{len(set(orig_sd) ^ set(new_sd))} symmetric difference")
    bad = [k for k in orig_sd if not torch.equal(orig_sd[k], new_sd[k])]
    check("all weights bit-identical after reload", not bad, f"{len(bad)} differing")
    check("lm_head tied to wte",
          torch.equal(reloaded.lm_head.weight, reloaded.transformer.wte.weight))

    return model, reloaded


def _logits_identical(model, reloaded):
    """Forward passes must agree exactly, and be deterministic."""
    print("\n[2] forward pass equivalence")

    torch.manual_seed(0)
    ids = torch.randint(0, TOY["vocab_size"], (2, TOY["block_size"]))

    with torch.no_grad():
        a = model(ids).logits
        b = reloaded(ids).logits
        b2 = reloaded(ids).logits

    check("logit shape", tuple(a.shape) == (2, TOY["block_size"], TOY["vocab_size"]),
          str(tuple(a.shape)))
    check("logits identical after reload", torch.equal(a, b),
          f"max|d| = {(a - b).abs().max():.3e}")
    check("inference is deterministic", torch.equal(b, b2))

    with torch.no_grad():
        loss = reloaded(ids, labels=ids).loss
    check("labels produce a finite loss", torch.isfinite(loss), f"loss = {loss.item():.4f}")

    gen = reloaded.generate(ids[:1, :4], max_new_tokens=6, do_sample=False,
                            use_cache=False, pad_token_id=0)
    check("generate() works", gen.shape == (1, 10), str(tuple(gen.shape)))


def _no_lazy_parameters():
    """
    No parameter may be created inside forward(). The old LatentAttention built
    q_compress lazily, which both excluded it from the optimizer (so it never
    trained) and made fresh models un-loadable from a trained checkpoint.
    """
    print("\n[3] no parameters are created on first forward")

    model = KimiK2ForCausalLM(KimiK2Config(**TOY))
    keys = set(model.state_dict())
    names = {n for n, _ in model.named_parameters()}

    for expected in ["transformer.h.0.attn.q_a_proj.weight",
                     "transformer.h.0.attn.q_b_proj.weight",
                     "transformer.h.0.attn.kv_a_proj_with_mqa.weight",
                     "transformer.h.0.attn.kv_b_proj.weight",
                     "transformer.h.0.attn.q_a_layernorm.weight",
                     "transformer.h.0.attn.kv_a_layernorm.weight"]:
        check(f"{expected.split('attn.')[-1]} present before forward", expected in names)

    with torch.no_grad():
        model(torch.zeros(1, 4, dtype=torch.long))
    check("forward adds no new parameters", set(model.state_dict()) == keys)

    # MLA uses RoPE, so there must be no learned position embedding to drift.
    check("no wpe on the MLA path", not any("wpe" in k for k in keys))
    # RoPE tables must stay out of the checkpoint so context length stays changeable.
    check("RoPE tables are not persisted", not any("rope_" in k for k in keys))


def _real_export(path: Path):
    """Validate an already-exported directory holding real trained weights."""
    print(f"\n[4] validating exported checkpoint at {path}")

    from safetensors.torch import load_file

    config = json.loads((path / "config.json").read_text())
    sd = load_file(path / "model.safetensors")

    check("no compile prefixes", not any(k.startswith("_orig_mod.") for k in sd))
    check("config vocab_size matches weights",
          config["vocab_size"] == sd["transformer.wte.weight"].shape[0],
          f"config={config['vocab_size']} weights={sd['transformer.wte.weight'].shape[0]}")

    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    check("AutoConfig loads", type(cfg).__name__ == "KimiK2Config")

    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True).eval()
    check("AutoModelForCausalLM loads", type(model).__name__ == "KimiK2ForCausalLM")

    loaded = model.state_dict()
    bad = [k for k in sd if k not in loaded or not torch.equal(loaded[k], sd[k])]
    check("every checkpoint tensor reached the model", not bad, f"{len(bad)} differing")
    check("lm_head tied", torch.equal(model.lm_head.weight, model.transformer.wte.weight))

    ids = torch.randint(0, config["vocab_size"], (1, min(16, config["block_size"])))
    with torch.no_grad():
        a, b = model(ids).logits, model(ids).logits
    check("inference is deterministic", torch.equal(a, b),
          f"max|d| = {(a - b).abs().max():.3e}")
    print(f"       {sum(p.numel() for p in model.parameters()):,} parameters loaded")


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """Export a toy model once and reuse it across the tests in this module."""
    out_dir = tmp_path_factory.mktemp("export")
    model, reloaded = _export_roundtrip(out_dir)
    return {"out_dir": out_dir, "model": model, "reloaded": reloaded}


def test_export_round_trip_preserves_weights(exported):
    assert exported["reloaded"] is not None


def test_logits_match_after_reload(exported):
    _logits_identical(exported["model"], exported["reloaded"])


def test_no_parameters_created_in_forward():
    _no_lazy_parameters()


def test_exported_directory_is_valid(exported):
    _real_export(exported["out_dir"] / "export")


def main():
    """Standalone runner, so this file also works as `python tests/test_hf_roundtrip.py`."""
    print("nanoKimi HuggingFace round-trip test")

    with tempfile.TemporaryDirectory() as tmp:
        model, reloaded = _export_roundtrip(Path(tmp))
        _logits_identical(model, reloaded)
        _no_lazy_parameters()

        if len(sys.argv) > 1:
            _real_export(Path(sys.argv[1]))

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
