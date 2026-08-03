"""Tests for the exported remote-code modeling file.

This file is uploaded to the Hub and loaded with trust_remote_code=True, so bugs in
it land on downstream users rather than on us. It deliberately duplicates the
architecture (it may not import from nanokimi), which means it can drift from
src/nanokimi/model/ without anything noticing.
"""

import pytest
import torch

from nanokimi.export.modeling_kimik2 import (
    KimiK2Config,
    KimiK2ForCausalLM,
    MoELayer,
    MultiHeadAttention,
    MultiHeadLatentAttention,
)

TOY = dict(
    vocab_size=257, block_size=32, n_layer=2, n_head=4, n_embd=64, dropout=0.0, bias=True,
    use_moe=True, num_experts=4, expert_capacity=8, top_k_experts=2,
    use_latent_attention=True, kv_lora_rank=64, q_lora_rank=96, qk_nope_head_dim=16,
    qk_rope_head_dim=8, v_head_dim=16, rope_theta=50000.0, attention_bias=False,
)

DENSE_ATTN = dict(TOY, use_latent_attention=False)
DENSE_FFN = dict(TOY, use_moe=False)
FULLY_DENSE = dict(TOY, use_latent_attention=False, use_moe=False)


def build(overrides: dict) -> KimiK2ForCausalLM:
    torch.manual_seed(0)
    return KimiK2ForCausalLM(KimiK2Config(**overrides)).eval()


@pytest.mark.parametrize("config", [TOY, DENSE_ATTN, DENSE_FFN, FULLY_DENSE],
                         ids=["mla_moe", "dense_attn", "dense_ffn", "fully_dense"])
def test_every_architecture_variant_runs(config):
    model = build(config)
    ids = torch.randint(0, config["vocab_size"], (2, 16))
    with torch.no_grad():
        out = model(ids)
    assert out.logits.shape == (2, 16, config["vocab_size"])
    assert torch.isfinite(out.logits).all()


@pytest.mark.parametrize("config", [TOY, DENSE_ATTN], ids=["mla", "dense_attn"])
def test_position_handling_matches_the_attention_variant(config):
    model = build(config)
    if config["use_latent_attention"]:
        assert model.use_rope
        assert not hasattr(model.transformer, "wpe")
    else:
        assert not model.use_rope
        assert hasattr(model.transformer, "wpe")


def test_kv_cache_property_on_the_config():
    assert KimiK2Config(**TOY).kv_cache_per_token == TOY["kv_lora_rank"] + TOY["qk_rope_head_dim"]
    dense = KimiK2Config(**DENSE_ATTN)
    assert dense.kv_cache_per_token == dense.n_head * (dense.n_embd // dense.n_head) * 2


def test_max_position_embeddings_mirrors_block_size():
    assert KimiK2Config(**TOY).max_position_embeddings == TOY["block_size"]


def test_loss_is_computed_when_labels_are_given():
    model = build(TOY)
    ids = torch.randint(0, TOY["vocab_size"], (2, 16))
    with torch.no_grad():
        out = model(ids, labels=ids)
    assert out.loss is not None and torch.isfinite(out.loss)


def test_return_dict_false_yields_a_tuple():
    model = build(TOY)
    ids = torch.randint(0, TOY["vocab_size"], (1, 8))
    with torch.no_grad():
        logits_only = model(ids, return_dict=False)
        with_loss = model(ids, labels=ids, return_dict=False)
    assert isinstance(logits_only, tuple) and len(logits_only) == 1
    assert isinstance(with_loss, tuple) and len(with_loss) == 2


def test_inputs_embeds_path():
    model = build(TOY)
    ids = torch.randint(0, TOY["vocab_size"], (1, 8))
    embeds = model.transformer.wte(ids)
    with torch.no_grad():
        from_ids = model(ids).logits
        from_embeds = model(inputs_embeds=embeds).logits
    assert torch.allclose(from_ids, from_embeds, atol=1e-6)


def test_forward_requires_some_input():
    model = build(TOY)
    with pytest.raises(ValueError, match="input_ids or inputs_embeds"):
        model()


def test_sequence_longer_than_block_size_is_rejected():
    model = build(TOY)
    ids = torch.randint(0, TOY["vocab_size"], (1, TOY["block_size"] + 1))
    with pytest.raises(ValueError, match="block size"):
        model(ids)


def test_padding_attention_mask_changes_the_result():
    """The 2D mask is converted to an additive mask; a masked token must be ignored."""
    model = build(TOY)
    ids = torch.randint(1, TOY["vocab_size"], (1, 8))
    mask = torch.ones(1, 8, dtype=torch.long)
    mask[0, -3:] = 0
    with torch.no_grad():
        unmasked = model(ids).logits
        masked = model(ids, attention_mask=mask).logits
    assert torch.isfinite(masked).all()
    assert not torch.allclose(unmasked, masked, atol=1e-6)


def test_embedding_accessors_round_trip():
    model = build(TOY)
    assert model.get_input_embeddings() is model.transformer.wte
    assert model.get_output_embeddings() is model.lm_head

    replacement = torch.nn.Embedding(TOY["vocab_size"], TOY["n_embd"])
    model.set_input_embeddings(replacement)
    assert model.get_input_embeddings() is replacement

    head = torch.nn.Linear(TOY["n_embd"], TOY["vocab_size"], bias=False)
    model.set_output_embeddings(head)
    assert model.get_output_embeddings() is head


def test_get_num_params_excludes_position_embeddings_only_when_present():
    mla = build(TOY)
    assert mla.get_num_params() == sum(p.numel() for p in mla.parameters())

    dense = build(DENSE_ATTN)
    assert dense.get_num_params() == sum(p.numel() for p in dense.parameters()) - dense.transformer.wpe.weight.numel()


@pytest.mark.parametrize("config", [TOY, DENSE_ATTN], ids=["mla", "dense_attn"])
def test_generate_works_and_crops_context(config):
    model = build(config)
    ids = torch.randint(0, config["vocab_size"], (1, config["block_size"] - 2))
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=8, do_sample=False, use_cache=False, pad_token_id=0)
    assert out.shape == (1, config["block_size"] - 2 + 8)
    assert int(out.max()) < config["vocab_size"]


def test_prepare_inputs_crops_beyond_block_size():
    model = build(TOY)
    long_ids = torch.randint(0, TOY["vocab_size"], (1, TOY["block_size"] + 10))
    mask = torch.ones_like(long_ids)
    prepared = model.prepare_inputs_for_generation(long_ids, attention_mask=mask)
    assert prepared["input_ids"].shape[1] == TOY["block_size"]
    assert prepared["attention_mask"].shape[1] == TOY["block_size"]


def test_qk_clip_in_the_exported_mla():
    attn = MultiHeadLatentAttention(n_embd=64, n_head=4, kv_lora_rank=32, q_lora_rank=48,
                                    qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
                                    max_seq_len=16)
    with torch.no_grad():
        attn.q_b_proj.weight.mul_(30.0)
        attn.kv_b_proj.weight.mul_(30.0)
    attn.train()
    x = torch.randn(1, 8, 64)
    attn(x)
    assert attn.qk_max_logit.max() > 100.0
    shared_before = attn.kv_a_proj_with_mqa.weight.clone()

    clipped = attn.qk_clip_(100.0)
    assert clipped > 0
    assert torch.equal(shared_before, attn.kv_a_proj_with_mqa.weight), "shared k^R must be untouched"

    attn(x)
    assert attn.qk_max_logit.max() <= 101.0


def test_qk_clip_in_the_exported_dense_attention():
    attn = MultiHeadAttention(n_embd=64, n_head=4)
    with torch.no_grad():
        attn.qkv_proj.weight.mul_(15.0)
    attn.train()
    x = torch.randn(1, 8, 64)
    attn(x)
    assert attn.qk_max_logit.max() > 100.0

    v_before = attn.qkv_proj.weight.view(3, 4, 16, -1)[2].clone()
    attn.qk_clip_(100.0)
    assert torch.equal(v_before, attn.qkv_proj.weight.view(3, 4, 16, -1)[2])

    attn(x)
    assert attn.qk_max_logit.max() <= 101.0


def test_exported_kv_cache_helpers_agree_with_the_config():
    mla = MultiHeadLatentAttention(n_embd=64, n_head=4, kv_lora_rank=32, q_lora_rank=48,
                                   qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
                                   max_seq_len=16)
    assert mla.kv_cache_per_token() == 32 + 8
    assert MultiHeadAttention(n_embd=64, n_head=4).kv_cache_per_token() == 4 * 16 * 2


def test_exported_moe_matches_the_library_aux_loss():
    """The two implementations must not drift; a published model would score differently."""
    from nanokimi.model.moe import MoELayer as LibraryMoE

    torch.manual_seed(0)
    exported = MoELayer(32, num_experts=4, expert_capacity=8, top_k=2).eval()
    library = LibraryMoE(32, num_experts=4, expert_capacity=8, top_k=2).eval()
    library.load_state_dict(exported.state_dict())

    x = torch.randn(2, 16, 32)
    with torch.no_grad():
        out_e, aux_e = exported(x)
        out_l, aux_l = library(x)
    assert torch.allclose(out_e, out_l, atol=1e-6)
    assert torch.allclose(aux_e, aux_l, atol=1e-6)


def test_exported_mla_matches_the_library_mla():
    from nanokimi.model.attention import MultiHeadLatentAttention as LibraryMLA

    kwargs = dict(n_embd=64, n_head=4, kv_lora_rank=32, q_lora_rank=48,
                  qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16, max_seq_len=16)
    torch.manual_seed(0)
    exported = MultiHeadLatentAttention(**kwargs).eval()
    library = LibraryMLA(**kwargs).eval()
    library.load_state_dict(exported.state_dict())

    x = torch.randn(2, 12, 64)
    with torch.no_grad():
        assert torch.allclose(exported(x), library(x), atol=1e-6)


def test_exported_moe_survives_bfloat16_autocast():
    """index_add_ with a bf16 source into an fp32 accumulator crashed every CUDA run."""
    layer = MoELayer(32, num_experts=4, expert_capacity=8, top_k=2).eval()
    x = torch.randn(2, 8, 32)
    with torch.no_grad(), torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        out, _ = layer(x)
    assert torch.isfinite(out.float()).all()


def test_exported_moe_capacity_path_matches_the_library():
    """Both files implement capacity dropping; they must drop the same tokens."""
    from nanokimi.model.moe import MoELayer as LibraryMoE

    torch.manual_seed(0)
    exported = MoELayer(32, num_experts=2, expert_capacity=4, top_k=1, apply_expert_capacity=True)
    library = LibraryMoE(32, num_experts=2, expert_capacity=4, top_k=1, apply_expert_capacity=True)
    library.load_state_dict(exported.state_dict())
    exported.train(), library.train()

    x = torch.randn(1, 32, 32)
    with torch.no_grad():
        out_e, _ = exported(x)
        out_l, _ = library(x)

    dropped = int((out_e.view(-1, 32).abs().sum(-1) == 0).sum())
    assert dropped > 0, "capacity should bind with 32 tokens and capacity 4"
    assert torch.allclose(out_e, out_l, atol=1e-6), "the two implementations dropped different tokens"
