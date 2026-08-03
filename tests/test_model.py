"""Model tests.

Covers properties that are easy to break silently: causality, the block_size boundary,
gradient reaching every parameter, and behaviour under autocast and torch.compile.
"""

import math

import pytest
import torch

from nanokimi.model.attention import MultiHeadAttention, MultiHeadLatentAttention, build_rope_cache
from nanokimi.model.moe import MoELayer
from nanokimi.model.transformer import KimiK2
from nanokimi.training.optimizer import create_muon_optimizer
from nanokimi.utils.config import OptimizerConfig
from nanokimi.utils.seeding import set_seed
from tests.conftest import TOY_MODEL

VOCAB = TOY_MODEL["vocab_size"]
BLOCK = TOY_MODEL["block_size"]


@pytest.fixture
def model():
    set_seed(0)
    return KimiK2(TOY_MODEL).eval()


# --- causality -------------------------------------------------------------


def test_model_is_strictly_causal(model):
    """A change at position t must not alter any output before t.

    Passing targets is what makes forward return logits for every position; the
    inference path deliberately computes only the last one.
    """
    ids = torch.randint(0, VOCAB, (1, BLOCK))
    with torch.no_grad():
        base, _ = model(ids, ids)
    assert base.shape == (1, BLOCK, VOCAB)

    cut = BLOCK // 2
    perturbed = ids.clone()
    perturbed[0, cut:] = (perturbed[0, cut:] + 1) % VOCAB
    with torch.no_grad():
        after, _ = model(perturbed, perturbed)

    assert torch.allclose(base[:, :cut], after[:, :cut], atol=1e-5)
    assert not torch.allclose(base[:, cut:], after[:, cut:], atol=1e-5)


def test_attention_variants_are_causal():
    for attn in (
        MultiHeadLatentAttention(n_embd=128, n_head=4, kv_lora_rank=64, q_lora_rank=96,
                                 qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32, max_seq_len=32),
        MultiHeadAttention(n_embd=128, n_head=4),
    ):
        attn.eval()
        x = torch.randn(1, 16, 128)
        with torch.no_grad():
            base = attn(x)
        x2 = x.clone()
        x2[:, 8:] += 5.0
        with torch.no_grad():
            after = attn(x2)
        assert torch.allclose(base[:, :8], after[:, :8], atol=1e-5), type(attn).__name__


# --- sequence length boundaries -------------------------------------------


def test_forward_accepts_exactly_block_size(model):
    ids = torch.randint(0, VOCAB, (1, BLOCK))
    with torch.no_grad():
        logits, _ = model(ids)
    assert logits.shape[1] == 1  # inference path returns only the last position


def test_forward_rejects_more_than_block_size(model):
    ids = torch.randint(0, VOCAB, (1, BLOCK + 1))
    with pytest.raises(AssertionError):
        model(ids)


def test_short_sequences_work(model):
    for length in (1, 2, 7):
        ids = torch.randint(0, VOCAB, (1, length))
        with torch.no_grad():
            logits, _ = model(ids)
        assert logits.shape[-1] == VOCAB


# --- rope ------------------------------------------------------------------


def test_rope_tables_rebuild_for_longer_sequences():
    """Tables are built lazily; a sequence longer than max_seq_len must not read garbage."""
    attn = MultiHeadLatentAttention(n_embd=128, n_head=4, kv_lora_rank=64, q_lora_rank=96,
                                    qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32,
                                    max_seq_len=8).eval()
    with torch.no_grad():
        out = attn(torch.randn(1, 24, 128))
    assert torch.isfinite(out).all()
    assert attn._rope_cache[0].size(0) >= 24


def test_rope_is_norm_preserving():
    cos, sin = build_rope_cache(16, 32, theta=50000.0)
    x = torch.randn(1, 1, 16, 32)
    y = x * cos.view(1, 1, 16, 32) + torch.cat(torch.chunk(x, 2, -1)[::-1], -1) * torch.tensor(0.0)
    # exercise the real helper rather than reimplementing it
    from nanokimi.model.attention import apply_rope

    y = apply_rope(x, cos.view(1, 1, 16, 32), sin.view(1, 1, 16, 32))
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)


def test_rope_tables_are_not_in_the_state_dict(model):
    """Persisting them would pin context length and bloat every checkpoint."""
    assert not any("rope_" in key for key in model.state_dict())


# --- gradients -------------------------------------------------------------


def test_every_parameter_receives_gradient(model):
    """q_compress was silently frozen for a whole training run; nothing may be unreachable."""
    model.train()
    ids = torch.randint(0, VOCAB, (2, BLOCK))
    _, loss = model(ids, ids)
    loss.backward()

    missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no gradient reached: {missing}"

    dead = [
        name
        for name, p in model.named_parameters()
        if p.grad is not None and p.grad.abs().max().item() == 0.0
    ]
    # the router can legitimately zero a few expert grads on a tiny batch, so allow a margin
    assert len(dead) <= TOY_MODEL["num_experts"], f"all-zero gradients: {dead}"


def test_loss_is_finite_and_near_uniform_at_init(model):
    model.train()
    ids = torch.randint(0, VOCAB, (2, BLOCK))
    _, loss = model(ids, ids)
    assert torch.isfinite(loss)
    # random init should sit near ln(vocab); far above means something is broken
    assert abs(loss.item() - math.log(VOCAB)) < 1.5


# --- moe -------------------------------------------------------------------


def test_moe_output_is_deterministic_at_inference():
    layer = MoELayer(64, num_experts=4, expert_capacity=8, top_k=2).eval()
    x = torch.randn(2, 16, 64)
    with torch.no_grad():
        assert torch.equal(layer(x)[0], layer(x)[0])


def test_moe_drops_no_tokens_at_inference():
    layer = MoELayer(64, num_experts=4, expert_capacity=2, top_k=2).eval()
    x = torch.randn(2, 32, 64)
    with torch.no_grad():
        out, _ = layer(x)
    assert int((out.view(-1, 64).abs().sum(-1) == 0).sum()) == 0


def test_moe_aux_loss_is_minimised_by_balanced_routing():
    layer = MoELayer(64, num_experts=4, expert_capacity=8, top_k=2)
    x = torch.randn(4, 64, 64)
    _, aux = layer(x)
    # normalised so a perfectly balanced router scores 1.0 before the coefficient
    assert aux.item() >= layer.load_balance_loss_coef * 0.999


# --- eval mode -------------------------------------------------------------


def test_dropout_is_inactive_in_eval():
    config = dict(TOY_MODEL, dropout=0.5)
    set_seed(0)
    model = KimiK2(config).eval()
    ids = torch.randint(0, VOCAB, (2, BLOCK))
    with torch.no_grad():
        assert torch.equal(model(ids)[0], model(ids)[0])


def test_generate_respects_block_size():
    """Generation past block_size must crop context, not crash or read past the table."""
    set_seed(0)
    model = KimiK2(TOY_MODEL).eval()
    ids = torch.randint(0, VOCAB, (1, BLOCK - 2))
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=10, temperature=0.8, top_k=20)
    assert out.shape == (1, BLOCK - 2 + 10)
    assert int(out.max()) < VOCAB


def test_greedy_generation_is_deterministic():
    set_seed(0)
    model = KimiK2(TOY_MODEL).eval()
    ids = torch.randint(0, VOCAB, (1, 8))
    with torch.no_grad():
        a = model.generate(ids, max_new_tokens=5, temperature=1.0, top_k=1)
        b = model.generate(ids, max_new_tokens=5, temperature=1.0, top_k=1)
    assert torch.equal(a, b)


# --- numerics --------------------------------------------------------------


def test_forward_is_finite_under_bfloat16_autocast(model):
    """bf16 is the training dtype on the H100; -inf masking would produce NaNs here."""
    ids = torch.randint(0, VOCAB, (2, BLOCK))
    with torch.no_grad(), torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits, _ = model(ids)
    assert torch.isfinite(logits.float()).all()


def test_model_init_is_reproducible():
    set_seed(123)
    a = KimiK2(TOY_MODEL)
    set_seed(123)
    b = KimiK2(TOY_MODEL)
    for (name, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.equal(pa, pb), name


def test_optimizer_step_keeps_weights_finite():
    set_seed(0)
    model = KimiK2(TOY_MODEL)
    optimizer = create_muon_optimizer(model, vars(OptimizerConfig(learning_rate=1e-2)))
    ids = torch.randint(0, VOCAB, (2, BLOCK))
    for _ in range(5):
        _, loss = model(ids, ids)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all(), name


@pytest.mark.slow
def test_torch_compile_matches_eager():
    """configs set compile: true, but it had never been exercised against MLA."""
    set_seed(0)
    model = KimiK2(TOY_MODEL).eval()
    ids = torch.randint(0, VOCAB, (1, 16))
    with torch.no_grad():
        eager = model(ids)[0]

    compiled = torch.compile(model)
    with torch.no_grad():
        got = compiled(ids)[0]
    assert torch.allclose(eager, got, atol=1e-4), f"max|d| = {(eager - got).abs().max():.3e}"
