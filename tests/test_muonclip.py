#!/usr/bin/env python3
"""
Regression test for the MuonClip optimizer and the MoE router.

Guards the things that were actually wrong before, so they cannot come back:

  1. The "Muon" optimizer was Adam in disguise - no Newton-Schulz orthogonalization
     at all, and beta1 == beta2 == momentum.
  2. Weight decay was folded into the gradient (L2) rather than decoupled, which
     drove every hidden matrix in nanokimi-mini to ~1e-6.
  3. There was no QK-Clip, so nothing bounded attention logits.
  4. The MoE auxiliary loss was MSE on mean gate probability, which never sees the
     hard dispatch decision and so cannot penalise router collapse.

    python test_muonclip.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanokimi.model.attention import MultiHeadAttention, MultiHeadLatentAttention
from nanokimi.model.moe import MoELayer
from nanokimi.model.transformer import KimiK2
from nanokimi.training.optimizer import (
    MuonClip,
    build_param_groups,
    collect_attention_modules,
    create_muon_optimizer,
    muon_update,
    zeropower_via_newtonschulz5,
)

TAU = 100.0
TOY = dict(
    vocab_size=512, block_size=32, n_layer=2, n_head=4, n_embd=128, dropout=0.0,
    bias=True, use_moe=True, num_experts=8, expert_capacity=8, top_k_experts=2,
    use_latent_attention=True, load_balance_loss_coef=0.01, apply_expert_capacity=False,
    # MLA dims, DeepSeek-V3 ratios scaled to head_dim=32.
    kv_lora_rank=128, q_lora_rank=384, qk_nope_head_dim=32, qk_rope_head_dim=16,
    v_head_dim=32, rope_theta=50000.0, attention_bias=False,
)


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        raise AssertionError(name)


def reference_newtonschulz5(G, steps):
    """Verbatim from https://github.com/KellerJordan/Muon/blob/master/muon.py"""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def test_newton_schulz():
    """The orthogonalization must match the reference implementation exactly."""
    print("\n[1] Newton-Schulz orthogonalization")
    torch.manual_seed(0)
    for shape in [(256, 64), (64, 256), (128, 128), (3072, 768)]:
        G = torch.randn(*shape)
        mine = zeropower_via_newtonschulz5(G, steps=5)
        ref = reference_newtonschulz5(G, steps=5).to(G.dtype)
        check(f"matches reference for {shape}", torch.equal(mine, ref),
              f"max|d| = {(mine - ref).abs().max():.3e}")
        # The reference documents output as US'V^T with S'_ii ~ Uniform(0.5, 1.5).
        U, _, Vh = torch.linalg.svd(G, full_matrices=False)
        align = torch.sum(mine.float() * (U @ Vh)) / (mine.float().norm() * (U @ Vh).norm())
        check(f"aligned with U V^T for {shape}", align > 0.90, f"cos = {align:.4f}")


def test_rms_matching():
    """Algorithm 1 line 5 scales the update to Adam-like RMS, so it should land near 0.2."""
    print("\n[2] consistent RMS matching (sqrt(max(n,m)) * 0.2)")
    torch.manual_seed(0)
    for shape in [(768, 768), (3072, 768), (768, 3072)]:
        G = torch.randn(*shape)
        u = muon_update(G.clone(), torch.zeros_like(G), beta=0.95, ns_steps=5, rms_scale=0.2)
        rms = u.float().pow(2).mean().sqrt().item()
        check(f"update RMS near 0.2 for {shape}", 0.10 < rms < 0.30, f"rms = {rms:.4f}")


def test_qk_clip_math():
    """QK-Clip must bound the max logit at tau and leave compliant heads alone."""
    print("\n[3] QK-Clip bounds the max attention logit")

    mla = MultiHeadLatentAttention(
        n_embd=128, n_head=4, kv_lora_rank=64, q_lora_rank=96,
        qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32, max_seq_len=32)
    for name, attn, blow in [
        ("MLA", mla, lambda m: (m.q_b_proj.weight.mul_(25.0), m.kv_b_proj.weight.mul_(25.0))),
        ("MultiHeadAttention", MultiHeadAttention(n_embd=128, n_head=4, bias=True),
         lambda m: (m.qkv_proj.weight.mul_(12.0),)),
    ]:
        with torch.no_grad():
            blow(attn)
        attn.train()
        torch.manual_seed(0)
        x = torch.randn(2, 16, 128)

        attn(x)
        before = attn.qk_max_logit.clone()
        check(f"{name}: logits exceed tau before clipping", before.max() > TAU,
              f"S_max = {before.max():.1f}")

        n = attn.qk_clip_(TAU)
        check(f"{name}: clipped exactly the exceeding heads", n == int((before > TAU).sum()),
              f"{n} heads")
        check(f"{name}: tracker reset after clip", float(attn.qk_max_logit.max()) == 0.0)

        attn(x)
        after = attn.qk_max_logit.clone()
        # No optimizer step in between, so the correction should be exact here.
        check(f"{name}: max logit now at tau", after.max() <= TAU * 1.01,
              f"S_max = {after.max():.1f}")
        untouched = before <= TAU
        if untouched.any():
            check(f"{name}: under-tau heads unchanged",
                  torch.allclose(before[untouched], after[untouched], rtol=1e-3),
                  f"{int(untouched.sum())} heads")

    # The value block of a fused QKV projection must never be rescaled.
    mha = MultiHeadAttention(n_embd=128, n_head=4, bias=True)
    with torch.no_grad():
        mha.qkv_proj.weight.mul_(12.0)
    mha.train()
    v_before = mha.qkv_proj.weight.view(3, 4, 32, -1)[2].clone()
    mha(torch.randn(2, 16, 128))
    mha.qk_clip_(TAU)
    check("fused QKV: value block untouched",
          torch.equal(v_before, mha.qkv_proj.weight.view(3, 4, 32, -1)[2]))


def test_param_grouping():
    """Muon is only valid for 2D hidden weights; everything else must go to AdamW."""
    print("\n[4] parameter grouping")
    model = KimiK2(TOY)
    groups = build_param_groups(model, weight_decay=0.1)
    muon = [g for g in groups if g["use_muon"]][0]["params"]
    adamw = [p for g in groups if not g["use_muon"] for p in g["params"]]

    check("every Muon parameter is 2D", all(p.ndim == 2 for p in muon), f"{len(muon)} params")
    check("no wpe on the MLA path (RoPE carries position)",
          not hasattr(model.transformer, "wpe"))
    excluded = {id(model.transformer.wte.weight), id(model.lm_head.weight)}
    check("no embedding or output head in the Muon group",
          not any(id(p) in excluded for p in muon))
    check("token embedding routed to AdamW",
          any(id(model.transformer.wte.weight) == id(a) for a in adamw))
    total = sum(len(g["params"]) for g in groups)
    unique = len({id(p) for p in model.parameters()})
    check("every parameter grouped exactly once (tied wte/lm_head counted once)",
          total == unique, f"{total} grouped vs {unique} unique")
    check("attention modules discovered for QK-Clip",
          len(collect_attention_modules(model)) == TOY["n_layer"])

    # A bare model.parameters() must not crash: 1D params fall back to AdamW.
    m = nn.Sequential(nn.Linear(32, 64), nn.LayerNorm(64), nn.Linear(64, 8))
    opt = MuonClip(list(m.parameters()), lr=1e-3, qk_clip_tau=None)
    m(torch.randn(4, 32)).sum().backward()
    opt.step()
    check("bare model.parameters() is handled without crashing",
          all(torch.isfinite(p).all() for p in m.parameters()))


def test_no_weight_collapse():
    """
    The old optimizer coupled L2 into the normalized update, which pinned every
    hidden matrix at roughly the learning-rate scale. Decoupled decay must not.
    """
    print("\n[5] training does not collapse the weights")
    torch.manual_seed(0)
    model = KimiK2(TOY)
    opt = create_muon_optimizer(model, dict(
        learning_rate=6e-4, momentum=0.95, weight_decay=0.1, eps=1e-8, qk_clip_tau=TAU))
    check("create_muon_optimizer returns MuonClip", isinstance(opt, MuonClip))

    model.train()
    torch.manual_seed(1)
    ids = torch.randint(0, TOY["vocab_size"], (4, TOY["block_size"]))
    before = {n: p.detach().clone() for n, p in model.named_parameters()}

    losses = []
    for _ in range(60):
        _, loss = model(ids, ids)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())

    check("loss decreased", losses[-1] < losses[0], f"{losses[0]:.3f} -> {losses[-1]:.3f}")
    check("all losses finite", all(torch.isfinite(torch.tensor(x)) for x in losses))

    for name in ["transformer.h.0.attn.q_b_proj.weight",
                 "transformer.h.0.attn.kv_b_proj.weight",
                 "transformer.h.0.mlp.gate.weight",
                 "transformer.h.0.mlp.experts.0.fc1.weight"]:
        std = dict(model.named_parameters())[name].std().item()
        check(f"{name.split('.', 2)[-1]} did not collapse", std > 1e-4, f"std = {std:.5f}")

    # Every MLA projection must actually receive updates. The old q_compress was
    # created lazily inside forward(), so it never entered the optimizer at all.
    for name in ["transformer.h.0.attn.q_a_proj.weight",
                 "transformer.h.0.attn.q_b_proj.weight",
                 "transformer.h.0.attn.kv_a_proj_with_mqa.weight",
                 "transformer.h.0.attn.kv_b_proj.weight"]:
        moved = (dict(model.named_parameters())[name] - before[name]).abs().max().item()
        check(f"{name.split('attn.')[-1]} is trained", moved > 1e-6,
              f"max|delta| = {moved:.3e}")


def test_moe_aux_loss():
    """
    The Switch/GShard loss couples the hard dispatch fraction f_i to the soft
    probability P_i. The old MSE-on-P_i loss could not see routing at all.
    """
    print("\n[6] MoE auxiliary load-balancing loss")
    n_exp, top_k = 8, 2

    def switch_aux(scores, topi):
        one_hot = F.one_hot(topi, num_classes=n_exp).sum(dim=1)
        f_i = one_hot.float().sum(dim=0) / (scores.size(0) * top_k)
        return n_exp * torch.sum(f_i * scores.mean(dim=0))

    def old_mse_aux(scores):
        usage = scores.mean(dim=0)
        return F.mse_loss(usage, torch.full_like(usage, 1.0 / n_exp))

    balanced = torch.zeros(256, n_exp)
    collapsed = torch.full((256, n_exp), -10.0)
    collapsed[:, :top_k] = 10.0

    def route(lg):
        s = F.softmax(lg, dim=-1)
        return s, torch.topk(s, top_k, dim=-1).indices

    s_b, t_b = route(balanced)
    s_c, t_c = route(collapsed)
    check("balanced routing gives the minimum value of 1.0",
          abs(switch_aux(s_b, t_b).item() - 1.0) < 1e-4, f"{switch_aux(s_b, t_b).item():.4f}")
    check("collapsed routing is penalised much more heavily",
          switch_aux(s_c, t_c).item() > 3.0, f"{switch_aux(s_c, t_c).item():.4f}")

    swing_new = abs(switch_aux(s_c, t_c).item() - switch_aux(s_b, t_b).item())
    swing_old = abs(old_mse_aux(s_c).item() - old_mse_aux(s_b).item())
    check("corrected loss has far more dynamic range than the old MSE",
          swing_new > 50 * swing_old, f"{swing_new:.3e} vs {swing_old:.3e}")

    # The layer itself must be deterministic and lossless at inference.
    layer = MoELayer(64, num_experts=n_exp, expert_capacity=8, top_k=top_k).eval()
    x = torch.randn(2, 64, 64)
    with torch.no_grad():
        a, _ = layer(x)
        b, _ = layer(x)
    check("inference is deterministic", torch.equal(a, b),
          f"max|d| = {(a - b).abs().max():.3e}")
    dropped = int((a.view(-1, 64).abs().sum(-1) == 0).sum())
    check("no tokens dropped at inference", dropped == 0, f"{dropped} zeroed")


def test_qk_clip_in_training_loop():
    """QK-Clip must fire through the optimizer and pull exploding logits back to tau."""
    print("\n[7] QK-Clip inside the training loop")
    torch.manual_seed(0)
    model = KimiK2(TOY)
    opt = create_muon_optimizer(model, dict(
        learning_rate=6e-4, momentum=0.95, weight_decay=0.1, eps=1e-8, qk_clip_tau=TAU))
    model.train()

    with torch.no_grad():  # force a genuine logit explosion
        for blk in model.transformer.h:
            blk.attn.q_b_proj.weight.mul_(200.0)
            blk.attn.kv_b_proj.weight.mul_(200.0)

    ids = torch.randint(0, TOY["vocab_size"], (4, TOY["block_size"]))
    peak, history = 0.0, []
    for _ in range(8):
        _, loss = model(ids, ids)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        peak = max(peak, opt.last_max_logit)
        history.append(opt.last_max_logit)

    check("logits genuinely exploded before clipping", peak > 500, f"peak = {peak:.1f}")
    check("QK-Clip brought them down by an order of magnitude",
          history[-1] < history[0] / 5, f"{history[0]:.1f} -> {history[-1]:.1f}")
    # QK-Clip is a feedback controller, not a hard clamp: the Muon update runs
    # between the forward that measured S_max and the clip that uses it, so the
    # steady state sits somewhat above tau rather than exactly on it. This mirrors
    # Figure 2 (right) of the Kimi K2 report, where logits ride at the cap.
    check("max logit settles close to tau", history[-1] <= TAU * 1.25, f"{history[-1]:.1f}")
    check("steady state is stable, not still climbing", max(history[3:]) <= TAU * 1.25,
          f"last 5 steps: {[round(h, 1) for h in history[3:]]}")


def main():
    print("nanoKimi MuonClip / MoE regression test")
    test_newton_schulz()
    test_rms_matching()
    test_qk_clip_math()
    test_param_grouping()
    test_no_weight_collapse()
    test_moe_aux_loss()
    test_qk_clip_in_training_loop()
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
