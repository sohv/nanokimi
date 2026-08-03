"""
MuonClip Optimizer Implementation for nanoKimi

Implements Algorithm 1 of the Kimi K2 technical report (arXiv:2507.20534):
Muon (momentum + Newton-Schulz orthogonalization + consistent RMS matching)
followed by QK-Clip, which rescales per-head query/key projection weights
whenever that head's max pre-softmax attention logit exceeds tau.

References:
  - Muon: https://kellerjordan.github.io/posts/muon/ (Keller Jordan, 2024)
  - MuonClip / QK-Clip: Kimi K2 Technical Report, Section 2.1, Algorithm 1
  - Moonlight (Muon at scale): arXiv:2502.16982
"""

import math                                                                    # Added to compute the sqrt(max(n,m)) RMS scaling factor.
import torch
import torch.nn as nn                                                          # Added so parameter grouping can test module types.
import torch.optim as optimizer
from typing import Any, Dict, Iterable, List, Optional                         # Added Iterable/List for the new param-group helpers.


# Coefficients of the quintic Newton-Schulz polynomial p(x) = ax + bx^3 + cx^5.
# Chosen because Keller Jordan tuned them to maximise the slope at zero, which
# inflates small singular values fastest and lets 5 iterations suffice.
NS_COEFFS = (3.4445, -4.7750, 2.0315)                                          # Added the reference Newton-Schulz coefficients, which the old code lacked entirely.


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Orthogonalize G via a quintic Newton-Schulz iteration.

    Returns approximately U V^T from the SVD G = U S V^T. The iteration is run in
    bfloat16 because it is numerically stable there and much faster on GPU.
    """
    assert G.ndim >= 2, "Newton-Schulz orthogonalization requires a matrix"     # Added a guard so 1D params can never reach this path.
    a, b, c = NS_COEFFS                                                        # Added unpacking of the tuned quintic coefficients.
    X = G.bfloat16()                                                           # Added bf16 cast because the reference runs the iteration in bf16 for speed and stability.
    transposed = G.size(-2) > G.size(-1)                                       # Added a shape check so we always iterate on the wider orientation.
    if transposed:                                                             # Added transpose handling to keep the Gram matrix as small as possible.
        X = X.mT                                                               # Added the transpose that makes X wide rather than tall.
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)                        # Added Frobenius normalization so the spectral norm starts at most 1, which the iteration requires.
    for _ in range(steps):                                                     # Added the iteration loop, which the previous "Muon" had no equivalent of.
        A = X @ X.mT                                                           # Added the Gram matrix used by both cubic and quintic terms.
        B = b * A + c * A @ A                                                  # Added the fused quintic term, which saves one matmul versus the naive form.
        X = a * X + B @ X                                                      # Added the Newton-Schulz update step itself.
    if transposed:                                                             # Added the inverse transpose so the result matches the input orientation.
        X = X.mT                                                               # Added the transpose back to the original shape.
    return X.to(G.dtype)                                                       # Added a cast back to the parameter dtype so the caller sees no dtype surprise.


@torch.no_grad()
def muon_update(
    grad: torch.Tensor,
    momentum_buffer: torch.Tensor,
    beta: float = 0.95,
    ns_steps: int = 5,
    rms_scale: float = 0.2,
    nesterov: bool = False,
) -> torch.Tensor:
    """
    One Muon update direction, following Algorithm 1 lines 4-5 of the Kimi K2 report.

        M_t = mu * M_{t-1} + G_t
        O_t = NewtonSchulz(M_t) * sqrt(max(n, m)) * 0.2
    """
    momentum_buffer.mul_(beta).add_(grad)                                      # Added heavy-ball momentum M = mu*M + G exactly as Algorithm 1 line 4 specifies.
    update = grad.add(momentum_buffer, alpha=beta) if nesterov else momentum_buffer  # Added an optional Nesterov look-ahead, which Moonlight and Keller Jordan's reference both use.
    if update.ndim > 2:                                                        # Added a flatten path so conv-style tensors can still be orthogonalized as matrices.
        update = update.view(update.size(0), -1)                               # Added the collapse of trailing dimensions into a single matrix dimension.
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)               # Added the orthogonalization that makes this Muon rather than Adam.
    n, m = update.size(-2), update.size(-1)                                    # Added shape capture for the RMS matching factor.
    update = update * (math.sqrt(max(n, m)) * rms_scale)                       # Added Kimi's consistent RMS matching (sqrt(max(n,m)) * 0.2) so Muon's update RMS matches AdamW's.
    return update                                                              # Added the return of the scaled orthogonal direction.


class MuonClip(optimizer.Optimizer):
    """
    MuonClip: Muon for 2D hidden weights, AdamW for everything else, plus QK-Clip.

    Muon is only correct for 2D hidden weight matrices. Embeddings, the output
    head, and all 1D parameters (biases, LayerNorm gains) must use AdamW - the
    orthogonalization has no meaning for them.

    Args:
        param_groups: groups produced by `build_param_groups`, each carrying `use_muon`.
        attention_modules: modules exposing `qk_clip_(tau)`; QK-Clip runs on these after each step.
        lr: Muon learning rate, in units of spectral norm per update.
        adamw_lr: separate learning rate for the AdamW group.
        momentum: Muon momentum mu.
        weight_decay: decoupled weight decay lambda.
        qk_clip_tau: attention-logit ceiling tau; set to None to disable QK-Clip.
    """

    def __init__(
        self,
        param_groups,
        attention_modules: Iterable[nn.Module] = (),                           # Added the attention module registry that QK-Clip needs to reach q/k weights.
        lr: float = 2e-4,                                                      # Modified the default to Kimi K2's pre-training LR of 2e-4 (constant for the first 10T tokens).
        adamw_lr: Optional[float] = None,                                      # Added a separate AdamW LR because embeddings want a different scale than orthogonalized matrices.
        momentum: float = 0.95,                                                # Modified the default to 0.95, the value used by both Kimi K2 and Keller Jordan's reference.
        weight_decay: float = 0.1,                                             # Modified the default to 0.1, the weight decay Kimi K2 held constant across all 15.5T tokens.
        betas: tuple = (0.9, 0.95),                                            # Added AdamW betas; (0.9, 0.95) is the standard LLM setting used by Moonlight alongside Muon.
        eps: float = 1e-8,
        ns_steps: int = 5,                                                     # Added the Newton-Schulz step count; 5 is what the tuned coefficients were designed for.
        rms_scale: float = 0.2,                                                # Added Kimi's RMS matching constant of 0.2 from Algorithm 1 line 5.
        nesterov: bool = False,                                                # Added a Nesterov toggle, off by default so we match Algorithm 1 literally.
        qk_clip_tau: Optional[float] = 100.0,                                  # Added tau = 100, the threshold Kimi K2 used for the whole run without ever retuning it.
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            adamw_lr=adamw_lr if adamw_lr is not None else lr,                 # Added a fallback so AdamW reuses the Muon LR when none is given.
            momentum=momentum,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            ns_steps=ns_steps,                                                 # Added Newton-Schulz steps to the defaults so they are visible per group.
            rms_scale=rms_scale,                                               # Added the RMS scale to the defaults so it can be overridden per group.
            nesterov=nesterov,                                                 # Added the Nesterov flag to the defaults.
            use_muon=True,                                                     # Added the per-group switch that decides Muon versus AdamW.
        )
        super(MuonClip, self).__init__(param_groups, defaults)

        self.attention_modules = list(attention_modules)                       # Added storage of the attention modules so step() can call QK-Clip on them.
        self.qk_clip_tau = qk_clip_tau                                         # Added storage of tau so it can be inspected or annealed by the training loop.
        self.last_max_logit = 0.0                                              # Added a record of the largest attention logit seen, since QK-Clip resets the trackers as it runs.
        self.last_clipped_heads = 0                                            # Added a record of how many heads were clipped on the most recent step.

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step, then applies QK-Clip."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            use_muon = group.get("use_muon", True)                             # Added the branch that routes each group to Muon or AdamW.
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # Newton-Schulz is only defined for matrices. Fall back to AdamW for
                # any 1D parameter even if its group is marked for Muon, so that
                # passing a bare model.parameters() can never crash the optimizer.
                use_muon_here = use_muon and p.ndim >= 2                       # Added a per-parameter guard so biases and LayerNorm gains can never reach the orthogonalization path.

                if use_muon_here:                                              # Modified the branch condition to use the per-parameter guard rather than the group flag alone.
                    if len(state) == 0:                                        # Added lazy state init for the single momentum buffer Muon needs.
                        state["momentum_buffer"] = torch.zeros_like(p)         # Added M_0 = 0 as specified in Algorithm 1.
                    update = muon_update(                                      # Added the call producing the orthogonalized, RMS-matched direction.
                        grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        ns_steps=group["ns_steps"],
                        rms_scale=group["rms_scale"],
                        nesterov=group["nesterov"],
                    )
                    if weight_decay != 0:                                      # Added decoupled weight decay, applied to the weight and not folded into the gradient.
                        p.mul_(1 - group["lr"] * weight_decay)                 # Added the AdamW-style shrink, replacing the old L2-into-gradient coupling that collapsed the weights.
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])        # Added the parameter update W = W - lr * O.
                else:                                                          # Added the AdamW branch for embeddings, the output head, and all 1D parameters.
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    beta1, beta2 = group["betas"]                              # Added separate beta1/beta2, fixing the old code which reused one momentum value for both moments.
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    bias_correction1 = 1 - beta1 ** state["step"]              # Added the correct beta1-based first-moment correction.
                    bias_correction2 = 1 - beta2 ** state["step"]              # Added the correct beta2-based second-moment correction, which the old code got wrong by reusing beta1.
                    denom = (exp_avg_sq / bias_correction2).sqrt_().add_(group["eps"])
                    if weight_decay != 0:                                      # Added decoupled weight decay for the AdamW group as well.
                        p.mul_(1 - group["adamw_lr"] * weight_decay)           # Added the decoupled shrink instead of adding wd*p into the gradient.
                    p.addcdiv_(exp_avg, denom, value=-group["adamw_lr"] / bias_correction1)

        if self.qk_clip_tau is not None:                                       # Added the QK-Clip phase, Algorithm 1 lines 8-17, which had no equivalent before.
            self.apply_qk_clip()                                               # Added the call that bounds attention logits after the weight update.

        return loss

    @torch.no_grad()
    def apply_qk_clip(self) -> int:
        """
        Algorithm 1 lines 9-17: for every attention head whose max logit exceeded
        tau during the forward pass, shrink that head's query and key weights.

        Returns the number of heads that were clipped this step.
        """
        clipped = 0                                                            # Added a counter so training can log how often clipping actually fires.
        max_logit = 0.0                                                        # Added a running max so the pre-clip logit value survives the reset inside qk_clip_.
        for module in self.attention_modules:                                  # Added the sweep over every registered attention layer.
            max_logit = max(max_logit, float(module.qk_max_logit.max()))       # Added the read of S_max before qk_clip_ zeroes the tracker.
            clipped += module.qk_clip_(self.qk_clip_tau)                       # Added the delegation to each attention module, which knows its own per-head weight layout.
        self.last_max_logit = max_logit                                        # Added the stored max so the training loop can log it after step() returns.
        self.last_clipped_heads = clipped                                      # Added the stored clip count for the same reason.
        return clipped                                                         # Added the return so the caller can log clipping activity over training.


# Kept the old name as an alias so existing scripts and checkpoints keep importing cleanly.
Muon = MuonClip                                                                # Added a backwards-compatible alias, since the class was renamed to reflect what it now does.


def build_param_groups(model: nn.Module, weight_decay: float = 0.1) -> List[Dict[str, Any]]:
    """
    Split parameters into a Muon group and an AdamW group.

    Muon is only defined for 2D hidden weight matrices. Embeddings and the output
    head are excluded because their rows are per-token and orthogonalizing across
    the vocabulary is meaningless; 1D parameters are excluded because they are not
    matrices at all.
    """
    muon_params, adamw_decay, adamw_no_decay = [], [], []                      # Added three buckets so each parameter class gets the treatment it needs.
    seen = set()                                                               # Added a dedup set because wte and lm_head are tied and share one tensor.

    for module_name, module in model.named_modules():                          # Added a module-level walk so embeddings can be identified by type.
        for param_name, p in module.named_parameters(recurse=False):           # Added recurse=False so each parameter is visited exactly once, at its owning module.
            if not p.requires_grad or id(p) in seen:                           # Added the tied-parameter guard that stops wte being optimized twice.
                continue
            seen.add(id(p))                                                    # Added the id record for the dedup check above.
            full_name = f"{module_name}.{param_name}" if module_name else param_name

            is_embedding = isinstance(module, nn.Embedding)                    # Added the embedding test, since embeddings must go to AdamW.
            is_output_head = full_name == "lm_head.weight"                     # Added the output-head test, since the unembedding must also go to AdamW.
            if p.ndim >= 2 and not is_embedding and not is_output_head:        # Added the condition selecting genuine hidden weight matrices for Muon.
                muon_params.append(p)                                          # Added the 2D hidden weight to the Muon bucket.
            elif p.ndim >= 2:                                                  # Added the branch catching embeddings and the head, which still want weight decay.
                adamw_decay.append(p)                                          # Added the embedding/head parameter to the decaying AdamW bucket.
            else:                                                              # Added the branch for biases and LayerNorm gains.
                adamw_no_decay.append(p)                                       # Added 1D parameters to the non-decaying bucket, since decaying gains and biases hurts.

    groups = [
        dict(params=muon_params, use_muon=True, weight_decay=weight_decay),    # Added the Muon group carrying the full weight decay.
        dict(params=adamw_decay, use_muon=False, weight_decay=weight_decay),   # Added the AdamW group for embeddings and the output head.
        dict(params=adamw_no_decay, use_muon=False, weight_decay=0.0),         # Added the AdamW group that must not be decayed.
    ]
    return [g for g in groups if len(g["params"]) > 0]                         # Added a filter so empty groups never reach the optimizer.


def collect_attention_modules(model: nn.Module) -> List[nn.Module]:
    """Find every attention module that implements QK-Clip."""
    return [m for m in model.modules() if hasattr(m, "qk_clip_")]              # Added discovery by capability, so both LatentAttention and MultiHeadAttention are picked up.


def create_muon_optimizer(model, config):
    """Create the MuonClip optimizer with the given configuration."""
    weight_decay = config.get('weight_decay', 0.1)                             # Modified to read weight decay first, because it is now needed to build the groups.
    param_groups = build_param_groups(model, weight_decay=weight_decay)        # Modified to pass structured param groups instead of a flat model.parameters(), which sent embeddings and biases through Muon.
    return MuonClip(                                                           # Modified to construct MuonClip rather than the old Adam-in-disguise class.
        param_groups,
        attention_modules=collect_attention_modules(model),                    # Added the attention module list so QK-Clip can run after each step.
        lr=config['learning_rate'],
        adamw_lr=config.get('adamw_learning_rate', config['learning_rate']),   # Added an optional separate AdamW LR, defaulting to the Muon LR.
        momentum=config.get('momentum', 0.95),                                 # Modified the fallback to 0.95, matching Kimi K2 and the Muon reference.
        weight_decay=weight_decay,
        betas=config.get('betas', (0.9, config.get('beta2', 0.95))),           # Added AdamW betas sourced from config, with the standard (0.9, 0.95) default.
        eps=config.get('eps', 1e-8),
        ns_steps=config.get('ns_steps', 5),                                    # Added the configurable Newton-Schulz step count, default 5 per the reference.
        rms_scale=config.get('rms_scale', 0.2),                                # Added the configurable RMS match factor, default 0.2 per Algorithm 1.
        nesterov=config.get('nesterov', False),                                # Added the configurable Nesterov toggle.
        qk_clip_tau=config.get('qk_clip_tau', 100.0),                          # Added tau from config, default 100.0 as used throughout Kimi K2 pre-training.
    )
