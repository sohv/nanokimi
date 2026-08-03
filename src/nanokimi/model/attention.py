"""
Multi-head Latent Attention (MLA) for nanoKimi

Implements the attention mechanism of DeepSeek-V3 (arXiv:2412.19437), which Kimi K2
adopts unchanged. The point of MLA is the KV cache: instead of caching K and V per
head, it caches one low-rank latent c_KV shared across all heads, plus one rotary
key k^R also shared across all heads. Cache cost is therefore
`kv_lora_rank + qk_rope_head_dim` per token regardless of how many heads there are.

    cache/token, n_embd=768, n_head=12:  plain MHA 1536  ->  MLA 288  (5.3x smaller)

The previous implementation in this file projected K and V to `n_head * latent_dim`,
i.e. still per head, so it delivered no cache saving at all (and with latent_dim=256
was 4x worse than plain MHA). That is why it was replaced.

Reference dimensions (DeepSeek-V3 and Kimi K2 both use these):
    hidden 7168, kv_lora_rank 512, q_lora_rank 1536,
    qk_nope_head_dim 128, qk_rope_head_dim 64, v_head_dim 128, attention_bias False
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    """
    RMS normalization, applied to the two LoRA bottlenecks.

    DeepSeek-V3 normalizes c_Q and c_KV before up-projecting. Without it the
    low-rank bottleneck is free to grow without bound, which is exactly the kind of
    drift QK-Clip then has to clean up.
    """

    def __init__(self, dim, eps=1e-6):                                             # Added RMSNorm, which the DeepSeek-V3 MLA block requires on both LoRA paths.
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))                                # Added the learnable gain, initialised to 1.0 as in the reference.
        self.eps = eps                                                             # Added the numerical-stability epsilon.

    def forward(self, x):
        dtype = x.dtype                                                            # Added dtype capture so the norm can run in fp32 and cast back.
        x = x.float()                                                              # Added an fp32 upcast, since computing the norm in bf16 loses precision.
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)            # Added the RMS normalization itself.
        return (x.to(dtype) * self.weight)                                         # Added the cast back and the learnable gain.


def build_rope_cache(seq_len, head_dim, theta=50000.0, device=None, dtype=torch.float32):
    """
    Precompute the rotary position embedding table.

    theta defaults to 50000.0 because that is what Kimi K2 uses (DeepSeek-V3 uses
    10000.0); the larger base stretches the frequency spectrum for longer contexts.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))  # Added the standard RoPE inverse-frequency schedule.
    t = torch.arange(seq_len, device=device).float()                               # Added the position index vector.
    freqs = torch.outer(t, inv_freq)                                               # Added the outer product giving one angle per (position, frequency) pair.
    emb = torch.cat((freqs, freqs), dim=-1)                                        # Added the half-and-half duplication used by the rotate_half convention.
    return emb.cos().to(dtype), emb.sin().to(dtype)                                # Added the cos/sin tables that get cached as buffers.


def rotate_half(x):
    """Rotate the two halves of the last dimension, the standard RoPE helper."""
    x1, x2 = x.chunk(2, dim=-1)                                                    # Added the split into the two halves RoPE rotates against each other.
    return torch.cat((-x2, x1), dim=-1)                                            # Added the rotation itself.


def apply_rope(x, cos, sin):
    """Apply rotary embeddings to a (B, n_head, T, d) tensor."""
    return x * cos + rotate_half(x) * sin                                          # Added the RoPE application, broadcasting cos/sin over batch and heads.


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-head Latent Attention, as in DeepSeek-V3 / Kimi K2.

    Query path:   h -> W^DQ -> RMSNorm -> W^UQ giving q^C, and W^QR giving q^R (per head)
    Key/value:    h -> W^DKV -> RMSNorm -> W^UK giving k^C and W^UV giving v (per head)
                  h -> W^KR -> k^R, a single rotary key SHARED across all heads
    Score:        q.k over the concatenation [nope ; rope], scaled by 1/sqrt(nope + rope)

    Only c_KV (kv_lora_rank) and k^R (qk_rope_head_dim) need caching at inference.

    Layer names follow HuggingFace's DeepseekV3Attention (q_a_proj, q_b_proj,
    kv_a_proj_with_mqa, kv_b_proj) so weights are interchangeable with that family.
    """

    def __init__(self, n_embd, n_head, kv_lora_rank=256, q_lora_rank=768,
                 qk_nope_head_dim=64, qk_rope_head_dim=32, v_head_dim=64,
                 max_seq_len=1024, rope_theta=50000.0, dropout=0.0, bias=False):
        super().__init__()

        self.n_embd = n_embd
        self.n_head = n_head
        self.kv_lora_rank = kv_lora_rank                                           # Added d_c, the shared KV latent width. 256 here = 4x qk_nope_head_dim, the same ratio DeepSeek-V3 uses (512 = 4x128).
        self.q_lora_rank = q_lora_rank                                             # Added d_c', the query LoRA width. 768 here = 3x kv_lora_rank, matching DeepSeek-V3's 1536 = 3x512.
        self.qk_nope_head_dim = qk_nope_head_dim                                   # Added the non-rotary per-head query/key width.
        self.qk_rope_head_dim = qk_rope_head_dim                                   # Added the rotary per-head width. 32 here = half of nope, matching DeepSeek-V3's 64 = half of 128.
        self.v_head_dim = v_head_dim                                               # Added the per-head value width, equal to nope as in DeepSeek-V3.
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim                     # Added the total query/key head width used for the score scale.

        # Query path: compress to q_lora_rank, normalize, then expand to per-head
        # nope+rope. The LoRA bottleneck is what keeps the query projection cheap.
        self.q_a_proj = nn.Linear(n_embd, q_lora_rank, bias=bias)                  # Added W^DQ, the query down-projection.
        self.q_a_layernorm = RMSNorm(q_lora_rank)                                  # Added the RMSNorm on c_Q that DeepSeek-V3 applies before up-projection.
        self.q_b_proj = nn.Linear(q_lora_rank, n_head * self.qk_head_dim, bias=False)  # Added W^UQ and W^QR fused into one up-projection, as the reference does.

        # KV path: one projection produces both the shared latent c_KV and the
        # shared rotary key k^R. Only these two need to be cached at inference.
        self.kv_a_proj_with_mqa = nn.Linear(n_embd, kv_lora_rank + qk_rope_head_dim, bias=bias)  # Added W^DKV and W^KR fused; the rope slice is shared across heads, hence "mqa".
        self.kv_a_layernorm = RMSNorm(kv_lora_rank)                                # Added the RMSNorm on c_KV, matching the reference.
        self.kv_b_proj = nn.Linear(kv_lora_rank, n_head * (qk_nope_head_dim + v_head_dim), bias=False)  # Added W^UK and W^UV fused, expanding the shared latent back out per head.

        # Output projection
        self.o_proj = nn.Linear(n_head * v_head_dim, n_embd, bias=bias)            # Added W^O, sized from v_head_dim rather than n_embd since v may differ.

        self.dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.qk_head_dim)                             # Added the score scale over the full nope+rope width, which is what the score is computed across.

        # RoPE tables are built lazily in forward rather than registered as
        # buffers. HuggingFace's from_pretrained materializes a model without
        # re-running __init__, so a non-persistent buffer comes back as
        # uninitialised memory and silently corrupts every position. Rebuilding on
        # demand is immune to how the module was constructed.
        self.max_seq_len = max_seq_len                                             # Added the default table length, kept so the cache can be sized up front.
        self.rope_theta = rope_theta                                               # Added the RoPE base, stored so the tables can be rebuilt at any time.
        self._rope_cache = None                                                    # Added a plain attribute (not a buffer) holding the cached cos/sin tables.

        # Per-head running max pre-softmax logit, read by QK-Clip after each step.
        # This one is safe as a buffer because it is reset to zero every step
        # anyway, so an uninitialised load cannot carry stale state for long.
        self.register_buffer("qk_max_logit", torch.zeros(n_head), persistent=False)  # Added the per-head logit tracker Algorithm 1 line 10 expects the forward pass to produce.

    def _rope_tables(self, seq_len, device, dtype):
        """Return cos/sin tables for `seq_len` positions, rebuilding only when needed."""
        cached = self._rope_cache                                                  # Added a read of the current cache entry.
        if (cached is None or cached[0].size(0) < seq_len                          # Added a length check so a longer sequence triggers a rebuild.
                or cached[0].device != device or cached[0].dtype != dtype):        # Added device/dtype checks so .to() and autocast are handled correctly.
            length = max(seq_len, self.max_seq_len)                                # Added sizing to the configured max so the table is normally built once.
            cos, sin = build_rope_cache(length, self.qk_rope_head_dim,             # Added the rebuild of the rotary tables.
                                        theta=self.rope_theta, device=device, dtype=dtype)
            self._rope_cache = (cos, sin)                                          # Added the cache store so subsequent steps reuse the tables.
        return self._rope_cache                                                    # Added the return of the cached tables.

    def forward(self, x, mask=None):
        B, T, C = x.size()

        # ---- query path -----------------------------------------------------
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))                    # Added the compress-normalize-expand query path replacing the old single q_proj.
        q = q.view(B, T, self.n_head, self.qk_head_dim).transpose(1, 2)            # Added the reshape to (B, n_head, T, nope+rope).
        q_nope, q_rope = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)  # Added the split into non-rotary and rotary query components.

        # ---- key/value path -------------------------------------------------
        compressed = self.kv_a_proj_with_mqa(x)                                    # Added the single projection producing both c_KV and the shared rotary key.
        c_kv, k_rope = compressed.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)  # Added the split separating the cached latent from the shared rotary key.
        k_rope = k_rope.view(B, T, 1, self.qk_rope_head_dim).transpose(1, 2)       # Added the reshape to a single head dimension, since k^R is shared across all heads.

        kv = self.kv_b_proj(self.kv_a_layernorm(c_kv))                             # Added the up-projection expanding the shared latent back out to per-head K and V.
        kv = kv.view(B, T, self.n_head, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)  # Added the reshape to (B, n_head, T, nope+v).
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)     # Added the split into the non-rotary key and the value.

        # ---- rotary embeddings ----------------------------------------------
        cos_t, sin_t = self._rope_tables(T, x.device, q.dtype)                     # Modified to fetch lazily built tables instead of reading a buffer that from_pretrained can leave uninitialised.
        cos = cos_t[:T].view(1, 1, T, self.qk_rope_head_dim)                       # Added the position slice of the cos table, broadcast over batch and heads.
        sin = sin_t[:T].view(1, 1, T, self.qk_rope_head_dim)                       # Added the matching sin slice.
        q_rope = apply_rope(q_rope, cos, sin)                                      # Added RoPE on the per-head rotary query component.
        k_rope = apply_rope(k_rope, cos, sin)                                      # Added RoPE on the shared rotary key.
        k_rope = k_rope.expand(-1, self.n_head, -1, -1)                            # Added the broadcast of the single shared rotary key across all heads.

        q = torch.cat([q_nope, q_rope], dim=-1)                                    # Added the concatenation forming the full query.
        k = torch.cat([k_nope, k_rope], dim=-1)                                    # Added the concatenation forming the full key.

        # ---- attention ------------------------------------------------------
        att = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        neg_inf = torch.finfo(att.dtype).min                                       # Added a finite floor instead of -inf, which produces NaNs under bf16 autocast when a row is fully masked.
        if mask is not None:
            att = att.masked_fill(mask == 0, neg_inf)
        else:
            causal_mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril().view(1, 1, T, T)  # Added a bool causal mask, cheaper than building it from float ones.
            att = att.masked_fill(~causal_mask, neg_inf)

        # Record the max logit per head for QK-Clip, after masking so future
        # positions cannot inflate it, and detached so it carries no gradient.
        if self.training:
            head_max = att.detach().amax(dim=(0, 2, 3)).float()                    # Added the per-head max matching the S_max definition in Kimi K2 Section 2.1.
            self.qk_max_logit.copy_(torch.maximum(self.qk_max_logit, head_max))    # Added a running max so gradient-accumulation micro-batches all feed one S_max per step.

        att = self.dropout(F.softmax(att, dim=-1))

        y = torch.matmul(att, v)                                                   # Added the value aggregation over v_head_dim rather than the old latent_dim.
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.v_head_dim)  # Added the merge back to (B, T, n_head*v_head_dim).
        return self.resid_dropout(self.o_proj(y))

    def kv_cache_per_token(self):
        """Values cached per token at inference: c_KV plus the shared rotary key."""
        return self.kv_lora_rank + self.qk_rope_head_dim                           # Added a helper making the actual cache cost inspectable, since that is the whole point of MLA.

    @torch.no_grad()
    def qk_clip_(self, tau=100.0):
        """
        QK-Clip for MLA (Kimi K2 Algorithm 1, lines 9-17).

        The report applies clipping only to unshared, head-specific components:
            q^C and k^C  ->  scaled by sqrt(gamma)
            q^R          ->  scaled by gamma
            k^R          ->  left untouched, because it is shared across heads and
                             rescaling it would affect every other head too.

        Returns the number of heads clipped this step.
        """
        s_max = self.qk_max_logit                                                  # Added a handle on the S_max values the forward pass recorded.
        over = s_max > tau                                                         # Added the per-head test from Algorithm 1 line 11.
        n_clipped = int(over.sum().item())                                         # Added a count so the training loop can log clipping frequency.

        if n_clipped > 0:
            gamma = torch.where(over, tau / s_max.clamp(min=1e-6), torch.ones_like(s_max))  # Added gamma = tau/S_max for exceeding heads and 1.0 elsewhere.
            g = gamma.to(self.q_b_proj.weight.dtype)                               # Added a dtype-matched gamma for the in-place scaling below.
            sqrt_g = g.sqrt()                                                      # Added sqrt(gamma), the factor for the non-rotary components (alpha = 0.5).

            # q_b_proj rows are (n_head, nope+rope, q_lora_rank): nope slice takes
            # sqrt(gamma), rope slice takes the full gamma.
            qw = self.q_b_proj.weight.view(self.n_head, self.qk_head_dim, -1)      # Added a per-head view of the fused query up-projection.
            qw[:, :self.qk_nope_head_dim, :].mul_(sqrt_g.view(-1, 1, 1))           # Added sqrt(gamma) scaling of the q^C block.
            qw[:, self.qk_nope_head_dim:, :].mul_(g.view(-1, 1, 1))                # Added full-gamma scaling of the q^R block, per the report's per-component rule.

            # kv_b_proj rows are (n_head, nope+v, kv_lora_rank): only the k^C slice
            # is scaled. The value slice must not be touched.
            kw = self.kv_b_proj.weight.view(self.n_head, self.qk_nope_head_dim + self.v_head_dim, -1)  # Added a per-head view of the fused KV up-projection.
            kw[:, :self.qk_nope_head_dim, :].mul_(sqrt_g.view(-1, 1, 1))           # Added sqrt(gamma) scaling of the k^C block only, leaving V untouched.

            # kv_a_proj_with_mqa holds the shared k^R and is deliberately left alone.

        self.qk_max_logit.zero_()                                                  # Added a reset so the next step measures a fresh S_max.
        return n_clipped                                                           # Added the return used for logging how many heads were clipped.


# The old name pointed at a per-head low-rank K/V module that gave no KV-cache
# saving. It now resolves to real MLA so existing imports keep working.
LatentAttention = MultiHeadLatentAttention                                         # Added an alias so `from attention import LatentAttention` still resolves after the rewrite.


class MultiHeadAttention(nn.Module):
    """
    Standard multi-head attention, kept as the dense baseline to compare MLA against.
    Uses learned absolute position embeddings from the model, not RoPE.
    """

    def __init__(self, n_embd, n_head, dropout=0.0, bias=True):
        super().__init__()
        assert n_embd % n_head == 0

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        # QKV projection
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd, bias=bias)

        # Output projection
        self.o_proj = nn.Linear(n_embd, n_embd, bias=bias)

        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Scale factor
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Same per-head max-logit tracker as MLA, so QK-Clip works either way.
        self.register_buffer("qk_max_logit", torch.zeros(n_head), persistent=False)  # Added the per-head logit tracker so this variant also supports QK-Clip.

    def forward(self, x, mask=None):
        B, T, C = x.size()

        # Compute QKV
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Compute attention
        att = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask
        neg_inf = torch.finfo(att.dtype).min                                        # Added a finite floor to replace -inf, which produces NaNs under bf16 autocast when a row is fully masked.
        if mask is not None:
            att = att.masked_fill(mask == 0, neg_inf)                               # Modified to fill with the dtype minimum rather than -inf.
        else:
            causal_mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril().view(1, 1, T, T)  # Modified to build a bool mask directly instead of comparing float ones.
            att = att.masked_fill(~causal_mask, neg_inf)                            # Modified to fill with the dtype minimum rather than -inf.

        # Record the max logit per head for QK-Clip.
        if self.training:                                                           # Added a training-only guard so inference pays nothing for this tracking.
            head_max = att.detach().amax(dim=(0, 2, 3)).float()                     # Added the per-head max over batch and both token axes, matching the S_max definition.
            self.qk_max_logit.copy_(torch.maximum(self.qk_max_logit, head_max))     # Added a running max so gradient-accumulation micro-batches all feed one S_max per step.

        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        # Apply attention to values
        y = torch.matmul(att, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.o_proj(y)
        y = self.resid_dropout(y)

        return y

    def kv_cache_per_token(self):
        """Values cached per token: full K and V for every head."""
        return self.n_head * self.head_dim * 2                                      # Added the same helper as MLA so the two can be compared directly.

    @torch.no_grad()
    def qk_clip_(self, tau=100.0):
        """
        QK-Clip for the fused-QKV baseline (Kimi K2 Algorithm 1, lines 9-17).

        The report notes clipping is "straightforward for regular multi-head
        attention": there is no shared rotary key, so both sides take sqrt(gamma).

        Returns the number of heads clipped this step.
        """
        s_max = self.qk_max_logit                                                   # Added a handle on the S_max values the forward pass recorded.
        over = s_max > tau                                                          # Added the per-head test from Algorithm 1 line 11.
        n_clipped = int(over.sum().item())                                          # Added a count so the training loop can log clipping frequency.

        if n_clipped > 0:                                                           # Added the guard so untouched heads cost nothing.
            gamma = torch.where(over, tau / s_max.clamp(min=1e-6), torch.ones_like(s_max))  # Added gamma = tau/S_max for exceeding heads and 1.0 elsewhere.
            sqrt_gamma = gamma.sqrt().to(self.qkv_proj.weight.dtype)                # Added the sqrt(gamma) factor corresponding to alpha = 0.5.

            # qkv_proj stacks [Q; K; V] along the output dim; V must be left alone.
            w = self.qkv_proj.weight.view(3, self.n_head, self.head_dim, -1)        # Added a view that separates the fused projection into its Q, K and V blocks.
            w[0].mul_(sqrt_gamma.view(-1, 1, 1))                                    # Added per-head scaling of the query block only.
            w[1].mul_(sqrt_gamma.view(-1, 1, 1))                                    # Added per-head scaling of the key block only, leaving the value block untouched.
            if self.qkv_proj.bias is not None:                                      # Added a bias check, since the reference formulation assumes no bias.
                b = self.qkv_proj.bias.view(3, self.n_head, self.head_dim)          # Added the matching view over the fused bias vector.
                b[0].mul_(sqrt_gamma.view(-1, 1))                                   # Added query bias scaling so the whole affine map shrinks.
                b[1].mul_(sqrt_gamma.view(-1, 1))                                   # Added key bias scaling for the same reason.

        self.qk_max_logit.zero_()                                                   # Added a reset so the next step measures a fresh S_max.
        return n_clipped                                                            # Added the return used for logging how many heads were clipped.
