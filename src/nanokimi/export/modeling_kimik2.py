"""
HuggingFace `transformers` implementation of the nanoKimi architecture.

Attention is Multi-head Latent Attention (MLA) as in DeepSeek-V3 (arXiv:2412.19437),
which Kimi K2 adopts unchanged: one low-rank latent c_KV plus one rotary key k^R are
shared across all heads, so the KV cache costs `kv_lora_rank + qk_rope_head_dim` per
token regardless of head count. Position comes from RoPE, so there are no learned
position embeddings on this path.

This file is self-contained on purpose: it is uploaded to the Hub and loaded via
`trust_remote_code=True`, so it may not import from the nanoKimi `src/` package.
Parameter names match `src/model.py` and follow HuggingFace's DeepseekV3Attention
naming, so checkpoints saved by `scripts/train.py` load without any key renaming.

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("<repo>", trust_remote_code=True)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import PreTrainedModel, PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutput

# transformers 5.x expects `_tied_weights_keys` as a {target: source} mapping;
# 4.x expects a plain list of target names.
_TRANSFORMERS_V5 = int(transformers.__version__.split(".")[0]) >= 5


class KimiK2Config(PretrainedConfig):
    model_type = "kimi-k2"

    def __init__(
        self,
        vocab_size=50257,
        block_size=256,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        bias=True,
        # --- MoE ---
        use_moe=True,
        num_experts=8,
        expert_capacity=32,
        top_k_experts=2,
        apply_expert_capacity=False,
        load_balance_loss_coef=0.01,
        # --- attention ---
        use_latent_attention=True,
        kv_lora_rank=256,
        q_lora_rank=768,
        qk_nope_head_dim=64,
        qk_rope_head_dim=32,
        v_head_dim=64,
        rope_theta=50000.0,
        attention_bias=False,
        tie_word_embeddings=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias

        self.use_moe = use_moe
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.top_k_experts = top_k_experts
        # The training-time capacity constraint drops tokens; that is a training-side
        # device, not inference behaviour, so it is off by default.
        self.apply_expert_capacity = apply_expert_capacity
        self.load_balance_loss_coef = load_balance_loss_coef

        self.use_latent_attention = use_latent_attention
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    # `block_size` is nanoGPT's name for the context length; also expose the name
    # the rest of `transformers` looks for.
    @property
    def max_position_embeddings(self):
        return self.block_size

    @property
    def kv_cache_per_token(self):
        """Values cached per token: the shared latent plus the shared rotary key."""
        if self.use_latent_attention:
            return self.kv_lora_rank + self.qk_rope_head_dim
        return self.n_head * (self.n_embd // self.n_head) * 2


class LayerNorm(nn.Module):
    """LayerNorm with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class RMSNorm(nn.Module):
    """RMS normalization, applied to MLA's two LoRA bottlenecks as in DeepSeek-V3."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight


def build_rope_cache(seq_len, head_dim, theta=50000.0, device=None, dtype=torch.float32):
    """Precompute the rotary tables. theta=50000.0 is Kimi K2's value."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-head Latent Attention (DeepSeek-V3 / Kimi K2).

    Query:  h -> W^DQ -> RMSNorm -> W^UQ (q^C) and W^QR (q^R), both per head
    Key/V:  h -> W^DKV -> RMSNorm -> W^UK (k^C) and W^UV (v), per head
            h -> W^KR -> k^R, a single rotary key SHARED across all heads
    Score:  over [nope ; rope], scaled by 1/sqrt(nope + rope)
    """

    def __init__(self, n_embd, n_head, kv_lora_rank=256, q_lora_rank=768,
                 qk_nope_head_dim=64, qk_rope_head_dim=32, v_head_dim=64,
                 max_seq_len=1024, rope_theta=50000.0, dropout=0.0, bias=False):
        super().__init__()

        self.n_embd = n_embd
        self.n_head = n_head
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

        self.q_a_proj = nn.Linear(n_embd, q_lora_rank, bias=bias)
        self.q_a_layernorm = RMSNorm(q_lora_rank)
        self.q_b_proj = nn.Linear(q_lora_rank, n_head * self.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(n_embd, kv_lora_rank + qk_rope_head_dim, bias=bias)
        self.kv_a_layernorm = RMSNorm(kv_lora_rank)
        self.kv_b_proj = nn.Linear(kv_lora_rank, n_head * (qk_nope_head_dim + v_head_dim), bias=False)

        self.o_proj = nn.Linear(n_head * v_head_dim, n_embd, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.qk_head_dim)

        # RoPE tables are built lazily, NOT registered as buffers. from_pretrained
        # materializes a model without re-running __init__, so a non-persistent
        # buffer comes back as uninitialised memory and silently corrupts every
        # position. Rebuilding on demand is immune to how the module was built.
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self._rope_cache = None

        self.register_buffer("qk_max_logit", torch.zeros(n_head), persistent=False)

    def _rope_tables(self, seq_len, device, dtype):
        """Return cos/sin tables for `seq_len` positions, rebuilding only when needed."""
        cached = self._rope_cache
        if (cached is None or cached[0].size(0) < seq_len
                or cached[0].device != device or cached[0].dtype != dtype):
            length = max(seq_len, self.max_seq_len)
            cos, sin = build_rope_cache(length, self.qk_rope_head_dim,
                                        theta=self.rope_theta, device=device, dtype=dtype)
            self._rope_cache = (cos, sin)
        return self._rope_cache

    def forward(self, x, attention_mask=None):
        B, T, C = x.size()

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.view(B, T, self.n_head, self.qk_head_dim).transpose(1, 2)
        q_nope, q_rope = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed = self.kv_a_proj_with_mqa(x)
        c_kv, k_rope = compressed.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_rope = k_rope.view(B, T, 1, self.qk_rope_head_dim).transpose(1, 2)

        kv = self.kv_b_proj(self.kv_a_layernorm(c_kv))
        kv = kv.view(B, T, self.n_head, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        cos_t, sin_t = self._rope_tables(T, x.device, q.dtype)
        cos = cos_t[:T].view(1, 1, T, self.qk_rope_head_dim)
        sin = sin_t[:T].view(1, 1, T, self.qk_rope_head_dim)
        q_rope = apply_rope(q_rope, cos, sin)
        k_rope = apply_rope(k_rope, cos, sin).expand(-1, self.n_head, -1, -1)

        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        att = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        att = att.masked_fill(~causal.view(1, 1, T, T), torch.finfo(att.dtype).min)
        if attention_mask is not None:
            att = att + attention_mask

        if self.training:
            head_max = att.detach().amax(dim=(0, 2, 3)).float()
            self.qk_max_logit.copy_(torch.maximum(self.qk_max_logit, head_max))

        att = self.dropout(F.softmax(att, dim=-1))

        y = torch.matmul(att, v)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.v_head_dim)
        return self.resid_dropout(self.o_proj(y))

    def kv_cache_per_token(self):
        return self.kv_lora_rank + self.qk_rope_head_dim

    @torch.no_grad()
    def qk_clip_(self, tau=100.0):
        """
        QK-Clip for MLA: q^C and k^C take sqrt(gamma), q^R takes gamma, and the
        shared k^R is left untouched so clipping one head cannot affect others.
        """
        s_max = self.qk_max_logit
        over = s_max > tau
        n_clipped = int(over.sum().item())

        if n_clipped > 0:
            gamma = torch.where(over, tau / s_max.clamp(min=1e-6), torch.ones_like(s_max))
            g = gamma.to(self.q_b_proj.weight.dtype)
            sqrt_g = g.sqrt()

            qw = self.q_b_proj.weight.view(self.n_head, self.qk_head_dim, -1)
            qw[:, :self.qk_nope_head_dim, :].mul_(sqrt_g.view(-1, 1, 1))
            qw[:, self.qk_nope_head_dim:, :].mul_(g.view(-1, 1, 1))

            kw = self.kv_b_proj.weight.view(
                self.n_head, self.qk_nope_head_dim + self.v_head_dim, -1)
            kw[:, :self.qk_nope_head_dim, :].mul_(sqrt_g.view(-1, 1, 1))

        self.qk_max_logit.zero_()
        return n_clipped


class MultiHeadAttention(nn.Module):
    """Dense causal MHA baseline, used when use_latent_attention=False."""

    def __init__(self, n_embd, n_head, dropout=0.0, bias=True):
        super().__init__()
        assert n_embd % n_head == 0

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.o_proj = nn.Linear(n_embd, n_embd, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.register_buffer("qk_max_logit", torch.zeros(n_head), persistent=False)

    def forward(self, x, attention_mask=None):
        B, T, C = x.size()

        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        att = att.masked_fill(~causal.view(1, 1, T, T), torch.finfo(att.dtype).min)
        if attention_mask is not None:
            att = att + attention_mask

        if self.training:
            head_max = att.detach().amax(dim=(0, 2, 3)).float()
            self.qk_max_logit.copy_(torch.maximum(self.qk_max_logit, head_max))

        att = self.dropout(F.softmax(att, dim=-1))

        y = torch.matmul(att, v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.o_proj(y))

    def kv_cache_per_token(self):
        return self.n_head * self.head_dim * 2

    @torch.no_grad()
    def qk_clip_(self, tau=100.0):
        s_max = self.qk_max_logit
        over = s_max > tau
        n_clipped = int(over.sum().item())
        if n_clipped > 0:
            gamma = torch.where(over, tau / s_max.clamp(min=1e-6), torch.ones_like(s_max))
            sqrt_gamma = gamma.sqrt().to(self.qkv_proj.weight.dtype)
            w = self.qkv_proj.weight.view(3, self.n_head, self.head_dim, -1)
            w[0].mul_(sqrt_gamma.view(-1, 1, 1))
            w[1].mul_(sqrt_gamma.view(-1, 1, 1))
            if self.qkv_proj.bias is not None:
                b = self.qkv_proj.bias.view(3, self.n_head, self.head_dim)
                b[0].mul_(sqrt_gamma.view(-1, 1))
                b[1].mul_(sqrt_gamma.view(-1, 1))
        self.qk_max_logit.zero_()
        return n_clipped


class ExpertFFN(nn.Module):
    """GPT-style 4x FFN serving as a single MoE expert."""

    def __init__(self, n_embd, dropout=0.0, bias=True):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class StandardFFN(nn.Module):
    """Dense FFN (used when use_moe=False)."""

    def __init__(self, n_embd, dropout=0.0, bias=True):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(F.gelu(self.fc1(x)))), x.new_zeros(())


class MoELayer(nn.Module):
    """Top-k routed mixture of experts with the Switch/GShard auxiliary loss."""

    def __init__(self, n_embd, num_experts=8, expert_capacity=32, top_k=2,
                 dropout=0.0, bias=True, apply_expert_capacity=False,
                 load_balance_loss_coef=0.01):
        super().__init__()

        self.n_embd = n_embd
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.top_k = top_k
        self.apply_expert_capacity = apply_expert_capacity
        self.load_balance_loss_coef = load_balance_loss_coef

        self.gate = nn.Linear(n_embd, num_experts, bias=bias)
        self.experts = nn.ModuleList(
            [ExpertFFN(n_embd, dropout=dropout, bias=bias) for _ in range(num_experts)]
        )

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)

        gate_scores = F.softmax(self.gate(x_flat), dim=-1)
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        top_k_scores = top_k_scores / top_k_scores.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(x_flat)
        for expert_idx in range(self.num_experts):
            hit = top_k_indices == expert_idx
            token_idx = hit.any(dim=-1).nonzero(as_tuple=True)[0]
            if token_idx.numel() == 0:
                continue
            if (self.apply_expert_capacity and self.training
                    and token_idx.numel() > self.expert_capacity):
                rank = (top_k_scores[token_idx] * hit[token_idx]).sum(dim=-1)
                token_idx = token_idx[torch.topk(rank, self.expert_capacity).indices]
            weight = (top_k_scores[token_idx] * hit[token_idx]).sum(dim=-1, keepdim=True)
            # Under autocast the experts emit bf16 while `output` stays fp32, and
            # index_add_ refuses mixed dtypes. Cast so the sum stays in fp32.
            contribution = (weight * self.experts[expert_idx](x_flat[token_idx])).to(output.dtype)
            output.index_add_(0, token_idx, contribution)

        return output.view(B, T, C), self._compute_load_balance_loss(gate_scores, top_k_indices)

    def _compute_load_balance_loss(self, gate_scores, top_k_indices):
        """
        Switch Transformer / GShard auxiliary loss, L = N * sum_i f_i * P_i,
        normalised so a balanced router gives 1.0. Coupling the hard dispatch
        fraction f_i to the soft probability P_i is what actually penalises
        router collapse; a loss on P_i alone does not.
        """
        num_tokens = gate_scores.size(0)
        one_hot = F.one_hot(top_k_indices, num_classes=self.num_experts).sum(dim=1)
        f_i = one_hot.float().sum(dim=0) / (num_tokens * self.top_k)
        p_i = gate_scores.mean(dim=0)
        return self.load_balance_loss_coef * self.num_experts * torch.sum(f_i * p_i)


class KimiBlock(nn.Module):
    """Pre-norm transformer block with optional MLA and optional MoE."""

    def __init__(self, config: KimiK2Config):
        super().__init__()

        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)

        if config.use_latent_attention:
            self.attn = MultiHeadLatentAttention(
                n_embd=config.n_embd,
                n_head=config.n_head,
                kv_lora_rank=config.kv_lora_rank,
                q_lora_rank=config.q_lora_rank,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                max_seq_len=config.block_size,
                rope_theta=config.rope_theta,
                dropout=config.dropout,
                bias=config.attention_bias,
            )
        else:
            self.attn = MultiHeadAttention(
                n_embd=config.n_embd,
                n_head=config.n_head,
                dropout=config.dropout,
                bias=config.bias,
            )

        if config.use_moe:
            self.mlp = MoELayer(
                n_embd=config.n_embd,
                num_experts=config.num_experts,
                expert_capacity=config.expert_capacity,
                top_k=config.top_k_experts,
                dropout=config.dropout,
                bias=config.bias,
                apply_expert_capacity=config.apply_expert_capacity,
                load_balance_loss_coef=config.load_balance_loss_coef,
            )
        else:
            self.mlp = StandardFFN(
                n_embd=config.n_embd, dropout=config.dropout, bias=config.bias
            )

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        mlp_out, load_balance_loss = self.mlp(self.ln_2(x))
        return x + mlp_out, load_balance_loss


class KimiK2PreTrainedModel(PreTrainedModel):
    config_class = KimiK2Config
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = False
    _no_split_modules = ["KimiBlock"]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (LayerNorm, RMSNorm)):
            nn.init.ones_(module.weight)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)


class KimiK2ForCausalLM(KimiK2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = (
        {"lm_head.weight": "transformer.wte.weight"} if _TRANSFORMERS_V5 else ["lm_head.weight"]
    )

    def __init__(self, config: KimiK2Config):
        super().__init__(config)

        # MLA carries position through RoPE, so learned position embeddings only
        # exist on the dense MultiHeadAttention baseline.
        self.use_rope = config.use_latent_attention

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([KimiBlock(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        if not self.use_rope:
            self.transformer.wpe = nn.Embedding(config.block_size, config.n_embd)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.post_init()

        # Scaled init on residual projections, per the GPT-2 paper.
        for pn, p in self.named_parameters():
            if pn.endswith("o_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def get_input_embeddings(self):
        return self.transformer.wte

    def set_input_embeddings(self, value):
        self.transformer.wte = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def get_num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.use_rope:
            n -= self.transformer.wpe.weight.numel()
        return n

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        inputs_embeds=None,
        return_dict=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else True

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("one of input_ids or inputs_embeds must be provided")
            inputs_embeds = self.transformer.wte(input_ids)

        b, t, _ = inputs_embeds.size()
        if t > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
            )

        if self.use_rope:
            x = self.transformer.drop(inputs_embeds)
        else:
            pos = torch.arange(0, t, dtype=torch.long, device=inputs_embeds.device)
            x = self.transformer.drop(inputs_embeds + self.transformer.wpe(pos))

        # (B, T) padding mask -> additive (B, 1, 1, T) mask
        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = (1.0 - attention_mask[:, None, None, :].to(x.dtype)) * torch.finfo(
                x.dtype
            ).min

        total_load_balance_loss = x.new_zeros(())
        for block in self.transformer.h:
            x, load_balance_loss = block(x, attention_mask=attention_mask)
            total_load_balance_loss = total_load_balance_loss + load_balance_loss

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            if self.config.use_moe:
                loss = loss + total_load_balance_loss

        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)

        return CausalLMOutput(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        # No KV cache: always re-run the full (cropped) context.
        if input_ids.size(1) > self.config.block_size:
            input_ids = input_ids[:, -self.config.block_size :]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -self.config.block_size :]
        return {"input_ids": input_ids, "attention_mask": attention_mask}


__all__ = [
    "KimiK2Config",
    "KimiK2PreTrainedModel",
    "KimiK2ForCausalLM",
    "MultiHeadLatentAttention",
    "MultiHeadAttention",
    "MoELayer",
    "RMSNorm",
]
