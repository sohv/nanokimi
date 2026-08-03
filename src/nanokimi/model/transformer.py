"""
Kimi-K2 Model Implementation for nanoKimi

This is the main model implementation that combines all the Kimi-K2 innovations:
- Latent Attention for memory efficiency
- Mixture of Experts for sparse scaling
- Compatible with Muon optimizer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from nanokimi.model.attention import MultiHeadAttention, MultiHeadLatentAttention
from nanokimi.model.moe import MoELayer, StandardFFN


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class KimiBlock(nn.Module):
    """
    Kimi-K2 Transformer Block
    
    Combines the innovations of Kimi-K2:
    - Latent Attention (optional)
    - Mixture of Experts (optional)
    - Standard layer normalization and residual connections
    """

    def __init__(self, config):
        super().__init__()
        
        # Layer normalization
        self.ln_1 = LayerNorm(config['n_embd'], bias=config['bias'])
        self.ln_2 = LayerNorm(config['n_embd'], bias=config['bias'])
        
        # Attention layer
        if config.get('use_latent_attention', False):
            self.attn = MultiHeadLatentAttention(                                   # Modified to build real MLA, which caches one shared latent instead of per-head K/V.
                n_embd=config['n_embd'],
                n_head=config['n_head'],
                kv_lora_rank=config.get('kv_lora_rank', 256),                       # Added d_c, the shared KV latent width that determines the cache cost.
                q_lora_rank=config.get('q_lora_rank', 768),                         # Added d_c', the query LoRA bottleneck width.
                qk_nope_head_dim=config.get('qk_nope_head_dim', 64),                # Added the non-rotary per-head query/key width.
                qk_rope_head_dim=config.get('qk_rope_head_dim', 32),                # Added the rotary per-head width.
                v_head_dim=config.get('v_head_dim', 64),                            # Added the per-head value width.
                max_seq_len=config['block_size'],                                   # Added the RoPE table length, sized from the model's context length.
                rope_theta=config.get('rope_theta', 50000.0),                       # Added the RoPE base; 50000.0 is the value Kimi K2 uses.
                dropout=config['dropout'],
                bias=config.get('attention_bias', False),                           # Added a separate attention bias flag, defaulting False as in DeepSeek-V3 and Kimi K2.
            )
        else:
            self.attn = MultiHeadAttention(
                n_embd=config['n_embd'],
                n_head=config['n_head'],
                dropout=config['dropout'],
                bias=config['bias']
            )
        
        # Feed-forward layer
        if config.get('use_moe', False):
            self.mlp = MoELayer(
                n_embd=config['n_embd'],
                num_experts=config.get('num_experts', 8),
                expert_capacity=config.get('expert_capacity', 32),
                top_k=config.get('top_k_experts', 2),
                dropout=config['dropout'],
                bias=config['bias'],
                load_balance_loss_coef=config.get('load_balance_loss_coef', 0.01),   # Added so the aux-loss weight comes from config instead of being hardcoded in the layer.
                apply_expert_capacity=config.get('apply_expert_capacity', False),    # Added so capacity dropping is opt-in and off by default.
            )
        else:
            self.mlp = StandardFFN(
                n_embd=config['n_embd'],
                dropout=config['dropout'],
                bias=config['bias']
            )

    def forward(self, x):
        # Attention with residual connection
        x = x + self.attn(self.ln_1(x))
        
        # MLP with residual connection
        mlp_out, load_balance_loss = self.mlp(self.ln_2(x))
        x = x + mlp_out
        
        return x, load_balance_loss


class KimiK2(nn.Module):
    """
    Kimi-K2 Model
    
    A transformer model incorporating the key innovations from Kimi-K2:
    - Latent Attention for memory efficiency
    - Mixture of Experts for sparse scaling
    - Optimized for use with Muon optimizer
    """

    def __init__(self, config):
        super().__init__()
        assert config['vocab_size'] is not None
        assert config['block_size'] is not None
        # Copy rather than alias: crop_block_size mutates self.config, and holding the
        # caller's dict meant cropping one model silently changed the config every
        # other model was built from.
        self.config = dict(config)

        # MLA carries position information through RoPE, so learned absolute
        # position embeddings are redundant there and are dropped. The dense
        # MultiHeadAttention baseline has no RoPE and still needs them.
        self.use_rope = config.get('use_latent_attention', False)                   # Added a flag recording whether position comes from RoPE inside MLA.

        # Embedding layers
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config['vocab_size'], config['n_embd']),  # token embeddings
            drop = nn.Dropout(config['dropout']),
            h = nn.ModuleList([KimiBlock(config) for _ in range(config['n_layer'])]),
            ln_f = LayerNorm(config['n_embd'], bias=config['bias']),
        ))
        if not self.use_rope:                                                       # Added the conditional so wpe only exists on the non-RoPE baseline path.
            self.transformer.wpe = nn.Embedding(config['block_size'], config['n_embd'])  # Modified to create position embeddings only when MLA/RoPE is not in use.
        self.lm_head = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)
        
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('o_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config['n_layer']))

        # Report number of parameters
        print("Number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.use_rope:                                     # Modified to guard the subtraction, since wpe does not exist on the RoPE/MLA path.
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config['block_size'], f"Cannot forward sequence of length {t}, block size is only {self.config['block_size']}"

        # Forward the model
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        if self.use_rope:                                                           # Added the RoPE branch, where MLA applies position inside the attention layer.
            x = self.transformer.drop(tok_emb)                                      # Modified to skip additive position embeddings entirely when RoPE is in use.
        else:
            pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
            pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
            x = self.transformer.drop(tok_emb + pos_emb)
        
        # Accumulate load balance losses from MoE layers
        total_load_balance_loss = 0.0
        
        for block in self.transformer.h:
            x, load_balance_loss = block(x)
            total_load_balance_loss += load_balance_loss
        
        x = self.transformer.ln_f(x)

        if targets is not None:
            # If we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            
            # Add load balance loss
            loss = loss + total_load_balance_loss
        else:
            # Inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config['block_size']
        self.config['block_size'] = block_size
        if not self.use_rope:                                                       # Modified to guard the wpe crop, since wpe does not exist on the RoPE/MLA path.
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, '_rope_cache'):                                  # Modified to reset the lazy RoPE cache, replacing a check for an `attn.bias` buffer that never existed.
                block.attn.max_seq_len = block_size                                 # Added the new table length so the next rebuild sizes to the cropped context.
                block.attn._rope_cache = None                                       # Added a cache clear so the tables are rebuilt at the new length on the next forward.

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config['block_size'] else idx[:, -self.config['block_size']:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
