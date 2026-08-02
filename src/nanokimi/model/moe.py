"""
Mixture of Experts (MoE) Implementation for nanoKimi

This module implements the MoE layer used in Kimi-K2, which allows
for efficient scaling by routing tokens to different expert networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoELayer(nn.Module):
    """
    Mixture of Experts Layer
    
    Routes input tokens to different expert networks based on a learned gating function.
    Only the top-k experts are activated for each token, making the computation sparse.
    
    Args:
        n_embd: embedding dimension
        num_experts: number of expert networks
        expert_capacity: capacity of each expert (max tokens per expert)
        top_k: number of experts to route each token to
        dropout: dropout probability
        bias: whether to use bias in linear layers
    """
    
    def __init__(self, n_embd, num_experts=8, expert_capacity=32, top_k=2, dropout=0.0, bias=True,
                 load_balance_loss_coef=0.01, apply_expert_capacity=False):        # Added the aux-loss coefficient and a capacity switch so both are configurable rather than hardcoded.
        super().__init__()

        self.n_embd = n_embd
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.top_k = top_k
        self.apply_expert_capacity = apply_expert_capacity                         # Added the flag that keeps token dropping out of the inference path entirely.

        # Gating network - decides which experts to use
        self.gate = nn.Linear(n_embd, num_experts, bias=bias)

        # Expert networks - simple FFN for each expert
        self.experts = nn.ModuleList([
            ExpertFFN(n_embd, dropout=dropout, bias=bias)
            for _ in range(num_experts)
        ])

        # Load balancing loss coefficient
        self.load_balance_loss_coef = load_balance_loss_coef                       # Modified to take the coefficient from the constructor instead of pinning it to 0.01.

    def forward(self, x):
        B, T, C = x.shape

        # Flatten to (B*T, C) for easier processing
        x_flat = x.view(-1, C)

        # Compute gating scores
        gate_logits = self.gate(x_flat)  # (B*T, num_experts)
        gate_scores = F.softmax(gate_logits, dim=-1)

        # Select top-k experts for each token
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)

        # Normalize top-k scores
        top_k_scores = top_k_scores / top_k_scores.sum(dim=-1, keepdim=True)

        # Initialize output
        output = torch.zeros_like(x_flat)

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens assigned to this expert
            hit = top_k_indices == expert_idx                                      # Added a reusable (tokens, top_k) hit mask so the routing weight can be recovered without a per-token loop.
            token_idx = hit.any(dim=-1).nonzero(as_tuple=True)[0]                  # Modified to collect expert token indices once, replacing the old boolean-mask-then-loop approach.

            if token_idx.numel() == 0:                                             # Modified the empty check to use the index tensor directly.
                continue

            # Apply capacity constraint. Only during training, and only when
            # explicitly enabled: dropping tokens is a training-side load-balancing
            # device, and doing it at inference made every forward pass random.
            if self.apply_expert_capacity and self.training and token_idx.numel() > self.expert_capacity:  # Added training and opt-in guards so inference is deterministic.
                weights_for_rank = (top_k_scores[token_idx] * hit[token_idx]).sum(dim=-1)                  # Added the routing weight per candidate token, used to rank them.
                keep = torch.topk(weights_for_rank, self.expert_capacity).indices                          # Modified to keep the highest-affinity tokens instead of a random subset, which is what capacity dropping is supposed to do.
                token_idx = token_idx[keep]                                                                # Added the reduction of the token list to those within capacity.

            # Weight by gating scores and add to output. A token can select the
            # same expert only once, so summing over the top_k axis picks out that
            # single routing weight.
            weight = (top_k_scores[token_idx] * hit[token_idx]).sum(dim=-1, keepdim=True)                  # Added a vectorized lookup of each token's routing weight for this expert.
            expert_output = self.experts[expert_idx](x_flat[token_idx])                                    # Modified to run the expert on the final token set after any capacity trim.
            output.index_add_(0, token_idx, weight * expert_output)                                        # Modified to scatter results with index_add_, replacing the Python loop that ran once per token.

        # Reshape back to original shape
        output = output.view(B, T, C)

        # Compute load balancing loss
        load_balance_loss = self._compute_load_balance_loss(gate_scores, top_k_indices)                    # Modified to also pass the hard top-k assignments, which the correct aux loss needs.

        return output, load_balance_loss

    def _compute_load_balance_loss(self, gate_scores, top_k_indices):
        """
        Switch Transformer / GShard auxiliary load-balancing loss.

            L = N * sum_i f_i * P_i

        f_i is the fraction of routing assignments that actually went to expert i,
        P_i is the mean gate probability for expert i. The product term is what
        makes this work: it couples the hard dispatch decision to the soft
        probability, so the gradient pushes probability mass away from experts that
        are already over-subscribed. The previous MSE-on-mean-probability loss saw
        only P_i and never f_i, so it could be driven near zero while every token
        still routed to the same two experts - which is exactly what happened.

        Normalized so a perfectly balanced router gives 1.0.
        """
        num_tokens = gate_scores.size(0)                                            # Added the token count used to normalize both terms.

        one_hot = F.one_hot(top_k_indices, num_classes=self.num_experts).sum(dim=1)  # Added a per-token expert assignment count from the hard top-k choice.
        f_i = one_hot.float().sum(dim=0) / (num_tokens * self.top_k)                 # Added f_i, the fraction of all assignments landing on each expert, summing to 1.
        p_i = gate_scores.mean(dim=0)                                                # Added P_i, the mean soft gate probability per expert, summing to 1.

        load_balance_loss = self.num_experts * torch.sum(f_i * p_i)                  # Added the Switch aux loss, which is minimized at 1.0 when routing is uniform.

        return self.load_balance_loss_coef * load_balance_loss                       # Modified to scale the correct aux loss by the configured coefficient.


class ExpertFFN(nn.Module):
    """
    Expert Feed-Forward Network
    
    A simple two-layer MLP that serves as an expert in the MoE layer.
    """
    
    def __init__(self, n_embd, dropout=0.0, bias=True):
        super().__init__()
        
        # Typical GPT-style FFN with 4x expansion
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class StandardFFN(nn.Module):
    """
    Standard Feed-Forward Network for comparison with MoE
    """
    
    def __init__(self, n_embd, dropout=0.0, bias=True):
        super().__init__()
        
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x, 0.0  # Return 0 load balance loss for consistency
