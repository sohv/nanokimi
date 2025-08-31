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
    
    def __init__(self, n_embd, num_experts=8, expert_capacity=32, top_k=2, dropout=0.0, bias=True):
        super().__init__()
        
        self.n_embd = n_embd
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        self.top_k = top_k
        
        # Gating network - decides which experts to use
        self.gate = nn.Linear(n_embd, num_experts, bias=bias)
        
        # Expert networks - simple FFN for each expert
        self.experts = nn.ModuleList([
            ExpertFFN(n_embd, dropout=dropout, bias=bias) 
            for _ in range(num_experts)
        ])
        
        # Load balancing loss coefficient
        self.load_balance_loss_coef = 0.01
        
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
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)
            
            if expert_mask.sum() == 0:
                continue
                
            # Get tokens for this expert
            expert_tokens = x_flat[expert_mask]
            
            # Apply capacity constraint
            if expert_tokens.size(0) > self.expert_capacity:
                # Random sampling if too many tokens
                perm = torch.randperm(expert_tokens.size(0))[:self.expert_capacity]
                expert_tokens = expert_tokens[perm]
                expert_mask_indices = torch.where(expert_mask)[0][perm]
            else:
                expert_mask_indices = torch.where(expert_mask)[0]
            
            # Process through expert
            expert_output = self.experts[expert_idx](expert_tokens)
            
            # Weight by gating scores and add to output
            for i, token_idx in enumerate(expert_mask_indices):
                # Find which position in top_k this expert is for this token
                expert_positions = (top_k_indices[token_idx] == expert_idx).nonzero(as_tuple=True)[0]
                if len(expert_positions) > 0:
                    weight = top_k_scores[token_idx, expert_positions[0]]
                    output[token_idx] += weight * expert_output[i]
        
        # Reshape back to original shape
        output = output.view(B, T, C)
        
        # Compute load balancing loss
        load_balance_loss = self._compute_load_balance_loss(gate_scores)
        
        return output, load_balance_loss
    
    def _compute_load_balance_loss(self, gate_scores):
        """
        Compute load balancing loss to encourage equal usage of experts
        """
        # Compute the fraction of tokens routed to each expert
        expert_usage = gate_scores.mean(dim=0)  # (num_experts,)
        
        # Target is uniform distribution
        target_usage = torch.ones_like(expert_usage) / self.num_experts
        
        # L2 loss between actual and target usage
        load_balance_loss = F.mse_loss(expert_usage, target_usage)
        
        return self.load_balance_loss_coef * load_balance_loss


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
