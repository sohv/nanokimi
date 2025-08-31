"""
Latent Attention Implementation for nanoKimi

This module implements the Latent Attention mechanism used in Kimi-K2,
which compresses attention representations to reduce memory footprint
while maintaining performance on long sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LatentAttention(nn.Module):
    """
    Latent Attention mechanism that compresses attention representations
    
    The key idea is to project keys and values into a lower-dimensional
    latent space, reducing memory usage while preserving attention quality.
    
    Args:
        n_embd: embedding dimension
        n_head: number of attention heads
        latent_dim: dimension of the latent space
        dropout: dropout probability
        bias: whether to use bias in linear layers
    """
    
    def __init__(self, n_embd, n_head, latent_dim=64, dropout=0.0, bias=True):
        super().__init__()
        assert n_embd % n_head == 0
        
        self.n_embd = n_embd
        self.n_head = n_head
        self.latent_dim = latent_dim
        self.head_dim = n_embd // n_head
        
        # Query projection (full dimension)
        self.q_proj = nn.Linear(n_embd, n_embd, bias=bias)
        
        # Key and Value projections to latent space
        self.k_proj = nn.Linear(n_embd, n_head * latent_dim, bias=bias)
        self.v_proj = nn.Linear(n_embd, n_head * latent_dim, bias=bias)
        
        # Output projection
        self.o_proj = nn.Linear(n_head * latent_dim, n_embd, bias=bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Scale factor for attention
        self.scale = 1.0 / math.sqrt(latent_dim)
        
    def forward(self, x, mask=None):
        B, T, C = x.size()  # batch, sequence length, embedding dim
        
        # Project to query, key, value
        q = self.q_proj(x)  # (B, T, n_embd)
        k = self.k_proj(x)  # (B, T, n_head * latent_dim)
        v = self.v_proj(x)  # (B, T, n_head * latent_dim)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.latent_dim).transpose(1, 2)  # (B, n_head, T, latent_dim)
        v = v.view(B, T, self.n_head, self.latent_dim).transpose(1, 2)  # (B, n_head, T, latent_dim)
        
        # Compress queries to latent dimension for attention computation
        # We use a learnable compression matrix
        if not hasattr(self, 'q_compress'):
            self.q_compress = nn.Linear(self.head_dim, self.latent_dim, bias=False).to(x.device)
        
        q_compressed = self.q_compress(q)  # (B, n_head, T, latent_dim)
        
        # Compute attention scores in latent space
        att = torch.matmul(q_compressed, k.transpose(-2, -1)) * self.scale  # (B, n_head, T, T)
        
        # Apply causal mask
        if mask is not None:
            att = att.masked_fill(mask == 0, float('-inf'))
        else:
            # Create causal mask
            causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
            att = att.masked_fill(causal_mask == 0, float('-inf'))
        
        # Apply softmax
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        # Apply attention to values
        y = torch.matmul(att, v)  # (B, n_head, T, latent_dim)
        
        # Reshape and project back
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.latent_dim)
        y = self.o_proj(y)
        y = self.resid_dropout(y)
        
        return y


class MultiHeadAttention(nn.Module):
    """
    Standard multi-head attention for comparison
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
        if mask is not None:
            att = att.masked_fill(mask == 0, float('-inf'))
        else:
            causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
            att = att.masked_fill(causal_mask == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        # Apply attention to values
        y = torch.matmul(att, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.o_proj(y)
        y = self.resid_dropout(y)
        
        return y
