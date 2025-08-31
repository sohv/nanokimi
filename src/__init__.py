"""
nanoKimi - The simplest, fastest repository for training/finetuning Kimi-K2 models

This package implements the Kimi-K2 architecture with key innovations:
- Muon Optimizer: Advanced optimization for faster convergence
- Mixture of Experts (MoE): Efficient scaling with expert routing  
- Latent Attention: Memory-efficient attention mechanism
"""

from .model import KimiK2
from .optimizer import Muon, create_muon_optimizer
from .attention import LatentAttention, MultiHeadAttention
from .moe import MoELayer, StandardFFN

__version__ = "0.1.0"
__author__ = "nanoKimi Team"

__all__ = [
    "KimiK2",
    "Muon", 
    "create_muon_optimizer",
    "LatentAttention",
    "MultiHeadAttention", 
    "MoELayer",
    "StandardFFN"
]
