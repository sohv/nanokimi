#!/usr/bin/env python3
"""
Example usage of nanoKimi components

This script demonstrates how to use the core nanoKimi components
for training and inference.
"""

import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model import KimiK2
from optimizer import create_muon_optimizer
from utils import get_device


def example_training():
    """Example of training a small Kimi-K2 model"""
    print("Example: Training a Kimi-K2 model")
    
    # Model configuration
    config = {
        'vocab_size': 1000,
        'block_size': 128,
        'n_layer': 4,
        'n_head': 8,
        'n_embd': 256,
        'dropout': 0.1,
        'bias': True,
        
        # Kimi-K2 features
        'use_moe': True,
        'num_experts': 8,
        'expert_capacity': 16,
        'top_k_experts': 2,
        'use_latent_attention': True,
        'latent_dim': 64,
    }
    
    # Create model
    device = get_device()
    model = KimiK2(config).to(device)
    
    # Create optimizer  
    muon_config = {
        'learning_rate': 3e-4,
        'momentum': 0.95,
        'weight_decay': 0.1,
        'eps': 1e-8,
    }
    optimizer = create_muon_optimizer(model, muon_config)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Dummy training loop
    model.train()
    for step in range(5):
        # Generate dummy batch
        x = torch.randint(0, config['vocab_size'], (4, config['block_size']), device=device)
        y = torch.randint(0, config['vocab_size'], (4, config['block_size']), device=device)
        
        # Forward pass
        logits, loss = model(x, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Step {step+1}: loss = {loss.item():.4f}")
    
    print("Training example completed!")
    return model


def example_generation(model):
    """Example of text generation"""
    print("\\nExample: Text generation")
    
    model.eval()
    
    # Create a prompt
    prompt = torch.randint(0, 1000, (1, 10), device=next(model.parameters()).device)
    
    # Generate text
    with torch.no_grad():
        generated = model.generate(
            prompt, 
            max_new_tokens=50, 
            temperature=0.8, 
            top_k=40
        )
    
    print(f"Prompt tokens: {prompt[0].tolist()}")
    print(f"Generated tokens: {generated[0].tolist()}")
    print("Generation example completed!")


def example_comparison():
    """Example comparing different architectures"""
    print("\\nExample: Architecture comparison")
    
    device = get_device()
    
    configs = {
        'Standard': {
            'vocab_size': 1000, 'block_size': 128, 'n_layer': 4, 
            'n_head': 8, 'n_embd': 256, 'dropout': 0.0, 'bias': True,
            'use_moe': False, 'use_latent_attention': False
        },
        'With MoE': {
            'vocab_size': 1000, 'block_size': 128, 'n_layer': 4,
            'n_head': 8, 'n_embd': 256, 'dropout': 0.0, 'bias': True,
            'use_moe': True, 'num_experts': 4, 'expert_capacity': 16, 'top_k_experts': 2,
            'use_latent_attention': False
        },
        'With Latent Attention': {
            'vocab_size': 1000, 'block_size': 128, 'n_layer': 4,
            'n_head': 8, 'n_embd': 256, 'dropout': 0.0, 'bias': True,
            'use_moe': False, 'use_latent_attention': True, 'latent_dim': 64
        },
        'Full Kimi-K2': {
            'vocab_size': 1000, 'block_size': 128, 'n_layer': 4,
            'n_head': 8, 'n_embd': 256, 'dropout': 0.0, 'bias': True,
            'use_moe': True, 'num_experts': 4, 'expert_capacity': 16, 'top_k_experts': 2,
            'use_latent_attention': True, 'latent_dim': 64
        }
    }
    
    for name, config in configs.items():
        model = KimiK2(config).to(device)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:20}: {params:,} parameters")
    
    print("Comparison example completed!")


def main():
    print("=" * 60)
    print("nanoKimi Usage Examples")
    print("Demonstrating Kimi-K2 features")
    print("=" * 60)
    
    # Training example
    model = example_training()
    
    # Generation example
    example_generation(model)
    
    # Comparison example
    example_comparison()
    
    print("\\n" + "=" * 60)
    print("All examples completed!")
    print("\\nFor real training, use: python train.py")
    print("For text generation, use: python generate.py")
    print("For benchmarking, use: python benchmark.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
