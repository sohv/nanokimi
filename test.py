#!/usr/bin/env python3
"""
Test script for nanoKimi

Simple tests to verify the installation and model components.
"""

import os
import sys
import torch

# Add src to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.join(project_root, 'config'))

from model import KimiK2
from optimizer import Muon
from utils import get_device, count_parameters
from default import model_config


def test_model_creation():
    """Test basic model creation"""
    print("Testing model creation...")
    
    # Small test config
    config = {
        'vocab_size': 1000,
        'block_size': 64,
        'n_layer': 2,
        'n_head': 4,
        'n_embd': 128,
        'dropout': 0.0,
        'bias': True,
        'use_moe': False,
        'use_latent_attention': False,
    }
    
    try:
        model = KimiK2(config)
        print(f"SUCCESS: Standard model created: {count_parameters(model):,} parameters")
        
        # Test with MoE
        config['use_moe'] = True
        config['num_experts'] = 4
        config['expert_capacity'] = 8
        config['top_k_experts'] = 2
        
        model_moe = KimiK2(config)
        print(f"SUCCESS: MoE model created: {count_parameters(model_moe):,} parameters")
        
        # Test with Latent Attention
        config['use_moe'] = False
        config['use_latent_attention'] = True
        config['latent_dim'] = 32
        
        model_latent = KimiK2(config)
        print(f"SUCCESS: Latent attention model created: {count_parameters(model_latent):,} parameters")
        
        return True
    except Exception as e:
        print(f"ERROR: Model creation failed: {e}")
        return False


def test_forward_pass():
    """Test forward pass"""
    print("\\nTesting forward pass...")
    
    config = {
        'vocab_size': 1000,
        'block_size': 64,
        'n_layer': 2,
        'n_head': 4,
        'n_embd': 128,
        'dropout': 0.0,
        'bias': True,
        'use_moe': True,
        'num_experts': 4,
        'expert_capacity': 8,
        'top_k_experts': 2,
        'use_latent_attention': True,
        'latent_dim': 32,
    }
    
    try:
        device = get_device()
        model = KimiK2(config).to(device)
        
        # Create dummy input
        batch_size, seq_len = 4, 32
        x = torch.randint(0, config['vocab_size'], (batch_size, seq_len), device=device)
        y = torch.randint(0, config['vocab_size'], (batch_size, seq_len), device=device)
        
        # Forward pass
        logits, loss = model(x, y)
        
        print(f"SUCCESS: Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Loss: {loss.item():.4f}")
        
        return True
    except Exception as e:
        print(f"ERROR: Forward pass failed: {e}")
        return False


def test_generation():
    """Test text generation"""
    print("\\nTesting text generation...")
    
    config = {
        'vocab_size': 1000,
        'block_size': 64,
        'n_layer': 2,
        'n_head': 4,
        'n_embd': 128,
        'dropout': 0.0,
        'bias': True,
        'use_moe': False,
        'use_latent_attention': False,
    }
    
    try:
        device = get_device()
        model = KimiK2(config).to(device)
        model.eval()
        
        # Create prompt
        prompt = torch.randint(0, config['vocab_size'], (1, 10), device=device)
        
        # Generate
        with torch.no_grad():
            generated = model.generate(prompt, max_new_tokens=20, temperature=1.0, top_k=100)
        
        print(f"SUCCESS: Text generation successful")
        print(f"  Prompt length: {prompt.shape[1]}")
        print(f"  Generated length: {generated.shape[1]}")
        
        return True
    except Exception as e:
        print(f"ERROR: Text generation failed: {e}")
        return False


def test_optimizer():
    """Test Muon optimizer"""
    print("\\nTesting Muon optimizer...")
    
    try:
        # Create a simple model
        model = torch.nn.Linear(100, 10)
        
        # Create Muon optimizer
        optimizer = Muon(model.parameters(), lr=1e-3)
        
        # Simple training step
        x = torch.randn(32, 100)
        y = torch.randn(32, 10)
        
        optimizer.zero_grad()
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        loss.backward()
        optimizer.step()
        
        print(f"SUCCESS: Muon optimizer test successful")
        print(f"  Loss: {loss.item():.4f}")
        
        return True
    except Exception as e:
        print(f"ERROR: Muon optimizer test failed: {e}")
        return False


def main():
    print("=" * 60)
    print("nanoKimi Test Suite")
    print("Testing core components")
    print("=" * 60)
    
    tests = [
        test_model_creation,
        test_forward_pass,
        test_generation,
        test_optimizer,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        if test_func():
            passed += 1
    
    print(f"\\n{'='*60}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("SUCCESS: All tests passed! nanoKimi is ready to use.")
    else:
        print("ERROR: Some tests failed. Check the error messages above.")
    
    print(f"{'='*60}")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
