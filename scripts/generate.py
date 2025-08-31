#!/usr/bin/env python3
"""
Text generation script for nanoKimi

Generate text using a trained Kimi-K2 model.
"""

import os
import sys
import pickle
import torch
import tiktoken
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import KimiK2
from utils import get_device

# Add config to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))


def load_model(checkpoint_path, device='auto'):
    """Load a trained model from checkpoint"""
    if device == 'auto':
        device = get_device()
    
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config
    model_args = checkpoint['model_args']
    
    # Create model
    model = KimiK2(model_args)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    return model, model_args, device


def generate_text(model, device, prompt="", max_new_tokens=500, temperature=0.8, top_k=200, encoding_name='gpt2'):
    """Generate text using the model"""
    
    # Get tokenizer
    enc = tiktoken.get_encoding(encoding_name)
    
    # Encode prompt
    if prompt:
        tokens = enc.encode(prompt)
        print(f"Prompt: '{prompt}'")
    else:
        tokens = [enc.encode("")[0]] if enc.encode("") else [0]  # Start with a default token
        print("Generating from scratch...")
    
    print(f"Input tokens: {len(tokens)}")
    print("-" * 50)
    
    # Convert to tensor
    x = torch.tensor(tokens, dtype=torch.long, device=device)[None, ...]
    
    # Generate
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.split(':')[0], dtype=torch.bfloat16):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    
    # Decode and print
    generated = enc.decode(y[0].tolist())
    print(generated)
    print("-" * 50)
    print(f"Generated {max_new_tokens} tokens")
    
    return generated


def main():
    parser = argparse.ArgumentParser(description='Generate text with nanoKimi')
    parser.add_argument('--checkpoint', type=str, default='out/ckpt.pt', 
                       help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, default='', 
                       help='Text prompt to start generation')
    parser.add_argument('--max_tokens', type=int, default=500, 
                       help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, 
                       help='Sampling temperature (0.0 = greedy, higher = more random)')
    parser.add_argument('--top_k', type=int, default=200, 
                       help='Top-k sampling (0 = no filtering)')
    parser.add_argument('--device', type=str, default='auto', 
                       help='Device to use (auto, cpu, cuda, mps)')
    parser.add_argument('--num_samples', type=int, default=1, 
                       help='Number of samples to generate')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("nanoKimi Text Generation")
    print("Generating text with Kimi-K2")
    print("=" * 60)
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        print("Make sure you've trained a model first by running: python scripts/train.py")
        return
    
    # Load model
    model, model_args, device = load_model(args.checkpoint, args.device)
    
    print(f"Model config:")
    print(f"  Layers: {model_args['n_layer']}")
    print(f"  Heads: {model_args['n_head']}")
    print(f"  Embedding dim: {model_args['n_embd']}")
    print(f"  Vocab size: {model_args['vocab_size']}")
    print(f"  Using MoE: {model_args.get('use_moe', False)}")
    print(f"  Using Latent Attention: {model_args.get('use_latent_attention', False)}")
    print()
    
    # Generate samples
    for i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"\\n{'='*20} Sample {i+1}/{args.num_samples} {'='*20}")
        
        generated = generate_text(
            model=model,
            device=device,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k
        )
        
        if args.num_samples > 1:
            print(f"{'='*60}")


if __name__ == "__main__":
    main()
