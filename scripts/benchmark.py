#!/usr/bin/env python3
"""
Benchmark script for nanoKimi vs nanoGPT

Compare performance, memory usage, and generation quality between nanoKimi and nanoGPT.
"""

import os
import sys
import time
import torch
import numpy as np
import argparse
from contextlib import nullcontext
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import KimiK2
from utils import get_device, Timer

# Add data to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
from prepare import get_batch


class ModelBenchmark:
    """Benchmark class for comparing models"""
    
    def __init__(self, model, device, model_name="Model"):
        self.model = model
        self.device = device
        self.model_name = model_name
        self.results = {}
    
    def benchmark_forward_pass(self, batch_size=32, block_size=256, num_iterations=100):
        """Benchmark forward pass speed"""
        print(f"\\nBenchmarking {self.model_name} forward pass...")
        
        # Create dummy data
        x = torch.randint(0, 50257, (batch_size, block_size), device=self.device)
        y = torch.randint(0, 50257, (batch_size, block_size), device=self.device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _, _ = self.model(x, y)
        
        # Synchronize
        if self.device.startswith('cuda'):
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        
        for _ in range(num_iterations):
            with torch.no_grad():
                logits, loss = self.model(x, y)
        
        if self.device.startswith('cuda'):
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_iterations
        tokens_per_second = (batch_size * block_size) / avg_time
        
        self.results['forward_time'] = avg_time
        self.results['tokens_per_second'] = tokens_per_second
        
        print(f"  Average forward time: {avg_time*1000:.2f}ms")
        print(f"  Tokens per second: {tokens_per_second:.0f}")
        
        return avg_time, tokens_per_second
    
    def benchmark_memory_usage(self, batch_size=32, block_size=256):
        """Benchmark memory usage"""
        print(f"\\nBenchmarking {self.model_name} memory usage...")
        
        if not self.device.startswith('cuda'):
            print("  Memory benchmarking only available on CUDA")
            return 0, 0
        
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Create dummy data
        x = torch.randint(0, 50257, (batch_size, block_size), device=self.device)
        y = torch.randint(0, 50257, (batch_size, block_size), device=self.device)
        
        # Forward pass
        logits, loss = self.model(x, y)
        
        # Backward pass
        loss.backward()
        
        # Get memory stats
        current_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        self.results['current_memory_mb'] = current_memory
        self.results['peak_memory_mb'] = peak_memory
        
        print(f"  Current memory: {current_memory:.1f} MB")
        print(f"  Peak memory: {peak_memory:.1f} MB")
        
        return current_memory, peak_memory
    
    def benchmark_generation(self, prompt_length=10, max_new_tokens=100, num_samples=5):
        """Benchmark text generation speed"""
        print(f"\\nBenchmarking {self.model_name} generation...")
        
        # Create dummy prompt
        prompt = torch.randint(0, 50257, (1, prompt_length), device=self.device)
        
        generation_times = []
        
        for i in range(num_samples):
            start_time = time.time()
            
            with torch.no_grad():
                generated = self.model.generate(prompt, max_new_tokens, temperature=0.8, top_k=200)
            
            if self.device.startswith('cuda'):
                torch.cuda.synchronize()
            
            end_time = time.time()
            generation_time = end_time - start_time
            generation_times.append(generation_time)
        
        avg_gen_time = np.mean(generation_times)
        tokens_per_second_gen = max_new_tokens / avg_gen_time
        
        self.results['generation_time'] = avg_gen_time
        self.results['generation_tokens_per_second'] = tokens_per_second_gen
        
        print(f"  Average generation time: {avg_gen_time:.2f}s")
        print(f"  Generation tokens per second: {tokens_per_second_gen:.1f}")
        
        return avg_gen_time, tokens_per_second_gen
    
    def get_model_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Calculate model size in MB
        param_size = sum(p.nelement() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in self.model.buffers())
        model_size_mb = (param_size + buffer_size) / 1024**2
        
        self.results['total_params'] = total_params
        self.results['trainable_params'] = trainable_params
        self.results['model_size_mb'] = model_size_mb
        
        print(f"\\n{self.model_name} Info:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: {model_size_mb:.1f} MB")
        
        return total_params, model_size_mb


def create_test_models(device):
    """Create test models for benchmarking"""
    
    # Common config
    base_config = {
        'vocab_size': 50304,
        'block_size': 256,
        'n_layer': 6,
        'n_head': 6,
        'n_embd': 384,
        'dropout': 0.0,
        'bias': True,
    }
    
    # Standard model (like nanoGPT)
    standard_config = base_config.copy()
    standard_config.update({
        'use_moe': False,
        'use_latent_attention': False,
    })
    
    # Kimi-K2 with MoE
    moe_config = base_config.copy()
    moe_config.update({
        'use_moe': True,
        'num_experts': 4,
        'expert_capacity': 16,
        'top_k_experts': 2,
        'use_latent_attention': False,
    })
    
    # Kimi-K2 with Latent Attention
    latent_config = base_config.copy()
    latent_config.update({
        'use_moe': False,
        'use_latent_attention': True,
        'latent_dim': 32,
    })
    
    # Kimi-K2 with both MoE and Latent Attention
    full_config = base_config.copy()
    full_config.update({
        'use_moe': True,
        'num_experts': 4,
        'expert_capacity': 16,
        'top_k_experts': 2,
        'use_latent_attention': True,
        'latent_dim': 32,
    })
    
    models = {
        'Standard (nanoGPT-like)': KimiK2(standard_config),
        'Kimi-K2 (MoE only)': KimiK2(moe_config),
        'Kimi-K2 (Latent Attn only)': KimiK2(latent_config),
        'Kimi-K2 (Full)': KimiK2(full_config),
    }
    
    # Move to device
    for model in models.values():
        model.to(device)
        model.eval()
    
    return models


def plot_results(benchmarks):
    """Plot benchmark results"""
    print("\\nCreating benchmark plots...")
    
    model_names = list(benchmarks.keys())
    
    # Extract metrics
    forward_times = [benchmarks[name].results.get('forward_time', 0) * 1000 for name in model_names]  # ms
    memory_usage = [benchmarks[name].results.get('peak_memory_mb', 0) for name in model_names]
    model_sizes = [benchmarks[name].results.get('model_size_mb', 0) for name in model_names]
    tokens_per_sec = [benchmarks[name].results.get('tokens_per_second', 0) for name in model_names]
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Forward pass time
    ax1.bar(model_names, forward_times)
    ax1.set_title('Forward Pass Time')
    ax1.set_ylabel('Time (ms)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Memory usage
    ax2.bar(model_names, memory_usage)
    ax2.set_title('Peak Memory Usage')
    ax2.set_ylabel('Memory (MB)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Model size
    ax3.bar(model_names, model_sizes)
    ax3.set_title('Model Size')
    ax3.set_ylabel('Size (MB)')
    ax3.tick_params(axis='x', rotation=45)
    
    # Throughput
    ax4.bar(model_names, tokens_per_sec)
    ax4.set_title('Training Throughput')
    ax4.set_ylabel('Tokens/second')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=150, bbox_inches='tight')
    print("  Saved plots to benchmark_results.png")


def main():
    parser = argparse.ArgumentParser(description='Benchmark nanoKimi vs nanoGPT')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for benchmarking')
    parser.add_argument('--block_size', type=int, default=256, help='Block size for benchmarking')
    parser.add_argument('--iterations', type=int, default=50, help='Number of iterations for timing')
    parser.add_argument('--skip_plots', action='store_true', help='Skip generating plots')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("nanoKimi vs nanoGPT Benchmark")
    print("Comparing Kimi-K2 innovations with standard transformer")
    print("=" * 80)
    
    # Setup device
    device = get_device() if args.device == 'auto' else args.device
    print(f"Using device: {device}")
    
    # Create test models
    print("\\nCreating test models...")
    models = create_test_models(device)
    
    # Create benchmarks
    benchmarks = {}
    for name, model in models.items():
        benchmarks[name] = ModelBenchmark(model, device, name)
    
    # Run benchmarks
    for name, benchmark in benchmarks.items():
        print(f"\\n{'='*60}")
        print(f"Benchmarking: {name}")
        print(f"{'='*60}")
        
        # Model info
        benchmark.get_model_info()
        
        # Forward pass benchmark
        benchmark.benchmark_forward_pass(
            batch_size=args.batch_size,
            block_size=args.block_size,
            num_iterations=args.iterations
        )
        
        # Memory benchmark
        benchmark.benchmark_memory_usage(
            batch_size=args.batch_size,
            block_size=args.block_size
        )
        
        # Generation benchmark
        benchmark.benchmark_generation()
    
    # Summary
    print(f"\\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")
    
    print(f"{'Model':<25} {'Params':<12} {'FWD(ms)':<10} {'Mem(MB)':<10} {'Tok/s':<10}")
    print("-" * 80)
    
    for name, benchmark in benchmarks.items():
        r = benchmark.results
        params = f"{r.get('total_params', 0)/1e6:.1f}M"
        fwd_time = f"{r.get('forward_time', 0)*1000:.1f}"
        memory = f"{r.get('peak_memory_mb', 0):.0f}"
        tokens_per_sec = f"{r.get('tokens_per_second', 0):.0f}"
        
        print(f"{name:<25} {params:<12} {fwd_time:<10} {memory:<10} {tokens_per_sec:<10}")
    
    # Plot results
    if not args.skip_plots:
        try:
            plot_results(benchmarks)
        except Exception as e:
            print(f"\\nERROR: Could not create plots: {e}")
            print("Install matplotlib to enable plotting: pip install matplotlib")
    
    print(f"\\n{'='*80}")
    print("Benchmark Complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
