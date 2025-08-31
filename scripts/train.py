#!/usr/bin/env python3
"""
Training script for nanoKimi

Train a Kimi-K2 model on the Shakespeare dataset.
"""

import os
import sys
import time
import math
import pickle
from contextlib import nullcontext

import torch
import torch.nn.functional as F

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import KimiK2
from optimizer import create_muon_optimizer
from utils import get_device, get_dtype, get_ctx, estimate_loss, get_lr, save_checkpoint, count_parameters, Timer

# Add data to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
from prepare import get_batch, prepare_shakespeare_data

# Import configurations
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
from default import model_config, train_config, muon_config, data_config, system_config, wandb_config


def main():
    print("=" * 60)
    print("nanoKimi Training Script")
    print("Training Kimi-K2 with Muon optimizer")
    print("=" * 60)
    
    # Setup directories
    out_dir = 'out'
    os.makedirs(out_dir, exist_ok=True)
    
    # Setup device and dtype
    device = get_device() if system_config['device'] == 'auto' else system_config['device']
    dtype = get_dtype(system_config['dtype'])
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")
    
    # Setup context for autocast
    ctx = get_ctx(device, dtype)
    
    # Prepare data if needed
    data_dir = data_config['data_dir']
    if not os.path.exists(os.path.join(data_dir, 'meta.pkl')):
        print("Preparing Shakespeare data...")
        prepare_shakespeare_data(data_dir)
    
    # Load meta info
    meta_path = os.path.join(data_dir, 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    # Update model config with vocab size
    model_config['vocab_size'] = meta['vocab_size']
    model_config['block_size'] = train_config['block_size']
    
    print(f"Vocabulary size: {model_config['vocab_size']}")
    print(f"Block size: {model_config['block_size']}")
    
    # Initialize model
    print("\nInitializing Kimi-K2 model...")
    model = KimiK2(model_config)
    model.to(device)
    
    # Print model info
    total_params = count_parameters(model)
    print(f"Model parameters: {total_params:,}")
    
    # Print architecture details
    print(f"Layers: {model_config['n_layer']}")
    print(f"Heads: {model_config['n_head']}")
    print(f"Embedding dim: {model_config['n_embd']}")
    print(f"Using MoE: {model_config.get('use_moe', False)}")
    print(f"Using Latent Attention: {model_config.get('use_latent_attention', False)}")
    
    # Initialize optimizer
    print("\nInitializing Muon optimizer...")
    optimizer = create_muon_optimizer(model, muon_config)
    
    # Compile model if requested
    if system_config['compile']:
        print("Compiling model with PyTorch 2.0...")
        model = torch.compile(model)
    
    # Initialize wandb if requested
    if wandb_config['wandb_log']:
        import wandb
        wandb.init(
            project=wandb_config['wandb_project'],
            name=wandb_config['wandb_run_name'],
            config={**model_config, **train_config, **muon_config}
        )
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    model.train()
    iter_num = 0
    best_val_loss = 1e9
    
    # Training metrics
    running_mfu = -1.0
    
    # Create a simple data iterator
    def get_train_batch():
        return get_batch(
            data_dir=data_dir,
            split='train',
            batch_size=train_config['batch_size'],
            block_size=train_config['block_size'],
            device=device
        )
    
    def get_val_batch():
        return get_batch(
            data_dir=data_dir,
            split='val',
            batch_size=train_config['batch_size'],
            block_size=train_config['block_size'],
            device=device
        )
    
    # Training loop
    while iter_num < train_config['max_iters']:
        # Determine learning rate
        lr = get_lr(
            iter_num,
            train_config['warmup_iters'],
            muon_config['learning_rate'],
            train_config['lr_decay_iters'],
            train_config['min_lr']
        )
        
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluate the loss on train/val sets
        if iter_num % train_config['eval_interval'] == 0 or iter_num == train_config['max_iters'] - 1:
            print(f"\nEvaluating at iter {iter_num}...")
            
            # Evaluate on validation set
            with Timer("Validation"):
                losses = {}
                model.eval()
                
                # Train loss
                train_loss = estimate_loss(model, get_train_batch, train_config['eval_iters'], device, ctx)
                losses['train'] = train_loss
                
                # Val loss
                val_loss = estimate_loss(model, get_val_batch, train_config['eval_iters'], device, ctx)
                losses['val'] = val_loss
                
                model.train()
            
            print(f"Iter {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            # Log to wandb
            if wandb_config['wandb_log']:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu*100,  # convert to percentage
                })
            
            # Save checkpoint if best validation loss
            if losses['val'] < best_val_loss or train_config['always_save_checkpoint']:
                best_val_loss = losses['val']
                if iter_num > 0:
                    save_checkpoint(model, optimizer, iter_num, best_val_loss, model_config, out_dir)
        
        # Training step
        with Timer(f"Iter {iter_num}") if iter_num % train_config['log_interval'] == 0 else nullcontext():
            # Get batch
            X, Y = get_train_batch()
            
            # Forward pass
            with ctx:
                logits, loss = model(X, Y)
                # Scale loss if using gradient accumulation
                loss = loss / system_config['gradient_accumulation_steps']
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (iter_num + 1) % system_config['gradient_accumulation_steps'] == 0:
                # Clip gradients
                if system_config['grad_clip'] != 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), system_config['grad_clip'])
                
                # Update weights
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        
        # Timing and logging
        if iter_num % train_config['log_interval'] == 0:
            # Estimate model flops utilization (MFU)
            lossf = loss.item() * system_config['gradient_accumulation_steps']
            
            print(f"Iter {iter_num}: loss {lossf:.4f}, lr {lr:.2e}")
        
        iter_num += 1
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    # Final checkpoint
    save_checkpoint(model, optimizer, iter_num, best_val_loss, model_config, out_dir)
    
    if wandb_config['wandb_log']:
        wandb.finish()


if __name__ == "__main__":
    main()
