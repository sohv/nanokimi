"""
Utility functions for nanoKimi
"""

import torch
import torch.nn.functional as F
import math
import time
import os
import pickle
from contextlib import nullcontext


def get_device():
    """Auto-detect the best available device"""
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def get_dtype(dtype_str):
    """Convert string to torch dtype"""
    dtype_map = {
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
    }
    return dtype_map.get(dtype_str, torch.float32)


def get_ctx(device, dtype):
    """Get the appropriate context for autocast"""
    if device == 'cpu':
        return nullcontext()
    elif device.startswith('cuda'):
        return torch.amp.autocast(device_type='cuda', dtype=dtype)
    elif device == 'mps':
        return torch.amp.autocast(device_type='cpu', dtype=dtype)
    else:
        return nullcontext()


def estimate_loss(model, data_loader, eval_iters, device, ctx):
    """Estimate loss over a few batches"""
    model.eval()
    losses = torch.zeros(eval_iters)
    
    with torch.no_grad():
        for k in range(eval_iters):
            try:
                # data_loader is a function that returns (X, Y) batches
                if callable(data_loader):
                    X, Y = data_loader()
                else:
                    X, Y = next(iter(data_loader))
                X, Y = X.to(device), Y.to(device)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            except StopIteration:
                break
    
    model.train()
    return losses.mean()


def get_lr(it, warmup_iters, learning_rate, lr_decay_iters, min_lr):
    """Learning rate schedule with warmup and cosine decay"""
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def save_checkpoint(model, optimizer, iter_num, best_val_loss, config, out_dir):
    """Save model checkpoint"""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': config,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
    }
    
    checkpoint_path = os.path.join(out_dir, 'ckpt.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    return iter_num, best_val_loss


def count_parameters(model):
    """Count the number of parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model):
    """Print model information"""
    total_params = count_parameters(model)
    print(f"Model parameters: {total_params:,}")
    
    # Calculate model size in MB
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    print(f"Model size: {size_mb:.2f} MB")


class Timer:
    """Simple timer context manager"""
    def __init__(self, name="Timer"):
        self.name = name
        
    def __enter__(self):
        self.start = time.time()
        return self
        
    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.start
        print(f"{self.name}: {self.elapsed:.4f}s")


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering"""
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits
