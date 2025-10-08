"""
Default configuration for nanoKimi training
"""

# Model configuration
model_config = {
    'vocab_size': 50304,  # GPT-2 vocab size (50257) rounded up for efficiency
    'n_layer': 12,
    'n_head': 12,
    'n_embd': 768,
    'dropout': 0.0,
    'bias': True,  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    
    # Kimi-K2 specific
    'use_moe': True,
    'num_experts': 8,
    'expert_capacity': 32,
    'top_k_experts': 2,
    'use_latent_attention': True,
    'latent_dim': 64,
}

# Training configuration
train_config = {
    'batch_size': 64,
    'block_size': 256,
    'max_iters': 5000,
    'lr_decay_iters': 5000,
    'min_lr': 6e-5,
    'beta2': 0.99,
    'warmup_iters': 100,
    'eval_interval': 250,
    'log_interval': 10,
    'eval_iters': 200,
    'eval_only': False,
    'always_save_checkpoint': True,
    'init_from': 'scratch',  # 'scratch' or 'resume' or 'gpt2*'
}

# Muon optimizer configuration
muon_config = {
    'learning_rate': 6e-4,
    'momentum': 0.95,
    'weight_decay': 0.1,
    'eps': 1e-8,
    'backend': 'triton',  # 'triton' or 'torch'
}

# Data configuration
data_config = {
    'dataset': 'shakespeare',
    'data_dir': 'data',
    'num_proc': 8,
    'num_proc_load_dataset': 4,
}

# System configuration
system_config = {
    'device': 'auto',  # 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc., or try 'mps' on macbooks
    'dtype': 'bfloat16',  # 'float32', 'bfloat16', or 'float16'
    'compile': True,  # use PyTorch 2.0 to compile the model to be faster
    'profile': False,  # use pytorch profiler
    'gradient_accumulation_steps': 1,
    'grad_clip': 1.0,
}

# Wandb logging
wandb_config = {
    'wandb_log': True,
    'wandb_project': 'nanokimi',
    'wandb_run_name': 'kimi-k2-toy',
}
