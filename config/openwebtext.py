"""
OpenWebText full-run configuration for nanoKimi

This file provides a recommended starting configuration for training on OpenWebText.
Adjust to your hardware and token budget.
"""

model_config = {
    'vocab_size': 50304,
    'n_layer': 12,
    'n_head': 12,
    'n_embd': 768,
    'dropout': 0.1,
    'bias': True,
    # Kimi-K2 specifics
    'use_moe': True,
    'num_experts': 16,
    'expert_capacity': 64,
    'top_k_experts': 2,
    'use_latent_attention': True,
    'latent_dim': 256,
}

train_config = {
    'batch_size': 512,                # effective global batch (use gradient accumulation if needed)
    'block_size': 1024,
    'max_iters': 200000,
    'lr_decay_iters': 200000,
    'min_lr': 1e-5,
    'beta2': 0.98,
    'warmup_iters': 20000,
    'eval_interval': 2000,
    'log_interval': 50,
    'eval_iters': 500,
    'eval_only': False,
    'always_save_checkpoint': False,
    'init_from': 'scratch',
}

muon_config = {
    'learning_rate': 3e-4,
    'momentum': 0.925,
    'weight_decay': 0.1,
    'eps': 1e-8,
    'backend': 'torch',  # triton may not be available on all platforms
}

data_config = {
    'dataset': 'openwebtext',
    'data_dir': 'data_openwebtext',
    'num_proc': 16,
    'num_proc_load_dataset': 8,
}

system_config = {
    'device': 'auto',
    'dtype': 'bfloat16',
    'compile': True,
    'profile': False,
    'gradient_accumulation_steps': 8,
    'grad_clip': 1.0,
}

wandb_config = {
    'wandb_log': True,
    'wandb_project': 'nanokimi-openwebtext',
    'wandb_run_name': 'kimi-k2-openwebtext-12L',
    'wandb_save_artifacts': True,  # if true, caller should log artifacts in training script
}
