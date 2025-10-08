# nanoKimi

The simplest, fastest repository for training/finetuning Kimi-K2 models.

Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT), nanoKimi implements the Kimi-K2 architecture with key innovations:
- **Muon Optimizer**: Advanced optimization technique for faster convergence
- **Mixture of Experts (MoE)**: Efficient scaling with expert routing
- **Latent Attention**: Memory-efficient attention mechanism

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/sohv/nanokimi.git
cd nanokimi

# 2. Setup environment and dependencies
python setup.py

# 3. Test the installation
python test.py

# 4. Train on toy dataset (Shakespeare)
python train.py

# 5. Generate text from trained model
python generate.py --prompt "To be or not to be"

# 6. Benchmark against standard transformer
python benchmark.py
```

## Project Structure

```
nanokimi/
├── src/
│   ├── model.py          
│   ├── optimizer.py       
│   ├── attention.py       
│   ├── moe.py            
│   └── utils.py         
├── data/
│   ├── prepare.py        
│   └── shakespeare.txt   
├── scripts/
│   ├── train.py          
│   ├── generate.py       
│   └── benchmark.py      
├── config/
│   └── default.py       
└── requirements.txt
```

## Features

- **Minimal Implementation**: Clean, readable code focusing on core concepts
- **Educational**: Well-commented code explaining Kimi-K2 innovations
- **Fast Training**: Optimized for quick experimentation
- **Benchmarking**: Direct comparison with nanoGPT

## Architecture Highlights

### Muon Optimizer
- Momentum-based optimization with adaptive learning rates
- Better convergence properties than Adam for large language models

### Mixture of Experts (MoE)
- Sparse expert routing for efficient scaling
- Top-k expert selection with load balancing

### Latent Attention
- Compressed attention representations
- Reduced memory footprint for long sequences
