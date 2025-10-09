---
language: en
license: mit
library_name: transformers
tags:
- text-generation
- shakespeare
- transformer
- pytorch
pipeline_tag: text-generation
model_type: kimi-k2
---

# nanokimi-mini

This repository contains a nanoKimi model checkpoint trained on Shakespeare dataset.

## Model Details

- **Architecture**: 12 layers, 12 heads, 768 embedding dimension
- **Training Data**: Shakespeare dataset 
- **Features**: Mixture of Experts (8 experts), Latent Attention
- **Model Type**: Kimi-K2 (custom transformer)

## Files

- `pytorch_model.bin` - Model weights
- `config.json` - Model configuration 
- `src/` - Source code for model architecture
- `modeling_kimik2.py` - HuggingFace wrapper

## Usage

```python
import torch
import json
from huggingface_hub import hf_hub_download

# Download files
config_path = hf_hub_download(repo_id="sohv/nanokimi-mini", filename="config.json")
weights_path = hf_hub_download(repo_id="sohv/nanokimi-mini", filename="pytorch_model.bin")

# Load config and weights
with open(config_path) as f:
    config = json.load(f)

weights = torch.load(weights_path, map_location="cpu")
print("Model downloaded successfully!")
```

## License

MIT License
