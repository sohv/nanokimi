"""
Data preparation for nanoKimi

Downloads and prepares the Shakespeare dataset for training.
Creates tokenized training and validation splits.
"""

import os
import pickle
import requests
import tiktoken
import numpy as np
from pathlib import Path


def download_shakespeare(data_dir):
    """Download the tiny Shakespeare dataset"""
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    
    input_file_path = data_dir / 'shakespeare.txt'
    
    if not input_file_path.exists():
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        print(f"Downloading {data_url}")
        
        response = requests.get(data_url)
        response.raise_for_status()
        
        with open(input_file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"Downloaded to {input_file_path}")
    else:
        print(f"Dataset already exists at {input_file_path}")
    
    return input_file_path


def prepare_shakespeare_data(data_dir='data', encoding_name='gpt2'):
    """
    Prepare the Shakespeare dataset for training
    
    Args:
        data_dir: directory to store data files
        encoding_name: tiktoken encoding to use ('gpt2', 'cl100k_base', etc.)
    """
    data_dir = Path(data_dir)
    
    # Download the dataset
    input_file = download_shakespeare(data_dir)
    
    # Load the text
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.read()
    
    print(f"Length of dataset in characters: {len(data):,}")
    
    # Get the tokenizer
    enc = tiktoken.get_encoding(encoding_name)
    
    # Encode the text
    train_ids = enc.encode_ordinary(data)
    print(f"Length of dataset in tokens: {len(train_ids):,}")
    
    # Create train/val split
    n = len(train_ids)
    train_ids = train_ids[:int(n*0.9)]
    val_ids = train_ids[int(n*0.9):]
    
    print(f"Train tokens: {len(train_ids):,}")
    print(f"Val tokens: {len(val_ids):,}")
    
    # Save to binary files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    
    train_file = data_dir / 'train.bin'
    val_file = data_dir / 'val.bin'
    
    train_ids.tofile(train_file)
    val_ids.tofile(val_file)
    
    print(f"Saved train data to {train_file}")
    print(f"Saved val data to {val_file}")
    
    # Save meta information
    meta = {
        'vocab_size': enc.n_vocab,
        'encoding_name': encoding_name,
        'train_tokens': len(train_ids),
        'val_tokens': len(val_ids),
    }
    
    meta_file = data_dir / 'meta.pkl'
    with open(meta_file, 'wb') as f:
        pickle.dump(meta, f)
    
    print(f"Saved meta info to {meta_file}")
    print(f"Vocab size: {meta['vocab_size']}")
    
    return meta


class ShakespeareDataset:
    """
    Simple dataset class for Shakespeare data
    """
    
    def __init__(self, data_dir='data', split='train', block_size=256):
        self.data_dir = Path(data_dir)
        self.split = split
        self.block_size = block_size
        
        # Load meta info
        meta_file = self.data_dir / 'meta.pkl'
        if meta_file.exists():
            with open(meta_file, 'rb') as f:
                meta = pickle.load(f)
            self.vocab_size = meta['vocab_size']
        else:
            self.vocab_size = 50304  # Default GPT-2 vocab size
        
        # Load data
        data_file = self.data_dir / f'{split}.bin'
        if not data_file.exists():
            raise FileNotFoundError(f"Data file {data_file} not found. Run prepare_data() first.")
        
        self.data = np.memmap(data_file, dtype=np.uint16, mode='r')
        print(f"Loaded {split} data: {len(self.data):,} tokens")
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        """Get a batch of data"""
        # Get a chunk of data
        chunk = self.data[idx:idx + self.block_size + 1]
        
        # Split into input and target
        x = chunk[:-1].astype(np.int64)
        y = chunk[1:].astype(np.int64)
        
        return x, y


def get_batch(data_dir='data', split='train', batch_size=64, block_size=256, device='cpu'):
    """
    Get a batch of data from the dataset
    """
    dataset = ShakespeareDataset(data_dir, split, block_size)
    
    # Generate random indices
    ix = np.random.randint(0, len(dataset), size=(batch_size,))
    
    # Get data
    x = np.stack([dataset[i][0] for i in ix])
    y = np.stack([dataset[i][1] for i in ix])
    
    # Convert to tensors
    import torch
    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)
    
    return x, y


if __name__ == "__main__":
    # Prepare the data when run as script
    meta = prepare_shakespeare_data()
    print("\nData preparation complete!")
    print(f"Vocabulary size: {meta['vocab_size']}")
    print(f"Training tokens: {meta['train_tokens']:,}")
    print(f"Validation tokens: {meta['val_tokens']:,}")
    
    # Test the dataset
    dataset = ShakespeareDataset()
    print(f"\nDataset length: {len(dataset):,}")
    
    # Show a sample
    x, y = dataset[0]
    print(f"Sample input shape: {x.shape}")
    print(f"Sample target shape: {y.shape}")
    print(f"First 10 tokens: {x[:10]}")
