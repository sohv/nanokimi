"""
Minimal Gradio app for Hugging Face Spaces deployment

This app downloads a checkpoint from disk (or you can modify to fetch from HF Hub),
loads the model and exposes a small text generation UI.

Keep this lightweight for CPU Spaces (consider quantization for speed).
"""

import os
import gradio as gr
import torch
import pickle
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

# Ensure src package is importable
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import KimiK2

# Environment/config
MODEL_REPO = os.environ.get('NANOKIMI_MODEL_REPO', None)  # e.g. 'user/nanokimi-mini'
MODEL_FILENAME = os.environ.get('NANOKIMI_MODEL_FILE', 'pytorch_model.bin')
LOCAL_DIR = os.environ.get('NANOKIMI_LOCAL_DIR', 'out')


def ensure_ckpt_local(repo_id=None, filename='pytorch_model.bin', dest_dir='out'):
    """Ensure checkpoint is available locally. If repo_id provided, download from HF Hub."""
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    local_path = os.path.join(dest_dir, filename)
    if os.path.exists(local_path):
        return local_path
    if repo_id is None:
        return None
    try:
        # hf_hub_download returns a local cached path
        print(f"Downloading {filename} from {repo_id}...")
        hub_path = hf_hub_download(repo_id=repo_id, filename=filename)
        return hub_path
    except Exception as e:
        print('Failed to download from Hub:', e)
        return None


def load_model(device='cpu'):
    """Load model either from local disk or from HF Hub (if env provided)."""
    # Try local ckpt first
    local_ckpt = ensure_ckpt_local(repo_id=MODEL_REPO, filename=MODEL_FILENAME, dest_dir=LOCAL_DIR)
    if local_ckpt is None or not os.path.exists(local_ckpt):
        return None, f"Checkpoint not found locally and could not download from Hub (repo={MODEL_REPO})"

    try:
        ckpt = torch.load(local_ckpt, map_location=device)
    except Exception as e:
        return None, f"Failed to load checkpoint: {e}"

    # Try several keys where config/state might be stored
    model_cfg = None
    state_dict = None
    if isinstance(ckpt, dict):
        # common patterns
        model_cfg = ckpt.get('model_config') or ckpt.get('model_args') or ckpt.get('config')
        state_dict = ckpt.get('model') or ckpt.get('model_state') or ckpt.get('state_dict') or ckpt
    else:
        state_dict = ckpt

    if model_cfg is None:
        return None, "Model config missing in checkpoint"

    try:
        model = KimiK2(model_cfg)
        model.load_state_dict(state_dict)
        # move to device
        model.to(device)
        model.eval()
        return model, "ok"
    except Exception as e:
        return None, f"Failed to instantiate/load model: {e}"


# Load model (CPU by default for Spaces)
MODEL, MSG = load_model(device='cpu')


def _model_device(model: torch.nn.Module):
    """Return device where model parameters live (defaults to cpu)."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device('cpu')


def generate(prompt: str, max_tokens: int = 64, temperature: float = 1.0, top_k: int = 40):
    if MODEL is None:
        return f"Model not loaded: {MSG}"

    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding('gpt2')
    except Exception:
        enc = None

    # encode
    if enc is not None:
        ids = enc.encode_ordinary(prompt)
        x = torch.tensor([ids], dtype=torch.long)
    else:
        # fallback naive char-level encoding (not recommended)
        x = torch.tensor([[ord(c) % 256 for c in prompt]], dtype=torch.long)

    device = _model_device(MODEL)
    x = x.to(device)

    out = MODEL.generate(x, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
    if enc is not None:
        return enc.decode(out[0].tolist())
    else:
        return ''.join(chr(int(t) % 256) for t in out[0].tolist())


with gr.Blocks() as demo:
    gr.Markdown("# nanoKimi demo (toy)")
    with gr.Row():
        inp = gr.Textbox(lines=3, placeholder="Enter prompt...", label="Prompt")
        out = gr.Textbox(lines=10, label="Generated")
    with gr.Row():
        max_t = gr.Slider(1, 512, value=64, step=1, label="Max tokens")
        temp = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Temperature")
        topk = gr.Slider(1, 200, value=40, step=1, label="Top-k")
    btn = gr.Button("Generate")

    def _call(prompt, max_tokens, temp, topk):
        return generate(prompt, int(max_tokens), float(temp), int(topk))

    btn.click(_call, inputs=[inp, max_t, temp, topk], outputs=out)


def main():
    demo.launch(server_name='0.0.0.0', server_port=7860)


if __name__ == '__main__':
    main()
