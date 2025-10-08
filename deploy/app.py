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

# Ensure src package is importable
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import KimiK2


MODEL_PATH = os.environ.get('NANOKIMI_CKPT', 'out/ckpt.pt')


def load_model(checkpoint_path=MODEL_PATH, device='cpu'):
    if not os.path.exists(checkpoint_path):
        return None, f"Checkpoint {checkpoint_path} not found"

    ckpt = torch.load(checkpoint_path, map_location=device)
    model_cfg = ckpt.get('model_config') or ckpt.get('model_state', {}).get('config')
    if model_cfg is None:
        return None, "Model config missing in checkpoint"

    model = KimiK2(model_cfg)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    return model, "ok"


MODEL, MSG = load_model()


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
        import torch
        x = torch.tensor([ids], dtype=torch.long)
    else:
        # fallback naive char-level encoding (not recommended)
        x = torch.tensor([[ord(c) % 256 for c in prompt]], dtype=torch.long)

    out = MODEL.generate(x.to(MODEL.device), max_tokens=max_tokens, temperature=temperature, top_k=top_k)
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
