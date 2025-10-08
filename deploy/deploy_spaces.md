# Deploying nanoKimi to Hugging Face Spaces

This folder contains a minimal Gradio app and requirements for deploying a small nanoKimi model to a Hugging Face Space.

Steps:
1. Add the trained checkpoint `out/ckpt.pt` to the repository or upload it to the Hugging Face Hub and set `NANOKIMI_CKPT` env var to the file path.
2. Ensure `deploy/requirements.txt` lists any packages you need (Gradio, tiktoken, torch). For CPU-only Spaces, consider quantizing the model to reduce memory.
3. Push the repository to a new Space and set the `NANOKIMI_CKPT` secret (or include the model in repo if small).
4. The app runs `deploy/app.py` by default. Adjust to download from HF Hub if you prefer.

Notes:
- Free Spaces are CPU-only and have memory/time limits. Use a small model or quantized weights.
- For large checkpoints, prefer storing model on the Hub and downloading at runtime.
