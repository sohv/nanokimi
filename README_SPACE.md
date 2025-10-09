# Deploying nanoKimi to Hugging Face Spaces

This repository contains a demo Gradio app at `/deploy/app.py` which can be used to host a Space. Follow the steps below to create a Space and deploy the app.

1. Create a new Space on Hugging Face (choose Gradio)
2. Add the following repository files (do NOT commit large checkpoints):
   - `deploy/` (app.py, README)
   - `src/` (model code)
   - `requirements.txt`
   - `config.json` (model config)
3. In the Space settings add environment variables:
   - `NANOKIMI_MODEL_REPO` = `your-username/nanokimi-mini`
   - `HUGGINGFACE_HUB_TOKEN` = (only if the model is private)
4. Push the repo to the Space git remote and wait for the build to finish.

Example usage and notes are in `deploy/README.md`.
