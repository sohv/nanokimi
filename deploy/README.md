# nanoKimi Space

This folder contains a Gradio app to demo the nanoKimi model.

How it works:
- On startup the app will try to download the model weights from the Hugging Face Model Hub (set `NANOKIMI_MODEL_REPO` environment variable in the Space settings).
- The app will use the `src/` folder for model code.

Environment variables to set in your Space settings:
- `NANOKIMI_MODEL_REPO`: `user/model-repo` (required)
- `NANOKIMI_MODEL_FILE`: filename (optional, default `pytorch_model.bin`)
- `HUGGINGFACE_HUB_TOKEN`: (if model repo is private)

Notes:
- Do not upload large model files directly into the Space repo. Use the Hub and download at runtime.
- For faster inference use a smaller checkpoint or a GPU Space.
