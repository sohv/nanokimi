# samples text from a trained nanoKimi checkpoint.
# uv run -m scripts.generate --ckpt results/raw/260802_25m_v1/checkpoints/ckpt.pt --prompt "To be, or not to be" --output_dir results/raw/260802_25m_v1

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import simple_parsing
import tiktoken
import torch

from nanokimi.model.transformer import KimiK2
from nanokimi.training.checkpoint import load_checkpoint
from nanokimi.training.schedule import get_device
from nanokimi.utils.logging import setup_logging
from nanokimi.utils.seeding import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    ckpt: str = ""
    output_dir: str = ""
    prompt: str = "\n"
    num_samples: int = 3
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 200
    seed: int = 42
    device: str = "auto"


def main() -> None:
    config = simple_parsing.parse(GenerateConfig, add_config_path_arg=True)
    if not config.ckpt or not config.output_dir:
        raise ValueError("--ckpt and --output_dir are both required")

    setup_logging(config.output_dir)
    set_seed(config.seed)

    device = get_device(config.device)
    ckpt = torch.load(config.ckpt, map_location="cpu", weights_only=False)
    model = KimiK2(ckpt["model_args"])
    load_checkpoint(config.ckpt, model)
    model.to(device).eval()

    enc = tiktoken.get_encoding("gpt2")
    ids = torch.tensor([enc.encode_ordinary(config.prompt)], dtype=torch.long, device=device)

    out_path = Path(config.output_dir) / "samples.jsonl"
    with out_path.open("w") as handle:
        for i in range(config.num_samples):
            with torch.no_grad():
                out = model.generate(ids, config.max_new_tokens, config.temperature, config.top_k)
            text = enc.decode(out[0].tolist())
            handle.write(json.dumps({"id": i, "prompt": config.prompt, "text": text}) + "\n")
            print(text)

    print(out_path)


if __name__ == "__main__":
    main()
