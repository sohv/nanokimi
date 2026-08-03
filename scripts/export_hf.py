# converts a nanoKimi training checkpoint into a HuggingFace-loadable repo directory.
# uv run -m scripts.export_hf --ckpt results/raw/260802_25m_v1/checkpoints/ckpt.pt --output_dir exports/nanokimi-25m

import logging
from dataclasses import dataclass

import simple_parsing

from nanokimi.export.hf import export_checkpoint
from nanokimi.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    ckpt: str = ""
    output_dir: str = ""
    push_to: str = ""
    private: bool = False


def main() -> None:
    config = simple_parsing.parse(ExportConfig, add_config_path_arg=True)
    if not config.ckpt or not config.output_dir:
        raise ValueError("--ckpt and --output_dir are both required")

    setup_logging(config.output_dir)
    out_dir = export_checkpoint(
        ckpt_path=config.ckpt,
        out_dir=config.output_dir,
        push_to=config.push_to or None,
        private=config.private,
    )
    print(out_dir)
    print(f"Verify with: uv run -m pytest tests/test_hf_roundtrip.py --export-dir {out_dir}")


if __name__ == "__main__":
    main()
