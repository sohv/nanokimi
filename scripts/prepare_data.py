# tokenizes a dataset once into flat uint16 .bin files that training memory-maps.
# uv run -m scripts.prepare_data --dataset openwebtext --output_dir data/processed/openwebtext --max_tokens 4_000_000_000

import logging
from dataclasses import dataclass

import simple_parsing

from nanokimi.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)


@dataclass
class PrepareConfig:
    dataset: str = "openwebtext"
    output_dir: str = ""
    max_tokens: int = 4_000_000_000
    val_tokens: int = 2_000_000
    num_proc: int = 8
    seed: int = 42


def main() -> None:
    config = simple_parsing.parse(PrepareConfig, add_config_path_arg=True)
    if not config.output_dir:
        raise ValueError("--output_dir is required")

    setup_logging(config.output_dir)

    if config.dataset == "openwebtext":
        from nanokimi.data.prepare_openwebtext import prepare

        meta = prepare(
            output_dir=config.output_dir,
            max_tokens=config.max_tokens,
            val_tokens=config.val_tokens,
            num_proc=config.num_proc,
            seed=config.seed,
        )
    elif config.dataset == "shakespeare":
        from nanokimi.data.prepare_shakespeare import prepare

        meta = prepare(output_dir=config.output_dir)
    else:
        raise ValueError(f"unknown dataset {config.dataset!r}, expected 'openwebtext' or 'shakespeare'")

    print(config.output_dir)
    print(f"train tokens: {meta['train_tokens']:,}  val tokens: {meta['val_tokens']:,}")


if __name__ == "__main__":
    main()
