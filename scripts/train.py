# trains one nanoKimi model size on a token budget, writing metrics and checkpoints to output_dir.
# uv run -m scripts.train --config_path configs/nanokimi_25m.yaml --output_dir results/raw/260802_25m_v1 --seed 42

import logging

import simple_parsing

from nanokimi.training.loop import train
from nanokimi.utils.config import RunConfig
from nanokimi.utils.logging import setup_logging
from nanokimi.utils.seeding import set_seed

LOGGER = logging.getLogger(__name__)


def main() -> None:
    config = simple_parsing.parse(RunConfig, add_config_path_arg=True)
    if not config.output_dir:
        raise ValueError("--output_dir is required")

    setup_logging(config.output_dir)
    set_seed(config.seed)

    summary = train(config)

    print(config.output_dir)
    print(
        f"val_loss {summary['final_val_loss']:.4f}  ppl {summary['final_val_ppl']:.2f}  "
        f"active_params {summary['active_params']:,}  tokens {summary['tokens_seen']:,}"
    )


if __name__ == "__main__":
    main()
