"""Run configuration and the config.json written beside every run's results."""

import json
import logging
import subprocess
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


def git_hash() -> str:
    """Current commit hash, with a -dirty suffix when the tree has uncommitted changes."""
    root = Path(__file__).resolve().parents[3]
    rev = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    )
    if rev.returncode != 0:
        return "unknown"
    commit = rev.stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True
    )
    return f"{commit}-dirty" if status.stdout.strip() else commit


@dataclass
class ModelConfig:
    """Architecture. MLA dims follow DeepSeek-V3 / Kimi K2 ratios at the given head_dim."""

    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    vocab_size: int = 50257
    block_size: int = 1024
    dropout: float = 0.0
    bias: bool = True

    use_moe: bool = True
    num_experts: int = 8
    top_k_experts: int = 2
    expert_capacity: int = 32
    apply_expert_capacity: bool = False
    load_balance_loss_coef: float = 0.01

    use_latent_attention: bool = True
    kv_lora_rank: int = 256
    q_lora_rank: int = 768
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 32
    v_head_dim: int = 64
    rope_theta: float = 50000.0
    attention_bias: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def kv_cache_per_token(self) -> int:
        """Values cached per token. For MLA this is independent of n_head."""
        if self.use_latent_attention:
            return self.kv_lora_rank + self.qk_rope_head_dim
        return self.n_head * self.head_dim * 2


@dataclass
class OptimizerConfig:
    """MuonClip. Defaults are the Kimi K2 technical report values (Algorithm 1)."""

    learning_rate: float = 2e-4
    adamw_learning_rate: float = 2e-4
    min_lr: float = 2e-5
    momentum: float = 0.95
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    ns_steps: int = 5
    rms_scale: float = 0.2
    nesterov: bool = False
    qk_clip_tau: float = 100.0
    grad_clip: float = 1.0


@dataclass
class TrainConfig:
    """Schedule and batching for one training run."""

    max_tokens: int = 0
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    warmup_iters: int = 100
    eval_interval: int = 250
    eval_iters: int = 100
    log_interval: int = 10
    checkpoint_interval: int = 1000


@dataclass
class RunConfig:
    """Top-level config for `scripts/train.py`, populated from a YAML in configs/."""

    output_dir: str = ""
    data_dir: str = "data/processed/openwebtext"
    seed: int = 42
    device: str = "auto"
    dtype: str = "bfloat16"
    compile: bool = True
    wandb_project: str = ""
    run_name: str = ""

    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _to_plain(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _to_plain(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def write_config_json(config: Any, output_dir: Path | str) -> Path:
    """Write config.json with the git hash beside a run's results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"git_hash": git_hash(), "config": _to_plain(config)}
    path = output_dir / "config.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    LOGGER.info("wrote %s", path)
    return path
