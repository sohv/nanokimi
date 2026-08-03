"""Config tests.

The four scaling configs are the experiment's independent variable. If a shape drifts,
or MLA dims stop matching across sizes, the study measures something other than scale.
"""

import json
import sys
from pathlib import Path

import pytest
import simple_parsing

from nanokimi.model.transformer import KimiK2
from nanokimi.training.schedule import count_active_parameters, count_parameters
from nanokimi.utils.config import ModelConfig, RunConfig, git_hash, write_config_json

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"

# Solved against active-parameter targets with head_dim pinned to 64.
EXPECTED = {
    "nanokimi_25m": dict(n_layer=7, n_embd=384, n_head=6, active=24_925_752, tokens=500_000_000),
    "nanokimi_50m": dict(n_layer=14, n_embd=384, n_head=6, active=49_850_736, tokens=1_000_000_000),
    "nanokimi_125m": dict(n_layer=21, n_embd=512, n_head=8, active=123_090_088, tokens=2_460_000_000),
    "nanokimi_200m": dict(n_layer=23, n_embd=640, n_head=10, active=200_540_856, tokens=4_010_000_000),
}

MLA_KEYS = ("kv_lora_rank", "q_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim")


def load_config(name: str) -> RunConfig:
    argv = sys.argv
    try:
        sys.argv = ["test", "--config_path", str(CONFIG_DIR / f"{name}.yaml"), "--output_dir", "/tmp/unused"]
        return simple_parsing.parse(RunConfig, add_config_path_arg=True)
    finally:
        sys.argv = argv


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_config_shape_matches_the_solved_target(name):
    config = load_config(name)
    want = EXPECTED[name]
    assert config.model.n_layer == want["n_layer"]
    assert config.model.n_embd == want["n_embd"]
    assert config.model.n_head == want["n_head"]
    assert config.train.max_tokens == want["tokens"]


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_active_parameter_count_matches(name):
    """Sizes are labelled by active params; drift here changes what the study measures."""
    config = load_config(name)
    model = KimiK2(config.model.as_dict())
    active = count_active_parameters(model)
    want = EXPECTED[name]["active"]
    assert active == want, f"{name}: {active:,} active, expected {want:,}"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_head_dim_is_64_everywhere(name):
    """head_dim must be constant so MLA dims do not vary with model size."""
    config = load_config(name)
    assert config.model.n_embd // config.model.n_head == 64


def test_mla_dims_are_identical_across_sizes():
    """Only depth and width may change between the four runs."""
    configs = {name: load_config(name) for name in EXPECTED}
    reference = configs["nanokimi_25m"].model
    for name, config in configs.items():
        for key in MLA_KEYS:
            assert getattr(config.model, key) == getattr(reference, key), f"{name}.{key} differs"
        assert config.model.kv_cache_per_token == reference.kv_cache_per_token


def test_moe_and_optimizer_settings_are_identical_across_sizes():
    configs = {name: load_config(name) for name in EXPECTED}
    reference = configs["nanokimi_25m"]
    for name, config in configs.items():
        assert config.model.num_experts == reference.model.num_experts, name
        assert config.model.top_k_experts == reference.model.top_k_experts, name
        assert config.optimizer.qk_clip_tau == reference.optimizer.qk_clip_tau, name
        assert config.optimizer.weight_decay == reference.optimizer.weight_decay, name
        assert config.optimizer.momentum == reference.optimizer.momentum, name


def test_token_budget_follows_chinchilla():
    """~20 tokens per active parameter, the budget the plan is built on."""
    for name, want in EXPECTED.items():
        ratio = want["tokens"] / want["active"]
        assert 19.0 <= ratio <= 21.0, f"{name}: {ratio:.1f} tokens/active-param"


def test_sizes_are_monotonic_and_reasonably_spaced():
    order = ["nanokimi_25m", "nanokimi_50m", "nanokimi_125m", "nanokimi_200m"]
    actives = [EXPECTED[n]["active"] for n in order]
    assert actives == sorted(actives)
    for smaller, larger in zip(actives, actives[1:]):
        assert 1.5 <= larger / smaller <= 3.0


def test_cli_flag_overrides_yaml():
    """Precedence must be CLI flag > --config_path YAML > dataclass default."""
    argv = sys.argv
    try:
        sys.argv = [
            "test", "--config_path", str(CONFIG_DIR / "nanokimi_25m.yaml"),
            "--output_dir", "/tmp/unused", "--n_layer", "3", "--seed", "7",
        ]
        config = simple_parsing.parse(RunConfig, add_config_path_arg=True)
    finally:
        sys.argv = argv
    assert config.model.n_layer == 3
    assert config.seed == 7
    assert config.model.n_embd == 384  # untouched YAML value still applies


def test_yaml_overrides_dataclass_default():
    config = load_config("nanokimi_200m")
    assert config.model.n_layer != ModelConfig().n_layer


def test_kv_cache_property_reflects_the_attention_variant():
    mla = ModelConfig(use_latent_attention=True, kv_lora_rank=256, qk_rope_head_dim=32)
    assert mla.kv_cache_per_token == 288

    dense = ModelConfig(use_latent_attention=False, n_embd=768, n_head=12)
    assert dense.kv_cache_per_token == 12 * 64 * 2

    # MLA's cache must not grow with head count; that is the whole point
    wide = ModelConfig(use_latent_attention=True, n_head=48, n_embd=3072, kv_lora_rank=256, qk_rope_head_dim=32)
    assert wide.kv_cache_per_token == mla.kv_cache_per_token


def test_write_config_json_records_git_hash(tmp_path):
    config = RunConfig(output_dir=str(tmp_path), seed=13)
    path = write_config_json(config, tmp_path)
    payload = json.loads(path.read_text())
    assert payload["git_hash"]
    assert payload["config"]["seed"] == 13
    assert payload["config"]["model"]["n_layer"] == ModelConfig().n_layer


def test_git_hash_marks_a_dirty_tree():
    value = git_hash()
    assert value == "unknown" or len(value.split("-")[0]) == 40


def test_smoke_config_is_small_enough_to_run_on_cpu():
    config = load_config("shakespeare_smoke")
    assert config.model.n_layer <= 6
    assert config.train.max_tokens <= 2_000_000
    assert config.compile is False
