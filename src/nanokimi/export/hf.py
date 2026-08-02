"""Export a nanoKimi checkpoint into a HuggingFace-loadable repo directory.

Everything is derived from the checkpoint itself, so this works unchanged across
model sizes. The CLI wrapper is `scripts/export_hf.py`.

Replaces the original uploader, which shipped a placeholder modeling file returning
random logits and a config.json whose vocab_size did not match the weights.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import torch

LOGGER = logging.getLogger(__name__)

MODELING_SRC = Path(__file__).resolve().parent / "modeling_kimik2.py"


def load_state_dict(ckpt_path: Path | str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a checkpoint and return (state_dict, model_args) with wrapper prefixes removed."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    model_args = ckpt.get("model_args", {}) if isinstance(ckpt, dict) else {}

    # torch.compile wraps the module, prefixing every key with `_orig_mod.`;
    # DDP prefixes with `module.`. Neither belongs in a published checkpoint.
    for prefix in ("_orig_mod.", "module."):
        if any(k.startswith(prefix) for k in state_dict):
            n = sum(1 for k in state_dict if k.startswith(prefix))
            state_dict = {k.removeprefix(prefix): v for k, v in state_dict.items()}
            LOGGER.info("stripped %r from %d keys", prefix, n)

    return state_dict, model_args


def infer_config(state_dict, model_args):
    """
    Build the HF config. Shapes read off the weights win over model_args, because
    scripts/train.py overrides vocab_size from the tokenizer at runtime and the
    static config files are therefore not authoritative.
    """
    vocab_size, n_embd = state_dict["transformer.wte.weight"].shape
    n_layer = 1 + max(
        int(k.split(".")[2]) for k in state_dict if k.startswith("transformer.h.")
    )

    use_moe = any(".mlp.experts." in k for k in state_dict)
    if use_moe:
        num_experts = 1 + max(
            int(k.split(".")[5]) for k in state_dict if ".mlp.experts." in k
        )
    else:
        num_experts = model_args.get("num_experts", 8)

    # MLA is identified by its LoRA projections. It uses RoPE, so there is no wpe
    # to read block_size from; that has to come from model_args instead.
    use_latent = "transformer.h.0.attn.kv_a_proj_with_mqa.weight" in state_dict   # Modified to detect MLA by its LoRA projection instead of the removed q_compress.
    if use_latent:
        kv_lora_rank = state_dict["transformer.h.0.attn.kv_a_layernorm.weight"].shape[0]  # Added: read d_c straight off the RMSNorm that sits on the latent.
        q_lora_rank = state_dict["transformer.h.0.attn.q_a_layernorm.weight"].shape[0]    # Added: read d_c' off the query-side RMSNorm.
        # kv_a_proj_with_mqa outputs kv_lora_rank + qk_rope_head_dim.
        qk_rope_head_dim = (                                                             # Added: recover the rotary width from the fused KV down-projection.
            state_dict["transformer.h.0.attn.kv_a_proj_with_mqa.weight"].shape[0] - kv_lora_rank
        )
        n_head = model_args.get("n_head")                                                # Added: head count is not recoverable from shapes alone, so it comes from model_args.
        if n_head is None:
            raise ValueError("n_head missing from checkpoint model_args; cannot infer MLA layout")
        qk_head_dim = state_dict["transformer.h.0.attn.q_b_proj.weight"].shape[0] // n_head  # Added: total per-head query width.
        qk_nope_head_dim = qk_head_dim - qk_rope_head_dim                                # Added: the non-rotary width is whatever is left.
        v_head_dim = (                                                                   # Added: recover v_head_dim from the fused KV up-projection.
            state_dict["transformer.h.0.attn.kv_b_proj.weight"].shape[0] // n_head
            - qk_nope_head_dim
        )
        block_size = model_args.get("block_size", 1024)                                  # Added: RoPE tables are not persisted, so block_size comes from model_args.
        attention_bias = "transformer.h.0.attn.q_a_proj.bias" in state_dict              # Added: detect whether MLA projections carry bias.
    else:
        block_size = state_dict["transformer.wpe.weight"].shape[0]                       # Modified: only the dense baseline still has wpe to read block_size from.
        n_head = model_args.get("n_head")
        kv_lora_rank, q_lora_rank = 256, 768
        qk_nope_head_dim, qk_rope_head_dim, v_head_dim = 64, 32, 64
        attention_bias = False

    bias = "transformer.h.0.ln_1.bias" in state_dict

    return {
        "model_type": "kimi-k2",
        "architectures": ["KimiK2ForCausalLM"],
        "auto_map": {
            "AutoConfig": "modeling_kimik2.KimiK2Config",
            "AutoModelForCausalLM": "modeling_kimik2.KimiK2ForCausalLM",
        },
        "vocab_size": vocab_size,
        "block_size": block_size,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "dropout": 0.0,  # inference default, regardless of the training value
        "bias": bias,
        "use_moe": use_moe,
        "num_experts": num_experts,
        "expert_capacity": model_args.get("expert_capacity", 32),
        "top_k_experts": model_args.get("top_k_experts", 2),
        "apply_expert_capacity": False,
        "load_balance_loss_coef": model_args.get("load_balance_loss_coef", 0.01),
        "use_latent_attention": use_latent,
        "kv_lora_rank": kv_lora_rank,                                                    # Added the MLA latent width to the exported config.
        "q_lora_rank": q_lora_rank,                                                      # Added the query LoRA width.
        "qk_nope_head_dim": qk_nope_head_dim,                                            # Added the non-rotary per-head width.
        "qk_rope_head_dim": qk_rope_head_dim,                                            # Added the rotary per-head width.
        "v_head_dim": v_head_dim,                                                        # Added the per-head value width.
        "rope_theta": model_args.get("rope_theta", 50000.0),                             # Added the RoPE base, which must match training or positions shift.
        "attention_bias": attention_bias,                                                # Added the attention bias flag.
        "tie_word_embeddings": True,
        "tokenizer": "tiktoken/gpt2",
        "torch_dtype": "float32",
    }


def write_export(state_dict: dict[str, Any], config: dict[str, Any], out_dir: Path | str) -> Path:
    """Write model.safetensors, config.json and the remote-code modeling file."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # safetensors rejects tensors that share storage. lm_head is tied to wte and
    # is restored by tie_weights() on load, so it is dropped here.
    if "lm_head.weight" in state_dict:
        wte = state_dict["transformer.wte.weight"]
        if not torch.equal(state_dict["lm_head.weight"], wte):
            raise ValueError(
                "lm_head.weight differs from transformer.wte.weight, but the config "
                "declares tie_word_embeddings=True. Refusing to drop it."
            )
        state_dict = {k: v for k, v in state_dict.items() if k != "lm_head.weight"}

    from safetensors.torch import save_file

    save_file(
        {k: v.contiguous() for k, v in state_dict.items()},
        out_dir / "model.safetensors",
        metadata={"format": "pt"},
    )
    LOGGER.info("wrote model.safetensors (%d tensors)", len(state_dict))

    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    shutil.copy2(MODELING_SRC, out_dir / "modeling_kimik2.py")
    LOGGER.info("wrote config.json and modeling_kimik2.py")
    return out_dir


def export_checkpoint(
    ckpt_path: Path | str,
    out_dir: Path | str,
    push_to: str | None = None,
    private: bool = False,
) -> Path:
    """Convert a training checkpoint into a HuggingFace repo directory, optionally uploading it."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    state_dict, model_args = load_state_dict(ckpt_path)
    config = infer_config(state_dict, model_args)
    LOGGER.info(
        "config: vocab=%d n_layer=%d n_head=%d n_embd=%d block=%d experts=%d kv_lora=%d",
        config["vocab_size"],
        config["n_layer"],
        config["n_head"],
        config["n_embd"],
        config["block_size"],
        config["num_experts"],
        config["kv_lora_rank"],
    )

    out_dir = write_export(state_dict, config, out_dir)

    if push_to:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id=push_to, private=private, exist_ok=True)
        api.upload_folder(folder_path=str(out_dir), repo_id=push_to)
        LOGGER.info("pushed to https://huggingface.co/%s", push_to)

    return out_dir
