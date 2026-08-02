# nanoKimi

A minimal implementation of the architectural techniques Kimi K2 popularised — **MuonClip**
(Muon + QK-Clip), **Mixture of Experts**, and **Multi-head Latent Attention** — at 25M–200M scale,
built as a research testbed for studying how those features behave across model scale.

This is **not** a scaled-down reproduction of Kimi K2. Kimi K2 is 1.04T total parameters with 32B
active, 384 experts, 61 layers, trained on 15.5T tokens. The models here are three to four orders
of magnitude smaller and trained on a few billion tokens. What they share is the architecture, not
the scale.

## Install

```bash
uv sync
```

## Quickstart

```bash
# tokenize once
uv run -m scripts.prepare_data --dataset shakespeare --output_dir data/processed/shakespeare

# smoke run: validates the whole stack on CPU before spending GPU time
uv run -m scripts.train --config_path configs/shakespeare_smoke.yaml \
  --output_dir results/raw/260802_smoke_v1 --seed 1337

# sample from it
uv run -m scripts.generate --ckpt results/raw/260802_smoke_v1/checkpoints/ckpt_final.pt \
  --output_dir results/raw/260802_smoke_v1 --prompt "KING RICHARD:"

# export to a HuggingFace-loadable directory
uv run -m scripts.export_hf --ckpt results/raw/260802_smoke_v1/checkpoints/ckpt_final.pt \
  --output_dir exports/nanokimi-smoke
```

## Architecture

| Component | What it is |
|---|---|
| `MuonClip` | Muon (momentum + Newton-Schulz orthogonalization + RMS matching) followed by QK-Clip, which rescales per-head query/key weights whenever a head's max attention logit exceeds `tau`. Kimi K2 Technical Report, Algorithm 1. |
| `MultiHeadLatentAttention` | DeepSeek-V3 MLA: one latent `c_KV` and one rotary key `k^R` shared across all heads, so the KV cache costs `kv_lora_rank + qk_rope_head_dim` per token regardless of head count. |
| `MoELayer` | Top-k routed experts with the Switch/GShard auxiliary loss, `N * sum_i f_i * P_i`, minimised at 1.0. |

Muon runs on 2D hidden weights only; embeddings, the output head, and every 1D parameter use AdamW.

## Scaling study

Four sizes, labelled by **active** parameters per token so they are comparable to dense reference
points like GPT-2 small. Total parameters are ~3.5x larger because the MoE experts dominate.

| config | n_layer | n_embd | n_head | active | total | tokens |
|---|---|---|---|---|---|---|
| `nanokimi_25m` | 7 | 384 | 6 | 24.9M | 93.9M | 0.50B |
| `nanokimi_50m` | 14 | 384 | 6 | 49.9M | 168.4M | 1.00B |
| `nanokimi_125m` | 21 | 512 | 8 | 123.1M | 413.4M | 2.46B |
| `nanokimi_200m` | 23 | 640 | 10 | 200.5M | 685.3M | 4.01B |

`head_dim` is 64 at every size, so the MLA dimensions are identical across all four runs and only
depth and width change. Token budgets follow Chinchilla at ~20 tokens per active parameter, and
each smaller run reads a strict prefix of the same tokenized slice so data never confounds scale.

## Testing

```bash
uv run -m pytest tests/ -v
```

## Layout

See `CLAUDE.md` for the full conventions. In short: `src/nanokimi/` is the package, `scripts/` are
thin CLI entry points, `configs/` holds one YAML per run, and every run writes
`results/raw/YYMMDD_description_v1/` with its config, log, metrics and checkpoints.
