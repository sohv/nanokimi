# CLAUDE.md — nanoKimi

Project-level rules. These override the general research-template rules where they conflict.

nanoKimi is an **engineering** project: a small model-training codebase, not an experiment built
out of LLM API calls. It implements the architectural techniques Kimi K2 popularised — MuonClip
(Muon + QK-Clip), Mixture of Experts, and Multi-head Latent Attention — at 25M–200M scale, as a
research testbed. It is not a scaled-down reproduction of Kimi K2, and nothing here should describe
it as one.

---

# What does not apply here

The general rules assume experiments made of API calls. This project makes **zero** LLM API calls,
so these sections are void:

- LLM API calls, `cached_llm_call`, `run_batch`, LiteLLM, the `cache/` directory.
- Debug/production model IDs (`claude-haiku-*`, `gpt-4o-*`).
- Derisking steps 1–2 (chat interface, few-shot prompting). The cheap first move here is the
  Shakespeare smoke run — see below.
- Mandatory `--model_id`, `--num_tasks`, `--dataset_path` CLI args.
- The `src/generation/`, `src/interp/`, `src/finetuning/`, `src/metrics/` layout. Those are research
  pipeline stages; this codebase is organised by component instead.

---

# Core rules

- All Python runs via `uv run -m ...`. Never `python -m ...`.
- Do or do not. No `try`/`except` around model, data, or training logic. A silent wrong loss curve
  is far worse than a stack trace.
- Type hints on all function signatures. `dict`, `list`, `X | None`, never `typing.Dict`.
- `LOGGER` for internal state. Reserve stdout for output paths and the final numbers a human reads.
- No decorative separators. No `print("=" * 60)`, no banners.
- Inline comments are short, lowercase, one line, and only on non-obvious logic.
- Formatting is `ruff-format`'s job. Line length 120.
- Never leave a number only in stdout. If it is printed, it is also in a structured file.

---

# Layout

```
src/nanokimi/          installable package — this is why nothing needs sys.path
  model/               attention.py (MLA) · moe.py · transformer.py
  training/            optimizer.py (MuonClip) · loop.py · schedule.py · checkpoint.py
  data/                prepare_openwebtext.py · prepare_shakespeare.py · loader.py
  export/              hf.py · modeling_kimik2.py
  utils/               config.py · logging.py · metrics.py · seeding.py
scripts/               thin CLI entry points, parse args and call into src/
configs/               one YAML per run
data/raw/ processed/   files only, never code
results/raw/YYMMDD_*/  one directory per run
tests/                 mirrors src/
```

`src/` holds reusable code; `scripts/` holds entry points. Never put a one-off in `src/`, never put
reusable logic in `scripts/`.

`src/nanokimi/export/modeling_kimik2.py` is uploaded to the Hub and loaded with
`trust_remote_code=True`. It must stay **self-contained** — it may not import from `nanokimi`. When
the architecture changes, it and `src/nanokimi/model/` change together or exports break silently.

---

# Configs

`RunConfig` in `src/nanokimi/utils/config.py` is authoritative: it defines every field, type and
default. `configs/*.yaml` seeds those defaults per run.

Precedence: explicit CLI flag > `--config_path` YAML > dataclass default.

simple_parsing flattens nested dataclasses, so nested fields take flat flags — `--max_tokens`, not
`--train.max_tokens`.

The four scaling configs differ only in `n_layer`, `n_embd`, `n_head` and `max_tokens`. `head_dim`
is 64 at every size, so the MLA dims are identical across all four and scale stays the only
variable. Do not change one size's architecture without changing all four.

---

# Runs

Every training run gets `results/raw/YYMMDD_description_v1/` containing:

- `config.json` — full config plus the git hash, written before training starts
- `run.log` — the run's log, so a crashed tmux job leaves a trace
- `metrics.jsonl` — step-level records, appended incrementally, floats at 4dp
- `summary.json` — final metrics
- `checkpoints/` — gitignored

`results/raw/` is append-only. A rerun writes a new dated directory; it never overwrites one.

Budgets are in **tokens**, not iterations, so sizes stay comparable. Each smaller model trains on a
strict prefix of the same tokenized slice — never a fresh sample.

Sizes are labelled by **active** parameters per token, which is what makes them comparable to dense
reference points like GPT-2 small. Total parameters are ~3.5x larger because the MoE experts
dominate. Report both; never report one as if it were the other.

Long runs go in tmux.

---

# Validating before spending GPU time

The cheap first move is the smoke run. It exercises the whole stack on real text on CPU:

```bash
uv run -m scripts.prepare_data --dataset shakespeare --output_dir data/processed/shakespeare
uv run -m scripts.train --config_path configs/shakespeare_smoke.yaml \
  --output_dir results/raw/YYMMDD_smoke_v1 --seed 1337
```

A healthy run shows loss falling well below `ln(50257) = 10.82`, all 8 experts receiving tokens in
every block, and weight std staying near its 0.02 init. If experts die or weights decay toward
zero, stop — that is the failure mode that wasted the first version of this project.

Never launch a multi-hour GPU run without a passing smoke run on the same commit.

---

# Testing

```bash
uv run -m pytest tests/ -v
```

Tests cover the things that were silently wrong before and must not regress:

- Newton-Schulz matches Keller Jordan's reference bit-for-bit.
- QK-Clip applies the MLA per-component rule: `q^C`/`k^C` take `sqrt(gamma)`, `q^R` takes `gamma`,
  the shared `k^R` is untouched.
- Muon only ever sees 2D hidden weights; embeddings and 1D params go to AdamW.
- Training does not collapse the weights.
- The MoE aux loss is the Switch/GShard form and is minimised at 1.0.
- HF export round-trips bit-exactly, including that no parameter is created inside `forward`.

Write the test first, confirm it fails, then implement. Tests must pass before committing.

**Non-persistent buffers computed in `__init__` are unsafe.** HuggingFace `from_pretrained`
materialises a model without re-running `__init__`, so such a buffer comes back as uninitialised
memory. This silently corrupted every RoPE position until a bit-exactness round-trip test caught
it. Build that kind of state lazily in `forward` instead.

---

# Plots

Any plot against parameter count, compute, or tokens uses **log-scaled axes** by default — that is
the whole point of a scaling study. Sentence case for titles and labels. Per-run figures go in the
run's `figures/`; polished ones in `results/figures/`.

---

# Research log

Append to `research_log.md` as soon as a run finishes, pulling numbers from `summary.json`, not
stdout:

```markdown
## YYMMDD — short description

**What:** one sentence
**Result:** one sentence, with the number
**Command:** uv run -m scripts.train --config_path ... --output_dir ...
**Output:** results/raw/YYMMDD_description_v1/
```

Include null and negative results. There is no paper pressure here; the bar is "did I learn
something true".

---

# Git

Branch per task, never commit to `main` directly. `git status` before staging; stage by name.
Short descriptive commit messages, no emoji. Run `pre-commit run --all-files` before committing.

---

# File management

`trash` not `rm`. `rg` not `grep`. `tree` not `ls`.
