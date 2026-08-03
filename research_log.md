# Research log

Append an entry as soon as a run finishes. Numbers come from `summary.json`, not stdout.

## 260803 — systematic test pass caught two real bugs before GPU time

**What:** Extended coverage from the model/optimizer into data, config, training and numerics —
89 tests across five files, plus a `scripts/preflight.py` that runs the real config for a few steps
on the target machine and reports throughput, memory and health.

**Result:** two genuine bugs, both silent and both fatal to the study.
(1) `get_batch` fell back to `np.random.default_rng()` with no seed, so `set_seed` never governed
data order and no run was reproducible — directly violating the Phase 2 requirement that data order
be fixed across sizes. Training now uses an explicitly seeded generator, with a separate stream for
eval so eval cadence cannot shift which tokens a run trains on.
(2) The MoE layer crashed under bf16 autocast: `index_add_` refuses a fp32 destination with a bf16
source. All four scaling configs specify `dtype: bfloat16`, so every H100 run would have died on its
first forward. The Shakespeare smoke run missed it because it uses float32 on CPU, where autocast is
a no-op. Also confirmed `torch.compile` matches eager on the MLA path, which had never been tested.

**Command:** `uv run -m pytest tests/ -q`

**Output:** 89 passed. Preflight on this Mac projects 121 h for the 25M budget, which is the argument
for the H100.

## 260802 — tokenized the fixed 4B-token OpenWebText slice

**What:** Phase 1. Streamed `Skylion007/openwebtext` (8.01M docs, parquet), shuffled with seed 42,
and tokenized once with tiktoken gpt2 into flat uint16 .bin files. The validation slice is drawn
first off the same shuffled stream so it matches the training distribution but never appears in any
model's training prefix.

**Result:** 4,010,276,740 train tokens (8.02 GB) and 2,029,483 val tokens. Sustained ~1.5M tokens/s,
~35 min wall clock. Validated: file sizes match meta.json exactly; max token id 50256 < vocab 50257
so uint16 is safe; 1,135 tokens/doc average, matching OpenWebText's known figure; decoded samples
from both splits are clean English; no train/val overlap; all four size budgets
(0.50B / 1.00B / 2.46B / 4.01B) fit as strict prefixes of the same train.bin.

**Command:** `uv run -m scripts.prepare_data --dataset openwebtext --output_dir data/processed/openwebtext --max_tokens 4_010_000_000 --val_tokens 2_000_000 --seed 42`

**Output:** `data/processed/openwebtext/` (gitignored; train.bin, val.bin, meta.json)

## 260802 — Shakespeare smoke run on the rebuilt stack

**What:** first end-to-end run after replacing the fake Muon with real MuonClip, the fake latent
attention with DeepSeek-V3 MLA, and the MSE routing loss with the Switch/GShard auxiliary loss.
250 iters, 4 layers, 256 embd, 8 experts, on tiny-shakespeare.

**Result:** healthy. Loss 10.88 -> 4.80 against a random baseline of ln(50257) = 10.82; all 8
experts received tokens in every block (7-36% spread around the 25% ideal); weight std stayed at
0.024-0.039 against a 0.02 init. Samples are recognisable English with Shakespearean structure.
Max attention logit rose to ~20 and plateaued, so QK-Clip never fired at this scale.

**Command:** `uv run -m scripts.train --config_path configs/shakespeare_smoke.yaml --output_dir results/raw/260802_smoke_v1 --seed 1337`

**Output:** `results/raw/260802_smoke_v1/`

## 260802 — the previously published checkpoint is collapsed

**What:** audited the weights of `sohv/nanokimi-mini` (520,657,248 params) while fixing the
HuggingFace import path.

**Result:** negative, and the reason the rebuild happened. 541 of 568 tensors have std < 1e-4;
attention, expert and gate matrices sit at ~8e-7 against a 0.02 init. Every `ln_1.weight` decayed
from 1.0 to a uniform 0.1766. The router had collapsed to 2 of 8 experts in nearly every block.
Loss on in-distribution text was 7.31 (ppl ~1488). Three causes: `q_compress` was created lazily
inside `forward` so it never entered the optimizer, the "Muon" optimizer was Adam with L2 folded
into the normalized update, and the MoE aux loss could not see the hard dispatch decision. The
checkpoint is not a usable baseline and the architecture has since changed, so it cannot be loaded
by the current code at all.

**Output:** n/a — audit, not a run.
