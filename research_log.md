# Research log

Append an entry as soon as a run finishes. Numbers come from `summary.json`, not stdout.

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
