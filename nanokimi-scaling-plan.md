# nanoKimi Scaling Study — Final Plan

Personal project plan, extending [nanoKimi](https://github.com/sohv/nanokimi) (a nanoGPT-style minimal implementation of the Kimi-K2 architecture: Muon optimizer, Mixture of Experts, latent attention) into a small-scale research testbed for studying how these architectural features behave across model scale. Not a paper, not a production release — a fast-iteration testbed, released on HuggingFace once working.

Compute: not cost-constrained. Single rented H100 (80GB), sequential runs.

---

## Model sizes

**25M / 50M / 125M / 200M parameters.**

Chosen to give roughly 2-2.5x spacing between adjacent sizes (except 125→200 at ~1.6x), close enough to known reference points (GPT-2 small ≈124M, Pythia-160M) that any pattern found is easier to contextualize against the literature.

**Possible future extension to 400M**, only if the four-point results specifically motivate a fifth, larger confirming point — not trained by default. Keep the codebase config-driven (model size, MoE expert count, latent attention dims all read from config, nothing hardcoded per size) so this stays a cheap addition later rather than a redo.

## Data budget (Chinchilla-scaled, ~20 tokens/param)

| Size | Tokens |
|---|---|
| 25M | ~500M |
| 50M | ~1B |
| 125M | ~2.5B |
| 200M | ~4B |

Take a single fixed 4B-token slice of OpenWebText, tokenize once, and use prefixes of it for the smaller runs — each smaller model's data is a strict subset of the larger ones', not a separate random sample, so any observed difference across sizes isn't confounded by different data.

## Compute

Single H100 (80GB), rented from a provider like Lambda Labs or RunPod, kept running through all four training phases plus sanity checks, then torn down. All four sizes fit comfortably on VRAM, even with MoE's extra expert-weight overhead and generous batch sizes — no need for multiple instances or parallel training, though nothing stops sequential-vs-parallel if preferred later. Use bf16 mixed precision throughout.

---

## Phase 0 — Fix the HF import bug

Current issue: shape mismatch preventing `AutoModel`/`AutoConfig` import from HuggingFace.

1. Reproduce the exact error with a minimal repro script.
2. Diagnose: most likely either a `config.json` field mismatch (something nanoKimi's config doesn't set that HF's loader expects) or a `state_dict` key naming mismatch between what's saved and what's expected on load.
3. Fix so `AutoModel.from_pretrained()` and `AutoConfig.from_pretrained()` work cleanly on a locally saved checkpoint.
4. Add a regression test: save → reload → forward pass → compare logits, so this doesn't silently break again.

This also doubles as release-format prep for Phase 5.

## Phase 1 — Data pipeline

5. Pull a 4B-token slice of OpenWebText via HF's streaming API (`Skylion007/openwebtext` or equivalent).
6. Tokenize once using nanoKimi's existing tokenizer (tiktoken/gpt2 per its config).
7. Save as a flat binary/memmap file (nanoGPT-style `.bin`) so training doesn't re-tokenize per run.
8. Hold out a small validation slice (~1-2M tokens) from the same distribution, used identically across all four sizes.

## Phase 2 — Training config

9. Lock what scales with model size (learning rate via a standard scaling rule, batch size, warmup steps) vs. what stays fixed across sizes (MoE expert count/top-k, latent attention compression dim, dropout, weight decay). Write one config file per size before training starts — no ad hoc tuning mid-run.
10. Fix the random seed strategy (same seed for init/data order where architecturally possible), so differences across sizes are attributable to scale, not randomness.

## Phase 3 — Compute setup

11. Provision the H100 instance.
12. Set up environment, confirm nanoKimi's data loader and MoE routing implementation actually utilize the GPU well at these small sizes — watch `nvidia-smi` during the first few minutes of the 25M run as a quick check before committing to the longer runs.

## Phase 4 — Train and sanity-check (sequential)

13. Train 25M. On completion: check the loss curve is sane, reload via the now-fixed HF path, generate a sample, confirm it's not gibberish.
14. Repeat for 50M, then 125M, then 200M — same checks each time, don't launch the next size until the current one passes.
15. Log training curves (loss, and any easy internal metrics like router entropy) for all four — useful for the model card and possibly the exploration phase.

## Phase 5 — Release to HuggingFace

16. Write one model card template: architecture, size, training data description (OpenWebText slice + token count), "research testbed, not production-grade" framing, known limitations, compute/data details from Phases 1-2.
17. Adapt the template per size, push all four together as a coherent set (e.g. `nanokimi-25m`, `nanokimi-50m`, `nanokimi-125m`, `nanokimi-200m`).

## Phase 6 — Exploration (time-boxed, 3-4 sessions)

18. Build hooks at the components that actually distinguish this architecture: MoE router (pre- and post-top-k gating logits), latent attention's compressed representation, and residual stream at each block boundary.
19. Log activations across all four sizes on a shared small evaluation set, so anything noticed can be checked for whether it holds across scale or is size-specific.
20. Run a handful of patching experiments once something in the logs looks worth a causal check.
21. Cap at 3-4 sessions. At the end, write down every candidate pattern noticed, however small — this list, not a finding, is the output of this phase.

## Phase 7 — Formalize a question

22. Pick one candidate from the list. Bring it back for a proper stress-test / red-team pass before committing real experiment time — the same rigor applied to other research projects, even without paper pressure.

## Phase 8 — Run and log

23. Run the formalized experiment(s) against the four checkpoints.
24. Log everything in the running project doc — dated entries, what was tried, what was found, including null/negative results, since there's no pressure to hide those here.

---

## Standing reminders

- Run a null/sanity check (untrained or randomly initialized version of the same hook point) before trusting any pattern noticed during exploration — most "interesting" activation patterns also show up in untrained models.
- Keep the codebase config-driven so a future 400M extension is cheap if genuinely motivated by the four-size results.
- No paper, no production pressure — the bar is "did I learn something true," not "is this publishable."
