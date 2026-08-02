# Coding guide

This is a short onboarding guide for anyone joining the project. It covers how the code is
organized, where new code and data should go, and the git workflow we follow. Project-level
conventions (Python style, logging, experiment scripts, LLM API calls) live in `CLAUDE.md` at the
repo root — read that first. This guide is about day-to-day navigation and collaboration habits.

## Navigating the code

- `src/` holds reusable code, organized by pipeline stage:
  - `src/data/` — dataset loading and preprocessing.
  - `src/generation/` — model inference, prompting, sampling.
  - `src/finetuning/` — training loops.
  - `src/interp/` — probes, features, hook points.
  - `src/metrics/` — metric computation and downstream analysis.
  - `src/utils/` — seeding, logging, the git hash and `config.json` written beside every run, and
    other shared helpers.
- `scripts/` holds entry points. Each script is a thin CLI wrapper that parses arguments and calls
  into `src/`. One script per pipeline stage, e.g. `scripts/run_generation.py`.
- `configs/` holds one YAML per experiment: model lists, hyperparameters, dataset paths.
- `notebooks/` holds de-risk exploration, never `src/` or `scripts/`. Outputs are stripped by
  nbstripout on commit, so don't rely on a saved cell result surviving.
- `data/raw/` is untouched source data. `data/processed/` is anything produced by `src/data/`.
  `data/splits/` holds fixed IDs, seeds, and eval splits saved as files.
- `results/raw/` holds run outputs and is append-only — a rerun writes a new subfolder, it never
  overwrites a previous one. Each run folder carries its `config.json` and `run.log` beside the
  results, so any number traces back to the commit that produced it. Output too large to commit
  goes in that run's `generations/` subfolder, which is gitignored — that's the escape valve when
  the large-file hook blocks a commit. `results/tables/` and `results/figures/` hold polished
  outputs for the paper.
- `docs/` holds design rationale: `experimental_design.md`, `decisions.md` (pre-registered
  thresholds and design choices), and `project_design_decisions.md` (background on the project).
  Read these before starting a new experiment so you don't repeat a discussion that's already been
  settled. `docs/project_claude_md/` holds override templates to copy into a project's own
  `CLAUDE.md` when it's a fine-tuning or interpretability project.
- `tests/` mirrors the `src/` modules it covers.
- `cache/` is the local LLM response cache. Gitignored, never shared, safe to delete — the only
  cost is re-paying for the calls.
- `research_log.md` at the repo root is the running record of every run: what was tested, what came
  back, the exact command, and the output path. Read it before starting a new experiment so you
  don't repeat one, and append to it as soon as a run finishes. This is what you'll write the paper
  and the rebuttal from.

## Deciding where new code and data go

- If logic will be called from more than one script, or by another collaborator, it belongs in
  `src/`, in the subfolder matching its pipeline stage.
- If it's a one-off script for a single run, it belongs in `scripts/`, calling into `src/` for
  anything reusable. Never hardcode one-off logic inside `src/`.
- Early, exploratory work (per `CLAUDE.md`'s de-risk mode) is fine as a notebook. Promote it into
  `src/` and `scripts/` once it needs to scale or be shared.
- New eval or metric code goes in `src/metrics/`, with a `scripts/run_<eval_name>.py` entry point.
  Its output goes in `results/raw/YYMMDD_description_v1/`, one subfolder per run.
- New datasets or generated training data go in `data/processed/`, produced by a script in
  `src/data/`. `data/raw/` is never edited directly.
- Any design choice fixed before seeing results (a threshold, a cutoff, a train/eval split rule)
  goes in `docs/decisions.md`, dated.

## Git workflow

We create a branch for each task we take on, and never commit directly to `main`. On a project with
collaborators, or with CI running on pull requests, we open a PR when the task is done, describing
what we did and why, and merge it into `main` once it's reviewed. Working solo with no CI, we merge
the branch locally — a PR you approve yourself is not a review. The habits below apply either way.

**Branch and sync habits**

- We name branches by type and topic, e.g. `feat/fake-capitals-sft`, `fix/tokenizer-mismatch`,
  `exp/dpo-extension`.
- We pull and rebase on the latest `main` before starting a new branch, so we're not working from
  stale code.
- We keep branches short-lived. A branch that lives for more than a few days is a sign the task
  should have been split up.

**Commit and PR hygiene**

- We write short, descriptive commit messages, no emoji.
- We run `pre-commit run --all-files` before every commit, not just at the end of a branch.
- We write PR descriptions that state what changed and why, not just a restatement of the diff —
  the "why" is what's useful to a reviewer or to us in three months.

**Safety nets**

- We run `git status` before staging anything, and stage files by name rather than with `git add -A`,
  so we don't accidentally commit something we didn't mean to.
- We never force-push or rewrite history on a shared branch. If a push is rejected, we pull and
  merge rather than override it.
- We never commit `.env` or any file containing a real API key. `.env.example` is the only file
  that should show a key, and only as a placeholder.
- We reload a file before editing it if anything else might have touched it — a pre-commit hook
  that reformats on commit, a rebase, another worktree, a tool editing alongside us. Saving a stale
  editor buffer over fresh changes is silent, and `git diff` before saving costs nothing.

**Command cheat sheet**

```bash
# start a new task
git checkout main
git pull
git checkout -b feat/short-topic-name

# work, then commit
git status
git add path/to/file.py
pre-commit run --all-files
git commit -m "short, descriptive message"
git push -u origin feat/short-topic-name

# with collaborators or CI: open a PR with gh
gh pr create --title "short title" --body "what changed and why"

# check on a PR
gh pr view --web
gh pr checks

# after review, merge and clean up
gh pr merge --squash
git checkout main
git pull
git branch -d feat/short-topic-name

# solo with no CI: merge locally instead of opening a PR
git checkout main
git merge --no-ff feat/short-topic-name
git branch -d feat/short-topic-name
```
