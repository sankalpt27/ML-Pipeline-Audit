# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A Claude **skill** (`ml-pipeline-audit/`) plus the evaluation workspace used to test it (`workspace/`). It was started in a Cowork session using Anthropic's `skill-creator` workflow (draft → test → review → iterate → package) and is being continued here in VS Code with Claude Code. There is no application code to build or run — this repo *is* a prompt/instructions package for Claude, plus its test fixtures and eval results.

## Repo layout

```
ml-pipeline-audit/              # the actual skill — this is what gets packaged/installed
├── SKILL.md                    # frontmatter (name, description) + the workflow instructions
├── references/
│   └── checklist.md            # detailed per-stage checks, code smells, stack-specific notes
└── evals/
    ├── evals.json              # test prompts + expected_output descriptions (skill-creator schema)
    └── files/
        ├── churn_pipeline.py           # test fixture: tabular pipeline w/ several planted bugs
        └── sales_forecast_pipeline.py  # test fixture: mostly-clean pipeline w/ one planted bug

workspace/iteration-1/          # results from the first (only, so far) test run
├── eval-1-churn-leakage/       # with_skill/ and without_skill/ runs, each with outputs/, grading.json, timing.json
├── eval-2-sales-forecast/      # same structure
├── benchmark.json / .md        # aggregated pass-rate / time / token comparison
└── review.html                 # static side-by-side viewer (open in a browser)
```

This directory sits one level below the repo root (`../CLAUDE.md`), which points here for full detail — this file is the canonical one; keep it up to date as the source of truth.

## What the skill does

`ml-pipeline-audit/SKILL.md` gives Claude a systematic way to audit an existing end-to-end ML pipeline (pandas/scikit-learn primarily, adaptable to other stacks) — reading pipeline code/notebooks and producing a structured report. It walks six stages in order, each stage building on the last (e.g. you can't judge whether scaling is correct until you know what model it's feeding):

1. **Data understanding** — data type, target distribution, split correctness (chronological for time series, stratified for imbalance, grouped where entities repeat)
2. **Data handling correctness** — train/test contamination, target leakage, reproducibility
3. **Feature engineering** — imputation fit-on-train-only, categorical encoding vs. cardinality/model, scaling vs. model type, encapsulation in `Pipeline`/`ColumnTransformer`
4. **Model selection & preprocessing fit** — does the model match the problem, does preprocessing match the model family
5. **Hyperparameter tuning** — search method, leakage-free CV, sane search space
6. **Evaluation** — right metric for the problem/imbalance, CV vs. single split, baseline comparison, provenance of the final reported number

Alongside the six stages, the skill also does a **code-correctness pass** — deprecated/removed API usage, version-mismatched calls, and (where inference/serving code is available) train/serving skew between how features are computed at training vs. serving time. This is explicitly *not* ML methodology; it exists because iteration 1 found a real gap (a baseline run without the skill caught a removed sklearn argument that a with-skill run missed, since the skill only reasoned about methodology). Findings here get folded into whichever of the six stage sections they belong to, using the same severity tags.

The detailed checks, code smells to grep for, a model-family/preprocessing table, the deprecated-API list, and stack-specific notes (PyTorch/TF, native XGBoost/LightGBM, Spark MLlib, Airflow/Kubeflow) live in `references/checklist.md` — `SKILL.md` is deliberately kept short and points there. Read `checklist.md` before editing audit logic; it's the substance of the skill, `SKILL.md` is just the map.

Output is a fixed-template markdown report: Summary → six stage sections with severity-tagged findings (**Critical**/**Warning**/**Note**) → Priority Fixes.

## Working on the skill

There's no build/lint/test tooling here (no package.json, no Python project config) — "testing" the skill means running it against the eval fixtures and reading the output critically, not running a test suite:

1. Edit `ml-pipeline-audit/SKILL.md` and/or `references/checklist.md`.
2. Add new test fixtures under `ml-pipeline-audit/evals/files/` and prompts to `ml-pipeline-audit/evals/evals.json` if adding coverage. `evals.json` follows the skill-creator schema (`skill_name`, `evals[].{id, prompt, expected_output, files}`); assertions checked against actual runs live separately in each `workspace/iteration-N/eval-*/eval_metadata.json`, not in `evals.json`.
3. Test by hand: point a fresh Claude Code session (or `claude -p`) at the skill and one of the eval prompts, and read the output against the assertions in the corresponding `eval_metadata.json`.
4. Package for distribution: the skill just needs to be a folder with `SKILL.md` at its root (plus `references/`) — zip `ml-pipeline-audit/` *excluding* `evals/` (test-only, not part of the shipped skill), or use `python -m scripts.package_skill ml-pipeline-audit` if that skill-creator tooling is available in the environment.
5. This repo has no `.git` yet — `git init && git add . && git commit -m "..."` is the starting point whenever it's time to push to GitHub.

## Status: iteration 1 complete, awaiting review

Both eval fixtures have deliberately planted bugs so audit output can be checked against known ground truth:

- **`churn_pipeline.py`** — imputer/scaler fit on the full dataframe before the split, a near-direct target-leakage feature (`days_since_cancellation_flag_set`), hyperparameter tuning that picks its winner by scoring on the test set, a ~4,000-category column one-hot encoded, and accuracy-only reporting on a ~5% imbalanced target. All five were injected so a correct audit explains the pipeline's suspiciously high (96%) reported accuracy.
- **`sales_forecast_pipeline.py`** — otherwise solid (proper `Pipeline`/`ColumnTransformer`, correctly skips scaling for a tree model, has a baseline comparison) except `train_test_split` uses the default random shuffle on time-ordered data instead of a chronological split. A single planted bug, used to test whether the skill can tell a mostly-clean pipeline from a broken one without inventing problems.

**Results** (`workspace/iteration-1/benchmark.md`): with-skill and without-skill (baseline Claude, no skill) runs both caught every planted issue in both cases — 100% pass rate either way, at a cost of roughly +19s / +9.7k tokens per run for the skill. One notable miss: on the sales-forecast case, only the *baseline* run caught a real runtime bug (`mean_squared_error(..., squared=False)`, removed in current sklearn) — the skill has no "does this code actually execute" check, since it's entirely about ML methodology. Open `workspace/iteration-1/review.html` in a browser for the full side-by-side reports.

**Update:** the skill now includes a code-correctness pass (deprecated/removed API checks) and a train/serving-skew check under Data Handling Correctness, added directly in response to the runtime-bug miss above. Neither has been re-run against the eval fixtures yet — do that before considering this closed, since `churn_pipeline.py`/`sales_forecast_pipeline.py` don't currently have a deprecated-API bug planted to verify the new check actually fires (the sales fixture's `squared=False` call is exactly this kind of bug and would be a natural real-world check here, though it wasn't planted as a test assertion in `evals.json`).

Open questions for the next iteration:

1. Whether a tie against baseline is actually a problem, or whether the skill's value is consistency/completeness rather than catching more than baseline Claude would anyway.
2. Whether the eval fixtures need to get harder (subtler leakage with no explanatory comment, or a genuinely-clean pipeline to stress-test false positives) — and now also whether a fixture is needed to verify the new code-correctness and train/serving-skew checks actually fire.

## Conventions

- Severity tags in report bodies: **Critical** (biases results, e.g. leakage), **Warning** (likely suboptimal/fragile), **Note** (worth knowing, not necessarily wrong).
- The skill explains *why* each check matters rather than issuing bare imperatives (per skill-creator's writing guidance) — keep that tone when extending it; avoid turning it into a wall of ALL-CAPS MUSTs.
