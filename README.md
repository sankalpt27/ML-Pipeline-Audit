# ml-pipeline-audit

A Claude Skill that audits an existing end-to-end ML pipeline (pandas/scikit-learn primarily, adaptable to other stacks) for correctness — data leakage, mismatched preprocessing, weak evaluation practices, and deprecated/broken API usage — and produces a structured, severity-tagged findings report.

ML pipelines fail silently: a pipeline with a leakage bug still runs, still trains a model, still reports a number — and that number is almost always *better* than it should be, because the most common bugs all bias results in the optimistic direction. This skill walks a pipeline stage by stage (data → features → model → tuning → evaluation) and asks, at each step, "could this step have seen information it shouldn't have?"

## What it checks

Six stages, worked through in order since later stages depend on understanding the earlier ones:

1. **Data understanding** — target distribution, split correctness (chronological for time series, stratified for imbalance, grouped where entities repeat)
2. **Data handling correctness** — train/test contamination, target leakage, reproducibility, train/serving skew
3. **Feature engineering** — imputation fit-on-train-only, categorical encoding vs. cardinality/model, scaling vs. model type, encapsulation in `Pipeline`/`ColumnTransformer`
4. **Model selection & preprocessing fit** — does the model match the problem, does preprocessing match the model family
5. **Hyperparameter tuning** — search method, leakage-free CV, sane search space
6. **Evaluation** — right metric for the problem/imbalance, CV vs. single split, baseline comparison, provenance of the final reported number

Alongside these, it also does a lightweight **code-correctness pass** — flagging deprecated/removed API calls (e.g. a removed `sklearn` argument) that would break the pipeline outright, independent of any ML-methodology issue.

Every finding is tagged **Critical** (biases results, e.g. leakage), **Warning** (likely suboptimal or fragile), or **Note** (worth knowing, not necessarily wrong), with a concrete fix — not just "this could be better." The report ends with a **Priority Fixes** list ranking what to address first.

## Using it

- **Claude Code, via the plugin marketplace (no manual file copying)**: this repo is registered as a plugin marketplace (`.claude-plugin/marketplace.json`), so you can install it with two commands:
  ```
  /plugin marketplace add sankalpt27/ML-Pipeline-Audit
  /plugin install ml-pipeline-audit@ml-pipeline-audit-marketplace
  ```
- **Claude Code, manually**: place `ml-pipeline-audit-project/ml-pipeline-audit/` at `.claude/skills/ml-pipeline-audit/` (project- or user-level). Either way, Claude auto-triggers it whenever you share pipeline code and ask for a review — including vaguer asks like "does this look right?" or "can you check this over?"
- **claude.ai**: zip the skill folder (excluding `evals/`, which is test-only) into a `.skill` file and upload it under Settings → Capabilities → Skills.
- **API**: skills can be attached to Messages API calls the same way.

Once installed, just point Claude at a pipeline — a single file, or a directory for it to trace from raw data to reported metric across files:

> "Can you go through this churn prediction pipeline and tell me if anything looks off? I'm getting 96% test accuracy which seems too good."

## Repo layout

```
ml-pipeline-audit-project/
├── CLAUDE.md                      # full technical documentation (canonical — start here for dev context)
├── ml-pipeline-audit/              # the skill itself — this is what gets packaged/installed
│   ├── SKILL.md                    # frontmatter + the six-stage audit workflow
│   ├── references/checklist.md     # detailed checks, code smells, deprecated-API list, stack-specific notes
│   └── evals/                      # test fixtures + prompts (not part of the shipped skill)
└── workspace/iteration-1/          # benchmark results comparing with-skill vs. without-skill runs
```

See [`ml-pipeline-audit-project/CLAUDE.md`](ml-pipeline-audit-project/CLAUDE.md) for the full breakdown of each piece, how to test changes, how to package the skill, and current status/open questions.

## Status

Iteration 1 (two planted-bug test fixtures, 3 runs each with and without the skill): both configurations caught 100% of the planted issues. The skill didn't measurably out-catch baseline Claude's own judgment on these two cases, at a cost of roughly +19s / +9.7k tokens per run — see [`workspace/iteration-1/benchmark.md`](ml-pipeline-audit-project/workspace/iteration-1/benchmark.md) for the numbers and [`review.html`](ml-pipeline-audit-project/workspace/iteration-1/review.html) for the full side-by-side reports. A code-correctness check and a train/serving-skew check were added afterward in response to a gap found in that run; harder eval fixtures are needed to validate them.
