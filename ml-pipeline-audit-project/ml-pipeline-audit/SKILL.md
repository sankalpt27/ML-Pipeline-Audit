---
name: ml-pipeline-audit
description: Audits an existing end-to-end ML/data-science pipeline (data loading, feature engineering, model training, hyperparameter tuning, evaluation) for correctness, data leakage, mismatched preprocessing, and weak evaluation practices, then produces a structured findings report. Use this skill whenever the user shares or points to pipeline code, a notebook, or a training script and asks you to review, audit, sanity-check, or find bugs in it — including vaguer asks like "does this look right", "can you check this over", "is there anything wrong with my model", or questions specifically about data leakage, imputation, categorical encoding, feature scaling, hyperparameter tuning, or evaluation metrics. Trigger even if the user only pastes code with no explicit request to "audit" — reviewing an ML pipeline for these issues is exactly what this skill is for.
---

# ML Pipeline Audit

## Why this skill exists

Machine learning pipelines fail silently. A pipeline with a bug in it still runs, still trains a model, still reports a number — and that number is often *better* than it should be, because the most common bugs (leakage, contamination, mismatched preprocessing) all bias results in the optimistic direction. Nobody gets an error message when their test accuracy is fake. The only way to catch this is to deliberately walk the pipeline stage by stage and ask, at each step, "could this step have seen information it shouldn't have?" and "does this step match what the model actually needs?"

This skill gives you that walk. It's organized around six stages that roughly mirror how any tabular/text/image ML pipeline is built, from raw data to reported metric. The goal of an audit is not to rewrite the user's pipeline — it's to read it carefully, form a specific, evidence-backed opinion about each stage, and hand back a report that tells them exactly what to check and why.

## Before you start: gather the pipeline

Find and read the actual code — training scripts, notebooks, `Pipeline`/`ColumnTransformer` definitions, config files, and anything that shows the data being loaded and split. If the user pasted a single file, work with that. If they pointed you at a repo or directory, look for the full path from raw data to final metric: data loading → preprocessing/feature engineering → train/test split → model definition → hyperparameter search → evaluation. A partial view leads to a partial (and sometimes wrong) audit, so if you can't find one of the six stages below, say so explicitly in the report rather than silently skipping it or assuming it's fine.

If the pipeline uses a framework other than pandas/scikit-learn (PyTorch/TensorFlow training loops, Spark MLlib, Airflow/Kubeflow DAGs, XGBoost/LightGBM native APIs), the same six questions still apply — only the code patterns you're looking for change. `references/checklist.md` has stack-specific notes; read it if the pipeline isn't plain sklearn.

## The six stages

Work through these in order — later stages are hard to judge correctly if you haven't understood the earlier ones (e.g. you can't judge whether scaling is correct until you know what model it's feeding). For each stage, `references/checklist.md` has the detailed checks, concrete code smells to grep for, and worked examples of bugs vs. correct patterns. Read it before writing findings — it's the substance of the audit, this file is just the map.

1. **Data understanding** — What kind of data is this (tabular / text / image / time series / mixed), how much of it is there, what's the target and its distribution, and is the train/test/validation split itself sound (done before any fitting, no leakage from the future in time-ordered data, stratified where the target is imbalanced)?

2. **Data handling correctness** — Beyond the split, is there train/test contamination anywhere (duplicate rows across splits, joins that pull in future or held-out information), target leakage (a feature that's a proxy for or derived from the label), are results reproducible (seeded randomness, deterministic splits), and — if inference/serving code is available — does feature computation at serving time actually match what ran during training (train/serving skew)?

3. **Feature engineering** — Is the feature extraction sensible for the data type? Does the imputation strategy match the actual pattern of missingness, and is it *fit* only on training data? Is categorical encoding (one-hot vs. ordinal vs. target/frequency encoding) appropriate for the cardinality and the downstream model? Is scaling applied correctly and only where the model needs it? And critically — are all of these steps encapsulated so they can't leak (a `Pipeline`/`ColumnTransformer`, or at minimum `fit` on train and `transform`-only on test), rather than being applied to the whole dataset before splitting?

4. **Model selection & preprocessing fit** — Does the model type match the problem (classification/regression/ranking/forecasting), and do the earlier preprocessing choices actually match what *this* model needs? Tree-based models don't need scaling but do need sensible encoding; linear and distance-based models need scaling and should be checked for multicollinearity; neural nets need scaling and often benefit from different missing-value handling than trees do.

5. **Hyperparameter tuning** — What search method is used (grid/random/Bayesian), and is the search itself leak-free — tuned via proper cross-validation on the training data only, with the held-out test set never touched until the very end? Is the search space actually sensible for the model and problem size?

6. **Evaluation** — Do the metrics match the problem type and what actually matters for the business goal (not just accuracy on an imbalanced classification problem), is cross-validation used where a single split would be noisy, is there a baseline to compare against (majority class, simple heuristic, previous model), and does the final reported number come from data the model and the tuning process never touched?

## Also check: does the code actually run

This is separate from the six stages above — it's not about ML methodology, it's about whether the pipeline executes at all against the library versions it imports. A pipeline that crashes, or silently changes behavior because an argument was renamed or removed, has no reported number worth auditing. Skim for deprecated/removed APIs and version-mismatched calls (`references/checklist.md` has a short list of common ones across sklearn/pandas/numpy/TF/XGBoost) as you read through each stage. Report anything found under whichever stage section it belongs to, with the same severity tags — a call that will outright crash is Critical, a deprecation warning that still works today is a Note.

## Writing the report

Produce a single report, organized by the six stages above (skip a stage's section only if that stage genuinely isn't present in what you were given — say so rather than omitting silently). For each stage:

- State briefly what the pipeline actually does at that stage (so the user can confirm you read it right).
- Flag concrete issues, each with a severity — **Critical** (biases results, e.g. leakage), **Warning** (likely suboptimal or fragile, e.g. mismatched preprocessing for the model type), or **Note** (worth knowing, not necessarily wrong) — and point at the specific line, cell, or code snippet.
- Give a concrete recommendation for each flagged issue, not just "this could be better." If you're proposing a fix, say what it is (e.g. "fit the `StandardScaler` inside the `Pipeline` so it's refit per CV fold" beats "watch out for leakage").

End with a short summary: the most important 1-3 things to fix first (usually the Critical items, since those are the ones quietly inflating the reported metric), and anything you couldn't assess because the code wasn't available.

Use this shape for the report itself:

```markdown
# ML Pipeline Audit

## Summary
[2-4 sentences: overall assessment, and the top priority fixes]

## 1. Data Understanding
[what the pipeline does] 
**Findings:** [severity-tagged issues, or "No issues found."]

## 2. Data Handling Correctness
...

## 3. Feature Engineering
...

## 4. Model Selection & Preprocessing Fit
...

## 5. Hyperparameter Tuning
...

## 6. Evaluation
...

## Priority Fixes
1. ...
2. ...
```

Be direct about severity — a pipeline with real leakage should read as urgent, not hedged. But don't invent problems to seem thorough: if a stage genuinely looks correct, say so plainly and move on. A report that's all red flags is as unhelpful as one that rubber-stamps everything; the value is in an accurate, differentiated read of what actually needs attention.
