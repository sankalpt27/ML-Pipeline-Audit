# Audit: `churn_pipeline.py`

Short answer: yes, something is off, and it fully explains the 96% accuracy. There's a severe target-leakage feature in the model, plus several methodology bugs (preprocessing leakage and test-set-based hyperparameter tuning) that would inflate the reported number even if the leakage feature weren't there. On top of that, accuracy is close to meaningless for a ~5%-positive-rate target. Details and fixes below, ordered roughly by how much of the 96% they explain.

## 1. Critical: target leakage via `days_since_cancellation_flag_set` (line 41)

The comment on lines 36-38 gives it away:

> `days_since_cancellation_flag_set` is only populated (non-null) once the internal "pending cancellation" flag has been set on the account, which happens right before a customer actually churns.

This field is only non-null for customers who are already in the process of churning — it's essentially a post-outcome marker, not a predictive feature. It's never excluded:

```python
feature_cols = [c for c in df.columns if c not in ("customer_id", "churned")]
```

`customer_id` and `churned` are dropped, but `days_since_cancellation_flag_set` is not, so it flows straight into `X`. A logistic regression will happily learn "non-null → churned" and get it right almost every time. This alone is very likely most of your 96%. This is the single biggest thing to fix before shipping — in production this field won't be available (or won't be predictive) at the time you actually need to score a customer, so the model will collapse in the real world.

**Fix:** drop this column entirely, or replace it with something that's legitimately available *before* the cancellation flag is set (e.g., a lagged/pre-flag version, or exclude it if no such signal exists). More generally, audit every column for "was this known at prediction time, before the outcome occurred?" — that's the standard target-leakage question and it's the one this pipeline fails.

There's also a secondary, more mundane bug tied to this same column: it's never imputed (only `monthly_charges`/`total_charges` go through `SimpleImputer`), so if it genuinely contains NaNs for the ~95% of non-churning customers, `LogisticRegression.fit` should raise a `ValueError` on missing values. If the code is actually running end-to-end for you, double check whether the NaNs are being silently coerced to something (e.g., a CSV read quirk) — either way this column needs explicit handling, not silence.

## 2. Preprocessing fit on the full dataset before the split (lines 19-29)

```python
imputer = SimpleImputer(strategy="mean")
df[[...]] = imputer.fit_transform(df[[...]])

scaler = StandardScaler()
df[[...]] = scaler.fit_transform(df[[...]])
...
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

The imputer's fill value and the scaler's mean/std are computed using the *entire* dataset, including rows that end up in the test set. That means test-set statistics leak into the transformation applied to training data (and vice versa). It's a smaller effect than #1, but it's still real leakage and will bias your test accuracy upward, especially if the classes/covariates aren't perfectly homogeneous across the eventual split.

**Fix:** split first, then `fit` the imputer/scaler only on `X_train`, and use `.transform` (not `.fit_transform`) on `X_test`. Wrapping this in an `sklearn.pipeline.Pipeline` makes this basically foolproof and is the standard way to avoid this class of bug.

## 3. Hyperparameter selection directly on the test set (lines 48-59)

```python
for C in param_grid["C"]:
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    # pick whichever C does best directly on the test set
    score = model.score(X_test, y_test)
    if score > best_score:
        best_score = score
        best_model = model
```

This picks the model that happens to score highest *on the test set itself*, then reports that same test set's accuracy as your headline metric. That's circular — you're no longer measuring generalization, you're measuring how well you can overfit to one particular 20% slice of data by trying 5 different C values against it. The `acc` printed at the end is not a valid estimate of out-of-sample performance.

Worth noting: you imported `GridSearchCV` but aren't using it — that's the tell that this loop is a manual (and leaky) reimplementation of what `GridSearchCV` does correctly.

**Fix:** use a validation split or cross-validation (e.g., `GridSearchCV` with `cv=5`, scored on a metric appropriate for imbalance — see #5) on the *training* data only to pick `C`. Touch the test set exactly once, at the very end, with the already-chosen model.

## 4. No stratification on the split (line 46)

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

With churn at ~5% positive, an unstratified split can easily produce a test set with a meaningfully different positive rate than train (or than the true population), adding noise/bias to your reported metric run-to-run. Also no `random_state`, so results aren't reproducible between runs.

**Fix:** `train_test_split(X, y, test_size=0.2, stratify=y, random_state=<fixed seed>)`.

## 5. Accuracy is the wrong metric for ~5% positive rate

Even without any leakage: a model that predicts "not churned" for every single customer gets ~95% accuracy on this data for free. A 96% accuracy number on a 5%-positive-rate target is barely better than a trivial baseline and tells you almost nothing about whether the model is actually catching churners.

**Fix:** report precision/recall/F1 for the churn class, PR-AUC (more informative than ROC-AUC under heavy imbalance), and a confusion matrix. Compare against the trivial "always predict majority class" baseline explicitly so it's obvious whether the model is adding value. If recall on churners is what the business cares about, consider `class_weight="balanced"` in `LogisticRegression` or resampling, and pick your decision threshold deliberately rather than using the default 0.5.

## 6. High-cardinality `zip_code` one-hot encoded (line 33)

`zip_code` has ~4,000 unique values and is one-hot encoded directly, adding ~4,000 sparse columns. This isn't leakage, but it's a real problem on its own:
- With this few features otherwise and this many dummy columns, you risk the model effectively memorizing zip codes associated with specific customers/outcomes in the training data (a form of overfitting distinct from #1).
- `pd.get_dummies` is applied to the whole `df` before the split, so the category set is built from both train and test — again minor, but combined with the fit-before-split pattern in #2, it's part of the same "touch the whole dataset before splitting" habit worth breaking.
- Practically, this will also blow up training time and memory for not much signal, since zip code is a weak/noisy proxy for whatever geographic signal actually matters.

**Fix:** don't one-hot-encode raw zip code at this cardinality. Options: bucket into a coarser geography (state/region, or first 2-3 digits of ZIP), target-encode with proper cross-validation, or drop it unless you've verified it adds value over a coarser feature.

## 7. Minor: mixed scaling in feature engineering (line 39)

```python
df["engagement_score"] = df["support_tickets_last_30d"] * 2 - df["tenure_months"]
```

By this point `tenure_months` has already been through `StandardScaler` (line 27-29) but `support_tickets_last_30d` hasn't been scaled at all. So this new feature is combining a standardized variable with a raw-count variable on arbitrary/incompatible scales, which makes the resulting feature's meaning hard to interpret and gives `support_tickets_last_30d` disproportionate weight relative to what you probably intended. Not a correctness bug, but worth fixing for it to actually mean what the name implies — either engineer this feature from raw values before scaling, or scale both consistently first.

## Bottom line

The 96% is not real. The `days_since_cancellation_flag_set` feature is the primary suspect and should be removed (or replaced with a pre-outcome signal) before anything else. After that, fix the fit-before-split preprocessing and the test-set-based hyperparameter loop, switch to a proper train/val/test (or CV) workflow, stratify the split, and re-evaluate using precision/recall/F1/PR-AUC instead of raw accuracy. I'd expect the honest number to drop substantially — quite possibly close to (or only modestly better than) the ~95% "always predict no-churn" baseline, which would tell you the model isn't yet adding much real signal and needs better features once the leaky one is gone.
