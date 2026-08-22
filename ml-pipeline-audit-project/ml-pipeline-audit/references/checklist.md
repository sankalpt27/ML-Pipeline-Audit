# Detailed Audit Checklist

Concrete checks and code smells for each of the six stages. This is the substance of the audit — read the relevant section for whatever stage you're currently reviewing.

## Table of contents
- [1. Data understanding](#1-data-understanding)
- [2. Data handling correctness](#2-data-handling-correctness)
- [3. Feature engineering](#3-feature-engineering)
- [4. Model selection & preprocessing fit](#4-model-selection--preprocessing-fit)
- [5. Hyperparameter tuning](#5-hyperparameter-tuning)
- [6. Evaluation](#6-evaluation)
- [Code correctness (does it actually run)](#code-correctness-does-it-actually-run)
- [Stack-specific notes](#stack-specific-notes)

---

## 1. Data understanding

**What to establish first:**
- Data type(s): tabular, text, image, time series, or a mix. This determines which checks below even apply.
- Target variable: what it is, its distribution (for classification, is it balanced or skewed 95/5?), and whether the task is really classification/regression/ranking/forecasting as framed.
- Volume: enough rows for the number of features and the model complexity being used (a 200-row dataset feeding a deep net is a red flag on its own).

**Split correctness — this is where leakage most often starts:**
- The split must happen *before* anything is fit on the data (scalers, encoders, imputers, feature selectors, even things like PCA). Look for `train_test_split` or an equivalent, and check what runs before it vs. after.
- Time series / any temporally-ordered data: the split must be chronological (train on the past, test on the future), never a random shuffle. `train_test_split(..., shuffle=True)` on time-ordered data is a leakage bug — the model gets to see the future during training.
- Imbalanced targets: check for `stratify=y` in the split call, or equivalent. Its absence isn't automatically wrong but is worth flagging as a Note if the target is skewed.
- Grouped data (e.g. multiple rows per patient/user/session): a random row-level split can put the same entity's rows in both train and test, which is a subtle leakage even though no single row is duplicated. Look for `GroupShuffleSplit`/`GroupKFold` if the data has an obvious entity ID column that isn't being grouped on.

---

## 2. Data handling correctness

**Train/test contamination:**
- Exact duplicate rows across the dataset before splitting (common with scraped or merged data) — if not deduplicated first, the same row can end up in both train and test, inflating test performance.
- Any preprocessing step applied to the *full* dataframe before the split (e.g. `df['x_scaled'] = scaler.fit_transform(df[['x']])` computed once at the top of the notebook, before `train_test_split` runs). This is the single most common leakage bug in pipelines that don't use `Pipeline`/`ColumnTransformer`.
- Joins/merges that pull in data that wouldn't have been available at prediction time (e.g. joining in an aggregate computed over the whole dataset, including future or test rows).

**Target leakage** (a feature that encodes the label, directly or indirectly):
- Features computed *from* the target (e.g. a "risk score" column derived from the same outcome being predicted).
- Timestamps or IDs that happen to correlate with outcome because of how the data was collected (e.g. account status changes right after a churn event).
- A suspiciously high-importance single feature is worth a specific look — ask "would this value actually be known before I need to make the prediction?"

**Reproducibility:**
- `random_state`/seed set on the split, on the model, and on any stochastic preprocessing (e.g. SMOTE). Missing seeds isn't a correctness bug but makes results non-reproducible and hides variance between runs — flag as a Note unless the pipeline claims reproducibility.

**Train/serving skew** (only checkable if inference or serving code is available alongside training code):
- The exact feature-computation logic (imputation values, encoding categories/mappings, scaling means/stds, derived-feature formulas) should be the *same code path* at training and inference time — ideally by serializing the fitted `Pipeline`/`ColumnTransformer` itself (`joblib`/`pickle`) and loading it at serving time, rather than re-implementing the same transforms in a separate service.
- Watch for hand-duplicated logic: a feature engineered one way in the training script (e.g. a pandas `groupby`/`shift` for a lag feature) and re-implemented separately in an API handler or batch-scoring script. Any drift between the two — a different rounding rule, a different null-handling default, a stale hard-coded category list — silently changes what the model sees at serving time versus what it was trained on, and the training-time evaluation number tells you nothing about it.
- If the fitted imputer/scaler/encoder's learned statistics (means, category lists, vocab) are hard-coded as literals in serving code instead of loaded from the saved artifact, treat that as a Warning even if the values are currently correct — they'll silently go stale the next time the model is retrained.
- If only training code is available (the common case for these audits), say explicitly that train/serving skew can't be assessed rather than assuming it's fine.

---

## 3. Feature engineering

**Imputation:**
- Check what's actually missing and why (missing completely at random vs. missing with a pattern, e.g. a survey question skipped based on a previous answer). Mean/median imputation for MCAR data is fine; for structured missingness, flag if there's no missingness indicator or more deliberate handling.
- The imputer must be `fit` on training data only and `transform`-only on test/validation — same principle as scalers. `SimpleImputer` fit on the full dataframe before the split is a leakage bug.

**Categorical encoding:**
- One-hot encoding: fine for low-cardinality categoricals feeding any model. Watch for high-cardinality columns (hundreds/thousands of categories, e.g. zip code or user ID) one-hot encoded — this explodes dimensionality and is usually a bug of habit rather than intent. Ordinal, target, frequency, or hashing encoding is usually more appropriate there.
- Target encoding specifically: must be fit with cross-validation or a similar leakage-safe scheme (encode each fold using stats from the *other* folds), otherwise it leaks the target directly into the feature. `category_encoders.TargetEncoder` fit naively on the whole training set before CV is a common subtle bug — check whether it's inside the CV loop/Pipeline or applied once upfront.
- Ordinal encoding of a truly nominal (unordered) categorical silently invents a false ordering — flag as a Warning if the model is sensitive to magnitude (e.g. linear regression) rather than just splits (trees).

**Scaling/normalization:**
- Needed: linear models (linear/logistic regression, SVM), distance-based models (KNN, k-means), neural networks, anything regularized (L1/L2 penalties are scale-sensitive).
- Not needed: tree-based models (decision trees, random forest, gradient boosting/XGBoost/LightGBM/CatBoost) — scaling them isn't wrong, just unnecessary, so don't flag its absence there.
- Must be fit on train only, same leakage principle as above.

**Encapsulation (the structural fix for most of the above):**
- The strongest signal a pipeline is leakage-safe is that preprocessing lives inside an `sklearn.pipeline.Pipeline` (optionally with `ColumnTransformer` for column-specific steps) that gets `fit` inside each cross-validation fold, rather than fit once on the whole dataset upfront. If you see manual `fit_transform` calls scattered before the split instead of a `Pipeline`, that's worth flagging even if you can't point to a specific leaked value — it's a structural risk, not just a one-off bug.

---

## 4. Model selection & preprocessing fit

**Model type vs. problem:**
- Classification target treated as regression (or vice versa) — check the loss function and metric match the actual task.
- Ranking/recommendation problems trained as plain classification without any pairwise/listwise objective — not wrong, but often leaves performance on the table; worth a Note.
- Severely imbalanced classification using a plain accuracy-optimizing model with no class weighting, resampling, or threshold adjustment — flag as a Warning tied to the evaluation section.

**Preprocessing-model match:**
| Model family | Needs scaling | Needs specific encoding | Other notes |
|---|---|---|---|
| Linear/logistic regression | Yes | One-hot (avoid ordinal on nominal) | Check multicollinearity (VIF) if many correlated features |
| Tree-based (RF, GBM, XGBoost) | No | One-hot or native categorical support | Handles missing values natively in some libraries — check if imputation is even needed |
| SVM / KNN | Yes | One-hot | Very sensitive to unscaled or high-cardinality features |
| Neural networks | Yes | One-hot or embeddings | Often benefits from more deliberate missing-value handling than mean imputation |
| Naive Bayes | Usually no | Depends on variant | Check the right NB variant is used for the feature types (Gaussian vs. Multinomial vs. Bernoulli) |

Flag any mismatch as a Warning, with the specific reason it matters for that model family.

---

## 5. Hyperparameter tuning

- **Search method**: grid search is fine for small spaces, random/Bayesian (`RandomizedSearchCV`, Optuna, Hyperopt) preferred for larger ones — not itself a bug, just note if the space is large enough that grid search is impractically slow or under-covers the space.
- **Leakage in tuning**: the tuning loop's cross-validation must only ever see the training data — the held-out test set should not appear anywhere in `GridSearchCV`/`RandomizedSearchCV`'s `X`/`y` arguments or in any manual tuning loop. If any preprocessing (scaling, encoding, imputation) is fit *before* the tuning CV rather than inside the pipeline being tuned, each fold is leaking information from the other folds too — a subtler but real form of leakage.
- **Search space sanity**: parameter ranges that are implausible for the data size (e.g. `max_depth` unbounded on a 500-row dataset, inviting overfitting) or that don't cover the region where the default hyperparameters would already sit.
- **Selecting the final model**: confirm the "best" hyperparameters are chosen by the CV score on training data, not by peeking at test-set performance across multiple candidate configs (that's tuning-via-test-set, a leakage pattern that's easy to miss because it looks like "just checking a few options").

---

## 6. Evaluation

- **Metric vs. problem**: accuracy alone on an imbalanced classification problem is close to meaningless (a model that always predicts the majority class can score 95%+ while being useless) — expect precision/recall/F1/PR-AUC or a business-relevant cost-weighted metric instead. Regression should generally report more than one of {MAE, RMSE, R²} since they emphasize different error patterns.
- **Validation strategy**: a single train/test split is noisy, especially on smaller datasets — cross-validation (k-fold, or `TimeSeriesSplit` for time-ordered data) gives a more reliable estimate and a sense of variance. Flag a single-split evaluation as a Note (not necessarily wrong, but worth acknowledging the uncertainty) unless the dataset is large enough that variance isn't a real concern.
- **Baseline comparison**: is there any baseline (majority-class predictor, simple heuristic, previous production model) the new model is compared against? A reported metric with no baseline makes it impossible to tell if the model is actually good.
- **Error analysis**: for classification, is there a confusion matrix or per-class breakdown (not just an aggregate score)? For regression, are residuals examined, or just a single aggregate error number?
- **The final number's provenance**: trace the reported metric back — was it computed on data that was genuinely held out through data prep, feature engineering, model training, *and* hyperparameter tuning? If the same test set was used earlier for any exploratory decision (e.g. feature selection based on test-set performance), the final number is optimistic. This is the single most important thing to verify, because it's the number the user will actually act on.

---

## Code correctness (does it actually run)

Not an ML-methodology check — a straightforward "would this raise an exception or silently misbehave against current library versions" pass. This matters because a reported metric from a script that doesn't actually run (or that runs differently than the author thinks) is worse than no metric at all. Flag anything found under whichever stage section it belongs to, using the same severity scale: a call that will outright crash is Critical, a working-but-deprecated call is a Warning or Note depending on urgency.

**Common deprecated/removed APIs to watch for** (not exhaustive — the general check is "does this exact function signature / argument / module path still exist in current versions of the library being imported"):
- `sklearn.metrics.mean_squared_error(..., squared=False)` — the `squared` parameter was removed; use `root_mean_squared_error(...)` instead.
- `sklearn.preprocessing.OneHotEncoder(sparse=...)` — renamed to `sparse_output` in scikit-learn 1.2+.
- `pandas.DataFrame.append(...)` — removed in pandas 2.0; use `pd.concat([...])`.
- `numpy` scalar aliases (`np.float`, `np.int`, `np.bool`, `np.object`) — removed in NumPy 1.24+; use the builtin types or explicit dtypes (`np.float64`, etc).
- `sklearn.preprocessing.LabelEncoder` applied to a 2D/multi-column input — `LabelEncoder` only supports 1D; this is a real bug, not just a deprecation, and silently produces wrong output rather than raising in some pandas/sklearn version combinations.
- Keras/TF: old Keras 1/2-style arguments like `nb_epoch` instead of `epochs`, or `tf.contrib.*` (removed entirely in TF2).
- XGBoost: calling sklearn-style methods (`.predict_proba`, `.fit(X, y)`) on the native `xgboost.Booster` API, which only exposes `.predict()` and requires `xgboost.DMatrix` inputs — mixing the two APIs is a common copy-paste bug.
- Any `import` of a module/function that has been moved or split into a separate package in the version implied by other imports in the same file (e.g. mixing old and new import paths for the same library).

If you can't determine which library version the pipeline targets (no `requirements.txt`/lockfile/version pin visible), say so and flag suspect calls as "verify against your installed version" rather than asserting they're broken.

---

## Stack-specific notes

**PyTorch / TensorFlow training loops:** look for the same six things, just in different code shapes — `Dataset`/`DataLoader` split logic instead of `train_test_split`, normalization stats (mean/std) computed on train-only data before being applied to validation/test, `torch.manual_seed`/`tf.random.set_seed` for reproducibility, and whether early stopping or checkpoint selection uses the validation set correctly (not the test set).

**XGBoost/LightGBM native APIs (not via sklearn wrapper):** check `early_stopping_rounds` is evaluated against a validation set distinct from the final test set, and that categorical handling (native categorical support vs. manual encoding) is consistent between train and inference.

**Spark MLlib:** the `Pipeline`/`PipelineModel` abstraction plays the same encapsulation role as sklearn's — check stages are fit only on the training `DataFrame`, and that any `StringIndexer`/`OneHotEncoder` stages are inside the pipeline rather than applied to the full dataset upfront.

**Airflow/Kubeflow/orchestrated pipelines:** the audit is less about a single script and more about the DAG — check which task computes train/test splits and confirm no downstream task (feature engineering, tuning) has access to an artifact derived from the full, unsplit dataset. Config-driven pipelines (YAML/JSON configs) may not show fit/transform calls directly — look at what artifacts each stage reads and writes.
