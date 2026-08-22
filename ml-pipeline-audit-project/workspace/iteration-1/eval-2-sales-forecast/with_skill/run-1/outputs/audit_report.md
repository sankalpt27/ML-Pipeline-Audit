# ML Pipeline Audit

## Summary

The pipeline is well-structured in several respects — lag features are computed causally (only from past sales), preprocessing is properly encapsulated in a `Pipeline`/`ColumnTransformer` fit only on the training set, the model/preprocessing choice (tree-based model, one-hot categoricals, no scaling) is internally consistent, and there's a sensible naive baseline for comparison. However, there is one **Critical** bug that undermines the validity of everything downstream: `train_test_split` is called with its default `shuffle=True` on time-ordered data (`sales_forecast_pipeline.py:32`), so the "test" set is not actually the future relative to the "train" set. For a next-day sales forecaster, this means the reported MAE/RMSE do not reflect real deployment performance and shouldn't be trusted until fixed. Fix the split first; the rest of the pipeline is in reasonably good shape modulo the Warnings/Notes below.

## 1. Data Understanding

The pipeline loads a daily per-store sales CSV, sorts it by date ascending, and engineers `lag_1_sales` and `lag_7_sales` per store via `groupby("store_id")["sales"].shift(...)`. Rows with missing lags (each store's first week) are dropped. The target is `sales` (continuous, regression), and features are `day_of_week`, `promo_active`, `holiday`, `lag_1_sales`, `lag_7_sales`, `store_id`.

**Findings:**
- **Critical** (`line 32`): `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)` uses the default `shuffle=True` on time-ordered data. Rows are split randomly across the whole date range rather than chronologically, so the training set can contain days *after* days that appear in the test set. This is a leakage bug for forecasting: the model effectively gets to "see" the general sales level/trend/seasonality of periods that should be strictly future relative to the test window, and the evaluation no longer reflects how the model would perform when forecasting genuinely unseen future days. **Fix:** split chronologically — e.g. pick a cutoff date (`cutoff = df["date"].quantile(0.8)`), then `train = df[df.date < cutoff]`, `test = df[df.date >= cutoff]`, or use `sklearn.model_selection.TimeSeriesSplit`. Do this per the whole dataset (a single global cutoff date across all stores) so the split mirrors how the model would actually be deployed — predicting all stores' near-future from a shared "today."
- **Note**: `store_id` is an entity column with many rows per store. Once the split is made chronological, decide explicitly whether you want one shared cutoff date across all stores (recommended — matches deployment, where you forecast all stores as of the same "today") versus a per-store cutoff; a per-store cutoff isn't wrong but should be a deliberate choice, not incidental.
- **Note**: The script doesn't show any inspection of the target's distribution, volume, or outliers (e.g., zero/negative sales days, promo-driven spikes, holiday effects). Worth a quick look before trusting any reported error metric, especially since MAE/RMSE are sensitive to a handful of large outlier days.

## 2. Data Handling Correctness

Lag features are computed with `groupby("store_id")["sales"].shift(1)` / `shift(7)` on the full (pre-split) dataframe, then rows with NaN lags are dropped.

**Findings:**
- No target-leakage issue with the lag features themselves — `shift(1)`/`shift(7)` only look backward in time relative to each row's own date, so they don't encode information from the current or future row's target. Computing them before the split is fine precisely because they're causal (unlike, say, fitting a scaler on the full dataset).
- **Note**: `shift(1)`/`shift(7)` are *positional* (previous row in the group), not calendar-based. If any store has gaps in its daily series (e.g. a closed day with no logged row), `lag_7_sales` will actually reference fewer or more than 7 calendar days back for that store, silently. Recommend reindexing each store's series to a continuous daily date range (`asfreq`/`reindex`) or joining on `date - pd.Timedelta(days=1)` / `date - pd.Timedelta(days=7)` explicitly, so the lag always means what it says.
- **Note**: There's no explicit check that `(date, store_id)` is unique before computing lags — duplicate rows for the same store/date would corrupt the shift alignment and inflate row counts. Worth a quick `df.duplicated(["date","store_id"]).sum()` check.
- Reproducibility: `random_state=42` is set on both the split and the model — good, no issue.

## 3. Feature Engineering

Categorical columns (`day_of_week`, `store_id`) are one-hot encoded via `ColumnTransformer`, with the numeric/lag columns passed through unchanged. This lives inside a `Pipeline` that's fit only on `X_train`.

**Findings:**
- No issues found with the encapsulation itself — preprocessing is correctly inside a `Pipeline`/`ColumnTransformer` and fit only on training data (via `pipeline.fit(X_train, y_train)`), so there's no scaler/encoder-fit-before-split leakage here.
- **Note**: `store_id` cardinality isn't shown in the script. If there are many stores (say, hundreds or more), one-hot encoding it will create a wide, sparse feature block that a `GradientBoostingRegressor` may not handle as efficiently as a more compact encoding (frequency encoding, or switching to `HistGradientBoostingRegressor` with native categorical support). If the store count is small (tens), this is fine as-is — just worth confirming the actual cardinality.
- No imputation step is present, which is fine as long as `promo_active`, `holiday`, and `day_of_week` have no missing values of their own — the only known source of NaNs (the lag columns) is already handled by the `dropna` upstream. Worth a quick `df[feature_cols].isna().sum()` sanity check to confirm this assumption holds.
- Scaling is correctly omitted — the model is tree-based (`GradientBoostingRegressor`), which doesn't need it. No issue.

## 4. Model Selection & Preprocessing Fit

`GradientBoostingRegressor` is used to predict a continuous `sales` target — a correct model-type-to-problem match (regression, not classification/ranking).

**Findings:**
- No issues found. The preprocessing matches what the model needs: no scaling (unnecessary for tree ensembles), one-hot encoding for categoricals (appropriate given `GradientBoostingRegressor` has no native categorical support). This is internally consistent.
- **Note**: Only `day_of_week`, `promo_active`, and `holiday` capture seasonality/calendar effects; there's no month/week-of-year feature, which could matter if sales have longer seasonal cycles beyond day-of-week. Not a defect, just a possible enhancement to consider once the split issue is fixed and you can re-baseline.

## 5. Hyperparameter Tuning

No hyperparameter search is present — `n_estimators=200` and `max_depth=3` are hardcoded on `GradientBoostingRegressor` with no `GridSearchCV`/`RandomizedSearchCV` or validation-based selection.

**Findings:**
- **Note**: Hardcoded hyperparameters aren't wrong, but there's no evidence they were chosen based on validation performance for this dataset, so the model may be leaving accuracy on the table (or mildly overfitting/underfitting). Recommend a modest `RandomizedSearchCV` (or manual sweep) over `n_estimators`, `max_depth`, and `learning_rate`, using **`TimeSeriesSplit`** rather than the default random K-fold as the CV splitter — given the temporal nature of this data, a random K-fold CV during tuning would reintroduce the same look-ahead problem flagged in Section 1.

## 6. Evaluation

The model is evaluated with MAE and RMSE against `y_test`, and compared to a naive baseline that just predicts `lag_1_sales` ("yesterday's sales").

**Findings:**
- **Critical** (ties back to Section 1): Because the train/test split is random rather than chronological, `mae_model`/`rmse_model` are computed on a test set that is temporally interleaved with the training set rather than strictly following it. This means the reported numbers are likely more optimistic than what you'd see forecasting genuinely future days in production, and they shouldn't be handed off to the team as-is until the split is fixed and the metrics are recomputed.
- Good practice, no issue: comparing against a naive "yesterday's sales" baseline (`mae_baseline`) is exactly the right sanity check for a forecasting problem — it lets you see whether the model earns its complexity. Keep this comparison after fixing the split.
- **Note**: Only a single train/test split is evaluated — no cross-validation, so there's no sense of the variance in MAE/RMSE across different time windows. Once the split is chronological, consider a rolling-origin evaluation (`TimeSeriesSplit` with multiple folds, each training on data up to some date and testing on the following window) to get a more robust estimate than one lucky/unlucky 20% slice.
- **Note**: Only aggregate MAE/RMSE are reported — there's no per-store or per-day-of-week breakdown. An aggregate number can mask stores where the model performs much worse (e.g., newer stores with less history, or stores with unusual promo patterns). Recommend at least a per-store MAE table or a residual plot before handoff.
- **Note**: Consider adding a relative-error metric (MAPE or WAPE) alongside MAE/RMSE — an absolute dollar/unit error is harder to interpret across stores with very different sales volumes than a percentage error would be, and the team will likely find that more directly actionable.

## Priority Fixes

1. **Fix the train/test split to be chronological** (`sales_forecast_pipeline.py:32`) — this is the one Critical issue, and it invalidates the currently reported MAE/RMSE. Use a date cutoff or `TimeSeriesSplit` instead of `train_test_split`'s default shuffle, then re-run the evaluation.
2. **Make the lag features calendar-based rather than positional** (guard against gaps in each store's daily series) so `lag_7_sales` always means "exactly 7 calendar days ago," not "7 rows ago."
3. Once the split is fixed, add a **`TimeSeriesSplit`-based hyperparameter search** and a **per-store/segmented error breakdown** so the team can trust both the model's tuning and its reported accuracy before it goes into production.
