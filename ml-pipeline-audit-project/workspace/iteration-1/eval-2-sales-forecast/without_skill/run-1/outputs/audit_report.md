# Review: `sales_forecast_pipeline.py`

**Scope:** sanity check of the modeling approach for a next-day store sales forecaster, ahead of team handoff.

**Bottom line:** The feature engineering (lags, calendar flags) and the instinct to benchmark against a naive baseline are both good. But there is one critical bug that will make the reported accuracy numbers meaningless for a forecasting use case, plus a code bug that will crash on current sklearn. I'd fix both before handing this off — the rest are hardening suggestions, not blockers.

---

## Critical issues

### 1. Random train/test split invalidates the evaluation (data leakage via time order)

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

`train_test_split` shuffles by default. Even though the data is loaded and sorted by date, that ordering is discarded at split time — `X_test` ends up as a random 20% sample of dates scattered throughout the whole history, interleaved with the training dates rather than held out as a genuine future period.

Why this matters for a forecasting model specifically: the model isn't being asked "how well do you predict a period you've never seen," it's being asked "how well do you interpolate between days you've already seen on either side." Sales are autocorrelated day-to-day and week-to-week, so a training set that contains days immediately before and after each test day makes the problem much easier than real deployment, where you only ever have the past to predict the future. The MAE/RMSE numbers this script prints are very likely optimistic — possibly substantially so — relative to what you'll see in production.

**Fix:** do a temporal split, not a random one. E.g.:

```python
cutoff = df["date"].quantile(0.8)  # or a fixed date
train_mask = df["date"] <= cutoff
X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[~train_mask], y[~train_mask]
```

or `train_test_split(X, y, test_size=0.2, shuffle=False)` on the already-sorted frame (equivalent as long as `df` stays sorted by date, but the explicit date cutoff is clearer and safer). For a more robust estimate, use `sklearn.model_selection.TimeSeriesSplit` to get several forward-chaining folds rather than a single split — with per-store panel data, a single global cutoff date is usually simplest and matches how the model will actually be used in production.

This is the one thing I'd insist on fixing before this goes in front of the team, because right now the headline "model beats the naive baseline by X" claim isn't trustworthy.

### 2. `mean_squared_error(..., squared=False)` will crash on current sklearn

```python
rmse_model = mean_squared_error(y_test, y_pred, squared=False)
```

The `squared` keyword was deprecated and has since been removed from `mean_squared_error` (confirmed: this raises `TypeError: got an unexpected keyword argument 'squared'` on sklearn 1.8.0, which is what's installed in this environment). Replace with:

```python
from sklearn.metrics import root_mean_squared_error
rmse_model = root_mean_squared_error(y_test, y_pred)
```

(or pin an older sklearn version, but the above is the forward-compatible fix). Worth double-checking whichever sklearn version the team's production environment actually runs, since this is exactly the kind of thing that works on a dev laptop and breaks in CI/deployment.

---

## Important gaps (not bugs, but worth addressing before handoff)

### 3. Lag features assume no gaps in the per-store date series

```python
df["lag_1_sales"] = df.groupby("store_id")["sales"].shift(1)
df["lag_7_sales"] = df.groupby("store_id")["sales"].shift(7)
```

`shift(1)`/`shift(7)` are positional, not date-aware. If any store is missing a row for a given date (closed store, missing data feed, etc.), the "lag_1" for the next available row silently becomes lag-2-or-more instead of yesterday's value — no error, just quietly wrong features. Worth adding a check that each store has one row per calendar day in the expected range (e.g., reindex each store's group on a complete date range, or assert `df.groupby("store_id")["date"].diff().dropna().eq(pd.Timedelta(days=1)).all()`) before trusting the lag columns.

### 4. No held-out validation for model selection / overfitting risk

`n_estimators=200, max_depth=3` are fixed with no validation curve, early stopping, or hyperparameter search. That's a reasonable starting point, but with only a single train/test evaluation (and one that's currently mis-specified per #1), there's no signal on whether the model is over- or under-fit. Once the split is fixed, it'd be worth adding a validation fold (or `TimeSeriesSplit` + `GridSearchCV`/`RandomizedSearchCV`) so the reported metric isn't the only piece of evidence about generalization.

### 5. Naive baseline could be strengthened

The current baseline (predict `lag_1_sales`, i.e., "same as yesterday") is a fine sanity check, but retail sales are usually strongly weekly-seasonal (weekday vs. weekend effects). A "same day last week" baseline (`lag_7_sales`) — or the better of the two — would be a tougher and more informative bar for the model to clear, especially since `lag_7_sales` is already computed.

### 6. No model/feature persistence or inference path

The script trains, evaluates, and prints metrics, then ends — nothing is saved (`joblib.dump` or similar) and there's no function/entry point showing how a next-day prediction would actually be produced in production (i.e., building the feature row for "tomorrow" from the latest known lags and the planned promo/holiday flags for that date). For a handoff, the team will need at least a `predict_next_day(store_id, date, ...)`-style function or a serialized pipeline artifact, otherwise this reads as a one-off analysis notebook rather than a pipeline.

### 7. Minor / polish items
- Hardcoded relative path `"daily_sales.csv"` with no existence check or CLI/config parameter — fine for a script you run locally, but worth parameterizing for handoff.
- No schema/sanity validation on the input (e.g., negative sales, nulls in `sales`, unexpected `store_id`s) before feature engineering.
- Only MAE/RMSE are reported; a scale-normalized metric (e.g., MAPE or WMAPE) would be more interpretable to non-technical stakeholders and more comparable across stores of very different sales volumes.
- New `store_id`s at inference time will be one-hot encoded to an all-zero vector (`handle_unknown="ignore"`) rather than erroring — that's a deliberate, reasonable choice, but worth documenting so it's not a surprise later (a new store will just get an unpersonalized prediction, not a failure).
- `remainder="passthrough"` in the `ColumnTransformer` works, but relies on column order/dtype implicitly; consider being explicit (list `numeric_cols` in the transformer) so the pipeline doesn't silently change behavior if `feature_cols` is edited later.

---

## What's already good

- Sorting by date and using `groupby("store_id")` before computing lags is the right idea (just needs the date-continuity check above).
- Comparing the model against a naive baseline on the same test set is exactly the right instinct for a forecasting sanity check — once the split is fixed, keep this.
- Tree-based model choice (gradient boosting) is reasonable for a mix of categorical/numeric tabular features without needing scaling, and the comment correctly notes scaling isn't needed.
- `OneHotEncoder(handle_unknown="ignore")` is the correct defensive choice for categorical columns that may see unseen categories at inference time.
- `random_state` is set for both the split and the model, so the current (flawed) result is at least reproducible — that'll make it easy to verify the fix in #1 by rerunning and comparing.

---

## Priority order for fixes

1. **Fix the train/test split to respect time order** (#1) — do this first; every other metric in the script is downstream of this.
2. **Fix the `mean_squared_error` call** (#2) — trivial, but blocks the script from even running on current sklearn.
3. Add the date-continuity check for lag features (#3).
4. Add a stronger baseline comparison and a validation strategy (#4, #5).
5. Add persistence + an inference entry point before calling this "handoff ready" (#6).
