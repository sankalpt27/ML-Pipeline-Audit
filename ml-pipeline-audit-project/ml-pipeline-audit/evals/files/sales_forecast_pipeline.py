"""
Daily store sales forecasting pipeline.
Predicts next-day sales from a time-ordered dataset using gradient boosting.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Load data, sorted by date ascending ---
df = pd.read_csv("daily_sales.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)
# columns: date, store_id, day_of_week, promo_active, holiday, sales (target)

# --- Feature engineering ---
df["lag_1_sales"] = df.groupby("store_id")["sales"].shift(1)
df["lag_7_sales"] = df.groupby("store_id")["sales"].shift(7)
df = df.dropna(subset=["lag_1_sales", "lag_7_sales"])

feature_cols = ["day_of_week", "promo_active", "holiday", "lag_1_sales", "lag_7_sales", "store_id"]
categorical_cols = ["day_of_week", "store_id"]
numeric_cols = [c for c in feature_cols if c not in categorical_cols]

X = df[feature_cols]
y = df["sales"]

# --- Split: 80/20 ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Pipeline: encoding + model (tree-based, so no scaling needed) ---
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ],
    remainder="passthrough",
)

pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42)),
    ]
)

pipeline.fit(X_train, y_train)

# --- Evaluate against a naive baseline (predict lag_1_sales) ---
y_pred = pipeline.predict(X_test)
baseline_pred = X_test["lag_1_sales"]

mae_model = mean_absolute_error(y_test, y_pred)
rmse_model = mean_squared_error(y_test, y_pred, squared=False)
mae_baseline = mean_absolute_error(y_test, baseline_pred)

print(f"Model MAE: {mae_model:.2f}, RMSE: {rmse_model:.2f}")
print(f"Naive baseline (yesterday's sales) MAE: {mae_baseline:.2f}")
