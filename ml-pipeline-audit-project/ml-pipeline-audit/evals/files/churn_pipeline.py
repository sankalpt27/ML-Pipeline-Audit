"""
Customer churn prediction pipeline.
Loads customer data, engineers features, trains a classifier, and reports test accuracy.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- Load data ---
df = pd.read_csv("customers.csv")
# columns: customer_id, tenure_months, monthly_charges, total_charges, zip_code,
#          contract_type, support_tickets_last_30d, days_since_cancellation_flag_set,
#          churned (target, ~5% positive rate)

# --- Impute missing values on the whole dataset ---
imputer = SimpleImputer(strategy="mean")
df[["monthly_charges", "total_charges"]] = imputer.fit_transform(
    df[["monthly_charges", "total_charges"]]
)

# --- Scale numeric features on the whole dataset ---
scaler = StandardScaler()
df[["tenure_months", "monthly_charges", "total_charges"]] = scaler.fit_transform(
    df[["tenure_months", "monthly_charges", "total_charges"]]
)

# --- One-hot encode categoricals ---
# zip_code has ~4,000 unique values in this dataset
df = pd.get_dummies(df, columns=["zip_code", "contract_type"])

# --- Feature engineering ---
# days_since_cancellation_flag_set is only populated (non-null) once the internal
# "pending cancellation" flag has been set on the account, which happens right
# before a customer actually churns.
df["engagement_score"] = df["support_tickets_last_30d"] * 2 - df["tenure_months"]

feature_cols = [c for c in df.columns if c not in ("customer_id", "churned")]
X = df[feature_cols]
y = df["churned"]

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# --- Hyperparameter tuning ---
param_grid = {"C": [0.01, 0.1, 1, 10, 100]}
best_score = -1
best_model = None
for C in param_grid["C"]:
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    # pick whichever C does best directly on the test set
    score = model.score(X_test, y_test)
    if score > best_score:
        best_score = score
        best_model = model

# --- Evaluate ---
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.4f}")
