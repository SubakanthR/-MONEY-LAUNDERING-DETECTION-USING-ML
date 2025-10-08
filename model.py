# Requirements:
# pip install pandas numpy scikit-learn xgboost scipy joblib

import os
import numpy as np
import pandas as pd
from scipy import sparse as sp
import joblib

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, average_precision_score, precision_recall_curve
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

# -------------------------------
# 1) Load data (set your filename)
# -------------------------------
csv_path = "/content/SAML-D.csv"  # <-- set to your CSV path
df = pd.read_csv(csv_path)

expected_cols = [
    "Time","Date","Sender_account","Receiver_account","Amount",
    "Payment_currency","Received_currency","Sender_bank_location",
    "Receiver_bank_location","Payment_type","Is_laundering","Laundering_type"
]
present_cols = [c for c in expected_cols if c in df.columns]
df = df[present_cols].copy()

# -------------------------------
# 2) Basic cleaning
# -------------------------------
df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
for c in df.columns:
    if df[c].dtype == "O":
        df[c] = df[c].fillna("UNK")
    else:
        df[c] = df[c].fillna(df[c].median())

# -----------------------------------------------
# 3) Combine Date + Time into a timestamp if valid
# -----------------------------------------------
time_col = None
if ("Date" in df.columns) and ("Time" in df.columns):
    try:
        df["__dt"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str), errors="coerce")
        if df["__dt"].notna().sum() > 0:
            time_col = "__dt"
    except Exception:
        time_col = None

# -----------------------------------
# 4) Time features if available
# -----------------------------------
if time_col is not None:
    df = df.sort_values(time_col)
    df["hour"] = df[time_col].dt.hour
    df["dayofweek"] = df[time_col].dt.dayofweek
    df["month"] = df[time_col].dt.month
else:
    df["hour"] = 0
    df["dayofweek"] = 0
    df["month"] = 0

# --------------------------------------
# 5) Amount transform and geo flag
# --------------------------------------
df["amount_log"] = np.log1p(df["Amount"].astype(float))
if {"Sender_bank_location","Receiver_bank_location"}.issubset(df.columns):
    df["cross_border"] = (df["Sender_bank_location"] != df["Receiver_bank_location"]).astype(int)
else:
    df["cross_border"] = 0

# -----------------------------------------------------
# 6) Behavior aggregates for sender/receiver accounts
# -----------------------------------------------------
def add_account_stats(frame, id_col, amount_col, time_col=None):
    f = frame.copy()
    if id_col not in f.columns:
        return f
    if time_col is not None and time_col in f.columns:
        f = f.sort_values(time_col)
        grp = f.groupby(id_col, group_keys=False)
        f[f"{id_col}_tx_count"] = grp.cumcount()
        f[f"{id_col}_amount_mean"] = grp[amount_col].apply(lambda s: s.shift().expanding().mean())
    else:
        grp = f.groupby(id_col, group_keys=False)
        f[f"{id_col}_tx_count"] = grp.cumcount()
        f[f"{id_col}_amount_mean"] = grp[amount_col].expanding().mean()
        f[f"{id_col}_amount_mean"] = grp[f"{id_col}_amount_mean"].shift().fillna(0.0)
    f[f"{id_col}_tx_count"] = f[f"{id_col}_tx_count"].fillna(0).astype(int)
    f[f"{id_col}_amount_mean"] = f[f"{id_col}_amount_mean"].fillna(0.0).astype(float)
    return f

df = add_account_stats(df, "Sender_account", "Amount", time_col)
df = add_account_stats(df, "Receiver_account", "Amount", time_col)

# --------------------------------
# 7) Target and features
# --------------------------------
target_col = "Is_laundering"
if target_col not in df.columns:
    raise ValueError("Column 'Is_laundering' not found in the file.")
y = df[target_col].astype(int)

# IMPORTANT: remove Laundering_type to avoid leakage
num_features = [c for c in [
    "Amount","amount_log","hour","dayofweek","month","cross_border",
    "Sender_account_tx_count","Sender_account_amount_mean",
    "Receiver_account_tx_count","Receiver_account_amount_mean"
] if c in df.columns]

cat_features = [c for c in [
    "Payment_currency","Received_currency","Sender_bank_location",
    "Receiver_bank_location","Payment_type"
] if c in df.columns]

X = df[num_features + cat_features].copy()

# ----------------------------------------
# 8) Train/validation/test split
# ----------------------------------------
if time_col is not None:
    q70 = df[time_col].quantile(0.7)
    q80 = df[time_col].quantile(0.8)
    train_mask = df[time_col] < q70
    val_mask = (df[time_col] >= q70) & (df[time_col] < q80)
    test_mask = df[time_col] >= q80

    X_tr, y_tr = X.loc[train_mask], y.loc[train_mask]
    X_val, y_val = X.loc[val_mask], y.loc[val_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]
else:
    groups = df["Sender_account"].astype(str) if "Sender_account" in df.columns else pd.Series(["G"]*len(df))
    gss_outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    trainval_idx, test_idx = next(gss_outer.split(X, y, groups=groups))
    X_trainval, y_trainval = X.iloc[trainval_idx], y.iloc[trainval_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    groups_trainval = groups.iloc[trainval_idx]
    gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=42)  # 0.125 of 0.8 ≈ 0.10 total
    tr_idx, val_idx = next(gss_inner.split(X_trainval, y_trainval, groups=groups_trainval))
    X_tr, y_tr = X_trainval.iloc[tr_idx], y_trainval.iloc[tr_idx]
    X_val, y_val = X_trainval.iloc[val_idx], y_trainval.iloc[val_idx]

# -----------------------------------------------------
# 9) Preprocessing: scale numerics, one-hot categoricals
# -----------------------------------------------------

# Cross-version compatible OneHotEncoder init:
# - prefer 'sparse_output' (>=1.2), fall back to 'sparse' for older versions.
try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)  # older sklearn

scaler = StandardScaler(with_mean=False)  # OK for sparse matrices

preprocess = ColumnTransformer(
    transformers=[
        ("num", scaler, num_features),
        ("cat", ohe, cat_features),
    ],
    remainder="drop",
    verbose_feature_names_out=True  # stable feature name prefixes
)

# Build full pipeline so downstream always references steps by name
model_placeholder = ("model", "will_be_set_later")
inference_pipeline = Pipeline(steps=[
    ("preprocess", preprocess),
    model_placeholder  # filled after training/calibration
])

# Fit transform using the pipeline step name to avoid manual ordering
preprocess.fit(X_tr)
X_tr_t = preprocess.transform(X_tr)
X_val_t = preprocess.transform(X_val)
X_test_t = preprocess.transform(X_test)

# ---------------------------------------
# 10) XGBoost with stronger regularization
# ---------------------------------------
pos = int(y_tr.sum())
total = len(y_tr)
neg = total - pos
scale_pos_weight = (neg / pos) if pos > 0 else 1.0

xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    learning_rate=0.03,
    n_estimators=350,
    max_depth=4,
    min_child_weight=10,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_lambda=12.0,
    reg_alpha=2.0,
    scale_pos_weight=scale_pos_weight,
    n_jobs=-1,
    random_state=42
)

try:
    xgb.fit(X_tr_t, y_tr, eval_set=[(X_val_t, y_val)], verbose=False)
except TypeError:
    xgb.fit(X_tr_t, y_tr)

# ---------------------------------------
# 11) Probability calibration (isotonic), cross-version safe
# ---------------------------------------
def make_calibrator(fitted_estimator):
    # Prefer newer keyword 'estimator'; fall back to 'base_estimator' on old versions
    try:
        return CalibratedClassifierCV(estimator=fitted_estimator, method="isotonic", cv="prefit")
    except TypeError:
        return CalibratedClassifierCV(base_estimator=fitted_estimator, method="isotonic", cv="prefit")

calibrated = make_calibrator(xgb)
calibrated.fit(X_val_t, y_val)

# Plug calibrator into the pipeline by name
inference_pipeline.steps[-1] = ("model", calibrated)

# ---------------------------------------------------------
# 12) Evaluation: PR-AUC and threshold tuning for precision
# ---------------------------------------------------------
proba_val = inference_pipeline.named_steps["model"].predict_proba(X_val_t)[:, 1]
proba_test = inference_pipeline.named_steps["model"].predict_proba(X_test_t)[:, 1]

val_apr = average_precision_score(y_val, proba_val)
test_apr = average_precision_score(y_test, proba_test)

prec, rec, thr = precision_recall_curve(y_val, proba_val)
target_precision = 0.2
best_thr = thr[-1] if len(thr) > 0 else 0.5
for p, r, t in zip(prec, rec, np.append(thr, 1.0)):
    if p >= target_precision:
        best_thr = t
        break

y_pred_test = (proba_test >= best_thr).astype(int)
print("Validation PR-AUC:", val_apr)
print("Test PR-AUC:", test_apr)
print(f"Chosen threshold (precision >= {target_precision:.0%}) on val: {best_thr:.6f}")
print(classification_report(y_test, y_pred_test, digits=4))

# ---------------------------------------
# 13) Save artifacts for future use
# ---------------------------------------
os.makedirs("artifacts", exist_ok=True)

# Save preprocessing transformer and calibrated model
joblib.dump(preprocess, "artifacts/preprocess.joblib")
joblib.dump(calibrated, "artifacts/calibrated_xgb.joblib")

# Save full inference pipeline with named steps (robust to ordering)
joblib.dump(inference_pipeline, "artifacts/inference_pipeline.joblib")

# Save native XGBoost model if available
underlying = getattr(calibrated, "base_estimator", None)
if underlying is None:
    underlying = getattr(calibrated, "estimator", None)
if hasattr(underlying, "save_model"):
    underlying.save_model("artifacts/xgb_model.json")

print("Saved artifacts to ./artifacts:")
print("- preprocess.joblib")
print("- calibrated_xgb.joblib")
print("- inference_pipeline.joblib")
print("- xgb_model.json (if available)")

# Optional: export feature names used by the ColumnTransformer for traceability
try:
    feat_names = preprocess.get_feature_names_out()
    pd.Series(feat_names).to_csv("artifacts/feature_names.csv", index=False)
except Exception:
    pass
