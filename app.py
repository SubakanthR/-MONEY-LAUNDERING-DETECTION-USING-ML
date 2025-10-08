import os
import io
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory

# Ensure xgboost is installed; required by saved model
from xgboost import XGBClassifier
import xgboost as xgb_native  # for DMatrix fallback if needed

# Raw columns from training CSVs
RAW_EXPECTED_COLS = [
    "Time","Date","Sender_account","Receiver_account","Amount",
    "Payment_currency","Received_currency","Sender_bank_location",
    "Receiver_bank_location","Payment_type","Is_laundering","Laundering_type"
]

# Engineered columns created during training
ENGINEERED_COLS = [
    "hour","dayofweek","month","amount_log","cross_border",
    "Sender_account_tx_count","Sender_account_amount_mean",
    "Receiver_account_tx_count","Receiver_account_amount_mean"
]

def create_app():
    app = Flask(__name__)
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev")

    # Config
    app.config["ARTIFACT_DIR"] = os.environ.get("ARTIFACT_DIR", "artifacts")
    ARTIFACT_DIR = app.config["ARTIFACT_DIR"]
    PREPROCESS_PATH = os.path.join(ARTIFACT_DIR, "preprocess.joblib")
    CALIBRATED_PATH = os.path.join(ARTIFACT_DIR, "calibrated_xgb.joblib")
    PIPELINE_PATH = os.path.join(ARTIFACT_DIR, "inference_pipeline.joblib")

    # State
    app.inference_pipeline = None
    app.preprocess = None
    app.calibrated = None

    def safe_joblib_load(path):
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                return joblib.load(f)
        except Exception as e:
            print(f"Error loading {path}: {e}", flush=True)
            return None

    # Eagerly load artifacts at startup
    app.inference_pipeline = safe_joblib_load(PIPELINE_PATH)
    if app.inference_pipeline is None:
        app.preprocess = safe_joblib_load(PREPROCESS_PATH)
        app.calibrated = safe_joblib_load(CALIBRATED_PATH)
    print("Artifacts loaded at startup.", flush=True)

    def ensure_inference_ready():
        if app.inference_pipeline is not None:
            return True, "ok"
        if app.preprocess is None or app.calibrated is None:
            return False, "Model artifacts are not loaded. Ensure artifacts/ files exist and match the training environment."
        return True, "ok"

    def read_csv_from_upload(file_storage):
        stream = io.BytesIO(file_storage.read())
        df = pd.read_csv(stream)
        return df

    # Reproduce training-time feature engineering for raw uploads
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        f = df.copy()
        # Coerce Amount numeric
        if "Amount" in f.columns:
            f["Amount"] = pd.to_numeric(f["Amount"], errors="coerce")
        # Fill NA like training
        for c in f.columns:
            if f[c].dtype == "O":
                f[c] = f[c].fillna("UNK")
            else:
                f[c] = f[c].fillna(f[c].median())
        # Combine Date + Time to timestamp
        time_col = None
        if ("Date" in f.columns) and ("Time" in f.columns):
            try:
                f["__dt"] = pd.to_datetime(f["Date"].astype(str) + " " + f["Time"].astype(str), errors="coerce")
                if f["__dt"].notna().sum() > 0:
                    time_col = "__dt"
            except Exception:
                time_col = None
        # Time features
        if time_col is not None:
            f = f.sort_values(time_col)
            f["hour"] = f[time_col].dt.hour
            f["dayofweek"] = f[time_col].dt.dayofweek
            f["month"] = f[time_col].dt.month
        else:
            f["hour"] = f.get("hour", 0)
            f["dayofweek"] = f.get("dayofweek", 0)
            f["month"] = f.get("month", 0)
        # Amount transform
        if "Amount" in f.columns:
            f["amount_log"] = np.log1p(f["Amount"].astype(float))
        else:
            f["amount_log"] = f.get("amount_log", 0.0)
        # Cross-border flag
        if {"Sender_bank_location","Receiver_bank_location"}.issubset(f.columns):
            f["cross_border"] = (f["Sender_bank_location"] != f["Receiver_bank_location"]).astype(int)
        else:
            f["cross_border"] = f.get("cross_border", 0).fillna(0).astype(int)
        # Running aggregates (prior stats per account)
        def add_account_stats(frame, id_col, amount_col, time_col=None):
            g = frame.copy()
            if id_col not in g.columns or amount_col not in g.columns:
                g[f"{id_col}_tx_count"] = g.get(f"{id_col}_tx_count", 0)
                g[f"{id_col}_amount_mean"] = g.get(f"{id_col}_amount_mean", 0.0)
                return g
            if time_col is not None and time_col in g.columns:
                g = g.sort_values(time_col)
                grp = g.groupby(id_col, group_keys=False)
                g[f"{id_col}_tx_count"] = grp.cumcount()
                def prior_mean(s):
                    return s.shift().expanding().mean()
                g[f"{id_col}_amount_mean"] = grp[amount_col].apply(prior_mean)
            else:
                grp = g.groupby(id_col, group_keys=False)
                g[f"{id_col}_tx_count"] = grp.cumcount()
                g[f"{id_col}_amount_mean"] = grp[amount_col].expanding().mean()
                g[f"{id_col}_amount_mean"] = grp[f"{id_col}_amount_mean"].shift().fillna(0.0)
            g[f"{id_col}_tx_count"] = g[f"{id_col}_tx_count"].fillna(0).astype(int)
            g[f"{id_col}_amount_mean"] = g[f"{id_col}_amount_mean"].fillna(0.0).astype(float)
            return g
        f = add_account_stats(f, "Sender_account", "Amount", time_col)
        f = add_account_stats(f, "Receiver_account", "Amount", time_col)
        return f

    # Access underlying XGB from calibrated wrapper/pipeline
    def get_underlying_xgb():
        if app.inference_pipeline is not None:
            calibrated = app.inference_pipeline.named_steps.get("model")
            base = getattr(calibrated, "base_estimator", None)
            if base is None:
                base = getattr(calibrated, "estimator", None)
            return base
        base = getattr(app.calibrated, "base_estimator", None)
        if base is None:
            base = getattr(app.calibrated, "estimator", None)
        return base

    # Get transformed feature names for readability
    def get_feature_names():
        if app.inference_pipeline is not None:
            try:
                return app.inference_pipeline.named_steps["preprocess"].get_feature_names_out()
            except Exception:
                return None
        if app.preprocess is not None:
            try:
                return app.preprocess.get_feature_names_out()
            except Exception:
                return None
        return None

    def humanize_feature_name(name):
        n = str(name)
        n = n.replace("num__", "").replace("cat__", "")
        n = n.replace("_", " ").replace("=", " = ")
        return n

    def summarize_reasons(contrib_row, feat_names, topk=4):
        if contrib_row is None or feat_names is None:
            return []
        vals = np.array(contrib_row[:-1])  # exclude bias
        idx = np.argsort(-vals)
        out = []
        for j in idx[:topk]:
            v = float(vals[j])
            if v <= 0:
                break
            out.append((humanize_feature_name(feat_names[j]), v))
        return out

    def predict_with_pipeline(df_input: pd.DataFrame):
        # Always engineer features from raw (Option A)
        X_raw = engineer_features(df_input)

        # Score using full pipeline if present
        if app.inference_pipeline is not None:
            preprocess = app.inference_pipeline.named_steps["preprocess"]
            Xt = preprocess.transform(X_raw)
            proba = app.inference_pipeline.named_steps["model"].predict_proba(Xt)[:, 1]
        else:
            Xt = app.preprocess.transform(X_raw)
            proba = app.calibrated.predict_proba(Xt)[:, 1]

        # Tuned default threshold with env override
        thr = float(os.environ.get("SCORE_THRESHOLD", "0.106383"))
        y_pred = (proba >= thr).astype(int)

        # Explanations via Tree SHAP contributions
        xgb = get_underlying_xgb()
        contribs = None
        try:
            # First try sklearn wrapper with pred_contribs
            contribs = xgb.predict(Xt, pred_contribs=True)
        except Exception:
            try:
                booster = xgb.get_booster()
                dmat = xgb_native.DMatrix(Xt)
                contribs = booster.predict(dmat, pred_contribs=True)
            except Exception:
                contribs = None
        feat_names = get_feature_names()
        reason_list = []
        if contribs is not None and feat_names is not None:
            for row in contribs:
                reason_list.append(summarize_reasons(row, feat_names, topk=4))
        else:
            reason_list = [[] for _ in range(len(X_raw))]
        return X_raw, proba, y_pred, thr, reason_list

    @app.route("/")
    def home():
        ok, msg = ensure_inference_ready()
        if not ok:
            flash(msg, "error")
        return render_template("index.html")

    @app.route("/predict", methods=["POST"])
    def predict():
        ok, msg = ensure_inference_ready()
        if not ok:
            flash(msg, "error")
            return redirect(url_for("home"))

        if "file" not in request.files or request.files["file"].filename == "":
            flash("Please choose a CSV file.", "error")
            return redirect(url_for("home"))

        file = request.files["file"]
        try:
            df = read_csv_from_upload(file)
        except Exception as e:
            flash(f"Failed to read CSV: {e}", "error")
            return redirect(url_for("home"))

        try:
            X_used, proba, y_pred, thr, reasons = predict_with_pipeline(df)
        except Exception as e:
            flash(f"Inference error: {e}", "error")
            return redirect(url_for("home"))

        labels_text = np.where(y_pred == 1, "Money laundering", "Not laundering")

        # Join reasons as text
        reasons_text = []
        for r in reasons:
            if not r:
                reasons_text.append("")
            else:
                reasons_text.append("; ".join([f"{k} (+{v:.03f})" for k, v in r]))

        out = X_used.copy()
        out["prediction"] = labels_text
        out["pred_proba"] = proba
        out["pred_label"] = y_pred
        out["reasons"] = reasons_text

        # Preview table
        preview_cols = [c for c in out.columns if c != "__dt"]
        preview_html = out[preview_cols].head(50).to_html(classes="table table-striped table-sm", index=False)

        # Save full results
        os.makedirs("tmp", exist_ok=True)
        out_path = os.path.join("tmp", "scored.csv")
        try:
            out.to_csv(out_path, index=False)
            download_ready = True
        except Exception:
            download_ready = False

        stats = {
            "threshold": float(thr),
            "n_rows": int(len(out)),
            "n_positive": int((y_pred == 1).sum()),
            "positive_rate": float((y_pred == 1).mean() if len(out) else 0.0),
        }

        # Top positives for panel
        positives = out.loc[out["pred_label"] == 1, ["pred_proba", "reasons"]].copy()
        positives = positives.sort_values("pred_proba", ascending=False).head(20)
        pos_records = [
            {"rank": i+1, "prob": float(row["pred_proba"]), "reasons": row["reasons"]}
            for i, (_, row) in enumerate(positives.iterrows())
        ]

        return render_template(
            "result.html",
            preview_table=preview_html,
            stats=stats,
            download_ready=download_ready,
            download_name="scored.csv",
            positives=pos_records
        )

    @app.route("/download/<path:filename>")
    def download(filename):
        return send_from_directory("tmp", filename, as_attachment=True)

    return app

if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=True)
