# %%
# === Cell 7: Point 9 — training script (logreg + random forest) ===
import os, re, json, hashlib
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib

# ------------------ CONFIG ------------------
FS_BASE = "/content/drive/MyDrive/DMML_Project/data/feature_store"
DATA_DIR = f"{FS_BASE}/feature_data"
MODELS_DIR = "/content/drive/MyDrive/DMML_Project/models"
REPORTS_DIR = "/content/drive/MyDrive/DMML_Project/reports"
SNAPSHOT_PREFIX = "generic_features"  # from earlier cells
TARGET_COL = "Churn"
RANDOM_STATE = 42
TEST_SIZE = 0.2
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
# -------------------------------------------

def _detect_snapshot(prefix: str, prefer_v="v2"):
    files = [f for f in os.listdir(DATA_DIR) if f.startswith(prefix) and f.endswith(".csv")]
    if not files:
        raise FileNotFoundError(f"No snapshots found in {DATA_DIR} for prefix '{prefix}'")
    # Prefer v2; else v1; else latest lexicographically
    cand_v2 = sorted([f for f in files if re.search(r"_v2\.csv$", f)])
    cand_v1 = sorted([f for f in files if re.search(r"_v1\.csv$", f)])
    if prefer_v == "v2" and cand_v2:
        pick = cand_v2[-1]
    elif cand_v1:
        pick = cand_v1[-1]
    else:
        files.sort()
        pick = files[-1]
    return os.path.join(DATA_DIR, pick)

def _hash_file(path: str) -> str:
    p = Path(path)
    h = hashlib.md5(p.read_bytes()).hexdigest()
    return h

def load_features_and_target():
    snap_path = _detect_snapshot(SNAPSHOT_PREFIX, prefer_v="v2")
    print("Using snapshot:", snap_path)
    df = pd.read_csv(snap_path)

    # Ensure row_id exists (not used as a feature)
    if "row_id" not in df.columns:
        df = df.reset_index(drop=True)
        df["row_id"] = df.index.astype(str)

    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not present in snapshot '{snap_path}'. "
                       "Add the label or join it before training.")
    # Build X, y
    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL, "row_id"], errors="ignore")

    # One-hot encode any non-numeric columns (e.g., gender)
    non_num = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    if non_num:
        X = pd.get_dummies(X, columns=non_num, drop_first=False)

    # Fill remaining NaNs
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    meta = {
        "snapshot_path": snap_path,
        "snapshot_md5": _hash_file(snap_path),
        "n_samples": int(df.shape[0]),
        "n_features": int(X.shape[1]),
        "target_positive_rate": float(y.mean()),
        "columns": X.columns.tolist()  # helpful for reproducibility
    }
    return X, y, meta

def evaluate_and_report(y_true, y_pred, y_prob=None):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None
    report_txt = classification_report(y_true, y_pred, digits=4)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return metrics, report_txt, cm

def train_and_save(model, model_name, X_train, y_train, X_test, y_test, data_meta):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics, report_txt, cm = evaluate_and_report(y_test, y_pred, y_prob)

    # Persist model and evaluation
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODELS_DIR, f"{model_name}_{ts}.pkl")
    joblib.dump(model, model_path)

    eval_payload = {
        "timestamp": ts,
        "model_name": model_name,
        "model_params": getattr(model, "get_params", lambda: {})(),
        "metrics": metrics,
        "classification_report": report_txt,
        "confusion_matrix": cm,
        "model_path": model_path,
        "feature_snapshot_path": data_meta["snapshot_path"],
        "feature_snapshot_md5": data_meta["snapshot_md5"],
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "feature_columns": data_meta["columns"]
    }
    report_path = os.path.join(REPORTS_DIR, f"evaluation_{model_name}_{ts}.json")
    with open(report_path, "w") as f:
        json.dump(eval_payload, f, indent=2)

    print(f"\n===== {model_name} =====")
    print(report_txt)
    print("Metrics:", metrics)
    print("Saved model:", model_path)
    print("Saved report:", report_path)

# ------------------ RUN ------------------
X, y, data_meta = load_features_and_target()
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# 1) Logistic Regression
logreg = LogisticRegression(max_iter=100000, class_weight="balanced", solver="liblinear")
train_and_save(logreg, "logreg", Xtr, ytr, Xte, yte, data_meta)

# 2) Random Forest
rf = RandomForestClassifier(
        n_estimators=30000,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )

train_and_save(rf, "random_forest", Xtr, ytr, Xte, yte, data_meta)



