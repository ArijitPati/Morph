#!/usr/bin/env python3
"""
Morph Model V1 — XGBoost binary classifier for voice-clone detection.

Trained on Gary Stafford + Deep Voice features only.

Usage:
    python scripts/train_model_v1.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_DIR = PROJECT_ROOT / "data/features"
MODEL_DIR = PROJECT_ROOT / "models"

# Input parquet files (Model V1 — Gary Stafford + Deep Voice only)
INPUT_FILES = [
    FEATURE_DIR / "gary_stafford_features.parquet",
    FEATURE_DIR / "deep_voice_features.parquet",
]

# Columns that must NOT be used as model features
METADATA_COLUMNS = {
    "sample_id",
    "dataset",
    "label",
    "label_name",
    "file_path",
}

# XGBoost hyperparameters
XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "eval_metric": "logloss",
}

# Train / validation / test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_v1")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """Load and concatenate the two feature Parquet files."""
    frames = []
    for fpath in INPUT_FILES:
        if not fpath.exists():
            log.error("Missing feature file: %s", fpath)
            sys.exit(1)
        df = pd.read_parquet(fpath)
        log.info("Loaded %s — %d rows, %d cols", fpath.name, *df.shape)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    log.info("Combined dataset — %d rows, %d cols", *combined.shape)
    return combined


def prepare_features(df: pd.DataFrame):
    """
    Separate features and labels.  All numeric columns not in
    METADATA_COLUMNS become model features.
    """
    feature_cols = sorted(
        c for c in df.columns
        if c not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    )

    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int32)

    # Replace any lingering NaN / inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y, feature_cols


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(X_train, y_train, X_val, y_val, scale_pos_weight: float):
    """Train the XGBoost classifier."""
    params = {**XGB_PARAMS, "scale_pos_weight": scale_pos_weight}

    model = xgb.XGBClassifier(**params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, X, y, split_name: str):
    """Evaluate model on a dataset split and print metrics."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    auc = roc_auc_score(y, y_prob)

    print(f"\n{'='*50}")
    print(f"  {split_name} SET EVALUATION")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print()
    print("  Classification Report:")
    print(classification_report(y, y_pred, target_names=["REAL", "FAKE"], digits=4))
    print("  Confusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    print(f"    REAL predicted:  TN={cm[0][0]:>5d}  FP={cm[0][1]:>5d}")
    print(f"    FAKE predicted:  FN={cm[1][0]:>5d}  TP={cm[1][1]:>5d}")
    print(f"{'='*50}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ---- Load data ----
    df = load_data()

    X, y, feature_cols = prepare_features(df)

    # ---- Print dataset summary ----
    n_real = int(np.sum(y == 0))
    n_fake = int(np.sum(y == 1))
    n_total = len(y)

    print(f"\n{'='*50}")
    print(f"  MORPH MODEL V1 — DATASET SUMMARY")
    print(f"{'='*50}")
    print(f"  Total samples  : {n_total}")
    print(f"  REAL (label=0) : {n_real}")
    print(f"  FAKE (label=1) : {n_fake}")
    print(f"  Imbalance ratio: {n_fake / n_real:.2f}x")
    print(f"  Num features   : {len(feature_cols)}")
    print(f"{'='*50}")

    # ---- Stratified train / val / test split ----
    # First split: 70% train, 30% temp (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(VAL_RATIO + TEST_RATIO), random_state=RANDOM_STATE, stratify=y
    )
    # Second split: split temp into 50/50 → 15% val, 15% test of total
    rel_val = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - rel_val), random_state=RANDOM_STATE, stratify=y_temp
    )

    print(f"\n  Train      : {X_train.shape[0]:>6d} samples")
    print(f"  Validation : {X_val.shape[0]:>6d} samples")
    print(f"  Test       : {X_test.shape[0]:>6d} samples")

    # ---- Compute scale_pos_weight from training split ----
    n_train_real = int(np.sum(y_train == 0))
    n_train_fake = int(np.sum(y_train == 1))
    scale_pos_weight = n_train_real / n_train_fake
    print(f"  scale_pos_weight : {scale_pos_weight:.4f}")
    print(f"{'='*50}\n")

    # ---- Train ----
    log.info("Training XGBoost Model V1 …")
    model = train(X_train, y_train, X_val, y_val, scale_pos_weight)

    # ---- Evaluate on all splits ----
    train_metrics = evaluate(model, X_train, y_train, "TRAIN")
    val_metrics = evaluate(model, X_val, y_val, "VALIDATION")
    test_metrics = evaluate(model, X_test, y_test, "TEST")

    # ---- Save model ----
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "morph_xgboost_v1.json"
    model.save_model(str(model_path))
    log.info("Model saved → %s", model_path)

    # ---- Save feature list ----
    feature_meta = {
        "model_name": "morph_xgboost_v1",
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "training_samples": int(X_train.shape[0]),
        "validation_samples": int(X_val.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "scale_pos_weight": float(scale_pos_weight),
        "test_metrics": test_metrics,
    }
    meta_path = MODEL_DIR / "morph_xgboost_v1_features.json"
    with open(meta_path, "w") as f:
        json.dump(feature_meta, f, indent=2)
    log.info("Feature metadata saved → %s", meta_path)

    print(f"\n  Model  → {model_path}")
    print(f"  Meta   → {meta_path}\n")


if __name__ == "__main__":
    main()
