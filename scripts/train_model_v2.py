#!/usr/bin/env python3
"""
Morph Model V2 — XGBoost binary classifier for voice-clone detection.

Trained on Gary Stafford + Deep Voice + stratified ASVspoof subset.
ASVspoof EVAL split is completely held out.

Usage:
    python scripts/train_model_v2.py
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

# Input parquet files
ASVSPOOF_FILE = FEATURE_DIR / "asvspoof2019_la_features.parquet"
GARY_FILE = FEATURE_DIR / "gary_stafford_features.parquet"
DEEP_VOICE_FILE = FEATURE_DIR / "deep_voice_features.parquet"

# Columns that must NOT be used as model features
METADATA_COLUMNS = {
    "sample_id",
    "dataset",
    "label",
    "label_name",
    "file_path",
    "split",
    "attack_id",
    "speaker_id",
}

# V2 split parameters
TRAIN_ASVSPOOF_COUNT = 50_000
RANDOM_STATE = 42

# XGBoost hyperparameters (same as V1)
XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "eval_metric": "logloss",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_v2")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three feature Parquet files."""
    for fpath in [ASVSPOOF_FILE, GARY_FILE, DEEP_VOICE_FILE]:
        if not fpath.exists():
            log.error("Missing feature file: %s", fpath)
            sys.exit(1)

    asvspoof = pd.read_parquet(ASVSPOOF_FILE)
    gary = pd.read_parquet(GARY_FILE)
    deep_voice = pd.read_parquet(DEEP_VOICE_FILE)

    log.info("ASVspoof  : %d rows, %d cols", *asvspoof.shape)
    log.info("Gary      : %d rows, %d cols", *gary.shape)
    log.info("Deep Voice: %d rows, %d cols", *deep_voice.shape)

    return asvspoof, gary, deep_voice


def build_v2_splits(
    asvspoof: pd.DataFrame,
    gary: pd.DataFrame,
    deep_voice: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build V2 TRAIN / DEV / EVAL splits.

    TRAIN: 50,000 stratified ASVspoof + all Gary + all Deep Voice
    DEV:   ~half remaining ASVspoof
    EVAL:  ~half remaining ASVspoof (completely unseen)
    """
    rng = np.random.RandomState(RANDOM_STATE)

    # --- Stratified sample 50,000 from ASVspoof for TRAIN ---
    # Group indices by label for stratification
    idx_real = asvspoof.index[asvspoof["label"] == 0].values
    idx_fake = asvspoof.index[asvspoof["label"] == 1].values

    # Compute per-class counts maintaining original ratio
    n_real_total = len(idx_real)
    n_fake_total = len(idx_fake)
    n_total = n_real_total + n_fake_total

    # Stratified: same proportion as original
    n_real_train = int(round(TRAIN_ASVSPOOF_COUNT * n_real_total / n_total))
    n_fake_train = TRAIN_ASVSPOOF_COUNT - n_real_train

    # Ensure we don't exceed available samples
    n_real_train = min(n_real_train, n_real_total)
    n_fake_train = min(n_fake_train, n_fake_total)
    actual_train_count = n_real_train + n_fake_train

    # Sample with fixed seed
    train_real_idx = rng.choice(idx_real, size=n_real_train, replace=False)
    train_fake_idx = rng.choice(idx_fake, size=n_fake_train, replace=False)
    train_asvspoof_idx = np.concatenate([train_real_idx, train_fake_idx])

    # Remaining ASVspoof indices
    all_asvspoof_idx = asvspoof.index.values
    remaining_idx = np.setdiff1d(all_asvspoof_idx, train_asvspoof_idx)

    # Split remaining ~50/50 into DEV and EVAL (stratified)
    remaining_df = asvspoof.loc[remaining_idx]
    remaining_labels = remaining_df["label"].values

    dev_idx, eval_idx = train_test_split(
        remaining_idx,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=remaining_labels,
    )

    # --- Assemble splits ---
    asvspoof_train = asvspoof.loc[train_asvspoof_idx]
    asvspoof_dev = asvspoof.loc[dev_idx]
    asvspoof_eval = asvspoof.loc[eval_idx]

    # TRAIN = ASVspoof train subset + Gary + Deep Voice
    train_dfs = [asvspoof_train, gary, deep_voice]
    train_df = pd.concat(train_dfs, ignore_index=True)
    dev_df = asvspoof_dev.reset_index(drop=True)
    eval_df = asvspoof_eval.reset_index(drop=True)

    return train_df, dev_df, eval_df


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return sorted numeric feature columns, excluding metadata."""
    return sorted(
        c for c in df.columns
        if c not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    )


def prepare_features(df: pd.DataFrame, feature_cols: list[str]):
    """Extract X and y arrays using the canonical feature column order."""
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(X_train, y_train, X_dev, y_dev, scale_pos_weight: float):
    """Train the XGBoost classifier."""
    params = {**XGB_PARAMS, "scale_pos_weight": scale_pos_weight}

    model = xgb.XGBClassifier(**params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_dev, y_dev)],
        verbose=50,
    )

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, X, y, split_name: str) -> dict:
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


def evaluate_by_class(model, X, y, split_name: str) -> dict:
    """Evaluate per-class metrics on a split."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    report = classification_report(
        y, y_pred, target_names=["bonafide", "spoof"], digits=4, output_dict=True
    )

    # Per-class detail
    classes = {"bonafide": 0, "spoof": 1}
    class_metrics = {}
    for cls_name, cls_label in classes.items():
        mask = y == cls_label
        n = int(mask.sum())
        cls_y = y[mask]
        cls_pred = y_pred[mask]
        cls_prob = y_prob[mask]

        cls_acc = accuracy_score(cls_y, cls_pred) if n > 0 else 0.0
        cls_prec = precision_score(cls_y, cls_pred, pos_label=cls_label, zero_division=0) if n > 0 else 0.0
        cls_rec = recall_score(cls_y, cls_pred, pos_label=cls_label, zero_division=0) if n > 0 else 0.0
        cls_f1 = f1_score(cls_y, cls_pred, pos_label=cls_label, zero_division=0) if n > 0 else 0.0

        class_metrics[cls_name] = {
            "count": n,
            "accuracy": cls_acc,
            "precision": cls_prec,
            "recall": cls_rec,
            "f1": cls_f1,
        }

    print(f"\n{'='*50}")
    print(f"  {split_name} — PERFORMANCE BY CLASS")
    print(f"{'='*50}")
    for cls_name, m in class_metrics.items():
        print(f"\n  [{cls_name}] (n={m['count']})")
        print(f"    Accuracy  : {m['accuracy']:.4f}")
        print(f"    Precision : {m['precision']:.4f}")
        print(f"    Recall    : {m['recall']:.4f}")
        print(f"    F1        : {m['f1']:.4f}")
    print(f"{'='*50}")

    return class_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ---- Load data ----
    log.info("Loading feature files …")
    asvspoof, gary, deep_voice = load_data()

    # ---- Build V2 splits ----
    log.info("Building V2 splits …")
    train_df, dev_df, eval_df = build_v2_splits(asvspoof, gary, deep_voice)

    # ---- Print split summary ----
    print(f"\n{'='*55}")
    print(f"  MORPH MODEL V2 — SPLIT SUMMARY")
    print(f"{'='*55}")
    for name, df in [("TRAIN", train_df), ("DEV", dev_df), ("EVAL", eval_df)]:
        n_real = int((df["label"] == 0).sum())
        n_fake = int((df["label"] == 1).sum())
        n_total = len(df)
        print(f"\n  {name}:")
        print(f"    Total      : {n_total:>6d}")
        print(f"    REAL (0)   : {n_real:>6d}")
        print(f"    FAKE (1)   : {n_fake:>6d}")
        print(f"    Imbalance  : {n_fake / n_real:.2f}x" if n_real > 0 else "    Imbalance  : N/A")

    print(f"\n  NOTE: EVAL set is completely unseen during training.")
    print(f"{'='*55}\n")

    # ---- Determine feature columns from TRAIN split ----
    feature_cols = get_feature_cols(train_df)
    log.info("Feature columns: %d", len(feature_cols))

    # ---- Prepare arrays ----
    X_train, y_train = prepare_features(train_df, feature_cols)
    X_dev, y_dev = prepare_features(dev_df, feature_cols)
    X_eval, y_eval = prepare_features(eval_df, feature_cols)

    # ---- Compute scale_pos_weight from TRAIN split ----
    n_train_real = int(np.sum(y_train == 0))
    n_train_fake = int(np.sum(y_train == 1))
    scale_pos_weight = n_train_real / n_train_fake
    print(f"  scale_pos_weight (TRAIN): {scale_pos_weight:.4f}")

    # ---- Train ----
    log.info("Training XGBoost Model V2 …")
    model = train_model(X_train, y_train, X_dev, y_dev, scale_pos_weight)

    # ---- Evaluate TRAIN / DEV ----
    train_metrics = evaluate(model, X_train, y_train, "TRAIN")
    dev_metrics = evaluate(model, X_dev, y_dev, "DEV")

    # ---- Evaluate EVAL (completely unseen) ----
    eval_metrics = evaluate(model, X_eval, y_eval, "EVAL")

    # ---- Evaluate EVAL by class ----
    eval_class_metrics = evaluate_by_class(model, X_eval, y_eval, "EVAL")

    # ---- Save model ----
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "morph_xgboost_v2.json"
    model.save_model(str(model_path))
    log.info("Model saved → %s", model_path)

    # ---- Save feature metadata ----
    feature_meta = {
        "model_name": "morph_xgboost_v2",
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "train_samples": int(X_train.shape[0]),
        "dev_samples": int(X_dev.shape[0]),
        "eval_samples": int(X_eval.shape[0]),
        "scale_pos_weight": float(scale_pos_weight),
        "train_metrics": train_metrics,
        "dev_metrics": dev_metrics,
        "eval_metrics": eval_metrics,
        "eval_class_metrics": eval_class_metrics,
    }
    meta_path = MODEL_DIR / "morph_xgboost_v2_features.json"
    with open(meta_path, "w") as f:
        json.dump(feature_meta, f, indent=2)
    log.info("Feature metadata saved → %s", meta_path)

    # ---- Save metrics (standalone) ----
    metrics_path = MODEL_DIR / "morph_xgboost_v2_metrics.json"
    metrics = {
        "train": train_metrics,
        "dev": dev_metrics,
        "eval": eval_metrics,
        "eval_by_class": eval_class_metrics,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Metrics saved → %s", metrics_path)

    print(f"\n  Model  → {model_path}")
    print(f"  Meta   → {meta_path}")
    print(f"  Metrics→ {metrics_path}\n")


if __name__ == "__main__":
    main()
