#!/usr/bin/env python3
"""
Train a robust Morph model using augmented + original data.

Uses the same V2 training configuration but adds augmented samples
to improve mic/channel robustness.
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
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FEATURE_DIR = PROJECT_ROOT / "data/features"
MODEL_DIR = PROJECT_ROOT / "models"

METADATA_COLUMNS = {
    "sample_id", "dataset", "label", "label_name", "file_path",
    "split", "attack_id", "speaker_id", "aug_id", "aug_params",
}

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_robust")


def get_feature_cols(df):
    return sorted(
        c for c in df.columns
        if c not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    )


def prepare_features(df, feature_cols):
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y


def evaluate(model, X, y, split_name):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    auc = roc_auc_score(y, y_prob)

    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fake_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    real_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    fake_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    print(f"\n  {split_name}:")
    print(f"    Accuracy:    {acc:.4f}")
    print(f"    Precision:   {prec:.4f}")
    print(f"    Recall:      {rec:.4f}")
    print(f"    F1:          {f1:.4f}")
    print(f"    ROC-AUC:     {auc:.4f}")
    print(f"    Fake Recall: {fake_recall:.4f}")
    print(f"    Real Recall: {real_recall:.4f}")
    print(f"    FPR:         {fake_fpr:.4f}")
    print(f"    FNR:         {fnr:.4f}")
    print(f"    CM: TN={tn} FP={fp} FN={fn} TP={tp}")

    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "roc_auc": auc, "fake_recall": fake_recall, "real_recall": real_recall,
        "fpr": fake_fpr, "fnr": fnr, "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),
    }


def main():
    # Load ASVspoof (original, large training set)
    asvspoof_path = FEATURE_DIR / "asvspoof2019_la_features.parquet"
    asvspoof = pd.read_parquet(asvspoof_path)
    log.info("ASVspoof: %d rows", len(asvspoof))

    # Load augmented Gary Stafford + Deep Voice
    gs_robust = pd.read_parquet(FEATURE_DIR / "gary_stafford_robust_features.parquet")
    dv_robust = pd.read_parquet(FEATURE_DIR / "deep_voice_robust_features.parquet")
    log.info("Gary Staff robust: %d rows", len(gs_robust))
    log.info("Deep Voice robust: %d rows", len(dv_robust))

    # Build splits same as V2
    rng = np.random.RandomState(42)

    # ASVspoof: 50k stratified train, rest split dev/eval
    idx_real = asvspoof.index[asvspoof["label"] == 0].values
    idx_fake = asvspoof.index[asvspoof["label"] == 1].values
    n_real_total = len(idx_real)
    n_fake_total = len(idx_fake)
    n_total = n_real_total + n_fake_total
    n_real_train = int(round(50000 * n_real_total / n_total))
    n_fake_train = 50000 - n_real_train
    n_real_train = min(n_real_train, n_real_total)
    n_fake_train = min(n_fake_train, n_fake_total)

    train_real_idx = rng.choice(idx_real, size=n_real_train, replace=False)
    train_fake_idx = rng.choice(idx_fake, size=n_fake_train, replace=False)
    train_asvspoof_idx = np.concatenate([train_real_idx, train_fake_idx])

    remaining_idx = np.setdiff1d(asvspoof.index.values, train_asvspoof_idx)
    remaining_df = asvspoof.loc[remaining_idx]
    dev_idx, eval_idx = train_test_split(
        remaining_idx, test_size=0.5, random_state=42,
        stratify=remaining_df["label"].values,
    )

    asvspoof_train = asvspoof.loc[train_asvspoof_idx]
    asvspoof_dev = asvspoof.loc[dev_idx]
    asvspoof_eval = asvspoof.loc[eval_idx]

    # TRAIN = ASVspoof train + augmented Gary + augmented Deep Voice
    # Only use augmented samples (aug_id > 0) + original Gary/DV for training
    gs_train = gs_robust  # includes both original and augmented
    dv_train = dv_robust

    train_df = pd.concat([asvspoof_train, gs_train, dv_train], ignore_index=True)
    dev_df = asvspoof_dev.reset_index(drop=True)
    eval_df = asvspoof_eval.reset_index(drop=True)

    feature_cols = get_feature_cols(train_df)
    log.info("Feature columns: %d", len(feature_cols))

    X_train, y_train = prepare_features(train_df, feature_cols)
    X_dev, y_dev = prepare_features(dev_df, feature_cols)
    X_eval, y_eval = prepare_features(eval_df, feature_cols)

    n_train_real = int(np.sum(y_train == 0))
    n_train_fake = int(np.sum(y_train == 1))
    scale_pos_weight = n_train_real / n_train_fake
    print(f"\n  Train: {len(X_train)} samples (REAL={n_train_real}, FAKE={n_train_fake})")
    print(f"  scale_pos_weight: {scale_pos_weight:.4f}")

    # Train
    params = {**XGB_PARAMS, "scale_pos_weight": scale_pos_weight}
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_dev, y_dev)], verbose=50)

    # Evaluate
    train_metrics = evaluate(model, X_train, y_train, "TRAIN")
    dev_metrics = evaluate(model, X_dev, y_dev, "DEV")
    eval_metrics = evaluate(model, X_eval, y_eval, "EVAL")

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "morph_xgboost_v2_robust.json"
    model.save_model(str(model_path))

    feature_meta = {
        "model_name": "morph_xgboost_v2_robust",
        "feature_count": len(feature_cols),
        "feature_columns": feature_cols,
        "train_samples": int(X_train.shape[0]),
        "augmented_gs": int((gs_robust["aug_id"] > 0).sum()),
        "augmented_dv": int((dv_robust["aug_id"] > 0).sum()),
        "dev_samples": int(X_dev.shape[0]),
        "eval_samples": int(X_eval.shape[0]),
        "scale_pos_weight": float(scale_pos_weight),
        "train_metrics": train_metrics,
        "dev_metrics": dev_metrics,
        "eval_metrics": eval_metrics,
    }
    meta_path = MODEL_DIR / "morph_xgboost_v2_robust_features.json"
    with open(meta_path, "w") as f:
        json.dump(feature_meta, f, indent=2)

    metrics_path = MODEL_DIR / "morph_xgboost_v2_robust_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"train": train_metrics, "dev": dev_metrics, "eval": eval_metrics}, f, indent=2)

    print(f"\n  Model  → {model_path}")
    print(f"  Meta   → {meta_path}")
    print(f"  Metrics→ {metrics_path}")


if __name__ == "__main__":
    main()
