#!/usr/bin/env python3
"""
Morph V2 — Feature Importance Analysis.

Loads the trained V2 XGBoost model and produces:
  - Per-feature importance (gain, weight, cover)
  - Feature-group importance
  - Top-30 bar chart
  - Feature-group bar chart
  - Optional SHAP analysis (if shap is installed)

Usage:
    python scripts/analyze_v2_features.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models/morph_xgboost_v2.json"
FEATURE_META_PATH = PROJECT_ROOT / "models/morph_xgboost_v2_features.json"
ASVSPOOF_PARQUET = PROJECT_ROOT / "data/features/asvspoof2019_la_features.parquet"

OUTPUT_DIR = PROJECT_ROOT / "data/features"

FEATURE_IMPORTANCE_CSV = OUTPUT_DIR / "v2_feature_importance.csv"
GROUP_IMPORTANCE_CSV = OUTPUT_DIR / "v2_feature_group_importance.csv"
TOP30_PNG = OUTPUT_DIR / "v2_feature_importance_top30.png"
GROUP_PNG = OUTPUT_DIR / "v2_feature_groups.png"
SHAP_IMPORTANCE_CSV = OUTPUT_DIR / "v2_shap_importance.csv"
SHAP_TOP30_PNG = OUTPUT_DIR / "v2_shap_top30.png"

SHAP_SAMPLE_SIZE = 10_000
SHAP_RANDOM_STATE = 42

METADATA_COLUMNS = {
    "sample_id", "dataset", "label", "label_name", "file_path",
    "split", "attack_id", "speaker_id",
}


# ---------------------------------------------------------------------------
# Feature grouping
# ---------------------------------------------------------------------------

def assign_group(feature_name: str) -> str:
    """Assign a human-readable group based on feature name prefix."""
    name = feature_name.lower()
    if name.startswith("mfcc"):
        return "MFCC"
    if name.startswith("cqcc"):
        return "CQCC"
    if name.startswith("spectral_"):
        return "Spectral"
    if name.startswith("chroma_"):
        return "Chroma"
    if name.startswith("f0") or name.startswith("voiced_frame"):
        return "F0/Pitch"
    if name.startswith("rms"):
        return "RMS"
    if name.startswith("zcr"):
        return "ZCR"
    if name == "duration":
        return "Duration"
    return "Other"


# ---------------------------------------------------------------------------
# Load model & feature order
# ---------------------------------------------------------------------------

def load_model_and_features():
    """Return (xgb.Booster, list[str] of feature columns)."""
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)
    if not FEATURE_META_PATH.exists():
        print(f"ERROR: Feature metadata not found at {FEATURE_META_PATH}")
        sys.exit(1)

    booster = xgb.Booster()
    booster.load_model(str(MODEL_PATH))
    print(f"Loaded model  → {MODEL_PATH}")

    with open(FEATURE_META_PATH) as f:
        meta = json.load(f)
    feature_cols = meta["feature_columns"]
    print(f"Loaded features → {FEATURE_META_PATH} ({len(feature_cols)} features)")
    return booster, feature_cols


# ---------------------------------------------------------------------------
# XGBoost native importance
# ---------------------------------------------------------------------------

def extract_importance(booster, feature_cols: list[str]) -> pd.DataFrame:
    """Extract gain, weight, cover from the XGBoost booster."""
    score_types = ["gain", "weight", "cover"]
    imp = booster.get_score(importance_type="gain")
    imp_w = booster.get_score(importance_type="weight")
    imp_c = booster.get_score(importance_type="cover")

    # Map f0..f131 back to real feature names
    records = []
    for i, feat in enumerate(feature_cols):
        fkey = f"f{i}"
        g = imp.get(fkey, 0.0)
        w = imp_w.get(fkey, 0.0)
        c = imp_c.get(fkey, 0.0)
        records.append({
            "feature": feat,
            "feature_group": assign_group(feat),
            "gain": g,
            "weight": w,
            "cover": c,
        })

    df = pd.DataFrame(records)

    # Percentages
    total_gain = df["gain"].sum()
    total_weight = df["weight"].sum()
    total_cover = df["cover"].sum()

    df["gain_percent"] = np.where(total_gain > 0, df["gain"] / total_gain * 100, 0.0)
    df["weight_percent"] = np.where(total_weight > 0, df["weight"] / total_weight * 100, 0.0)
    df["cover_percent"] = np.where(total_cover > 0, df["cover"] / total_cover * 100, 0.0)

    df = df.sort_values("gain", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Group analysis
# ---------------------------------------------------------------------------

def group_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate importance by feature group."""
    grp = (
        df.groupby("feature_group", sort=False)
        .agg(
            total_gain=("gain", "sum"),
            gain_percent=("gain_percent", "sum"),
            feature_count=("feature", "count"),
        )
        .sort_values("total_gain", ascending=False)
        .reset_index()
    )
    return grp


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(df_feat: pd.DataFrame, grp: pd.DataFrame, feature_cols: list[str]):
    """Run sanity checks before saving."""
    errors = []

    # Exactly 132 features
    if len(df_feat) != 132:
        errors.append(f"Expected 132 features, got {len(df_feat)}")

    # No missing features
    model_features = set(feature_cols)
    analysis_features = set(df_feat["feature"])
    missing = model_features - analysis_features
    extra = analysis_features - model_features
    if missing:
        errors.append(f"Missing features: {missing}")
    if extra:
        errors.append(f"Extra features: {extra}")

    # No duplicates
    dups = df_feat["feature"][df_feat["feature"].duplicated()].tolist()
    if dups:
        errors.append(f"Duplicate features: {dups}")

    # No NaN/Inf
    for col in ["gain", "weight", "cover", "gain_percent", "weight_percent", "cover_percent"]:
        n_bad = df_feat[col].isna().sum() + np.isinf(df_feat[col]).sum()
        if n_bad > 0:
            errors.append(f"{col} has {n_bad} NaN/Inf values")

    # Gain percentages sum ~100%
    gain_sum = df_feat["gain_percent"].sum()
    if abs(gain_sum - 100.0) > 0.01:
        errors.append(f"gain_percent sums to {gain_sum:.4f}, expected ~100")

    # Group gain percentages sum ~100%
    grp_gain_sum = grp["gain_percent"].sum()
    if abs(grp_gain_sum - 100.0) > 0.01:
        errors.append(f"Group gain_percent sums to {grp_gain_sum:.4f}, expected ~100")

    if errors:
        for e in errors:
            print(f"  VALIDATION ERROR: {e}")
        sys.exit(1)
    else:
        print("  Validation: PASSED (132 features, no NaN/Inf, sums ~100%)")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_top30(df: pd.DataFrame, out_path: Path):
    """Bar chart of top 30 features by gain."""
    top = df.head(30).copy()
    top = top.sort_values("gain", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top)))
    ax.barh(top["feature"], top["gain"], color=colors)
    ax.set_xlabel("Gain", fontsize=12)
    ax.set_title("Morph V2 — Top 30 Features by Gain", fontsize=14, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


def plot_groups(grp: pd.DataFrame, out_path: Path):
    """Bar chart of total gain percentage by feature group."""
    grp_sorted = grp.sort_values("gain_percent", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(grp_sorted)))
    bars = ax.barh(grp_sorted["feature_group"], grp_sorted["gain_percent"], color=colors)

    for bar, pct in zip(bars, grp_sorted["gain_percent"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=10)

    ax.set_xlabel("Gain %", fontsize=12)
    ax.set_title("Morph V2 — Feature Group Importance (Gain %)", fontsize=14, fontweight="bold")
    ax.set_xlim(0, max(grp_sorted["gain_percent"]) * 1.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# SHAP analysis (optional)
# ---------------------------------------------------------------------------

def try_shap(booster, feature_cols: list[str]):
    """Attempt SHAP analysis. Returns (bool, int) = (performed, n_samples)."""
    try:
        import shap
    except ImportError:
        print("\n  SHAP: SKIPPED (shap not installed)")
        return False, 0

    print("\n  SHAP: Available — computing TreeExplainer …")

    # Load EVAL split from ASVspoof
    if not ASVSPOOF_PARQUET.exists():
        print(f"  SHAP: SKIPPED (missing {ASVSPOOF_PARQUET})")
        return False, 0

    df_all = pd.read_parquet(ASVSPOOF_PARQUET)
    df_eval = df_all[df_all["split"] == "eval"].copy()
    print(f"  SHAP: Total EVAL samples available: {len(df_eval)}")

    # Subsample if too large
    if len(df_eval) > SHAP_SAMPLE_SIZE:
        df_eval = df_eval.sample(n=SHAP_SAMPLE_SIZE, random_state=SHAP_RANDOM_STATE)
        print(f"  SHAP: Subsampled to {SHAP_SAMPLE_SIZE} (seed={SHAP_RANDOM_STATE})")

    X_eval = df_eval[feature_cols].values.astype(np.float32)
    X_eval = np.nan_to_num(X_eval, nan=0.0, posinf=0.0, neginf=0.0)
    n_samples = len(X_eval)

    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_eval)

    # Mean absolute SHAP per feature
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    shap_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs,
    })
    shap_df["feature_group"] = shap_df["feature"].apply(assign_group)
    shap_df["shap_percent"] = shap_df["mean_abs_shap"] / shap_df["mean_abs_shap"].sum() * 100
    shap_df = shap_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    shap_df.to_csv(SHAP_IMPORTANCE_CSV, index=False)
    print(f"  SHAP CSV  → {SHAP_IMPORTANCE_CSV}")

    # Top 30 plot
    top = shap_df.head(30).copy().sort_values("mean_abs_shap", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.magma(np.linspace(0.3, 0.9, len(top)))
    ax.barh(top["feature"], top["mean_abs_shap"], color=colors)
    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.set_title("Morph V2 — Top 30 Features by SHAP Importance", fontsize=14, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(SHAP_TOP30_PNG, dpi=150)
    plt.close(fig)
    print(f"  SHAP plot → {SHAP_TOP30_PNG}")

    return True, n_samples


# ---------------------------------------------------------------------------
# Terminal report
# ---------------------------------------------------------------------------

def print_report(df_feat: pd.DataFrame, grp: pd.DataFrame,
                 shap_done: bool, shap_n: int):
    """Final summary to stdout."""
    print(f"\n{'='*60}")
    print(f"  MORPH V2 — FEATURE IMPORTANCE REPORT")
    print(f"{'='*60}")

    print(f"\n  Features analyzed: {len(df_feat)}")

    # Top 20
    print(f"\n  TOP 20 FEATURES BY GAIN:")
    print(f"  {'Rank':<5} {'Feature':<35} {'Gain':>10} {'Gain%':>8}")
    print(f"  {'-'*5} {'-'*35} {'-'*10} {'-'*8}")
    for i, row in df_feat.head(20).iterrows():
        print(f"  {i+1:<5} {row['feature']:<35} {row['gain']:>10.2f} {row['gain_percent']:>7.2f}%")

    # Bottom 10
    print(f"\n  BOTTOM 10 FEATURES BY GAIN:")
    print(f"  {'Rank':<5} {'Feature':<35} {'Gain':>10} {'Gain%':>8}")
    print(f"  {'-'*5} {'-'*35} {'-'*10} {'-'*8}")
    tail = df_feat.tail(10).iloc[::-1]
    rank_start = len(df_feat) - 9
    for idx, (_, row) in enumerate(tail.iterrows()):
        print(f"  {rank_start + idx:<5} {row['feature']:<35} {row['gain']:>10.2f} {row['gain_percent']:>7.2f}%")

    # Group ranking
    print(f"\n  FEATURE GROUP RANKING (by gain):")
    print(f"  {'Rank':<5} {'Group':<15} {'Features':>9} {'Total Gain':>12} {'Gain%':>8}")
    print(f"  {'-'*5} {'-'*15} {'-'*9} {'-'*12} {'-'*8}")
    for i, row in grp.iterrows():
        print(f"  {i+1:<5} {row['feature_group']:<15} {row['feature_count']:>9} {row['total_gain']:>12.2f} {row['gain_percent']:>7.2f}%")

    total_gain_pct = grp["gain_percent"].sum()
    print(f"\n  Total gain % across groups: {total_gain_pct:.2f}%")

    # SHAP status
    print(f"\n  SHAP analysis: {'YES' if shap_done else 'SKIPPED (shap not installed)'}")
    if shap_done:
        print(f"  SHAP samples used: {shap_n}")

    # Output files
    print(f"\n  GENERATED FILES:")
    for p in [FEATURE_IMPORTANCE_CSV, GROUP_IMPORTANCE_CSV, TOP30_PNG, GROUP_PNG]:
        print(f"    {p}")
    if shap_done:
        print(f"    {SHAP_IMPORTANCE_CSV}")
        print(f"    {SHAP_TOP30_PNG}")

    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Load model + feature order
    booster, feature_cols = load_model_and_features()

    # 2. Extract XGBoost importance
    print("\nExtracting XGBoost feature importance …")
    df_feat = extract_importance(booster, feature_cols)

    # 3. Group analysis
    grp = group_analysis(df_feat)

    # 4. Validate
    print("\nValidating …")
    validate(df_feat, grp, feature_cols)

    # 5. Save CSVs
    df_feat.to_csv(FEATURE_IMPORTANCE_CSV, index=False)
    grp.to_csv(GROUP_IMPORTANCE_CSV, index=False)
    print(f"\nSaved → {FEATURE_IMPORTANCE_CSV}")
    print(f"Saved → {GROUP_IMPORTANCE_CSV}")

    # 6. Save plots
    print("\nGenerating plots …")
    plot_top30(df_feat, TOP30_PNG)
    plot_groups(grp, GROUP_PNG)

    # 7. SHAP (optional)
    shap_done, shap_n = try_shap(booster, feature_cols)

    # 8. Terminal report
    print_report(df_feat, grp, shap_done, shap_n)


if __name__ == "__main__":
    main()
