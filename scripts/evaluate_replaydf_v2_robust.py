#!/usr/bin/env python3
"""
Offline evaluation of Morph V2 Robust on 100 selected ReplayDF recordings.

- Reuses EXACT 132-feature extraction (ai_engine.features.extraction.extract_features)
- Reuses existing V2 Robust model (morph_xgboost_v2_robust.json + metadata)
- Reads data/raw/replaydf/selected/selection.csv (| delimited)
- For each file: load WAV -> extract features -> predict -> compare vs ground truth
  spoof -> FAKE (1), bona-fide -> REAL (0)
- Computes accuracy, precision, recall, F1, ROC-AUC, FPR, FNR,
  spoof detection rate (TPR), bona-fide detection rate (TNR)
- Saves per-file results to data/features/replaydf_v2_robust_results.csv
- Prints concise summary
- Skips/reports errors per-file without crashing whole run

Usage:
    venv_new/bin/python scripts/evaluate_replaydf_v2_robust.py
    venv_new/bin/python scripts/evaluate_replaydf_v2_robust.py --selection data/raw/replaydf/selected/selection.csv --output data/features/replaydf_v2_robust_results.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Ensure project root on sys.path for ai_engine imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_engine.features.extraction import SR, extract_features
from ai_engine.inference.model import MorphModel

SELECTION_CSV = PROJECT_ROOT / "data/raw/replaydf/selected/selection.csv"
REPLAYDF_ROOT = PROJECT_ROOT / "data/raw/replaydf"
OUTPUT_CSV = PROJECT_ROOT / "data/features/replaydf_v2_robust_results.csv"

LABEL_MAP = {"spoof": 1, "bona-fide": 0}
LABEL_STR_MAP = {1: "FAKE", 0: "REAL"}

def compute_metrics(y_true: List[int], y_pred: List[int], y_score: List[float]) -> Dict[str, float]:
    """Calculate required metrics. FAKE=1 positive."""
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    y_score_arr = np.array(y_score)

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()
    # confusion_matrix returns TN, FP, FN, TP for binary 0/1
    accuracy = accuracy_score(y_true_arr, y_pred_arr)
    # zero_division=0 to avoid warnings when no predicted positives
    precision = precision_score(y_true_arr, y_pred_arr, zero_division=0)
    recall = recall_score(y_true_arr, y_pred_arr, zero_division=0)
    f1 = f1_score(y_true_arr, y_pred_arr, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true_arr, y_score_arr)
    except ValueError:
        roc_auc = float("nan")

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    # Spoof detection rate = TPR = recall
    spoof_dr = recall
    # Bona-fide detection rate = TNR = tn / (tn+fp)
    bona_fide_dr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "spoof_detection_rate": float(spoof_dr),
        "bona_fide_detection_rate": float(bona_fide_dr),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

def evaluate(selection_path: Path = SELECTION_CSV, output_path: Path = OUTPUT_CSV) -> Tuple[pd.DataFrame, Dict[str, float]]:
    print(f"[ReplayDF Eval] Selection: {selection_path}")
    print(f"[ReplayDF Eval] Output   : {output_path}")
    print(f"[ReplayDF Eval] Model    : v2_robust")
    print(f"[ReplayDF Eval] Features : 132 (extract_features)")

    if not selection_path.exists():
        raise FileNotFoundError(f"Selection CSV not found: {selection_path}")

    # Load model once
    print("[ReplayDF Eval] Loading V2 Robust model...")
    model = MorphModel(version="v2_robust")
    print(f"[ReplayDF Eval] Model loaded: {model.version} features={model.feature_count}")
    assert model.feature_count == 132, f"Expected 132 features, got {model.feature_count}"

    # Read selection
    rows: List[Dict[str, str]] = []
    with open(selection_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for r in reader:
            rows.append(r)
    print(f"[ReplayDF Eval] Selected entries: {len(rows)} (expected 100)")
    if len(rows) != 100:
        print(f"[WARN] Expected 100 entries, found {len(rows)}")

    per_file: List[Dict] = []
    y_true: List[int] = []
    y_pred: List[int] = []
    y_score: List[float] = []
    errors: List[Dict] = []

    for idx, row in enumerate(rows, 1):
        uid = row.get("uid", "")
        recorded_file = row.get("recorded_file", "")
        label_str_raw = row.get("label", "")
        true_label = LABEL_MAP.get(label_str_raw)
        if true_label is None:
            print(f"[WARN] Row {idx} uid={uid} unknown label '{label_str_raw}' -> skip")
            errors.append({"uid": uid, "recorded_file": recorded_file, "error": f"unknown label {label_str_raw}"})
            continue

        # Resolve absolute wav path: REPLAYDF_ROOT / recorded_file
        # recorded_file is like wav/<uid>/benign/en/<file>.wav
        wav_path = REPLAYDF_ROOT / recorded_file
        original_file = row.get("original_file", "")

        if not wav_path.exists():
            msg = f"WAV not found: {wav_path}"
            print(f"[ERROR] {msg} (uid={uid}) -> skip")
            errors.append({"uid": uid, "recorded_file": recorded_file, "error": msg})
            per_file.append({
                "uid": uid,
                "recorded_file": recorded_file,
                "original_file": original_file,
                "label": label_str_raw,
                "true_label": true_label,
                "true_str": LABEL_STR_MAP[true_label],
                "pred_label": "",
                "pred_str": "",
                "real_prob": "",
                "fake_prob": "",
                "confidence": "",
                "duration": "",
                "correct": "",
                "error": msg,
                "architecture": row.get("architecture", ""),
                "language": row.get("language", ""),
                "mic": row.get("mic", ""),
                "speaker": row.get("speaker", ""),
            })
            continue

        try:
            y, sr = librosa.load(str(wav_path), sr=SR, mono=True)
            duration = librosa.get_duration(y=y, sr=SR)
            feats = extract_features(y, SR, duration)
            # Strict feature count check
            if len(feats) != 132:
                raise ValueError(f"extract_features returned {len(feats)} features, expected 132")
            pred = model.predict(feats)
            real_prob, fake_prob = model.predict_proba(feats)
            confidence = real_prob if pred == 0 else fake_prob
            pred_str = LABEL_STR_MAP[pred]
            correct = int(pred == true_label)

            y_true.append(true_label)
            y_pred.append(pred)
            y_score.append(float(fake_prob))

            per_file.append({
                "uid": uid,
                "recorded_file": recorded_file,
                "original_file": original_file,
                "label": label_str_raw,
                "true_label": true_label,
                "true_str": LABEL_STR_MAP[true_label],
                "pred_label": pred,
                "pred_str": pred_str,
                "real_prob": round(float(real_prob), 6),
                "fake_prob": round(float(fake_prob), 6),
                "confidence": round(float(confidence), 6),
                "duration": round(float(duration), 3),
                "correct": correct,
                "error": "",
                "architecture": row.get("architecture", ""),
                "language": row.get("language", ""),
                "mic": row.get("mic", ""),
                "speaker": row.get("speaker", ""),
            })
            print(f"[{idx:3d}/{len(rows)}] {uid} {label_str_raw:9s} -> {pred_str:4s} P(FAKE)={fake_prob:.4f} {'✓' if correct else '✗'}")

        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            print(f"[ERROR] Row {idx} uid={uid} {wav_path} -> {msg} -> skip")
            import traceback
            traceback.print_exc()
            errors.append({"uid": uid, "recorded_file": recorded_file, "error": msg})
            per_file.append({
                "uid": uid,
                "recorded_file": recorded_file,
                "original_file": original_file,
                "label": label_str_raw,
                "true_label": true_label,
                "true_str": LABEL_STR_MAP[true_label],
                "pred_label": "",
                "pred_str": "",
                "real_prob": "",
                "fake_prob": "",
                "confidence": "",
                "duration": "",
                "correct": "",
                "error": msg,
                "architecture": row.get("architecture", ""),
                "language": row.get("language", ""),
                "mic": row.get("mic", ""),
                "speaker": row.get("speaker", ""),
            })
            continue

    # Compute metrics on successfully evaluated files
    if len(y_true) == 0:
        raise RuntimeError("No files successfully evaluated — cannot compute metrics")

    metrics = compute_metrics(y_true, y_pred, y_score)
    # Add counts
    metrics["n_total"] = len(y_true)
    metrics["n_errors"] = len(errors)
    metrics["n_selected"] = len(rows)

    # Save per-file csv
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(per_file)
    # Define column order
    cols = ["uid","recorded_file","original_file","label","true_label","true_str","pred_label","pred_str","real_prob","fake_prob","confidence","duration","correct","error","architecture","language","mic","speaker"]
    # Only keep existing cols
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    df.to_csv(output_path, index=False)
    print(f"\n[ReplayDF Eval] Per-file results saved to {output_path} ({len(df)} rows)")

    if errors:
        print(f"[ReplayDF Eval] {len(errors)} errors/skips (see error column)")

    # Print concise summary
    print("\n" + "="*68)
    print(" ReplayDF V2 Robust — Offline Evaluation Summary (100 selected)")
    print("="*68)
    print(f" Evaluated : {metrics['n_total']}/{metrics['n_selected']} (errors: {metrics['n_errors']})")
    print(f" Confusion : TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']} TP={metrics['tp']}")
    print(f" Accuracy  : {metrics['accuracy']:.4f}")
    print(f" Precision : {metrics['precision']:.4f}  (FAKE as positive)")
    print(f" Recall    : {metrics['recall']:.4f}  (spoof detection rate)")
    print(f" F1        : {metrics['f1']:.4f}")
    print(f" ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f" FPR       : {metrics['fpr']:.4f}  (bona-fide false alarm)")
    print(f" FNR       : {metrics['fnr']:.4f}  (spoof miss)")
    print(f" Spoof DR  : {metrics['spoof_detection_rate']:.4f}")
    print(f" BonaF DR  : {metrics['bona_fide_detection_rate']:.4f}")
    print("="*68 + "\n")

    return df, metrics

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Morph V2 Robust on ReplayDF selected set")
    parser.add_argument("--selection", type=str, default=str(SELECTION_CSV), help="Path to selection.csv")
    parser.add_argument("--output", type=str, default=str(OUTPUT_CSV), help="Output CSV path")
    args = parser.parse_args()
    evaluate(Path(args.selection), Path(args.output))

if __name__ == "__main__":
    main()
