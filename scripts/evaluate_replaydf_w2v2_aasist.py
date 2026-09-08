#!/usr/bin/env python3
"""
Offline evaluation of W2V2-AASIST on 100 selected ReplayDF recordings.

- Reuses official pretrained W2V2-AASIST (XLS-R 300M + AASIST) via
  ai_engine/inference/w2v2_aasist.py (thin wrapper around
  TakHemlata/SSL_Anti-spoofing LA_model.pth)
- Reads data/raw/replaydf/selected/selection.csv (| delimited)
- For each file: mono 16kHz, pad/truncate to 64600, infer, convert
  bona-fide logit -> P(fake) (higher = spoof)
- Computes confusion matrix, accuracy, precision, spoof recall/F1,
  ROC-AUC, FPR, FNR, bona-fide DR, spoof DR, REAL correct/50, SPOOF correct/50,
  average inference time
- Saves per-file results to data/features/replaydf_w2v2_aasist_results.csv
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai_engine.inference.w2v2_aasist import W2V2AASISTModel, TARGET_SR

SELECTION_CSV = PROJECT_ROOT / "data/raw/replaydf/selected/selection.csv"
REPLAYDF_ROOT = PROJECT_ROOT / "data/raw/replaydf"
OUTPUT_CSV = PROJECT_ROOT / "data/features/replaydf_w2v2_aasist_results.csv"

LABEL_MAP = {"spoof": 1, "bona-fide": 0}
LABEL_STR_MAP = {1: "FAKE", 0: "REAL"}

def compute_metrics(y_true: List[int], y_pred: List[int], y_score: List[float]) -> Dict[str, float]:
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    y_score_arr = np.array(y_score)
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()
    accuracy = accuracy_score(y_true_arr, y_pred_arr)
    precision = precision_score(y_true_arr, y_pred_arr, zero_division=0)
    recall = recall_score(y_true_arr, y_pred_arr, zero_division=0)
    f1 = f1_score(y_true_arr, y_pred_arr, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true_arr, y_score_arr)
    except ValueError:
        roc_auc = float("nan")
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    bona_fide_dr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    spoof_dr = recall
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "bona_fide_detection_rate": float(bona_fide_dr),
        "spoof_detection_rate": float(spoof_dr),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

def evaluate(selection_path: Path = SELECTION_CSV, output_path: Path = OUTPUT_CSV) -> Tuple[pd.DataFrame, Dict[str, float]]:
    print(f"[ReplayDF W2V2-AASIST] Selection: {selection_path}")
    print(f"[ReplayDF W2V2-AASIST] Output: {output_path}")
    print(f"[ReplayDF W2V2-AASIST] Model: w2v2_aasist (XLS-R 300M + AASIST, LA_model.pth)")
    print(f"[ReplayDF W2V2-AASIST] Handling: mono 16kHz, 64600 samples")

    if not selection_path.exists():
        raise FileNotFoundError(f"Selection CSV not found: {selection_path}")

    # Load model once
    print("[ReplayDF W2V2-AASIST] Loading W2V2-AASIST...")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ReplayDF W2V2-AASIST] Device: {device} (cuda available: {torch.cuda.is_available()})")
    model = W2V2AASISTModel(device=device)
    print(f"[ReplayDF W2V2-AASIST] Model loaded: version={model.version} checkpoint={model.checkpoint_path} loaded={model.is_loaded}")

    rows: List[Dict[str, str]] = []
    with open(selection_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for r in reader:
            rows.append(r)
    print(f"[ReplayDF W2V2-AASIST] Selected: {len(rows)}")

    per_file: List[Dict] = []
    y_true: List[int] = []
    y_pred: List[int] = []
    y_score: List[float] = []
    errors: List[Dict] = []
    infer_times: List[float] = []

    for idx, row in enumerate(rows, 1):
        uid = row.get("uid", "")
        recorded_file = row.get("recorded_file", "")
        label_str_raw = row.get("label", "")
        true_label = LABEL_MAP.get(label_str_raw)
        if true_label is None:
            print(f"[WARN] Row {idx} unknown label")
            continue
        wav_path = REPLAYDF_ROOT / recorded_file
        original_file = row.get("original_file", "")
        if not wav_path.exists():
            msg = f"WAV not found: {wav_path}"
            print(f"[ERROR] {msg}")
            per_file.append({
                "uid": uid, "recorded_file": recorded_file, "original_file": original_file,
                "label": label_str_raw, "true_label": true_label, "true_str": LABEL_STR_MAP[true_label],
                "pred_label": "", "pred_str": "", "real_prob": "", "fake_prob": "", "confidence": "", "duration": "", "correct": "", "error": msg,
                "architecture": row.get("architecture", ""), "language": row.get("language", ""), "mic": row.get("mic", ""), "speaker": row.get("speaker", ""), "infer_time": ""
            })
            errors.append({"uid": uid, "error": msg})
            continue
        try:
            y, sr = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
            duration = librosa.get_duration(y=y, sr=TARGET_SR)
            t0 = time.time()
            p_real, p_fake = model.predict_proba(y, sr)
            infer_time = time.time() - t0
            infer_times.append(infer_time)
            pred = 1 if p_fake >= 0.5 else 0
            pred_str = LABEL_STR_MAP[pred]
            correct = int(pred == true_label)
            y_true.append(true_label)
            y_pred.append(pred)
            y_score.append(float(p_fake))
            per_file.append({
                "uid": uid, "recorded_file": recorded_file, "original_file": original_file,
                "label": label_str_raw, "true_label": true_label, "true_str": LABEL_STR_MAP[true_label],
                "pred_label": pred, "pred_str": pred_str,
                "real_prob": round(float(p_real), 6), "fake_prob": round(float(p_fake), 6),
                "confidence": round(float(max(p_real, p_fake)), 6),
                "duration": round(float(duration), 3),
                "correct": correct, "error": "",
                "architecture": row.get("architecture", ""), "language": row.get("language", ""), "mic": row.get("mic", ""), "speaker": row.get("speaker", ""), "infer_time": round(float(infer_time), 4)
            })
            print(f"[{idx:3d}/{len(rows)}] {uid} {label_str_raw:9s} -> {pred_str:4s} P(FAKE)={p_fake:.4f} {'✓' if correct else '✗'} {infer_time:.2f}s")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            msg = f"{type(exc).__name__}: {exc}"
            print(f"[ERROR] {msg}")
            per_file.append({
                "uid": uid, "recorded_file": recorded_file, "original_file": original_file,
                "label": label_str_raw, "true_label": true_label, "true_str": LABEL_STR_MAP[true_label],
                "pred_label": "", "pred_str": "", "real_prob": "", "fake_prob": "", "confidence": "", "duration": "", "correct": "", "error": msg,
                "architecture": row.get("architecture", ""), "language": row.get("language", ""), "mic": row.get("mic", ""), "speaker": row.get("speaker", ""), "infer_time": ""
            })
            errors.append({"uid": uid, "error": msg})

    if len(y_true) == 0:
        raise RuntimeError("No files evaluated")

    metrics = compute_metrics(y_true, y_pred, y_score)
    metrics["n_total"] = len(y_true)
    metrics["n_errors"] = len(errors)
    metrics["n_selected"] = len(rows)
    metrics["avg_infer_time"] = float(np.mean(infer_times)) if infer_times else 0.0
    # REAL/SPoof correct
    # y_true 0=REAL, 1=FAKE
    # Count
    real_correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    spoof_correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    metrics["real_correct"] = int(real_correct)
    metrics["spoof_correct"] = int(spoof_correct)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(per_file)
    cols = ["uid","recorded_file","original_file","label","true_label","true_str","pred_label","pred_str","real_prob","fake_prob","confidence","duration","correct","error","architecture","language","mic","speaker","infer_time"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    df.to_csv(output_path, index=False)
    print(f"\n[ReplayDF W2V2-AASIST] Saved {len(df)} rows to {output_path}")
    if errors:
        print(f"[ReplayDF W2V2-AASIST] {len(errors)} errors")

    print("\n" + "="*68)
    print(" ReplayDF W2V2-AASIST — Offline Evaluation Summary (100 selected)")
    print("="*68)
    print(f" Evaluated : {metrics['n_total']}/{metrics['n_selected']} (errors: {metrics['n_errors']})")
    print(f" Device    : {device}")
    print(f" Checkpoint: {model.checkpoint_path}")
    print(f" Confusion : TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']} TP={metrics['tp']}")
    print(f" REAL correct : {metrics['real_correct']}/50")
    print(f" SPOOF correct: {metrics['spoof_correct']}/50")
    print(f" Accuracy  : {metrics['accuracy']:.4f}")
    print(f" Precision : {metrics['precision']:.4f}")
    print(f" Recall    : {metrics['recall']:.4f} (spoof DR)")
    print(f" F1        : {metrics['f1']:.4f}")
    print(f" ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f" FPR       : {metrics['fpr']:.4f}")
    print(f" FNR       : {metrics['fnr']:.4f}")
    print(f" BonaF DR  : {metrics['bona_fide_detection_rate']:.4f}")
    print(f" Spoof DR  : {metrics['spoof_detection_rate']:.4f}")
    print(f" Avg infer : {metrics['avg_infer_time']:.4f}s / file")
    print("="*68 + "\n")

    return df, metrics

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate W2V2-AASIST on ReplayDF")
    parser.add_argument("--selection", type=str, default=str(SELECTION_CSV))
    parser.add_argument("--output", type=str, default=str(OUTPUT_CSV))
    args = parser.parse_args()
    evaluate(Path(args.selection), Path(args.output))

if __name__ == "__main__":
    main()
