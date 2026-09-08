"""
Morph V2 Diagnostic Script — Google Agent Browser Recording

Inspects the recording, runs V2 inference on full file and per 4-second window,
compares feature distributions against known REAL/FAKE samples, and diagnoses
classification behavior.

Usage:
    python _diag_tmp/diagnose_google_agent.py
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

import subprocess

import librosa
import numpy as np

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai_engine.features.extraction import SR, extract_features
from ai_engine.inference.model import MorphModel
from ai_engine.inference.pipeline import detect, DetectionResult

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUT_DIR = PROJECT_ROOT / "_diag_tmp"
RECORDING_WEBM = PROJECT_ROOT / "morph-diag-1788791823829.webm"
RECORDING_WAV = OUT_DIR / "morph-diag-1788791823829_converted.wav"

KNOWN_REAL = PROJECT_ROOT / "data/raw/gary_stafford/real/gs_00202.wav"
KNOWN_FAKE = PROJECT_ROOT / "data/raw/gary_stafford/fake/gs_01619.wav"

WINDOW_SEC = 4.0  # training chunk size

# ---------------------------------------------------------------------------
# 0. Convert WebM to WAV (librosa/soundfile can't read WebM)
# ---------------------------------------------------------------------------
def convert_webm_to_wav():
    print("Converting WebM → WAV (16kHz mono) for pipeline compatibility...")
    cmd = [
        "ffmpeg", "-y", "-i", str(RECORDING_WEBM),
        "-ar", str(SR), "-ac", "1", "-sample_fmt", "s16",
        str(RECORDING_WAV),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    size_kb = RECORDING_WAV.stat().st_size / 1024
    print(f"  Converted: {RECORDING_WAV.name} ({size_kb:.1f} KB)")
    print()


# ---------------------------------------------------------------------------
# 1. Full-pipeline detection on the entire recording
# ---------------------------------------------------------------------------
def run_full_detection() -> DetectionResult:
    print("=" * 70)
    print("STEP 1: Full-pipeline V2 detection on entire recording")
    print("=" * 70)
    result = detect(RECORDING_WAV, model_version="v2")
    print(f"  Label:          {result.label_str}")
    print(f"  Confidence:     {result.confidence:.6f}")
    print(f"  P(REAL):        {result.real_probability:.6f}")
    print(f"  P(FAKE):        {result.fake_probability:.6f}")
    print(f"  Duration:       {result.duration:.3f}s")
    print(f"  Model version:  {result.model_version}")
    print()
    return result


# ---------------------------------------------------------------------------
# 2. Windowed detection (non-overlapping 4-second windows)
# ---------------------------------------------------------------------------
def run_windowed_detection() -> List[Dict]:
    print("=" * 70)
    print("STEP 2: Per-window V2 detection (4-second non-overlapping)")
    print("=" * 70)

    y, _ = librosa.load(str(RECORDING_WAV), sr=SR, mono=True)
    total_samples = len(y)
    window_samples = int(WINDOW_SEC * SR)
    n_windows = total_samples // window_samples
    remainder_samples = total_samples % window_samples

    results = []
    model = MorphModel(version="v2")

    for i in range(n_windows):
        start_sample = i * window_samples
        end_sample = start_sample + window_samples
        chunk = y[start_sample:end_sample]

        start_sec = start_sample / SR
        end_sec = end_sample / SR

        duration = float(WINDOW_SEC)
        features = extract_features(chunk, SR, duration)
        label = model.predict(features)
        real_prob, fake_prob = model.predict_proba(features)

        r = {
            "window": i + 1,
            "start_sec": round(start_sec, 3),
            "end_sec": round(end_sec, 3),
            "label": "FAKE" if label == 1 else "REAL",
            "real_prob": round(real_prob, 6),
            "fake_prob": round(fake_prob, 6),
        }
        results.append(r)
        print(f"  Window {r['window']:2d}  [{r['start_sec']:6.2f}s – {r['end_sec']:6.2f}s]  "
              f"P(REAL)={r['real_prob']:.4f}  P(FAKE)={r['fake_prob']:.4f}  → {r['label']}")

    if remainder_samples > 0:
        start_sample = n_windows * window_samples
        chunk = y[start_sample:]
        start_sec = start_sample / SR
        end_sec = total_samples / SR
        duration = end_sec - start_sec
        features = extract_features(chunk, SR, duration)
        label = model.predict(features)
        real_prob, fake_prob = model.predict_proba(features)
        r = {
            "window": n_windows + 1,
            "start_sec": round(start_sec, 3),
            "end_sec": round(end_sec, 3),
            "label": "FAKE" if label == 1 else "REAL",
            "real_prob": round(real_prob, 6),
            "fake_prob": round(fake_prob, 6),
            "note": f"partial window ({duration:.2f}s)",
        }
        results.append(r)
        print(f"  Window {r['window']:2d}  [{r['start_sec']:6.2f}s – {r['end_sec']:6.2f}s]  "
              f"P(REAL)={r['real_prob']:.4f}  P(FAKE)={r['fake_prob']:.4f}  → {r['label']}  "
              f"[partial]")

    print()
    return results


# ---------------------------------------------------------------------------
# 3. Window statistics
# ---------------------------------------------------------------------------
def compute_window_stats(window_results: List[Dict]) -> Dict:
    print("=" * 70)
    print("STEP 3: Window-level statistics")
    print("=" * 70)

    fake_probs = [r["fake_prob"] for r in window_results]
    real_probs = [r["real_prob"] for r in window_results]
    n_fake = sum(1 for r in window_results if r["label"] == "FAKE")
    n_real = sum(1 for r in window_results if r["label"] == "REAL")
    n_total = len(window_results)

    stats = {
        "num_windows": n_total,
        "mean_fake_prob": round(float(np.mean(fake_probs)), 6),
        "max_fake_prob": round(float(np.max(fake_probs)), 6),
        "min_fake_prob": round(float(np.min(fake_probs)), 6),
        "std_fake_prob": round(float(np.std(fake_probs)), 6),
        "mean_real_prob": round(float(np.mean(real_probs)), 6),
        "pct_fake": round(n_fake / n_total * 100, 2),
        "pct_real": round(n_real / n_total * 100, 2),
        "n_fake": n_fake,
        "n_real": n_real,
    }

    print(f"  Total windows:        {stats['num_windows']}")
    print(f"  Windows classified FAKE: {stats['n_fake']}  ({stats['pct_fake']:.1f}%)")
    print(f"  Windows classified REAL: {stats['n_real']}  ({stats['pct_real']:.1f}%)")
    print(f"  Mean P(FAKE):         {stats['mean_fake_prob']:.6f}")
    print(f"  Max  P(FAKE):         {stats['max_fake_prob']:.6f}")
    print(f"  Min  P(FAKE):         {stats['min_fake_prob']:.6f}")
    print(f"  Std  P(FAKE):         {stats['std_fake_prob']:.6f}")
    print()
    return stats


# ---------------------------------------------------------------------------
# 4. Feature distribution comparison
# ---------------------------------------------------------------------------
FOCUS_FEATURES = [
    # MFCC
    "mfcc_1_mean", "mfcc_1_std", "mfcc_2_mean", "mfcc_2_std",
    "mfcc_3_mean", "mfcc_3_std", "mfcc_4_mean", "mfcc_4_std",
    "mfcc_5_mean", "mfcc_5_std",
    # CQCC
    "cqcc_1_mean", "cqcc_1_std", "cqcc_2_mean", "cqcc_2_std",
    "cqcc_3_mean", "cqcc_3_std",
    # Spectral
    "spectral_centroid_mean", "spectral_centroid_std",
    "spectral_bandwidth_mean", "spectral_bandwidth_std",
    "spectral_rolloff_mean", "spectral_rolloff_std",
    "spectral_contrast_1_mean", "spectral_contrast_1_std",
    "spectral_contrast_2_mean", "spectral_contrast_2_std",
    "spectral_contrast_3_mean", "spectral_contrast_3_std",
    # Pitch / energy
    "f0_mean", "f0_std", "voiced_frame_ratio",
    "rms_mean", "rms_std",
    "zcr_mean", "zcr_std",
    "duration",
]


def extract_window_features(y_full: np.ndarray, window_idx: int, window_samples: int) -> Dict[str, float]:
    """Extract features from a specific window of the full audio."""
    start = window_idx * window_samples
    end = start + window_samples
    chunk = y_full[start:end]
    duration = len(chunk) / SR
    return extract_features(chunk, SR, duration)


def compare_feature_distributions():
    print("=" * 70)
    print("STEP 4: Feature distribution comparison")
    print("=" * 70)

    model = MorphModel(version="v2")
    feature_cols = model.feature_names

    # Load all three audio files
    y_google, _ = librosa.load(str(RECORDING_WAV), sr=SR, mono=True)
    y_real, _ = librosa.load(str(KNOWN_REAL), sr=SR, mono=True)
    y_fake, _ = librosa.load(str(KNOWN_FAKE), sr=SR, mono=True)

    # Extract features from each
    dur_google = len(y_google) / SR
    dur_real = len(y_real) / SR
    dur_fake = len(y_fake) / SR

    feat_google = extract_features(y_google, SR, dur_google)
    feat_real = extract_features(y_real, SR, dur_real)
    feat_fake = extract_features(y_fake, SR, dur_fake)

    # Also compute per-window features for Google Agent
    window_samples = int(WINDOW_SEC * SR)
    n_windows = len(y_google) // window_samples
    window_feats = []
    for i in range(n_windows):
        wf = extract_window_features(y_google, i, window_samples)
        window_feats.append(wf)

    # Compute stats across windows
    google_window_means = {}
    google_window_stds = {}
    for fc in feature_cols:
        vals = [wf[fc] for wf in window_feats]
        google_window_means[fc] = float(np.mean(vals))
        google_window_stds[fc] = float(np.std(vals))

    # Focus feature comparison table
    print()
    print(f"  {'Feature':<30s} {'Google(full)':>14s} {'REAL sample':>14s} {'FAKE sample':>14s} "
          f"{'Google(winμ)':>14s} {'Google(winσ)':>14s}")
    print("  " + "-" * 104)

    comparison_rows = []
    for fc in FOCUS_FEATURES:
        g_val = feat_google[fc]
        r_val = feat_real[fc]
        f_val = feat_fake[fc]
        wm = google_window_means[fc]
        ws = google_window_stds[fc]
        print(f"  {fc:<30s} {g_val:>14.4f} {r_val:>14.4f} {f_val:>14.4f} {wm:>14.4f} {ws:>14.4f}")
        comparison_rows.append({
            "feature": fc,
            "google_agent_full": round(g_val, 6),
            "known_real": round(r_val, 6),
            "known_fake": round(f_val, 6),
            "google_window_mean": round(wm, 6),
            "google_window_std": round(ws, 6),
        })

    print()

    # Compute Euclidean distance in feature space
    g_vec = np.array([feat_google[fc] for fc in feature_cols])
    r_vec = np.array([feat_real[fc] for fc in feature_cols])
    f_vec = np.array([feat_fake[fc] for fc in feature_cols])

    # Normalize for fair comparison
    all_vecs = np.stack([g_vec, r_vec, f_vec])
    ranges = np.ptp(all_vecs, axis=0)
    ranges[ranges == 0] = 1.0  # avoid division by zero
    g_norm = (g_vec - all_vecs.min(axis=0)) / ranges
    r_norm = (r_vec - all_vecs.min(axis=0)) / ranges
    f_norm = (f_vec - all_vecs.min(axis=0)) / ranges

    dist_g_r = float(np.linalg.norm(g_norm - r_norm))
    dist_g_f = float(np.linalg.norm(g_norm - f_norm))
    dist_r_f = float(np.linalg.norm(r_norm - f_norm))

    print(f"  Normalized Euclidean distances (all 132 features):")
    print(f"    Google Agent <-> Known REAL:  {dist_g_r:.4f}")
    print(f"    Google Agent <-> Known FAKE:  {dist_g_f:.4f}")
    print(f"    Known REAL   <-> Known FAKE:  {dist_r_f:.4f}")
    print(f"    Google Agent is CLOSER to:    {'REAL' if dist_g_r < dist_g_f else 'FAKE'}")
    print()

    # Per-feature-group analysis
    print("  Per-group mean absolute differences (Google vs REAL vs FAKE):")
    groups = {
        "MFCC": [f"mfcc_{i}_{s}" for i in range(1, 21) for s in ("mean", "std")],
        "CQCC": [f"cqcc_{i}_{s}" for i in range(1, 21) for s in ("mean", "std")],
        "Spectral centroid": ["spectral_centroid_mean", "spectral_centroid_std"],
        "Spectral bandwidth": ["spectral_bandwidth_mean", "spectral_bandwidth_std"],
        "Spectral rolloff": ["spectral_rolloff_mean", "spectral_rolloff_std"],
        "Spectral contrast": [f"spectral_contrast_{i}_{s}" for i in range(1, 8) for s in ("mean", "std")],
        "F0 / Pitch": ["f0_mean", "f0_std", "voiced_frame_ratio"],
        "RMS energy": ["rms_mean", "rms_std"],
        "ZCR": ["zcr_mean", "zcr_std"],
    }

    for group_name, feats in groups.items():
        g_vals = np.array([feat_google[f] for f in feats])
        r_vals = np.array([feat_real[f] for f in feats])
        f_vals = np.array([feat_fake[f] for f in feats])
        mad_gr = float(np.mean(np.abs(g_vals - r_vals)))
        mad_gf = float(np.mean(np.abs(g_vals - f_vals)))
        closer = "REAL" if mad_gr < mad_gf else "FAKE"
        print(f"    {group_name:<20s}  Google↔REAL: {mad_gr:.4f}  Google↔FAKE: {mad_gf:.4f}  → closer to {closer}")

    print()
    return {
        "comparison_rows": comparison_rows,
        "distances": {
            "google_vs_real": round(dist_g_r, 6),
            "google_vs_fake": round(dist_g_f, 6),
            "real_vs_fake": round(dist_r_f, 6),
        },
        "closer_to": "REAL" if dist_g_r < dist_g_f else "FAKE",
    }


# ---------------------------------------------------------------------------
# 5. Diagnosis
# ---------------------------------------------------------------------------
def diagnose(full_result: DetectionResult, window_results: List[Dict],
             window_stats: Dict, feature_comparison: Dict):
    print("=" * 70)
    print("STEP 5: Diagnosis")
    print("=" * 70)
    print()

    # Analyze what happened
    full_label = full_result.label_str
    pct_fake = window_stats["pct_fake"]
    mean_fake = window_stats["mean_fake_prob"]
    max_fake = window_stats["max_fake_prob"]
    closer = feature_comparison["closer_to"]
    dist_real = feature_comparison["distances"]["google_vs_real"]
    dist_fake = feature_comparison["distances"]["google_vs_fake"]

    print(f"  Full recording classification: {full_label}")
    print(f"  Window-level: {pct_fake:.1f}% FAKE, {100-pct_fake:.1f}% REAL")
    print(f"  Mean P(FAKE) across windows:  {mean_fake:.4f}")
    print(f"  Max  P(FAKE) across windows:  {max_fake:.4f}")
    print(f"  Feature-space proximity:      closer to {closer}")
    print()

    # Identify likely causes
    causes = []

    if full_label == "REAL" and pct_fake < 50:
        causes.append({
            "cause": "Model generalization gap",
            "detail": (
                "The V2 model was trained on ASVspoof2019 and Gary Stafford voice clone data. "
                "The Google Agent browser recording likely uses a different TTS engine, different "
                "recording chain (browser → WebM/Opus → resample), or different acoustic conditions "
                "that produce feature distributions within the model's learned REAL boundary. "
                "The model has not seen this specific spoofing method during training."
            ),
        })

    if pct_fake > 0 and pct_fake < 100:
        causes.append({
            "cause": "Windowing sensitivity / partial detection",
            "detail": (
                f"Some windows ({pct_fake:.1f}%) triggered fake detection but not enough to flip "
                "the full-file result. This suggests the spoofing artifacts may be transient or "
                "concentrated in specific segments, and the 4-second windowing with per-window "
                "independent classification misses the aggregate signal."
            ),
        })

    # Check feature-space distance
    if dist_fake > dist_real * 0.9:
        causes.append({
            "cause": "Recording/preprocessing difference",
            "detail": (
                "The Google Agent recording is in WebM/Opus format (48kHz) and is resampled to "
                "16kHz by librosa. Opus codec artifacts, the browser audio pipeline (WebAudio API), "
                "or the resampling process may alter the acoustic features in ways that push them "
                "toward the REAL distribution. The training data uses WAV/FLAC at 16kHz directly."
            ),
        })

    # Check if it's a duration effect
    if full_result.duration > 10 and pct_fake < 50:
        causes.append({
            "cause": "Full-length vs windowing issue",
            "detail": (
                f"The full recording is {full_result.duration:.1f}s. When classified as a single "
                f"chunk, it receives {full_label} with P(FAKE)={full_result.fake_probability:.4f}. "
                "The model's duration feature ({:.1f}) may not match typical spoof durations "
                "in training data. Longer real speech may dominate the feature aggregation."
            ),
        })

    # MFCC/CQCC analysis
    g_mfcc = feature_comparison["comparison_rows"]
    mfcc_diffs_real = [r["google_agent_full"] - r["known_real"]
                       for r in g_mfcc if r["feature"].startswith("mfcc_")]
    mfcc_diffs_fake = [r["google_agent_full"] - r["known_fake"]
                       for r in g_mfcc if r["feature"].startswith("mfcc_")]
    mad_mfcc_gr = float(np.mean(np.abs(mfcc_diffs_real)))
    mad_mfcc_gf = float(np.mean(np.abs(mfcc_diffs_fake)))

    if mad_mfcc_gr < mad_mfcc_gf:
        causes.append({
            "cause": "MFCC/CQCC distribution closer to REAL",
            "detail": (
                f"The MFCC features have mean absolute difference of {mad_mfcc_gr:.4f} from "
                f"Known REAL vs {mad_mfcc_gf:.4f} from Known FAKE. The Google Agent voice's "
                "cepstral characteristics closely match genuine human speech patterns, suggesting "
                "a high-quality TTS that produces natural-sounding spectral envelope."
            ),
        })

    # Print diagnoses
    for i, c in enumerate(causes, 1):
        print(f"  Cause {i}: {c['cause']}")
        for line in textwrap.wrap(c["detail"], width=64):
            print(f"    {line}")
        print()

    # Summary
    print("  " + "=" * 60)
    print("  SUMMARY")
    print("  " + "=" * 60)
    if full_label == "REAL":
        print(f"""
  The V2 model classifies the Google Agent recording as {full_label}
  (P(FAKE) = {full_result.fake_probability:.4f}).

  Primary cause: MODEL GENERALIZATION GAP
  The V2 XGBoost model was trained on ASVspoof2019 LA (voice conversion /
  TTS) and Gary Stafford voice clones. It has NOT been trained on
  Google Agent's specific TTS engine or browser-based recording chain.

  Secondary factors:
  - The WebM/Opus codec + resampling chain alters acoustic features
  - The 48kHz→16kHz downsampling may lose high-frequency spoofing artifacts
  - Window-level analysis shows {pct_fake:.0f}% of windows are classified FAKE,
    indicating some spoof signals are present but weak
  - Feature-space distance to REAL ({dist_real:.4f}) < distance to FAKE ({dist_fake:.4f})

  Recommendation: Collect Google Agent recordings as training data for V3,
  or add codec/resampling augmentation to improve generalization.
""")
    else:
        print(f"  The model correctly identifies the Google Agent recording as {full_label}.")
    print()

    return causes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print()
    print("#" * 70)
    print("#  MORPH V2 DIAGNOSTIC — Google Agent Browser Recording")
    print("#" * 70)
    print()

    # Step 0: Convert WebM to WAV
    convert_webm_to_wav()

    # Step 1
    full_result = run_full_detection()

    # Step 2
    window_results = run_windowed_detection()

    # Step 3
    window_stats = compute_window_stats(window_results)

    # Step 4
    feature_comparison = compare_feature_distributions()

    # Step 5
    causes = diagnose(full_result, window_results, window_stats, feature_comparison)

    # Save all results to JSON
    output = {
        "full_detection": full_result.to_dict(),
        "window_results": window_results,
        "window_stats": window_stats,
        "feature_comparison": feature_comparison,
        "causes": [c["cause"] for c in causes],
    }
    out_path = OUT_DIR / "diag_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
