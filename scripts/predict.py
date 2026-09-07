#!/usr/bin/env python3
"""
Morph Voice Detector — CLI prediction demo.

Supports both V1 and V2 models. V2 is the default.

Usage:
    python scripts/predict.py path/to/audio.wav
    python scripts/predict.py path/to/audio.wav --model v2
    python scripts/predict.py path/to/audio.wav --model v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import xgboost as xgb

# ---------------------------------------------------------------------------
# Import feature extraction from the shared module (no duplication)
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from extract_features import SR, extract_features  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = SCRIPTS_DIR.parent

MODEL_CONFIGS = {
    "v1": {
        "model": PROJECT_ROOT / "models" / "morph_xgboost_v1.json",
        "features": PROJECT_ROOT / "models" / "morph_xgboost_v1_features.json",
    },
    "v2": {
        "model": PROJECT_ROOT / "models" / "morph_xgboost_v2.json",
        "features": PROJECT_ROOT / "models" / "morph_xgboost_v2_features.json",
    },
}

# ---------------------------------------------------------------------------
# Model / metadata loading
# ---------------------------------------------------------------------------

def load_model(version: str):
    cfg = MODEL_CONFIGS[version]
    if not cfg["model"].exists():
        print(f"Error: Model file not found — {cfg['model']}")
        sys.exit(1)
    model = xgb.XGBClassifier()
    model.load_model(str(cfg["model"]))
    return model


def load_feature_columns(version: str) -> list[str]:
    cfg = MODEL_CONFIGS[version]
    if not cfg["features"].exists():
        print(f"Error: Feature metadata not found — {cfg['features']}")
        sys.exit(1)
    with open(cfg["features"]) as f:
        meta = json.load(f)
    return meta["feature_columns"]

# ---------------------------------------------------------------------------
# Strict feature validation
# ---------------------------------------------------------------------------

def validate_features(feats: dict, expected_columns: list[str]) -> None:
    """
    Verify that extracted features exactly match the expected feature set.
    Fails hard on any mismatch — never silently fills with zero.
    """
    extracted = set(feats.keys())
    expected = set(expected_columns)

    missing = expected - extracted
    extra = extracted - expected

    if missing:
        print(f"Error: {len(missing)} expected features missing from extraction:")
        for m in sorted(missing):
            print(f"  - {m}")
        sys.exit(1)

    if extra:
        print(f"Error: {len(extra)} unexpected features produced by extraction:")
        for e in sorted(extra):
            print(f"  - {e}")
        sys.exit(1)

    # Check for NaN/Inf values
    bad_keys = [k for k, v in feats.items() if not np.isfinite(v)]
    if bad_keys:
        print(f"Error: {len(bad_keys)} features contain NaN/Inf values:")
        for k in bad_keys:
            print(f"  - {k}: {feats[k]}")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(audio_path: Path, model, feature_columns: list[str]):
    """Extract features from a single audio file and return prediction."""
    y, sr = librosa.load(str(audio_path), sr=SR, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    feats = extract_features(y, sr, duration)

    # Strict validation: extracted features must exactly match expected set
    validate_features(feats, feature_columns)

    # Build a single-row array in the exact feature column order
    X = np.array([[feats[col] for col in feature_columns]], dtype=np.float32)

    label = int(model.predict(X)[0])
    probs = model.predict_proba(X)[0]
    real_prob = float(probs[0])
    fake_prob = float(probs[1])
    confidence = real_prob if label == 0 else fake_prob

    return label, confidence, real_prob, fake_prob, duration

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Morph Voice Detector — detect synthetic/deepfake audio"
    )
    parser.add_argument("audio", help="Path to WAV or FLAC audio file")
    parser.add_argument(
        "--model",
        choices=["v1", "v2"],
        default="v2",
        help="Model version to use (default: v2)",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio)

    if not audio_path.exists():
        print(f"Error: File not found — {audio_path}")
        sys.exit(1)

    if audio_path.suffix.lower() not in (".wav", ".flac"):
        print(f"Error: Unsupported format '{audio_path.suffix}' — use WAV or FLAC.")
        sys.exit(1)

    version = args.model
    model = load_model(version)
    feature_columns = load_feature_columns(version)

    try:
        label, confidence, real_prob, fake_prob, duration = predict(
            audio_path, model, feature_columns
        )
    except SystemExit:
        raise
    except Exception as e:
        print(f"Error: Could not process audio — {e}")
        sys.exit(1)

    label_str = "REAL" if label == 0 else "FAKE / DEEPFAKE"

    print()
    print("=" * 44)
    print(f"        MORPH VOICE DETECTOR — Model {version.upper()}")
    print("=" * 44)
    print(f"  File       : {audio_path.name}")
    print(f"  Duration   : {duration:.2f}s")
    print(f"  Prediction : {label_str}")
    print(f"  Confidence : {confidence * 100:.2f}%")
    print("-" * 44)
    print(f"  REAL  prob : {real_prob * 100:.2f}%")
    print(f"  FAKE  prob : {fake_prob * 100:.2f}%")
    print("=" * 44)
    print()


if __name__ == "__main__":
    main()
