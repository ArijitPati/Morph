"""
Unified inference pipeline for Morph voice-clone detection.

This is the reusable entry point for the future FastAPI backend.

Pipeline:
    raw audio → feature extraction → 132 features → strict validation
    → XGBoost inference → DetectionResult
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import librosa
import numpy as np

from ai_engine.features.extraction import SR, extract_features
from ai_engine.inference.model import MorphModel

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectionResult:
    """Structured result from the Morph detection pipeline."""

    label: int                    # 0 = REAL, 1 = FAKE
    confidence: float             # probability of the predicted class
    real_probability: float       # P(REAL)
    fake_probability: float       # P(FAKE)
    duration: float               # audio duration in seconds
    model_version: str            # "v1" or "v2"

    @property
    def label_str(self) -> str:
        return "REAL" if self.label == 0 else "FAKE"

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "label_str": self.label_str,
            "confidence": self.confidence,
            "real_probability": self.real_probability,
            "fake_probability": self.fake_probability,
            "duration": self.duration,
            "model_version": self.model_version,
        }


# ---------------------------------------------------------------------------
# Strict feature validation
# ---------------------------------------------------------------------------


def _validate_features(features: dict, expected_columns: list[str]) -> None:
    """
    Verify that extracted features exactly match the expected feature set.
    Fails hard on any mismatch.
    """
    extracted = set(features.keys())
    expected = set(expected_columns)

    missing = expected - extracted
    extra = extracted - expected

    if missing:
        raise ValueError(
            f"{len(missing)} expected features missing: {sorted(missing)}"
        )
    if extra:
        raise ValueError(
            f"{len(extra)} unexpected features produced: {sorted(extra)}"
        )

    # Check for NaN/Inf
    bad_keys = [k for k, v in features.items() if not np.isfinite(v)]
    if bad_keys:
        raise ValueError(
            f"{len(bad_keys)} features contain NaN/Inf: {bad_keys}"
        )


# ---------------------------------------------------------------------------
# Model cache (loaded once, reused across calls)
# ---------------------------------------------------------------------------

_model_cache: dict[str, MorphModel] = {}


def _get_model(version: str) -> MorphModel:
    """Get or create a cached MorphModel for the given version."""
    if version not in _model_cache:
        _model_cache[version] = MorphModel(version=version)
    return _model_cache[version]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect(
    audio_path: str | Path,
    *,
    model_version: str = "v2",
    sr: int = SR,
) -> DetectionResult:
    """
    Run the full Morph detection pipeline on an audio file.

    Parameters
    ----------
    audio_path : str or Path
        Path to a WAV or FLAC audio file.
    model_version : str
        Model version to use ("v1" or "v2"). Default: "v2".
    sr : int
        Sample rate for loading audio. Default: 16000.

    Returns
    -------
    DetectionResult
        Structured prediction result.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load model
    model = _get_model(model_version)

    # Load audio
    y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    # Extract features
    features = extract_features(y, sr, duration)

    # Strict validation
    _validate_features(features, model.feature_names)

    # Inference
    label = model.predict(features)
    real_prob, fake_prob = model.predict_proba(features)
    confidence = real_prob if label == 0 else fake_prob

    return DetectionResult(
        label=label,
        confidence=confidence,
        real_probability=real_prob,
        fake_probability=fake_prob,
        duration=duration,
        model_version=model_version,
    )


def detect_from_array(
    y: np.ndarray,
    *,
    sr: int = SR,
    model_version: str = "v2",
) -> DetectionResult:
    """
    Run detection on a pre-loaded audio array (for backend use).

    Parameters
    ----------
    y : np.ndarray
        Mono audio samples.
    sr : int
        Sample rate. Default: 16000.
    model_version : str
        Model version ("v1" or "v2"). Default: "v2".

    Returns
    -------
    DetectionResult
    """
    model = _get_model(model_version)
    duration = librosa.get_duration(y=y, sr=sr)
    features = extract_features(y, sr, duration)
    _validate_features(features, model.feature_names)

    label = model.predict(features)
    real_prob, fake_prob = model.predict_proba(features)
    confidence = real_prob if label == 0 else fake_prob

    return DetectionResult(
        label=label,
        confidence=confidence,
        real_probability=real_prob,
        fake_probability=fake_prob,
        duration=duration,
        model_version=model_version,
    )
