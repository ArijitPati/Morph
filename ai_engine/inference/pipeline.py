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
from ai_engine.inference.aggregation import AggregatedRisk, aggregate
from ai_engine.inference.model import MorphModel
from ai_engine.inference.windowing import (
    DEFAULT_HOP_SEC,
    DEFAULT_WINDOW_SEC,
    MIN_WINDOW_SEC,
    StreamingBuffer,
    WindowSlice,
    get_window_boundaries,
)

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


@dataclass(frozen=True)
class WindowResult:
    """Per-window detection result for real-time pipeline."""

    window_index: int
    start_sec: float
    end_sec: float
    duration: float
    is_partial: bool
    label: int
    confidence: float
    real_probability: float
    fake_probability: float
    model_version: str

    @property
    def label_str(self) -> str:
        return "REAL" if self.label == 0 else "FAKE"

    def to_dict(self) -> dict:
        return {
            "window_index": self.window_index,
            "window": self.window_index + 1,  # 1-based alias for frontend
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "duration": self.duration,
            "chunk_duration": self.duration,
            "is_partial": self.is_partial,
            "label": self.label,
            "label_str": self.label_str,
            "confidence": self.confidence,
            "real_probability": self.real_probability,
            "fake_probability": self.fake_probability,
            "real_prob": round(self.real_probability, 6),
            "fake_prob": round(self.fake_probability, 6),
            "model_version": self.model_version,
        }


@dataclass(frozen=True)
class WindowedDetectionResult:
    """Full windowed result with per-window scores + temporal aggregation."""

    total_duration: float
    window_sec: float
    hop_sec: float
    model_version: str
    windows: list[WindowResult]
    aggregation: AggregatedRisk
    # Full-file single-shot result for comparison/diagnostics
    full_result: DetectionResult | None = None

    def to_dict(self) -> dict:
        return {
            "total_duration": self.total_duration,
            "duration": self.total_duration,
            "window_sec": self.window_sec,
            "hop_sec": self.hop_sec,
            "model_version": self.model_version,
            "windows": [w.to_dict() for w in self.windows],
            "aggregation": self.aggregation.to_dict(),
            "full_result": self.full_result.to_dict() if self.full_result else None,
            # Convenience top-level fields mirroring DetectionResponse for backward compat
            "verdict": self.aggregation.aggregated_label_str,
            "risk_score": self.aggregation.risk_score,
            "risk_level": self.aggregation.risk_level.value,
            "real_probability": self.aggregation.final_real_prob,
            "fake_probability": self.aggregation.final_fake_prob,
            "feature_count": 132,
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


# ---------------------------------------------------------------------------
# Windowed detection — real-time architecture
# ---------------------------------------------------------------------------


def _infer_window(
    chunk: np.ndarray,
    sr: int,
    duration: float,
    model: MorphModel,
) -> tuple[int, float, float, float]:
    """Run 132-feature extraction + XGBoost on one window."""
    features = extract_features(chunk, sr, duration)
    _validate_features(features, model.feature_names)
    label = model.predict(features)
    real_prob, fake_prob = model.predict_proba(features)
    confidence = real_prob if label == 0 else fake_prob
    return label, confidence, real_prob, fake_prob


def detect_windows_from_array(
    y: np.ndarray,
    *,
    sr: int = SR,
    model_version: str = "v2",
    window_sec: float = DEFAULT_WINDOW_SEC,
    hop_sec: float | None = None,
    min_window_sec: float = MIN_WINDOW_SEC,
    include_partial: bool = True,
    aggregation_method: str = "mean",
) -> WindowedDetectionResult:
    """
    Windowed detection on a pre-loaded mono array.

    Splits audio into fixed windows, runs Morph inference per window,
    and aggregates scores temporally.

    Does NOT modify existing single-shot detect() / detect_from_array().
    """
    if hop_sec is None:
        hop_sec = window_sec

    model = _get_model(model_version)
    total_duration = librosa.get_duration(y=y, sr=sr)

    slices = get_window_boundaries(
        len(y),
        sr=sr,
        window_sec=window_sec,
        hop_sec=hop_sec,
        min_window_sec=min_window_sec,
        include_partial=include_partial,
    )

    windows: list[WindowResult] = []
    for sl in slices:
        chunk = y[sl.start_sample : sl.end_sample]
        dur = sl.duration_sec
        label, confidence, real_prob, fake_prob = _infer_window(chunk, sr, dur, model)
        windows.append(
            WindowResult(
                window_index=sl.index,
                start_sec=sl.start_sec,
                end_sec=sl.end_sec,
                duration=dur,
                is_partial=sl.is_partial,
                label=label,
                confidence=confidence,
                real_probability=real_prob,
                fake_probability=fake_prob,
                model_version=model_version,
            )
        )

    # Temporal aggregation — handle empty (e.g. file shorter than min window)
    if windows:
        fake_probs = [w.fake_probability for w in windows]
        agg: AggregatedRisk = aggregate(fake_probs, method=aggregation_method)  # type: ignore[arg-type]
    else:
        # Fallback: treat whole file as one window if too short for windowing
        label, confidence, real_prob, fake_prob = _infer_window(y, sr, total_duration, model)
        windows.append(
            WindowResult(
                window_index=0,
                start_sec=0.0,
                end_sec=round(total_duration, 4),
                duration=total_duration,
                is_partial=True,
                label=label,
                confidence=confidence,
                real_probability=real_prob,
                fake_probability=fake_prob,
                model_version=model_version,
            )
        )
        agg = aggregate([fake_prob], method=aggregation_method)  # type: ignore[arg-type]

    # Full-file single-shot for diagnostics/comparison
    try:
        full = detect_from_array(y, sr=sr, model_version=model_version)
    except Exception:
        full = None

    return WindowedDetectionResult(
        total_duration=total_duration,
        window_sec=window_sec,
        hop_sec=hop_sec,
        model_version=model_version,
        windows=windows,
        aggregation=agg,
        full_result=full,
    )


def detect_windows(
    audio_path: str | Path,
    *,
    model_version: str = "v2",
    sr: int = SR,
    window_sec: float = DEFAULT_WINDOW_SEC,
    hop_sec: float | None = None,
    min_window_sec: float = MIN_WINDOW_SEC,
    include_partial: bool = True,
    aggregation_method: str = "mean",
) -> WindowedDetectionResult:
    """
    Windowed detection on an audio file.

    Convenience wrapper over detect_windows_from_array().
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    return detect_windows_from_array(
        y,
        sr=sr,
        model_version=model_version,
        window_sec=window_sec,
        hop_sec=hop_sec,
        min_window_sec=min_window_sec,
        include_partial=include_partial,
        aggregation_method=aggregation_method,
    )
