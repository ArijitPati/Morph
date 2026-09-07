"""
Reusable feature extraction for the Morph voice-clone detection project.

This module contains the EXACT feature extraction logic used by
scripts/extract_features.py, preserved without modification.

Extracts 132 numerical acoustic features from a mono audio signal:

    MFCC 1-20 mean/std                          (40)
    Spectral centroid mean/std                   (2)
    Spectral bandwidth mean/std                  (2)
    Spectral rolloff mean/std                    (2)
    Spectral contrast 7 bands mean/std           (14)
    Chroma 12 classes mean/std                   (24)
    RMS energy mean/std                          (2)
    Zero-crossing rate mean/std                  (2)
    F0 mean/std + voiced-frame ratio             (3)
    CQCC 1-20 mean/std                          (40)
    Duration                                     (1)
                                      Total = 132
"""

from __future__ import annotations

from typing import Dict

import librosa
import numpy as np
from scipy.fftpack import dct

# ---------------------------------------------------------------------------
# Constants — identical to scripts/extract_features.py
# ---------------------------------------------------------------------------

SR = 16000
N_MFCC = 20
N_CQCC = 20
CQT_N_BINS = 84
CQT_BINS_PER_OCTAVE = 12
CQT_HOP_LENGTH = 512


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_stats(arr: np.ndarray, prefix: str) -> Dict[str, float]:
    """Compute mean and std for a 1-D array, replacing NaN/inf with 0."""
    mean_val = float(np.mean(arr)) if arr.size > 0 else 0.0
    std_val = float(np.std(arr)) if arr.size > 0 else 0.0
    if not np.isfinite(mean_val):
        mean_val = 0.0
    if not np.isfinite(std_val):
        std_val = 0.0
    return {f"{prefix}_mean": mean_val, f"{prefix}_std": std_val}


def extract_cqcc(y: np.ndarray, sr: int, n_cqcc: int = N_CQCC) -> Dict[str, float]:
    """
    Constant-Q Cepstral Coefficients (CQCC).

    1. Compute the Constant-Q Transform via librosa.cqt.
    2. Take log-magnitudes (with numerical floor).
    3. Apply DCT-II along the frequency axis.
    4. Keep the first n_cqcc coefficients.
    5. Return mean and std across time frames.
    """
    C = np.abs(
        librosa.cqt(
            y,
            sr=sr,
            hop_length=CQT_HOP_LENGTH,
            n_bins=CQT_N_BINS,
            bins_per_octave=CQT_BINS_PER_OCTAVE,
        )
    )
    log_C = np.log(C + 1e-10)
    cqcc_full = dct(log_C, type=2, axis=0, norm="ortho")
    cqcc = cqcc_full[:n_cqcc, :]

    features: Dict[str, float] = {}
    for i in range(n_cqcc):
        features[f"cqcc_{i+1}_mean"] = float(np.mean(cqcc[i]))
        features[f"cqcc_{i+1}_std"] = float(np.std(cqcc[i]))

    return features


def extract_pitch(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Fundamental frequency (F0) estimation using librosa.pyin.

    Returns mean F0, std F0, and voiced-frame ratio.
    """
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
    )

    f0_valid = f0[~np.isnan(f0)]

    features: Dict[str, float] = {}
    if f0_valid.size > 0:
        features["f0_mean"] = float(np.mean(f0_valid))
        features["f0_std"] = float(np.std(f0_valid))
    else:
        features["f0_mean"] = 0.0
        features["f0_std"] = 0.0

    if f0.size > 0:
        features["voiced_frame_ratio"] = float(np.sum(voiced_flag) / f0.size)
    else:
        features["voiced_frame_ratio"] = 0.0

    return features


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_features(y: np.ndarray, sr: int, duration: float) -> Dict[str, float]:
    """
    Extract the full 132-dimensional feature vector from a mono audio signal.

    This function is IDENTICAL to scripts/extract_features.py:extract_features().
    """
    features: Dict[str, float] = {}

    # MFCC (20 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        features[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))

    # Spectral features
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

    features.update(_safe_stats(centroid, "spectral_centroid"))
    features.update(_safe_stats(bandwidth, "spectral_bandwidth"))
    features.update(_safe_stats(rolloff, "spectral_rolloff"))

    # Spectral contrast (7 bands)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
    for i in range(contrast.shape[0]):
        features.update(_safe_stats(contrast[i], f"spectral_contrast_{i+1}"))

    # Chroma (12 classes)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(12):
        features.update(_safe_stats(chroma[i], f"chroma_{i+1}"))

    # RMS energy
    rms = librosa.feature.rms(y=y)[0]
    features.update(_safe_stats(rms, "rms"))

    # Zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features.update(_safe_stats(zcr, "zcr"))

    # Pitch / F0
    features.update(extract_pitch(y, sr))

    # CQCC
    features.update(extract_cqcc(y, sr))

    # Duration
    features["duration"] = duration

    # Final sanitisation
    for k, v in features.items():
        if not np.isfinite(v):
            features[k] = 0.0

    return features
