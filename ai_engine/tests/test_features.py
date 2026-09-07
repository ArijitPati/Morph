"""
Tests for ai_engine.features.extraction — verify extraction is unchanged.
"""

import json
from pathlib import Path

import librosa
import numpy as np
import pytest

from ai_engine.features.extraction import SR, extract_features

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

SAMPLE_WAV = PROJECT_ROOT / "data/raw/gary_stafford/real/gs_00202.wav"

V2_FEATURE_META = (
    PROJECT_ROOT / "ai_engine/models/morph_xgboost_v2_features.json"
)


@pytest.fixture(scope="module")
def sample_audio():
    """Load a known-good WAV file."""
    y, sr = librosa.load(str(SAMPLE_WAV), sr=SR, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    return y, sr, duration


@pytest.fixture(scope="module")
def features(sample_audio):
    """Extract features from the sample."""
    y, sr, duration = sample_audio
    return extract_features(y, sr, duration)


@pytest.fixture(scope="module")
def expected_columns():
    """Load expected feature columns from V2 JSON."""
    with open(V2_FEATURE_META) as f:
        meta = json.load(f)
    return meta["feature_columns"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFeatureCount:
    def test_exactly_132_features(self, features):
        assert len(features) == 132

    def test_expected_count_matches_json(self, features, expected_columns):
        assert len(expected_columns) == 132


class TestFeatureNames:
    def test_no_missing_features(self, features, expected_columns):
        extracted = set(features.keys())
        expected = set(expected_columns)
        missing = expected - extracted
        assert not missing, f"Missing features: {missing}"

    def test_no_extra_features(self, features, expected_columns):
        extracted = set(features.keys())
        expected = set(expected_columns)
        extra = extracted - expected
        assert not extra, f"Extra features: {extra}"

    def test_exact_match(self, features, expected_columns):
        assert sorted(features.keys()) == sorted(expected_columns)


class TestFeatureValues:
    def test_all_finite(self, features):
        bad = {k: v for k, v in features.items() if not np.isfinite(v)}
        assert not bad, f"Non-finite values: {bad}"

    def test_sr_is_16000(self):
        assert SR == 16000

    def test_duration_feature_matches_input(self, sample_audio, features):
        _, _, duration = sample_audio
        assert abs(features["duration"] - duration) < 0.01
