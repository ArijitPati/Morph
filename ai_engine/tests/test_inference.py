"""
Tests for ai_engine.inference — model loading and prediction.
"""

import json
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import pytest

from ai_engine.features.extraction import SR, extract_features
from ai_engine.inference.model import MorphModel
from ai_engine.inference.pipeline import DetectionResult, detect

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

SAMPLES = {
    "gary_real": PROJECT_ROOT / "data/raw/gary_stafford/real/gs_00202.wav",
    "gary_fake": PROJECT_ROOT / "data/raw/gary_stafford/fake/gs_01619.wav",
    "asvspoof_real": (
        PROJECT_ROOT
        / "data/raw/asvspoof2019_la/LA/ASVspoof2019_LA_eval/flac/LA_E_5849185.flac"
    ),
    "asvspoof_fake": (
        PROJECT_ROOT
        / "data/raw/asvspoof2019_la/LA/ASVspoof2019_LA_eval/flac/LA_E_1000147.flac"
    ),
}


@pytest.fixture(scope="module")
def v1_model():
    return MorphModel(version="v1")


@pytest.fixture(scope="module")
def v2_model():
    return MorphModel(version="v2")


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

class TestModelLoading:
    def test_v1_loads(self, v1_model):
        assert v1_model.version == "v1"
        assert v1_model.feature_count == 132

    def test_v2_loads(self, v2_model):
        assert v2_model.version == "v2"
        assert v2_model.feature_count == 132

    def test_v1_v2_same_feature_count(self, v1_model, v2_model):
        assert v1_model.feature_count == v2_model.feature_count

    def test_v1_v2_same_feature_names(self, v1_model, v2_model):
        assert v1_model.feature_names == v2_model.feature_names

    def test_invalid_version_raises(self):
        with pytest.raises(ValueError, match="Unknown model version"):
            MorphModel(version="v3")


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

class TestPrediction:
    @pytest.mark.parametrize(
        "sample_key,expected_label",
        [
            ("gary_real", 0),
            ("gary_fake", 1),
            ("asvspoof_real", 0),
            ("asvspoof_fake", 1),
        ],
    )
    def test_v2_predict_correct_label(self, v2_model, sample_key, expected_label):
        path = SAMPLES[sample_key]
        y, sr = librosa.load(str(path), sr=SR, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        features = extract_features(y, sr, duration)
        label = v2_model.predict(features)
        assert label == expected_label, (
            f"{sample_key}: expected {expected_label}, got {label}"
        )

    def test_predict_proba_sums_to_one(self, v2_model):
        path = SAMPLES["gary_real"]
        y, sr = librosa.load(str(path), sr=SR, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        features = extract_features(y, sr, duration)
        real_prob, fake_prob = v2_model.predict_proba(features)
        assert abs(real_prob + fake_prob - 1.0) < 1e-6

    def test_predict_returns_int(self, v2_model):
        path = SAMPLES["gary_real"]
        y, sr = librosa.load(str(path), sr=SR, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        features = extract_features(y, sr, duration)
        label = v2_model.predict(features)
        assert isinstance(label, int)
        assert label in (0, 1)


# ---------------------------------------------------------------------------
# Pipeline (detect)
# ---------------------------------------------------------------------------

class TestPipeline:
    @pytest.mark.parametrize(
        "sample_key,expected_label",
        [
            ("gary_real", 0),
            ("gary_fake", 1),
            ("asvspoof_real", 0),
            ("asvspoof_fake", 1),
        ],
    )
    def test_detect_correct_label(self, sample_key, expected_label):
        path = SAMPLES[sample_key]
        result = detect(path, model_version="v2")
        assert result.label == expected_label

    def test_detect_result_fields(self):
        path = SAMPLES["gary_real"]
        result = detect(path, model_version="v2")
        assert isinstance(result, DetectionResult)
        assert result.label in (0, 1)
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.real_probability <= 1.0
        assert 0.0 <= result.fake_probability <= 1.0
        assert result.duration > 0
        assert result.model_version == "v2"
        assert result.label_str in ("REAL", "FAKE")

    def test_detect_to_dict(self):
        path = SAMPLES["gary_real"]
        result = detect(path, model_version="v2")
        d = result.to_dict()
        assert "label" in d
        assert "confidence" in d
        assert "real_probability" in d
        assert "fake_probability" in d
        assert "duration" in d
        assert "model_version" in d
        assert "label_str" in d

    def test_detect_v1_works(self):
        path = SAMPLES["gary_real"]
        result = detect(path, model_version="v1")
        assert result.model_version == "v1"
        assert result.label == 0


# ---------------------------------------------------------------------------
# Comparison against existing predict.py
# ---------------------------------------------------------------------------

class TestComparisonWithPredictPy:
    """Verify the new pipeline produces identical results to scripts/predict.py."""

    @pytest.mark.parametrize(
        "sample_key",
        ["gary_real", "gary_fake", "asvspoof_real", "asvspoof_fake"],
    )
    def test_v2_matches_predict_py(self, sample_key):
        path = SAMPLES[sample_key]

        # New pipeline
        result = detect(path, model_version="v2")

        # Existing predict.py (subprocess)
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/predict.py"),
            str(path),
            "--model", "v2",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, f"predict.py failed: {proc.stderr}"

        # Parse predict.py output
        output = proc.stdout
        for line in output.splitlines():
            line = line.strip()
            if line.startswith("REAL  prob :"):
                real_pct = float(line.split(":")[1].strip().rstrip("%"))
            elif line.startswith("FAKE  prob :"):
                fake_pct = float(line.split(":")[1].strip().rstrip("%"))

        # Compare within floating-point tolerance
        assert abs(result.real_probability * 100 - real_pct) < 0.01, (
            f"REAL prob mismatch: pipeline={result.real_probability:.6f}, "
            f"predict.py={real_pct/100:.6f}"
        )
        assert abs(result.fake_probability * 100 - fake_pct) < 0.01, (
            f"FAKE prob mismatch: pipeline={result.fake_probability:.6f}, "
            f"predict.py={fake_pct/100:.6f}"
        )
