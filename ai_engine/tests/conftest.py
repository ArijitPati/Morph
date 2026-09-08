"""
Pytest configuration for ai_engine tests.

Handles the case where raw audio data (data/raw/) is not present.
Tests that depend on real audio files are skipped with a clear message
when the data directory is missing. This keeps CI green on clean
checkouts while preserving full test coverage locally.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# ---------------------------------------------------------------------------
# Audio samples referenced by tests
# ---------------------------------------------------------------------------

REQUIRED_SAMPLES = [
    "gary_stafford/real/gs_00202.wav",
    "gary_stafford/fake/gs_01619.wav",
    "asvspoof2019_la/LA/ASVspoof2019_LA_eval/flac/LA_E_5849185.flac",
    "asvspoof2019_la/LA/ASVspoof2019_LA_eval/flac/LA_E_1000147.flac",
]


def _has_real_data() -> bool:
    """Return True if all required audio samples exist on disk."""
    return all((DATA_DIR / s).exists() for s in REQUIRED_SAMPLES)


HAS_REAL_DATA = _has_real_data()

# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

# Classes / test names that require real audio data to run.
# Models loading tests (TestModelLoading) do NOT need audio.
_DATA_DEPENDENT_TESTS = {
    # test_features.py — every class except the one bare assertion
    "TestFeatureCount",
    "TestFeatureNames",
    "TestFeatureValues",
    # test_inference.py
    "TestPrediction",
    "TestPipeline",
    "TestComparisonWithPredictPy",
}


def pytest_collection_modifyitems(config, items):
    """Skip data-dependent tests when audio samples are unavailable."""
    if HAS_REAL_DATA:
        return

    skip_marker = pytest.mark.skip(
        reason="data/raw/ audio samples not found — skipping integration tests"
    )
    for item in items:
        parent_name = getattr(item, "cls", None)
        if parent_name is not None and parent_name.__name__ in _DATA_DEPENDENT_TESTS:
            item.add_marker(skip_marker)
