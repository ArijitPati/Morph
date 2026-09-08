"""
Test for ReplayDF offline evaluation helper — light, no audio I/O.
Verifies metric computation and selection CSV sanity.
"""
from pathlib import Path
import csv

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SELECTION_CSV = PROJECT_ROOT / "data/raw/replaydf/selected/selection.csv"

# Import helper from evaluation script
import sys
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

def test_compute_metrics_basic():
    from scripts.evaluate_replaydf_v2_robust import compute_metrics

    # Perfect predictions
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.8, 0.9]
    m = compute_metrics(y_true, y_pred, y_score)
    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["fpr"] == 0.0
    assert m["fnr"] == 0.0
    assert m["spoof_detection_rate"] == 1.0
    assert m["bona_fide_detection_rate"] == 1.0
    assert m["roc_auc"] == 1.0
    assert m["tp"] == 2 and m["tn"] == 2

def test_compute_metrics_imperfect():
    from scripts.evaluate_replaydf_v2_robust import compute_metrics
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 0]  # 1 FP, 1 FN
    y_score = [0.2, 0.6, 0.7, 0.3]
    m = compute_metrics(y_true, y_pred, y_score)
    assert m["accuracy"] == 0.5
    assert m["precision"] == 0.5  # TP=1, FP=1
    assert m["recall"] == 0.5     # TP=1, FN=1
    assert m["fpr"] == 0.5
    assert m["fnr"] == 0.5
    assert m["spoof_detection_rate"] == 0.5
    assert m["bona_fide_detection_rate"] == 0.5

def test_selection_csv_exists_and_has_100():
    assert SELECTION_CSV.exists(), f"Missing {SELECTION_CSV}"
    with open(SELECTION_CSV, newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f, delimiter="|"))
    assert len(reader) == 100, f"Expected 100 rows, got {len(reader)}"
    labels = [r["label"] for r in reader]
    assert labels.count("spoof") == 50
    assert labels.count("bona-fide") == 50
    # Check recorded_file paths are non-empty
    for r in reader:
        assert r["recorded_file"].startswith("wav/")
        assert r["uid"]

def test_feature_count_is_132():
    # Verify reusable extraction still yields 132 features
    import librosa
    import numpy as np
    from ai_engine.features.extraction import SR, extract_features
    y = np.zeros(SR * 2, dtype=np.float32)  # 2 sec silence
    feats = extract_features(y, SR, 2.0)
    assert len(feats) == 132
    assert "duration" in feats
    assert all(v == 0.0 or isinstance(v, float) for v in feats.values())
