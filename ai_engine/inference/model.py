"""
Model loader for Morph V1/V2 XGBoost classifiers.

Provides a single MorphModel class that handles:
- Loading the serialized XGBoost model
- Loading the corresponding feature-order JSON
- Validating the feature list
- Keeping the model in memory
- Providing a clean predict/predict_proba interface
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import xgboost as xgb

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_AI_ENGINE_ROOT = Path(__file__).resolve().parent.parent
_MODELS_DIR = _AI_ENGINE_ROOT / "models"

_MODEL_FILES = {
    "v1": {
        "model": "morph_xgboost_v1.json",
        "features": "morph_xgboost_v1_features.json",
    },
    "v2": {
        "model": "morph_xgboost_v2.json",
        "features": "morph_xgboost_v2_features.json",
    },
    "v2_robust": {
        "model": "morph_xgboost_v2_robust.json",
        "features": "morph_xgboost_v2_robust_features.json",
    },
}

# ---------------------------------------------------------------------------
# MorphModel
# ---------------------------------------------------------------------------


class MorphModel:
    """
    Reusable wrapper around a trained Morph XGBoost classifier.

    Parameters
    ----------
    version : str
        Model version to load ("v1" or "v2"). Default: "v2".
    models_dir : Path, optional
        Override the directory containing model files.
    """

    def __init__(self, version: str = "v2", models_dir: Path | None = None) -> None:
        if version not in _MODEL_FILES:
            raise ValueError(
                f"Unknown model version '{version}'. Supported: {list(_MODEL_FILES)}"
            )

        self._version = version
        self._models_dir = models_dir or _MODELS_DIR
        self._model: xgb.XGBClassifier | None = None
        self._feature_columns: List[str] = []

        self._load()

    # -- Loading -----------------------------------------------------------

    def _load(self) -> None:
        cfg = _MODEL_FILES[self._version]
        model_path = self._models_dir / cfg["model"]
        features_path = self._models_dir / cfg["features"]

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not features_path.exists():
            raise FileNotFoundError(f"Feature metadata not found: {features_path}")

        # Load XGBoost model
        self._model = xgb.XGBClassifier()
        self._model.load_model(str(model_path))

        # Load feature column order
        with open(features_path) as f:
            meta = json.load(f)
        self._feature_columns = meta["feature_columns"]

        # Validate
        if not self._feature_columns:
            raise ValueError(f"Empty feature list in {features_path}")
        if len(set(self._feature_columns)) != len(self._feature_columns):
            raise ValueError(f"Duplicate feature names in {features_path}")

    # -- Properties --------------------------------------------------------

    @property
    def version(self) -> str:
        return self._version

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_columns)

    @property
    def feature_count(self) -> int:
        return len(self._feature_columns)

    # -- Prediction --------------------------------------------------------

    def predict(self, features: dict | np.ndarray) -> int:
        """
        Predict label for a single sample.

        Parameters
        ----------
        features : dict or np.ndarray
            If dict: keys must match self.feature_names exactly.
            If ndarray: must have shape (1, n_features) in the correct order.

        Returns
        -------
        int
            0 = REAL/bonafide, 1 = FAKE/spoof.
        """
        X = self._to_array(features)
        return int(self._model.predict(X)[0])

    def predict_proba(self, features: dict | np.ndarray) -> tuple[float, float]:
        """
        Predict class probabilities for a single sample.

        Returns
        -------
        tuple[float, float]
            (real_probability, fake_probability).
        """
        X = self._to_array(features)
        probs = self._model.predict_proba(X)[0]
        return float(probs[0]), float(probs[1])

    # -- Internal ----------------------------------------------------------

    def _to_array(self, features: dict | np.ndarray) -> np.ndarray:
        """Convert a feature dict or array to the expected model input."""
        if isinstance(features, dict):
            # Validate feature set
            extracted = set(features.keys())
            expected = set(self._feature_columns)
            missing = expected - extracted
            extra = extracted - expected
            if missing:
                raise ValueError(f"Missing features: {sorted(missing)}")
            if extra:
                raise ValueError(f"Unexpected features: {sorted(extra)}")

            row = np.array([[features[col] for col in self._feature_columns]],
                           dtype=np.float32)
        else:
            row = np.asarray(features, dtype=np.float32)
            if row.ndim == 1:
                row = row.reshape(1, -1)

        # Sanitize NaN/Inf
        row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
        return row
