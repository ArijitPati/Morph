"""
Morph AI Engine — reusable ML inference for voice-clone detection.

Provides feature extraction, model loading, and a unified inference pipeline
for the Morph V1/V2 XGBoost classifiers.
"""

from ai_engine.features.extraction import SR, extract_features
from ai_engine.inference.model import MorphModel
from ai_engine.inference.pipeline import DetectionResult, detect

__all__ = ["SR", "extract_features", "MorphModel", "DetectionResult", "detect"]
