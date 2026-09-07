"""Inference module."""

from ai_engine.inference.model import MorphModel
from ai_engine.inference.pipeline import DetectionResult, detect, detect_from_array

__all__ = ["MorphModel", "DetectionResult", "detect", "detect_from_array"]
