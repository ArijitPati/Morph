"""Inference module."""

from ai_engine.inference.aggregation import AggregatedRisk, RiskLevel, aggregate
from ai_engine.inference.model import MorphModel
from ai_engine.inference.pipeline import (
    DetectionResult,
    WindowedDetectionResult,
    WindowResult,
    detect,
    detect_from_array,
    detect_windows,
    detect_windows_from_array,
)
from ai_engine.inference.windowing import (
    DEFAULT_HOP_SEC,
    DEFAULT_WINDOW_SEC,
    MIN_WINDOW_SEC,
    StreamingBuffer,
    WindowSlice,
    get_window_boundaries,
    slice_audio,
)

__all__ = [
    "MorphModel",
    "DetectionResult",
    "WindowResult",
    "WindowedDetectionResult",
    "AggregatedRisk",
    "RiskLevel",
    "aggregate",
    "detect",
    "detect_from_array",
    "detect_windows",
    "detect_windows_from_array",
    "DEFAULT_WINDOW_SEC",
    "DEFAULT_HOP_SEC",
    "MIN_WINDOW_SEC",
    "WindowSlice",
    "StreamingBuffer",
    "get_window_boundaries",
    "slice_audio",
]
