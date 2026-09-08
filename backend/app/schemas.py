"""Pydantic response schemas."""

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    engine: str
    model_version: str
    feature_count: int


class DetectionResponse(BaseModel):
    verdict: str
    risk_score: float
    real_probability: float
    fake_probability: float
    model_version: str
    duration: float
    feature_count: int


# ─── Windowed detection ──────────────────────────────────────────


class WindowResultSchema(BaseModel):
    window_index: int
    window: int  # 1-based alias
    start_sec: float
    end_sec: float
    duration: float
    chunk_duration: float
    is_partial: bool
    label: int
    label_str: Literal["REAL", "FAKE"]
    confidence: float
    real_probability: float
    fake_probability: float
    real_prob: float
    fake_prob: float
    model_version: str


class AggregationSchema(BaseModel):
    total_windows: int
    n_fake: int
    n_real: int
    pct_fake: float
    pct_real: float
    mean_fake_prob: float
    median_fake_prob: float
    max_fake_prob: float
    min_fake_prob: float
    std_fake_prob: float
    final_fake_prob: float
    final_real_prob: float
    aggregated_label: int
    aggregated_label_str: Literal["REAL", "FAKE"]
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    risk_score: float
    confidence: float


class WindowedDetectionResponse(BaseModel):
    total_duration: float
    duration: float
    window_sec: float
    hop_sec: float
    model_version: str
    windows: list[WindowResultSchema]
    aggregation: AggregationSchema
    full_result: dict | None = None
    verdict: str
    risk_score: float
    risk_level: str
    real_probability: float
    fake_probability: float
    feature_count: int
