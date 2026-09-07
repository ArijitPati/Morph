"""Pydantic response schemas."""

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
