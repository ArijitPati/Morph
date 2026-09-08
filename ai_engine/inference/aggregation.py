"""
Temporal risk aggregation for Morph real-time inference.

Given a sequence of per-window P(FAKE) scores, produces a final
call-level risk decision.

No DL, no speaker verification — purely statistical aggregation over
XGBoost window scores.

Designed to be replaceable with a learned temporal model later
(RNN/Transformer on window embeddings), but for V2 the rules are
deterministic and auditable.

Example
-------
    from ai_engine.inference.aggregation import aggregate

    result = aggregate(fake_probs=[0.9, 0.85, 0.2, 0.88])
    # result.mean_fake_prob ≈ 0.707
    # result.risk_level == "HIGH" / "MEDIUM" / "LOW"
    # result.aggregated_label == 1 (FAKE)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass(frozen=True)
class AggregatedRisk:
    """Aggregated call-level risk from windowed inference."""

    total_windows: int
    n_fake: int
    n_real: int
    pct_fake: float  # 0–100
    pct_real: float  # 0–100
    mean_fake_prob: float  # 0–1
    median_fake_prob: float
    max_fake_prob: float
    min_fake_prob: float
    std_fake_prob: float
    # The "final" probability used for the top-level verdict.
    # Default = mean_fake_prob, but caller may choose max/median.
    final_fake_prob: float
    final_real_prob: float
    aggregated_label: int  # 0=REAL, 1=FAKE
    aggregated_label_str: str  # "REAL"/"FAKE"
    risk_level: RiskLevel
    risk_score: float  # 0–100 (= final_fake_prob * 100)
    confidence: float  # |final_fake_prob - 0.5| * 2  (distance from decision boundary)

    def to_dict(self) -> dict:
        return {
            "total_windows": self.total_windows,
            "n_fake": self.n_fake,
            "n_real": self.n_real,
            "pct_fake": self.pct_fake,
            "pct_real": self.pct_real,
            "mean_fake_prob": self.mean_fake_prob,
            "median_fake_prob": self.median_fake_prob,
            "max_fake_prob": self.max_fake_prob,
            "min_fake_prob": self.min_fake_prob,
            "std_fake_prob": self.std_fake_prob,
            "final_fake_prob": self.final_fake_prob,
            "final_real_prob": self.final_real_prob,
            "aggregated_label": self.aggregated_label,
            "aggregated_label_str": self.aggregated_label_str,
            "risk_level": self.risk_level.value,
            "risk_score": self.risk_score,
            "confidence": self.confidence,
        }


def _risk_level(fake_prob: float) -> RiskLevel:
    """
    Heuristic thresholds — tune after V2 Robust evaluation.

    LOW    : final P(FAKE) < 0.35  → likely real, no alert
    MEDIUM : 0.35 ≤ P < 0.65        → uncertain, soft warning
    HIGH   : P ≥ 0.65               → likely spoof, hard alert
    """
    if fake_prob >= 0.65:
        return RiskLevel.HIGH
    if fake_prob >= 0.35:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def aggregate(
    fake_probs: list[float] | np.ndarray,
    *,
    method: Literal["mean", "median", "max", "vote"] = "mean",
    threshold: float = 0.5,
) -> AggregatedRisk:
    """
    Aggregate per-window fake probabilities into a call-level verdict.

    Parameters
    ----------
    fake_probs : list[float]
        One P(FAKE) per window (values in [0, 1]).
    method : str
        How to compute final_fake_prob:
        - "mean"   : arithmetic mean (default, robust)
        - "median" : median (outlier-robust)
        - "max"    : most suspicious window dominates
        - "vote"   : fraction of windows labeled FAKE
    threshold : float
        Decision threshold for labeling (default 0.5).

    Returns
    -------
    AggregatedRisk
    """
    arr = np.asarray(fake_probs, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("fake_probs must not be empty")

    # Clamp to [0,1] for safety
    arr = np.clip(arr, 0.0, 1.0)

    n = int(arr.size)
    n_fake = int(np.sum(arr >= threshold))
    n_real = n - n_fake

    mean_p = float(np.mean(arr))
    median_p = float(np.median(arr))
    max_p = float(np.max(arr))
    min_p = float(np.min(arr))
    std_p = float(np.std(arr)) if n > 1 else 0.0

    if method == "mean":
        final_fake = mean_p
    elif method == "median":
        final_fake = median_p
    elif method == "max":
        final_fake = max_p
    elif method == "vote":
        final_fake = n_fake / n if n > 0 else 0.0
    else:
        raise ValueError(f"Unknown method '{method}'; expected mean/median/max/vote")

    final_fake = float(np.clip(final_fake, 0.0, 1.0))
    final_real = 1.0 - final_fake
    label = 1 if final_fake >= threshold else 0

    return AggregatedRisk(
        total_windows=n,
        n_fake=n_fake,
        n_real=n_real,
        pct_fake=round(n_fake / n * 100, 2) if n else 0.0,
        pct_real=round(n_real / n * 100, 2) if n else 0.0,
        mean_fake_prob=round(mean_p, 6),
        median_fake_prob=round(median_p, 6),
        max_fake_prob=round(max_p, 6),
        min_fake_prob=round(min_p, 6),
        std_fake_prob=round(std_p, 6),
        final_fake_prob=round(final_fake, 6),
        final_real_prob=round(final_real, 6),
        aggregated_label=label,
        aggregated_label_str="FAKE" if label == 1 else "REAL",
        risk_level=_risk_level(final_fake),
        risk_score=round(final_fake * 100, 2),
        confidence=round(abs(final_fake - 0.5) * 2, 4),
    )


def aggregate_from_labels(
    labels: list[int],
    fake_probs: list[float] | None = None,
    *,
    method: Literal["mean", "median", "max", "vote"] = "mean",
    threshold: float = 0.5,
) -> AggregatedRisk:
    """
    Convenience: aggregate when you already have discrete labels but
    optionally also have probabilities.  If fake_probs is provided it
    takes precedence; otherwise uses vote over labels.
    """
    if fake_probs is not None:
        return aggregate(fake_probs, method=method, threshold=threshold)
    # Vote-only fallback
    if not labels:
        raise ValueError("labels must not be empty")
    n = len(labels)
    n_fake = sum(1 for lb in labels if lb == 1)
    vote_prob = n_fake / n
    label = 1 if vote_prob >= threshold else 0
    return AggregatedRisk(
        total_windows=n,
        n_fake=n_fake,
        n_real=n - n_fake,
        pct_fake=round(n_fake / n * 100, 2),
        pct_real=round((n - n_fake) / n * 100, 2),
        mean_fake_prob=round(vote_prob, 6),
        median_fake_prob=round(vote_prob, 6),
        max_fake_prob=round(float(max(labels)), 6),
        min_fake_prob=round(float(min(labels)), 6),
        std_fake_prob=0.0,
        final_fake_prob=round(vote_prob, 6),
        final_real_prob=round(1 - vote_prob, 6),
        aggregated_label=label,
        aggregated_label_str="FAKE" if label == 1 else "REAL",
        risk_level=_risk_level(vote_prob),
        risk_score=round(vote_prob * 100, 2),
        confidence=round(abs(vote_prob - 0.5) * 2, 4),
    )
