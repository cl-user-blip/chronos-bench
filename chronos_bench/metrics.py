"""Point and interval metrics with explicit edge-case contracts."""

import math

import numpy as np


def point_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float | None]:
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    if y.shape != pred.shape or not y.size or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Metrics require nonempty finite arrays of equal shape")
    error = y - pred
    denominator = np.abs(y).sum()
    return {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt((error ** 2).mean())),
            "wape": float(np.abs(error).sum() / denominator) if denominator else None}


def conformal_radius(residuals: np.ndarray, alpha: float = 0.1) -> float:
    """Split-conformal finite-sample rank; exchangeability is not assumed here."""
    residuals = np.asarray(residuals, dtype=float).ravel()
    if not 0 < alpha < 1 or not residuals.size or not np.isfinite(residuals).all() or np.any(residuals < 0):
        raise ValueError("Need finite nonnegative residuals and 0 < alpha < 1")
    rank = math.ceil((len(residuals) + 1) * (1 - alpha))
    if rank > len(residuals):
        raise ValueError("Too few calibration examples for this alpha")
    return float(np.partition(residuals, rank - 1)[rank - 1])


def interval_metrics(y: np.ndarray, pred: np.ndarray, radius: float, alpha: float) -> dict[str, float]:
    lower, upper = pred - radius, pred + radius
    score = (upper - lower) + 2 / alpha * np.maximum(lower - y, 0) + 2 / alpha * np.maximum(y - upper, 0)
    return {"coverage": float(np.mean((y >= lower) & (y <= upper))),
            "mean_width": float(2 * radius), "interval_score": float(np.mean(score))}
