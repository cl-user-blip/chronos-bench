"""Strict input contracts and a fully local, explicitly synthetic example."""

from pathlib import Path

import numpy as np
import pandas as pd


def synthetic_load(rows: int = 3360, seed: int = 42) -> pd.DataFrame:
    """Hourly arbitrary-unit load with seasonality, correlated noise and drift."""
    if rows < 720:
        raise ValueError("Demo needs at least 720 hourly observations")
    rng = np.random.default_rng(seed)
    timestamp = pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC")
    t = np.arange(rows)
    hour = timestamp.hour.to_numpy()
    weekend = (timestamp.dayofweek.to_numpy() >= 5).astype(float)
    daily = 26 * np.sin(2 * np.pi * (hour - 7) / 24)
    evening_peak = 20 * np.exp(-((hour - 19) / 2.5) ** 2)
    seasonal = 8 * np.sin(2 * np.pi * t / (24 * 30))
    innovations = rng.normal(0, 3.0, rows)
    residual = np.zeros(rows)
    for i in range(1, rows):
        residual[i] = 0.72 * residual[i - 1] + innovations[i]
    # Continuous drift and changing daily amplitude deliberately challenge calibration.
    drift = 0.004 * t
    y = 110 + daily * (1 + 0.000025 * t) + evening_peak - 13 * weekend
    y += seasonal + residual + drift
    return pd.DataFrame({"timestamp": timestamp, "y": y})


def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "y"}
    if not required.issubset(frame.columns):
        raise ValueError("CSV must contain timestamp and y columns")
    result = frame.loc[:, ["timestamp", "y"]].copy()
    try:
        result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="raise")
        result["y"] = pd.to_numeric(result["y"], errors="raise").astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid timestamp or numeric target") from exc
    if result.isna().any().any() or not np.isfinite(result["y"]).all():
        raise ValueError("Missing values and infinite targets must be handled explicitly")
    ts = result["timestamp"]
    if len(result) < 720:
        raise ValueError("At least 720 observations are required")
    if not ts.is_monotonic_increasing or ts.duplicated().any():
        raise ValueError("Timestamps must be strictly increasing and unique; no implicit sorting")
    deltas = ts.diff().dropna()
    if not (deltas == pd.Timedelta(hours=1)).all():
        raise ValueError("This version requires hourly data without gaps; resample explicitly")
    return result.reset_index(drop=True)


def load_csv(path: str | Path) -> pd.DataFrame:
    return validate_frame(pd.read_csv(path))
