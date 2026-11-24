"""Feature provenance is expressed in observation-index space."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

LAGS = (0, 1, 2, 23, 24, 47, 167)
WINDOWS = (6, 24, 168)
LOOKBACK = 168


@dataclass(frozen=True)
class Supervised:
    x: np.ndarray
    y: np.ndarray
    origins: np.ndarray
    targets: np.ndarray
    names: tuple[str, ...]


def make_features(frame: pd.DataFrame, horizon: int, calendar: bool = True) -> Supervised:
    """At origin t observe y[:t+1], predict y[t+h]; target calendar is known."""
    if not isinstance(horizon, int) or horizon < 1 or isinstance(horizon, bool):
        raise ValueError("horizon must be a positive integer")
    y = frame["y"].to_numpy(dtype=float)
    origins = np.arange(LOOKBACK - 1, len(y) - horizon)
    if not len(origins):
        raise ValueError("Insufficient history for the requested horizon")
    targets = origins + horizon
    columns = [y[origins - lag] for lag in LAGS]
    names = [f"lag_{lag}" for lag in LAGS]
    series = pd.Series(y)
    for window in WINDOWS:
        # Right-aligned windows include t and never use t+1 or later.
        columns += [series.rolling(window).mean().to_numpy()[origins],
                    series.rolling(window).std(ddof=0).to_numpy()[origins]]
        names += [f"mean_{window}", f"std_{window}"]
    if calendar:
        times = frame["timestamp"].iloc[targets]
        for name, values, period in (("hour", times.dt.hour.to_numpy(), 24),
                                     ("weekday", times.dt.dayofweek.to_numpy(), 7)):
            columns += [np.sin(2 * np.pi * values / period),
                        np.cos(2 * np.pi * values / period)]
            names += [f"{name}_sin", f"{name}_cos"]
        columns.append((times.dt.dayofweek.to_numpy() >= 5).astype(float))
        names.append("weekend")
    return Supervised(np.column_stack(columns), y[targets], origins, targets, tuple(names))


def purged_splits(data: Supervised, rows: int) -> dict[str, np.ndarray]:
    """Keep both origin and target in their assigned chronological block."""
    edges = (LOOKBACK - 1, int(rows * 0.55), int(rows * 0.70), int(rows * 0.85), rows)
    splits = {}
    for name, start, end in zip(("train", "validation", "calibration", "test"), edges, edges[1:]):
        mask = (data.origins >= start) & (data.targets < end)
        if mask.sum() < 24:
            raise ValueError(f"{name} has fewer than 24 examples after horizon purge; use more data or a shorter horizon")
        splits[name] = np.flatnonzero(mask)
    return splits


def seasonal_naive(values: np.ndarray, origins: np.ndarray, horizon: int, period: int = 24) -> np.ndarray:
    """Use latest same-hour observation known at t, even for h > one season."""
    if horizon < 1 or period < 1:
        raise ValueError("horizon and seasonal period must be positive")
    seasonal_cycles = (horizon + period - 1) // period
    reference = origins + horizon - seasonal_cycles * period
    if np.any(reference < 0):
        raise ValueError("Insufficient history for seasonal baseline")
    return values[reference]
