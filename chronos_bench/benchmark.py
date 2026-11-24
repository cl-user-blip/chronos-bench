"""A single explicit protocol: fit -> select -> refit -> calibrate -> test."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .data import validate_frame
from .features import make_features, purged_splits, seasonal_naive
from .metrics import conformal_radius, interval_metrics, point_metrics
from .report import render_report


@dataclass(frozen=True)
class Config:
    horizons: tuple[int, ...] = (1, 6, 24)
    alpha: float = 0.1
    seed: int = 42

    def __post_init__(self):
        if not self.horizons or any(not isinstance(h, int) or isinstance(h, bool) or h < 1 for h in self.horizons):
            raise ValueError("Horizons must be positive integers")
        if len(set(self.horizons)) != len(self.horizons):
            raise ValueError("Horizons must be unique")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between 0 and 1")


def _model(family: str, parameter: float | int, seed: int):
    if family == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=parameter))
    if family == "histgb":
        # No random internal holdout: model selection uses our chronological block.
        return HistGradientBoostingRegressor(max_leaf_nodes=int(parameter), max_iter=80,
                                            learning_rate=0.08, l2_regularization=2,
                                            early_stopping=False, random_state=seed)
    raise ValueError(f"Unknown model family: {family}")


def run_benchmark(frame: pd.DataFrame, output: str | Path, config: Config = Config(),
                  source: str = "user_csv") -> dict:
    frame = validate_frame(frame)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()
    report = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "config": asdict(config), "dataset": {"source": source, "rows": len(frame),
                  "sha256": fingerprint, "sha256_representation": "Validated UTC frame serialized by pandas.to_csv(index=False), UTF-8", "start": str(frame.timestamp.iloc[0]),
                  "end": str(frame.timestamp.iloc[-1]), "frequency": "hourly"},
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
              "protocol": "55% fit / 15% validation / 15% calibration / 15% test; purged origins; refit on first 70%",
              "interval_note": "Empirical coverage only: dependent, drifting time-series data do not guarantee exchangeability.",
              "splits": {}, "results": []}
    prediction_frames = []
    values = frame.y.to_numpy()
    with threadpool_limits(limits=1):
        for horizon in config.horizons:
            data = make_features(frame, horizon, calendar=True)
            splits = purged_splits(data, len(frame))
            report["splits"][str(horizon)] = {
                name: {"count": len(index), "first_origin": int(data.origins[index[0]]),
                       "last_origin": int(data.origins[index[-1]]),
                       "first_target": int(data.targets[index[0]]),
                       "last_target": int(data.targets[index[-1]])}
                for name, index in splits.items()}
            train, val, cal, test = (splits[name] for name in ("train", "validation", "calibration", "test"))
            # Refit excludes every calibration/test label, including their boundary origins.
            refit = np.flatnonzero(data.targets < int(len(frame) * 0.70))
            horizon_rows = []

            def record(name, validation_mae, pred_cal, pred_test, params, candidates, seconds):
                radius = conformal_radius(np.abs(data.y[cal] - pred_cal), config.alpha)
                row = {"model": name, "horizon": horizon, "validation_mae": float(validation_mae),
                       **point_metrics(data.y[test], pred_test),
                       **interval_metrics(data.y[test], pred_test, radius, config.alpha),
                       "conformal_radius": radius, "params": params, "selection_trials": candidates,
                       "refit_examples": int(len(refit)) if name != "seasonal_naive" else 0,
                       "refit_last_target": int(data.targets[refit[-1]]) if name != "seasonal_naive" else None,
                       "calibration_examples": int(len(cal)), "test_examples": int(len(test)),
                       "fit_and_predict_seconds": round(seconds, 5)}
                horizon_rows.append(row)
                prediction_frames.append(pd.DataFrame({
                    "model": name, "horizon": horizon, "origin_index": data.origins[test],
                    "target_index": data.targets[test],
                    "target_timestamp": frame.timestamp.iloc[data.targets[test]].to_numpy(),
                    "actual": data.y[test], "prediction": pred_test,
                    "lower": pred_test - radius, "upper": pred_test + radius}))

            started = perf_counter()
            seasonal_val = seasonal_naive(values, data.origins[val], horizon)
            record("seasonal_naive", point_metrics(data.y[val], seasonal_val)["mae"],
                   seasonal_naive(values, data.origins[cal], horizon),
                   seasonal_naive(values, data.origins[test], horizon),
                   {"period": 24}, [], perf_counter() - started)
            for calendar in (False, True):
                variant = make_features(frame, horizon, calendar=calendar)
                for family, parameters in (("ridge", (0.1, 10.0, 100.0)), ("histgb", (7, 15))):
                    started = perf_counter()
                    candidates = []
                    for parameter in parameters:
                        model = _model(family, parameter, config.seed)
                        model.fit(variant.x[train], data.y[train])
                        score = point_metrics(data.y[val], model.predict(variant.x[val]))["mae"]
                        candidates.append({"parameter": parameter, "validation_mae": score})
                    best = min(candidates, key=lambda item: item["validation_mae"])
                    model = _model(family, best["parameter"], config.seed)
                    model.fit(variant.x[refit], data.y[refit])
                    pred_cal = model.predict(variant.x[cal])
                    pred_test = model.predict(variant.x[test])
                    record(f"{family}_{'calendar' if calendar else 'lags'}", best["validation_mae"],
                           pred_cal, pred_test,
                           {"alpha" if family == "ridge" else "max_leaf_nodes": best["parameter"],
                            "calendar": calendar, "features": list(variant.names)},
                           candidates, perf_counter() - started)
            winner = min(horizon_rows, key=lambda item: item["validation_mae"])["model"]
            for row in horizon_rows:
                row["selected_by_validation"] = row["model"] == winner
            report["results"].extend(horizon_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(output / "predictions.csv", index=False)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    svg = render_report(report, predictions, output / "report.html")
    (output / "preview.svg").write_text(svg, encoding="utf-8")
    return report
