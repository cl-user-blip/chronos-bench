import json

import numpy as np
import pandas as pd
import pytest

from chronos_bench.benchmark import Config, run_benchmark
from chronos_bench.data import synthetic_load, validate_frame
from chronos_bench.features import make_features, purged_splits, seasonal_naive
from chronos_bench.metrics import conformal_radius, interval_metrics, point_metrics


def test_features_cannot_read_future_targets():
    original = synthetic_load(1008)
    changed = original.copy()
    cutoff = 500
    changed.loc[cutoff + 1:, "y"] += 10000
    before = make_features(original, horizon=24)
    after = make_features(changed, horizon=24)
    past = before.origins <= cutoff
    np.testing.assert_array_equal(before.x[past], after.x[past])
    assert np.any(before.y[past] != after.y[past])  # Counterfactual really changed future labels.


@pytest.mark.parametrize("horizon", [1, 6, 24, 48])
def test_purged_splits_never_cross_label_boundaries(horizon):
    frame = synthetic_load(1008)
    data = make_features(frame, horizon)
    blocks = purged_splits(data, len(frame))
    previous = None
    for indices in blocks.values():
        if previous is not None:
            assert data.targets[previous].max() < data.origins[indices].min()
        assert np.all(data.targets[indices] - data.origins[indices] == horizon)
        previous = indices


def test_right_aligned_rolling_statistics_are_auditable():
    frame = synthetic_load(720)
    data = make_features(frame, horizon=6)
    first_origin = data.origins[0]
    mean_col = data.names.index("mean_24")
    assert data.x[0, mean_col] == pytest.approx(frame.y.iloc[first_origin - 23:first_origin + 1].mean())
    assert data.x[0, 0] == frame.y.iloc[first_origin]


@pytest.mark.parametrize("horizon", [1, 24, 25, 48, 72])
def test_seasonal_naive_never_uses_unobserved_value(horizon):
    values = np.arange(400)
    origins = np.arange(168, 250)
    predicted = seasonal_naive(values, origins, horizon)
    assert np.all(predicted <= origins)
    assert np.all((origins + horizon - predicted) % 24 == 0)
    assert np.all(origins - predicted < 24)


def test_finite_sample_calibration_rank_is_not_naive_percentile():
    residuals = np.arange(1, 10, dtype=float)
    # ceil((9+1)*0.9) = 9: the correct finite sample order statistic is 9.
    assert conformal_radius(residuals, 0.1) == 9
    with pytest.raises(ValueError, match="Too few"):
        conformal_radius(residuals, 0.01)


def test_metrics_zeros_and_interval_penalty():
    assert point_metrics(np.zeros(2), np.ones(2)) == {"mae": 1, "rmse": 1, "wape": None}
    result = interval_metrics(np.array([0., 4.]), np.array([0., 0.]), 1, 0.1)
    assert result["coverage"] == 0.5
    assert result["interval_score"] == 32
    with pytest.raises(ValueError):
        point_metrics(np.array([1]), np.array([np.inf]))


@pytest.mark.parametrize("defect", ["gap", "duplicate", "nan", "infinite", "unsorted"])
def test_input_contract_rejects_ambiguous_data(defect):
    frame = synthetic_load(744)
    if defect == "gap":
        frame = frame.drop(index=400)
    elif defect == "duplicate":
        frame.loc[400, "timestamp"] = frame.loc[399, "timestamp"]
    elif defect == "nan":
        frame.loc[400, "y"] = np.nan
    elif defect == "infinite":
        frame.loc[400, "y"] = np.inf
    else:
        frame = frame.iloc[::-1]
    with pytest.raises(ValueError):
        validate_frame(frame)


def test_test_targets_cannot_change_tuning_or_calibration(tmp_path):
    frame = synthetic_load(720)
    changed = frame.copy()
    changed.loc[int(0.85 * len(frame)):, "y"] += 1000
    one = run_benchmark(frame, tmp_path / "one", Config((1,)))
    two = run_benchmark(changed, tmp_path / "two", Config((1,)))
    for before, after in zip(one["results"], two["results"]):
        assert before["params"] == after["params"]
        assert before["selection_trials"] == after["selection_trials"]
        assert before["conformal_radius"] == after["conformal_radius"]
        assert before["selected_by_validation"] == after["selected_by_validation"]
        assert before["mae"] != after["mae"]
        if before["refit_last_target"] is not None:
            assert before["refit_last_target"] < one["splits"]["1"]["calibration"]["first_origin"]
    stored = json.loads((tmp_path / "one" / "report.json").read_text(encoding="utf-8"))
    assert stored["dataset"]["rows"] == 720
    assert len(stored["results"]) == 5
    predictions = pd.read_csv(tmp_path / "one" / "predictions.csv")
    assert set(predictions.model) == {r["model"] for r in stored["results"]}
    assert "Forecasts you can audit" in (tmp_path / "one" / "report.html").read_text(encoding="utf-8")


def test_invalid_configuration_fails_before_running():
    for horizons in ((0,), (1, 1), (1.5,), (True,), ()):
        with pytest.raises(ValueError):
            Config(horizons)
    with pytest.raises(ValueError):
        Config(alpha=1)
    frame = synthetic_load(720)
    with pytest.raises(ValueError, match="fewer than 24"):
        purged_splits(make_features(frame, 100), len(frame))
