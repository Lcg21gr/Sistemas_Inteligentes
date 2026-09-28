import warnings

import numpy as np
import pandas as pd
import pytest

from nusecast.data import make_features
from nusecast.experiment import evaluate
from nusecast.metrics import metrics
from nusecast.models import BASELINES, TUNED, GammaPoisson, candidates, make_model
from test_data import config, panel


@pytest.mark.parametrize("family", [*TUNED, *BASELINES])
def test_all_families_produce_nonnegative_predictions(family, panel, config):
    frame = make_features(panel, pd.Series(1.0, index=panel.index), config)
    train = frame.loc[frame.period <= "2023-12"]
    test = frame.loc[frame.period == "2025-01"]
    profile = dict(config["profiles"]["demo"], mlp_iterations=10, evolution_generations=2, trees=5)
    params = candidates(family, 1)[0] if family in TUNED else {}
    model = make_model(family, params, 42, profile)
    with warnings.catch_warnings(record=True):
        model.fit(train)
    prediction = model.predict(test)
    assert prediction.shape == (2,)
    assert np.isfinite(prediction).all()
    assert (prediction >= 0).all()


def test_gamma_poisson_intervals_and_no_prior_double_counting(panel, config):
    frame = make_features(panel, pd.Series(1.0, index=panel.index), config)
    train = frame.loc[frame.period < "2025-01"].copy()
    target = frame.loc[frame.period == "2025-01"]
    model = GammaPoisson({"window": 3, "prior_strength": 0.5}, 0.9).fit(train)
    original = model.predict(target)

    train.loc[train.period >= "2024-10", "y"] = 999999
    model.fit(train)
    np.testing.assert_allclose(original, model.predict(target))
    low, high = model.intervals(target)
    assert (low <= high).all() and (low >= 0).all()
    assert (low <= original).all() and (high >= original).all()


def test_evaluation_cannot_see_future_targets(panel, config):
    frame = make_features(panel, pd.Series(1.0, index=panel.index), config)
    diagnostics = []
    profile = config["profiles"]["demo"]
    first = evaluate(frame, ["2025-01"], "GLM", {}, [0], profile, 0.9, diagnostics)
    frame.loc[frame.period >= "2025-01", "y"] = 999999
    second = evaluate(frame, ["2025-01"], "GLM", {}, [0], profile, 0.9, diagnostics)
    np.testing.assert_allclose(first.prediction, second.prediction)
    assert all(d["training_end"] < d["period"] for d in diagnostics)


def test_metrics_known_values():
    frame = pd.DataFrame({"y": [0., 10.], "prediction": [2., 6.], "lower": [0., 5.], "upper": [3., 8.]})
    result = metrics(frame)
    assert result["mae"] == 3
    assert result["bias"] == -1
    assert result["coverage"] == 0.5
    assert result["mean_underprediction"] == 2
    assert result["mean_overprediction"] == 1


def test_metrics_reject_nan():
    with pytest.raises(ValueError):
        metrics(pd.DataFrame({"y": [1], "prediction": [np.nan]}))
