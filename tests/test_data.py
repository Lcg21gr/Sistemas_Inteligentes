import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nusecast.data import FEATURES, causal_history, load_config, load_panel, make_features, split_label


@pytest.fixture
def config():
    return load_config(Path(__file__).parents[1] / "config.json")


@pytest.fixture
def panel():
    dates = pd.period_range("2020-01", "2026-07", freq="M")
    return pd.DataFrame({"UPZ47": np.arange(len(dates)) + 100,
                         "UPZ48": np.arange(len(dates)) + 200}, index=dates, dtype=float)


def test_no_future_leakage_even_across_suspect_month(panel, config):
    coverage = pd.Series(0.8, index=panel.index)
    before = make_features(panel, coverage, config)
    for target in ("2024-02", "2024-05", "2025-01", "2026-07"):
        changed = panel.copy()
        changed.loc[changed.index >= pd.Period(target, "M")] += 50_000
        modified_coverage = coverage.copy()
        modified_coverage.loc[modified_coverage.index >= pd.Period(target, "M")] = 0.1
        after = make_features(changed, modified_coverage, config)
        pd.testing.assert_frame_equal(before.loc[before.period <= target, FEATURES],
                                      after.loc[after.period <= target, FEATURES])


def test_correction_uses_only_previous_three_and_preserves_target(panel, config):
    original = panel.copy()
    clean, flags = causal_history(panel, config)
    date = pd.Period("2024-01", "M")
    assert clean.loc[date, "UPZ47"] == panel.loc[date-3:date-1, "UPZ47"].mean()
    assert flags.loc[date].all()
    pd.testing.assert_frame_equal(panel, original)
    frame = make_features(panel, pd.Series(1.0, index=panel.index), config)
    assert not frame.loc[frame.period == "2024-01", "eligible"].any()
    assert frame.loc[(frame.period == "2024-01") & (frame.upz == "UPZ47"), "y"].iloc[0] == panel.loc[date, "UPZ47"]


def test_next_month_has_no_fabricated_target(panel, config):
    frame = make_features(panel, pd.Series(1.0, index=panel.index), config, include_next=True)
    forecast = frame.loc[frame.period == "2026-08"]
    assert len(forecast) == 2
    assert forecast.y.isna().all()
    assert forecast.lag1.tolist() == panel.iloc[-1].tolist()


def test_temporal_partitions(panel, config):
    frame = make_features(panel, pd.Series(1.0, index=panel.index), config)
    frame = frame.loc[frame.eligible]
    groups = frame.period.map(lambda p: split_label(p, config))
    assert (groups == "entrenamiento").sum() == 36 * 2
    assert (groups == "validacion").sum() == 10 * 2
    assert (groups == "prueba").sum() == 19 * 2
    assert frame.loc[groups == "entrenamiento", "period"].max() < frame.loc[groups == "validacion", "period"].min()
    assert frame.loc[groups == "validacion", "period"].max() < frame.loc[groups == "prueba", "period"].min()


def test_keep_suspect_is_explicit_sensitivity(panel, config):
    config["exclude_suspect"] = False
    clean, flags = causal_history(panel, config)
    pd.testing.assert_frame_equal(clean, panel)
    assert not flags.any().any()


def test_invalid_dates_rejected(tmp_path, config):
    config["validation_end"] = "2022-12"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="ordenadas"):
        load_config(path)


def write_raw(path, duplicate=False, missing=False, negative=False):
    dates = pd.period_range("2020-01", "2021-02", freq="M")
    rows = [{"ANIO": d.year, "MES": d.month, "TIPO_INCIDENTE": "942", "LOCALIDAD": "KENNEDY",
             "COD_UPZ": u, "UPZ": u, "CANT_INCIDENTES": 100} for d in dates for u in ["UPZ47", "UPZ48", "UPZ999"]]
    if duplicate:
        rows.append(rows[0].copy())
    if missing:
        rows.pop(0)
    if negative:
        rows[0]["CANT_INCIDENTES"] = -1
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.mark.parametrize("problem,match", [("duplicate", "duplicadas"), ("missing", "incompleto"), ("negative", "no negativos")])
def test_bad_raw_is_rejected(tmp_path, config, problem, match):
    path = tmp_path / "raw.csv"
    write_raw(path, **{problem: True})
    config["end"] = "2021-02"
    with pytest.raises(ValueError, match=match):
        load_panel(path, config)


def test_unknown_upz_reported_not_silently_dropped(tmp_path, config):
    path = tmp_path / "raw.csv"
    write_raw(path)
    config["end"] = "2021-02"
    loaded, coverage, names, audit = load_panel(path, config)
    assert loaded.shape == (14, 2)
    assert "UPZ999" not in names
    assert audit["excluded_unknown_upz_incidents"] == 1400
    assert np.allclose(coverage, 2 / 3)
