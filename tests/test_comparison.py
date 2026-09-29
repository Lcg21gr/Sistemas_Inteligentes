import copy

import pandas as pd
import pytest

from nusecast.comparison import Version, checked_predictions, comparable, comparison_scores
from nusecast.training import train_interactive
from nusecast.uploads import ingest
from test_data import config
from test_workbench import app_at, custom, sample


def versions():
    return [Version(f"Prueba {i}", checked_predictions(pd.DataFrame([
        {"fase": phase, "family": "RN", "period": period, "upz": "UPZ47", "y": 20.0,
         "prediction": 20.0 + error}
        for phase, period in [("validacion", "2024-01"), ("prueba", "2025-01")]
    ])), {"RN": {"hidden": [neurons]}}, "Datos sintéticos de prueba", "mismo-archivo")
        for i, (neurons, error) in enumerate([(8, 6), (16, 4), (32, 2)], 1)]


def test_three_versions_metrics_and_same_evaluation():
    data = versions()
    assert comparable(data, "RN", "prueba")
    assert comparable(data, "RN", "validacion")
    scores = comparison_scores(data, "RN", "prueba")
    assert scores.mae.tolist() == pytest.approx([6, 4, 2])
    assert scores.n.tolist() == [1, 1, 1]
    assert scores["Δ MAE vs primera"].tolist() == [0, -2, -4]
    assert scores["Reducción MAE (%)"].tolist() == pytest.approx([0, 100 / 3, 200 / 3])


def test_comparison_rejects_more_than_three():
    data = versions()
    with pytest.raises(ValueError, match="tres"):
        comparison_scores(data + data[:1], "RN", "prueba")
    with pytest.raises(ValueError, match="tres"):
        comparison_scores([], "RN", "prueba")


def test_mismatched_targets_do_not_offer_improvement_percentage():
    data = versions()[:2]
    data[1].predictions.loc[data[1].predictions.fase == "prueba", "y"] += 1
    assert not comparable(data, "RN", "prueba")
    assert "Reducción MAE (%)" not in comparison_scores(data, "RN", "prueba")
    assert comparable(data, "RN", "validacion")


def test_mismatched_periods_do_not_compare():
    data = versions()[:2]
    data[1].predictions.loc[data[1].predictions.fase == "prueba", "period"] = "2025-02"
    assert not comparable(data, "RN", "prueba")


def test_zero_reference_error_does_not_divide_by_zero():
    data = versions()
    data[0].predictions["prediction"] = data[0].predictions.y
    scores = comparison_scores(data, "RN", "prueba")
    assert scores["Reducción MAE (%)"].isna().all()
    assert scores["Δ MAE vs primera"].tolist() == [0, 4, 2]


def test_prediction_validation_rejects_duplicates():
    data = versions()[0].predictions
    with pytest.raises(ValueError, match="duplicadas"):
        checked_predictions(pd.concat([data, data.iloc[:1]]))


def test_prediction_validation_rejects_nonfinite_or_negative_values():
    data = versions()[0].predictions
    for value in [float("nan"), float("inf"), -1.0]:
        with pytest.raises(ValueError):
            checked_predictions(data.assign(prediction=value))


def test_history_ui_can_compare_own_runs_change_phase_and_empty_selection(tmp_path, sample, custom):
    source, _ = ingest(tmp_path, sample.to_csv(index=False).encode(), "sample.csv")
    for i, neurons in enumerate([8, 16, 32], 1):
        settings = copy.deepcopy(custom)
        settings["experiment_name"] = f"Prueba {i}"
        settings["model_grid"]["RN"][0]["hidden"] = [neurons]
        train_interactive(tmp_path, source, settings, lambda value, message: None)
    ui = app_at(tmp_path).run(timeout=30)
    ui.radio(key="main_page").set_value("Historial").run(timeout=30)
    assert not ui.exception
    assert not ui.checkbox
    assert len(ui.multiselect[0].value) == 3
    assert all(name.startswith("Prueba ") for name in ui.multiselect[0].options)
    assert ui.multiselect[0].proto.max_selections == 3
    assert len(ui.get("plotly_chart")) == 2
    assert len(ui.table) == 1
    ui.radio[1].set_value("prueba").run(timeout=30)
    assert not ui.exception
    assert len(ui.dataframe[0].value) == 3
    ui.multiselect[0].set_value([]).run()
    assert not ui.exception
    assert any("Selecciona al menos" in item.value for item in ui.info)


def test_history_does_not_load_external_results_even_if_present(tmp_path):
    external = tmp_path / "experimentos_rn" / "referencia_anterior"
    external.mkdir(parents=True)
    (external / "status.json").write_text('{"state": "complete"}')
    (external / "config.json").write_text("not valid JSON")
    ui = app_at(tmp_path).run(timeout=30)
    ui.radio(key="main_page").set_value("Historial").run(timeout=30)
    assert not ui.exception
    assert not ui.checkbox
    assert not ui.multiselect
    assert not ui.error
    assert not ui.warning
    assert any("Todavía no hay ejecuciones" in item.value for item in ui.info)
