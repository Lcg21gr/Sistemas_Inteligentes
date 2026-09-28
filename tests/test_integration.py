import json
from pathlib import Path

import numpy as np
import pandas as pd

from nusecast.experiment import run_experiment
from nusecast.models import FAMILIES
from test_data import config


def test_end_to_end_creates_complete_consistent_artifacts(tmp_path, config):
    dates = pd.period_range("2020-01", "2021-12", freq="M")
    source = tmp_path / "raw.csv"
    pd.DataFrame([{"ANIO": d.year, "MES": d.month, "TIPO_INCIDENTE": "942", "LOCALIDAD": "KENNEDY",
                   "COD_UPZ": u, "UPZ": u, "CANT_INCIDENTES": 50 + i + j * 20}
                  for i, d in enumerate(dates) for j, u in enumerate(["UPZ47", "UPZ48", "UPZ999"])]).to_csv(source, index=False)
    config.update(end="2021-12", train_end="2021-06", validation_end="2021-09", test_end="2021-12", suspect_months=[])
    config["profiles"]["demo"].update(candidates=1, mlp_iterations=8, evolution_generations=2, trees=5)
    result = run_experiment(tmp_path, source, config)
    status = json.loads((result / "status.json").read_text())
    assert status["state"] == "complete"
    predictions = pd.read_csv(result / "predictions_test.csv")
    assert set(predictions.family) == set(FAMILIES)
    assert len(predictions) == len(FAMILIES) * 3 * 2
    assert not predictions.duplicated(["family", "period", "upz"]).any()
    assert not predictions[["y", "prediction"]].isna().any().any()
    forecast = pd.read_csv(result / "forecast.csv")
    assert set(forecast.period) == {"2022-01"}
    assert len(forecast) == len(FAMILIES) * 2
    val = pd.read_csv(result / "metrics_validation.csv")
    selection = json.loads((result / "selection.json").read_text())
    assert selection["chosen_by_validation"] == val.iloc[0].family
    fits = json.loads((result / "fit_diagnostics.json").read_text())
    assert all(fit["training_end"] < fit["period"] for fit in fits)
    assert "plotly" in (result / "informe.html").read_text()
    assert len((result / "informe.html").read_text()) > 10_000
    ri = predictions.loc[predictions.family == "RI"]
    assert np.isfinite(ri[["lower", "upper"]]).all().all()
    from streamlit.testing.v1 import AppTest
    app = (Path(__file__).parents[1] / "app.py").read_text()
    app = app.replace("ROOT = Path(__file__).resolve().parent", f"ROOT = Path({str(tmp_path)!r})")
    ui = AppTest.from_string(app).run(timeout=30)
    assert not ui.exception
    assert len(ui.tabs) == 6
    assert len(ui.metric) == 7
    ui.selectbox[1].select("RI").run(timeout=30)
    assert not ui.exception


def test_empty_dashboard_is_informative(tmp_path):
    from streamlit.testing.v1 import AppTest
    app = (Path(__file__).parents[1] / "app.py").read_text()
    app = app.replace("ROOT = Path(__file__).resolve().parent", f"ROOT = Path({str(tmp_path)!r})")
    ui = AppTest.from_string(app).run(timeout=30)
    assert not ui.exception
    assert "Todavía no hay ejecuciones" in ui.info[0].value
