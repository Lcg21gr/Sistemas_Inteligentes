import io
import json
from pathlib import Path

import openpyxl
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from nusecast.assistant import Settings, ask, build_context, load_settings, save_settings
from nusecast.models import FAMILIES, make_model
from nusecast.training import train_interactive, training_lock, validate_source, validate_training
from nusecast.uploads import ingest, normalize, sheet_names
from test_assistant import transport
from test_data import config


@pytest.fixture
def sample():
    return pd.DataFrame([{"ANIO": p.year, "MES": p.month, "TIPO_INCIDENTE": 942,
                          "LOCALIDAD": "KENNEDY", "COD_UPZ": u, "UPZ": u,
                          "CANT_INCIDENTES": 50 + i + j * 20}
                         for i, p in enumerate(pd.period_range("2020-01", "2021-12", freq="M"))
                         for j, u in enumerate(["UPZ47", "UPZ48", "UPZ999"])])


@pytest.fixture
def custom(config):
    config.update(end="2021-12", train_end="2021-06", validation_end="2021-09", test_end="2021-12",
                  suspect_months=[], models=["RN"])
    config["profiles"]["custom"] = dict(seeds=[42], candidates=1, mlp_iterations=10, trees=5, evolution_generations=2)
    config["model_grid"] = {"RN": [{"hidden": [8], "alpha": 0.2, "learning_rate": 0.01, "activation": "tanh"}]}
    return config


def workbook_bytes(sample, sheet="Incidentes", formula=False):
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.title = sheet
    worksheet.append(list(sample.columns))
    for row in sample.itertuples(index=False, name=None):
        worksheet.append(row)
    if formula:
        worksheet["G2"] = "=1+1"
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


@pytest.mark.parametrize("extension", [".xlsx", ".xlsm"])
def test_excel_import_matches_csv_without_mutating_original(tmp_path, sample, extension):
    content = workbook_bytes(sample)
    assert sheet_names(content) == ["Incidentes"]
    result, metadata = ingest(tmp_path, content, "../../source" + extension, sheet="Incidentes")
    pd.testing.assert_frame_equal(pd.read_csv(result), sample)
    assert metadata["rows"] == 72
    assert metadata["original_name"] == "source" + extension
    assert result.is_relative_to(tmp_path / "data" / "uploads")


def test_csv_semicolon_import_isolated_and_readable(tmp_path, sample):
    content = sample.to_csv(index=False, sep=";").encode()
    one, meta = ingest(tmp_path, content, "one.csv", separator=";")
    two, _ = ingest(tmp_path, content, "one.csv", separator=";")
    assert one != two and one.read_bytes() == two.read_bytes()
    assert len(meta["normalized_sha256"]) == 64


def test_excel_formulas_need_saved_values(tmp_path, sample):
    with pytest.raises(ValueError, match="Faltan valores"):
        ingest(tmp_path, workbook_bytes(sample, formula=True), "formula.xlsx", sheet="Incidentes")
    assert not list(tmp_path.rglob("datos.csv"))


@pytest.mark.parametrize("column,value,match", [("MES", 13, "mes"), ("ANIO", 2020.5, "año"),
                                               ("CANT_INCIDENTES", -1, "conteos"), ("CANT_INCIDENTES", None, "Faltan")])
def test_import_rejects_invalid_cells(sample, column, value, match):
    frame = sample.astype(object)
    frame.loc[0, column] = value
    with pytest.raises(ValueError, match=match):
        normalize(frame)


def test_import_rejects_missing_headers_and_unsupported_file(tmp_path, sample):
    with pytest.raises(ValueError, match="Faltan columnas"):
        normalize(sample.drop(columns=["MES"]))
    with pytest.raises(ValueError, match="XLS antiguo"):
        ingest(tmp_path, b"not a workbook", "bad.xls")


def test_validation_rejects_duplicate_and_missing_month(tmp_path, sample, custom):
    for frame, message in [(pd.concat([sample, sample.iloc[:1]]), "duplicadas"), (sample.iloc[1:], "incompleto")]:
        source, _ = ingest(tmp_path, frame.to_csv(index=False).encode(), "sample.csv")
        with pytest.raises(ValueError, match=message):
            validate_source(source, custom)


@pytest.mark.parametrize("field,value", [("hidden", [200]), ("hidden", []), ("alpha", -1),
                                         ("learning_rate", 1.0), ("activation", "unknown")])
def test_training_rejects_invalid_hyperparameters(custom, field, value):
    custom["model_grid"]["RN"][0][field] = value
    with pytest.raises(ValueError):
        validate_training(custom, "custom")


def test_manual_parameters_reach_estimator(custom):
    model = make_model("RN", custom["model_grid"]["RN"][0], 42, custom["profiles"]["custom"])
    estimator = model.pipeline.named_steps["model"]
    assert estimator.hidden_layer_sizes == (8,)
    assert estimator.learning_rate_init == 0.01
    assert estimator.alpha == 0.2
    assert estimator.activation == "tanh"


def test_training_lock_rejects_second_job(tmp_path):
    with training_lock(tmp_path):
        with pytest.raises(ValueError, match="activo"):
            with training_lock(tmp_path):
                pass


def test_manual_run_progress_and_dashboard_without_intervals(tmp_path, sample, custom):
    source, _ = ingest(tmp_path, sample.to_csv(index=False).encode(), "sample.csv")
    validated = validate_source(source, custom)
    assert validated["fits"] == 35
    events = []
    result = train_interactive(tmp_path, source, custom, lambda value, message: events.append((value, message)))
    assert events[0][0] == 0 and events[-1][0] == 1
    assert len([message for _, message in events if message.startswith("Ajuste:")]) == 35
    assert all(a[0] <= b[0] for a, b in zip(events, events[1:]))
    scores = pd.read_csv(result / "metrics_test.csv")
    assert set(scores.family) == {"RN", "BASE_LAST", "BASE_MA3", "BASE_SEASONAL", "GLM"}
    selection = json.loads((result / "selection.json").read_text())
    assert selection["models"]["RN"]["params"] == custom["model_grid"]["RN"][0]
    context = build_context(validated["audit"], scores, selection, pd.read_csv(result / "forecast.csv"), 0.9, FAMILIES)
    json.dumps(context, allow_nan=False)
    assert "No se incluyó Gamma-Poisson" in (result / "informe.html").read_text()
    ui = app_at(tmp_path).run(timeout=30)
    assert not ui.exception
    assert any("no incluye Gamma-Poisson" in info.value for info in ui.info)


def app_at(root):
    source = (Path(__file__).parents[1] / "app.py").read_text()
    source = source.replace("ROOT = Path(__file__).resolve().parent", f"ROOT = Path({str(root)!r})")
    return AppTest.from_string(source)


def test_training_ui_validation_invalidates_on_parameter_change(tmp_path, sample, custom):
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    sample.to_csv(raw / "Numero_incidentes.csv", index=False)
    (tmp_path / "config.json").write_text(json.dumps(custom))
    ui = app_at(tmp_path).run()
    ui.radio(key="main_page").set_value("Datos y entrenamiento").run()
    assert not ui.exception
    train = next(button for button in ui.button if button.label == "Entrenar y guardar experimento")
    assert train.disabled
    next(button for button in ui.button if button.label == "Validar datos y configuración").click().run()
    assert not ui.exception
    assert not next(button for button in ui.button if button.label == "Entrenar y guardar experimento").disabled
    next(field for field in ui.number_input if field.label == "Semilla inicial").set_value(43).run()
    assert next(button for button in ui.button if button.label == "Entrenar y guardar experimento").disabled
    assert not list((tmp_path / "outputs").glob("*/status.json"))
    next(button for button in ui.button if button.label == "Validar datos y configuración").click().run()
    next(button for button in ui.button if button.label == "Entrenar y guardar experimento").click().run(timeout=30)
    assert not ui.exception
    assert any("Experimento disponible" in message.value for message in ui.success)
    statuses = list((tmp_path / "outputs").glob("*/status.json"))
    assert len(statuses) == 1
    assert json.loads(statuses[0].read_text())["state"] == "complete"
    ui.run()
    assert len(list((tmp_path / "outputs").glob("*/status.json"))) == 1
    ui.radio(key="main_page").set_value("Historial").run()
    assert not ui.exception and len(ui.dataframe) == 1


def test_settings_persist_without_secret_and_temperature_is_sent(tmp_path, monkeypatch, transport):
    monkeypatch.setenv("OPENAI_API_KEY", "private-test-value")
    save_settings(tmp_path, {"model": "gpt-4.1-mini", "enabled": True, "temperature": 0.7, "max_tokens": 300})
    loaded = load_settings(tmp_path)
    assert loaded.ready and loaded.temperature == 0.7 and loaded.max_tokens == 300
    assert "private-test-value" not in (tmp_path / "assistant_settings.json").read_text()
    ask(loaded, "¿Qué significa MAE?", {})
    assert transport.call_args.kwargs["json"]["temperature"] == 0.7
    ask(Settings("test", model="unknown-model", enabled=True), "Pregunta", {})
    assert "temperature" not in transport.call_args.kwargs["json"]


def test_invalid_public_settings_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        save_settings(tmp_path, {"model": "gpt-4.1-mini", "enabled": True, "temperature": float("nan"), "max_tokens": 300})
    with pytest.raises(ValueError):
        save_settings(tmp_path, {"api_key": "should-not-save"})
    assert not (tmp_path / "assistant_settings.json").exists()


def test_settings_ui_does_not_consult_api(tmp_path, transport):
    ui = app_at(tmp_path).run()
    ui.radio(key="main_page").set_value("Configuración IA").run()
    assert not ui.exception
    ui.slider[0].set_value(0.4)
    next(button for button in ui.button if button.label == "Guardar configuración").click().run()
    assert not ui.exception
    assert load_settings(tmp_path).temperature == 0.4
    transport.assert_not_called()
