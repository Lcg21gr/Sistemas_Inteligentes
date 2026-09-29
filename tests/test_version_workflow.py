import json
from pathlib import Path

import pytest

from nusecast.comparison import comparable, comparison_scores, load_run
from nusecast.training import factorial_grid, train_interactive, validate_source, validate_training
from nusecast.uploads import ingest
from test_data import config
from test_workbench import app_at, custom, sample


def control(items, label):
    return next(item for item in items if item.label == label)


def test_user_can_import_once_train_three_manual_versions_and_compare(tmp_path, sample, custom):
    source, metadata = ingest(tmp_path, sample.to_csv(index=False).encode(), "Mi_archivo.csv")
    (tmp_path / "config.json").write_text(json.dumps(custom))
    ui = app_at(tmp_path)
    ui.session_state["imported_data"] = ("original-import", str(source), metadata)
    ui.run()
    ui.radio(key="main_page").set_value("Datos y entrenamiento").run()
    assert control(ui.radio, "Fuente de datos").value == "Reutilizar último archivo importado"
    control(ui.multiselect, "Modelos a entrenar").set_value(["RN"]).run()
    control(ui.radio, "Ajuste de hiperparámetros").set_value("Configuración manual").run()
    control(ui.number_input, "Iteraciones máximas MLP").set_value(10).run()
    for i, neurons in enumerate([8, 16, 32], 1):
        control(ui.text_input, "Nombre de esta prueba").set_value(f"Prueba {i}")
        control(ui.text_input, "Neuronas por capa oculta").set_value(str(neurons)).run()
        assert control(ui.button, "Entrenar y guardar experimento").disabled
        control(ui.button, "Validar datos y configuración").click().run(timeout=30)
        assert not ui.exception
        control(ui.button, "Entrenar y guardar experimento").click().run(timeout=60)
        assert not ui.exception
        assert len(list((tmp_path / "outputs").glob("*/status.json"))) == i
    paths = sorted((tmp_path / "outputs").glob("*/status.json"),
                   key=lambda p: json.loads((p.parent / "manifest.json").read_text())["created_utc"])
    versions = [load_run(path.parent) for path in paths]
    assert len({version.source_hash for version in versions}) == 1
    assert [v.parameters["RN"]["hidden"] for v in versions] == [[8], [16], [32]]
    assert comparable(versions, "RN", "prueba")
    assert len(comparison_scores(versions, "RN", "prueba")) == 3
    ui.radio(key="main_page").set_value("Historial").run(timeout=30)
    assert not ui.exception
    assert len(control(ui.multiselect, "Ejecuciones a comparar").value) == 3
    assert len(ui.get("plotly_chart")) == 2
    ui.radio(key="main_page").set_value("Datos y entrenamiento").run(timeout=30)
    assert not ui.exception
    assert control(ui.radio, "Fuente de datos").value == "Reutilizar último archivo importado"
    assert len(list((tmp_path / "data" / "uploads").glob("*/datos.csv"))) == 1
    assert len(list((tmp_path / "outputs").glob("*/status.json"))) == 3


def test_factorial_grid_and_budget_are_real(tmp_path, sample, custom):
    grid = factorial_grid([0.0003, 0.003, 0.03], [8, 32, 128], [0.01, 1.0, 10.0])
    assert len(grid) == 27
    custom["model_grid"] = {"RN": grid}
    with pytest.raises(ValueError):
        validate_training(custom, "custom")
    custom["experiment_design"] = "factorial_rn"
    validate_training(custom, "custom")
    for rates, neurons, alphas in [([0.003, 0.003], [8], [1.0]), ([0], [8], [1.0]),
                                  ([0.003], [129], [1.0]), ([0.003], [8], [])]:
        with pytest.raises(ValueError):
            factorial_grid(rates, neurons, alphas)
    source, _ = ingest(tmp_path, sample.to_csv(index=False).encode(), "data.csv")
    assert validate_source(source, custom)["fits"] == 113


def test_factorial_nine_combinations_are_executed_and_saved(tmp_path, sample, custom):
    custom["model_grid"] = {"RN": factorial_grid([0.0003, 0.003, 0.03], [4, 8, 16], [1.0])}
    custom["experiment_design"] = "factorial_rn"
    custom["experiment_name"] = "Factorial 3x3"
    source, _ = ingest(tmp_path, sample.to_csv(index=False).encode(), "data.csv")
    events = []
    output = train_interactive(tmp_path, source, custom, lambda p, m: events.append(m))
    version = load_run(output)
    assert len(version.search) == 9
    assert version.parameters["RN"]["configuraciones"] == 9
    assert version.name.startswith("Factorial 3x3")
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["actual_configs_per_family"]["RN"] == 9
    assert sum(message.startswith("Ajuste:") for message in events) == 59


def test_factorial_ui_uses_current_uploaded_data_and_rejects_bad_factors(tmp_path, sample, custom):
    source, metadata = ingest(tmp_path, sample.to_csv(index=False).encode(), "Mi_archivo.csv")
    (tmp_path / "config.json").write_text(json.dumps(custom))
    ui = app_at(tmp_path)
    ui.session_state["imported_data"] = ("original-import", str(source), metadata)
    ui.run()
    ui.radio(key="main_page").set_value("Datos y entrenamiento").run()
    control(ui.radio, "Ajuste de hiperparámetros").set_value("Factorial controlado RN").run()
    control(ui.button, "Validar datos y configuración").click().run(timeout=30)
    assert not ui.exception
    assert not control(ui.button, "Entrenar y guardar experimento").disabled
    control(ui.text_input, "Neuronas a probar (una capa)").set_value("129").run()
    assert any("Revisa los factores" in item.value for item in ui.error)
