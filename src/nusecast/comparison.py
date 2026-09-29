import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from nusecast.metrics import metrics
from nusecast.models import FAMILIES


@dataclass
class Version:
    name: str
    predictions: pd.DataFrame
    parameters: dict
    provenance: str
    source_hash: str = ""
    search: pd.DataFrame | None = None


def checked_predictions(frame):
    required = ["fase", "family", "period", "upz", "y", "prediction"]
    if not set(required).issubset(frame.columns) or frame.empty:
        raise ValueError("Faltan predicciones o columnas de comparación.")
    data = frame[required].copy()
    if data.isna().any().any() or set(data.fase) != {"validacion", "prueba"}:
        raise ValueError("Se requieren validación y prueba sin valores ausentes.")
    for column in ["y", "prediction"]:
        data[column] = pd.to_numeric(data[column], errors="raise")
        if not np.isfinite(data[column]).all() or data[column].lt(0).any():
            raise ValueError("Conteos o predicciones no válidos.")
    data["period"] = pd.PeriodIndex(data.period, freq="M").astype(str)
    if data.duplicated(["fase", "family", "period", "upz"]).any():
        raise ValueError("Hay predicciones duplicadas para una misma unidad de evaluación.")
    return data.sort_values(["fase", "family", "period", "upz"]).reset_index(drop=True)


def load_run(path, name=None):
    config = json.loads((path / "config.json").read_text())
    manifest = json.loads((path / "manifest.json").read_text())
    selection = json.loads((path / "selection.json").read_text())
    predictions = checked_predictions(pd.concat([
        pd.read_csv(path / "predictions_validation.csv").assign(fase="validacion"),
        pd.read_csv(path / "predictions_test.csv").assign(fase="prueba")], ignore_index=True))
    profile = config["profiles"][manifest["profile"]]
    parameters = {}
    for family, spec in selection["models"].items():
        value = dict(spec["params"])
        value.update(profile=manifest["profile"], semillas=manifest["seeds"],
                     max_iter=profile["mlp_iterations"] if family == "RN" else None,
                     configuraciones=manifest.get("actual_configs_per_family", {}).get(family, profile["candidates"]),
                     entrenamiento_hasta=config["train_end"], validacion_hasta=config["validation_end"],
                     prueba_hasta=config["test_end"], correccion_causal=config["exclude_suspect"])
        if family == "RN":
            value.setdefault("learning_rate", 0.003)
            value.setdefault("activation", "relu")
            value["other_upz_lag1"] = True
        parameters[family] = value
    label = f"{config['experiment_name']} · {path.name}" if config.get("experiment_name") else path.name
    search = pd.read_csv(path / "search_validation.csv") if (path / "search_validation.csv").is_file() else None
    return Version(name or label, predictions, parameters,
                   "Ejecución del laboratorio con configuración y manifiesto guardados.",
                   manifest.get("source_sha256", ""), search)


def targets(version, family, phase):
    data = version.predictions
    return data.loc[(data.family == family) & (data.fase == phase), ["period", "upz", "y"]].sort_values(
        ["period", "upz"]).reset_index(drop=True)


def comparable(versions, family, phase):
    first = targets(versions[0], family, phase)
    return not first.empty and all(first.equals(targets(version, family, phase)) for version in versions[1:])


def comparison_scores(versions, family, phase):
    if not 1 <= len(versions) <= 3:
        raise ValueError("Selecciona de una a tres versiones.")
    rows = []
    for version in versions:
        data = version.predictions
        subset = data.loc[(data.family == family) & (data.fase == phase)]
        if subset.empty:
            raise ValueError("El modelo no está presente en todas las versiones.")
        rows.append({"Versión": version.name, **metrics(subset)})
    result = pd.DataFrame(rows)
    if comparable(versions, family, phase):
        reference = result.iloc[0].mae
        result["Δ MAE vs primera"] = result.mae - reference
        result["Reducción MAE (%)"] = (reference - result.mae) / reference * 100 if reference else np.nan
    return result


def render_comparison(runs):
    st.header("Historial y comparación de versiones")
    st.caption("Compara hasta tres ejecuciones realizadas en esta herramienta. Se conservan las anteriores; no se reentrena al comparar.")
    versions = []
    for run in runs:
        try:
            versions.append(load_run(run))
        except (OSError, ValueError, KeyError, TypeError) as error:
            st.warning(f"No se pudo leer {run.name}: {error}")
    if not versions:
        st.info("Todavía no hay ejecuciones terminadas.")
        return
    catalog = {version.name: version for version in versions}
    defaults = list(reversed(list(catalog)[:3]))
    chosen = st.multiselect("Ejecuciones a comparar", list(catalog), default=defaults,
                            max_selections=3)
    if not chosen:
        st.info("Selecciona al menos una versión.")
        return
    selected = [catalog[name] for name in chosen]
    shared = set.intersection(*(set(v.predictions.family) for v in selected))
    if not shared:
        st.warning("Las versiones elegidas no tienen un modelo en común.")
        return
    families = sorted(shared, key=lambda f: (f != "RN", f))
    family = st.selectbox("Modelo a comparar", families, format_func=lambda f: FAMILIES.get(f, f))
    phase = st.radio("Periodo de evaluación", ["validacion", "prueba"], horizontal=True,
                     format_func=lambda p: "Validación" if p == "validacion" else "Prueba")
    same = comparable(selected, family, phase)
    if same:
        st.success("Coinciden meses, UPZ y valores reales de evaluación en las versiones elegidas.")
    else:
        st.warning("Las filas o los valores reales difieren. No se calcula una mejora porcentual ni una curva comparativa.")
    hashes = {v.source_hash for v in selected if v.source_hash}
    if len(hashes) > 1 or any(not v.source_hash for v in selected):
        st.info("Coincidir en evaluación no demuestra que la preparación o los datos de entrenamiento sean idénticos. "
                "El archivo de origen cambió o alguna ejecución no conserva su identificador.")
    score = comparison_scores(selected, family, phase)
    shown = score.rename(columns={"mae": "MAE", "rmse": "RMSE", "bias": "Sesgo", "n": "Filas"})
    shown = shown[["Versión", "MAE", "RMSE", "Sesgo", "Filas"] +
                  [name for name in ["Δ MAE vs primera", "Reducción MAE (%)"] if name in shown]]
    st.dataframe(shown.round(3), hide_index=True, width="stretch")
    st.download_button("Descargar historial seleccionado", shown.to_csv(index=False),
                       "comparacion_experimentos.csv", "text/csv")
    st.caption("MAE sobre el promedio de predicciones de las semillas, no promedio de sus MAE. "
               "Δ negativo indica menor error frente a la primera versión seleccionada. Elige parámetros con validación.")
    with st.expander("Parámetros y procedencia", expanded=True):
        parameters = pd.DataFrame({v.name: v.parameters[family] for v in selected}).fillna("No informado").map(
            lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value))
        parameters = parameters.rename(index={"hidden": "Neuronas por capa", "learning_rate": "Tasa de aprendizaje",
            "alpha": "Regularización L2", "activation": "Activación", "max_iter": "Iteraciones máximas",
            "semillas": "Semillas", "configuraciones": "Configuraciones evaluadas", "other_upz_lag1": "Señal de otras UPZ",
            "profile": "Perfil", "entrenamiento_hasta": "Entrenamiento hasta", "validacion_hasta": "Validación hasta",
            "prueba_hasta": "Prueba hasta", "correccion_causal": "Corrección causal", "semillas_seleccion": "Semillas de selección"})
        st.table(parameters)
        for v in selected:
            st.caption(f"{v.name}: {v.provenance}")
        st.caption("Cambiar a la vez arquitectura, iteraciones y semillas no aísla el efecto de un hiperparámetro. "
                   "La comparación histórica es descriptiva; el factorial permite contrastes controlados.")
    for version in selected:
        if version.search is not None:
            with st.expander(f"Combinaciones evaluadas en validación · {version.name}"):
                render_search(version.search, family)
    if same:
        colors = ["#062F46", "#B68200", "#27835B"]
        combined = pd.concat([v.predictions.loc[(v.predictions.family == family) &
            (v.predictions.fase == phase)].assign(Versión=v.name) for v in selected], ignore_index=True)
        combined["Error absoluto"] = (combined.prediction - combined.y).abs()
        monthly = combined.groupby(["Versión", "period"], as_index=False)["Error absoluto"].mean()
        st.subheader("Cómo cambia el error mes a mes")
        st.plotly_chart(px.line(monthly, x="period", y="Error absoluto", color="Versión", markers=True,
            color_discrete_sequence=colors, labels={"period": "Mes", "Error absoluto": "MAE mensual"}), width="stretch")
        upz = st.selectbox("UPZ para comparar pronósticos", sorted(combined.upz.unique()))
        local = combined.loc[combined.upz == upz]
        figure = px.line(local, x="period", y="prediction", color="Versión", markers=True,
            color_discrete_sequence=colors, labels={"period": "Mes", "prediction": "Registros de incidentes"})
        real = local.loc[local["Versión"] == selected[0].name].sort_values("period")
        figure.add_scatter(x=real.period, y=real.y, mode="lines+markers", name="Observado", line=dict(color="#777777", dash="dash"))
        st.plotly_chart(figure, width="stretch")
        st.caption("Predicciones retrospectivas a un mes. La variación entre semillas no es un intervalo de confianza.")


def render_search(search, family):
    subset = search.loc[search.family == family].copy()
    if subset.empty:
        st.caption("Modelo de referencia sin búsqueda de hiperparámetros.")
        return
    parameters = pd.json_normalize(subset.params.map(json.loads)).reset_index(drop=True)
    table = pd.concat([subset.drop(columns="params").reset_index(drop=True), parameters], axis=1)
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption("MAE medio y desviación poblacional entre semillas. Selección hecha solo con validación.")
    if len(table) > 1 and {"hidden", "learning_rate", "alpha"}.issubset(table.columns):
        table["Capas"] = table.hidden.map(lambda value: ",".join(map(str, value)))
        table["L2"] = table.alpha.astype(str)
        st.plotly_chart(px.line(table.sort_values("learning_rate"), x="learning_rate", y="mae_mean_seeds",
            color="Capas", facet_col="L2", markers=True, log_x=True, error_y="mae_std_seeds",
            labels={"learning_rate": "Tasa de aprendizaje", "mae_mean_seeds": "MAE medio de validación"}), width="stretch")
