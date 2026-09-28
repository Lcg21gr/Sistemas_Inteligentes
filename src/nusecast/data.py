from __future__ import annotations

import hashlib
import json
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

LAGS = [f"lag{i}" for i in range(1, 13)]
NUMERIC = LAGS + ["ma3", "month_sin", "month_cos", "workdays", "coverage_lag1",
                  "coverage_lag12", "other_upz_lag1", "imputed_lags"]
FEATURES = NUMERIC + ["upz"]
REQUIRED = ["ANIO", "MES", "TIPO_INCIDENTE", "LOCALIDAD", "COD_UPZ", "UPZ", "CANT_INCIDENTES"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    return validate_config(config)


def validate_config(config: dict) -> dict:
    first, train, val, test = (pd.Period(config[k], "M") for k in
                               ["first_target", "train_end", "validation_end", "test_end"])
    if not first <= train < val < test <= pd.Period(config["end"], "M"):
        raise ValueError("Las fechas de entrenamiento, validación y prueba no están ordenadas.")
    if first < pd.Period(config["start"], "M") + 12:
        raise ValueError("Se necesitan doce meses de historia antes del primer objetivo.")
    if not 0 < config["interval_level"] < 1:
        raise ValueError("interval_level debe estar entre 0 y 1.")
    for month in config["suspect_months"]:
        pd.Period(month, "M")
    return config


def load_panel(path: Path, config: dict) -> tuple[pd.DataFrame, pd.Series, dict, dict]:
    pieces, rows = [], 0
    for chunk in pd.read_csv(path, dtype=str, usecols=REQUIRED, chunksize=150_000):
        rows += len(chunk)
        subset = chunk.loc[(chunk.TIPO_INCIDENTE == config["incident_code"]) &
                           (chunk.LOCALIDAD == config["locality"])].copy()
        if not subset.empty:
            pieces.append(subset)
    if not pieces:
        raise ValueError("No hay registros para el código y la localidad configurados.")
    selected = pd.concat(pieces, ignore_index=True)
    selected["period"] = pd.PeriodIndex(selected.ANIO + "-" + selected.MES.str.zfill(2), freq="M")
    selected = selected.loc[selected.period.between(pd.Period(config["start"], "M"),
                                                    pd.Period(config["end"], "M"))].copy()
    if selected.empty:
        raise ValueError("No hay registros en el periodo solicitado.")
    selected["count"] = pd.to_numeric(selected.CANT_INCIDENTES, errors="raise")
    if selected["count"].isna().any() or (selected["count"] < 0).any() or (selected["count"] % 1 != 0).any():
        raise ValueError("Los conteos deben ser enteros no negativos, sin valores faltantes.")
    located = selected.loc[selected.COD_UPZ.notna() &
                           selected.COD_UPZ.ne(config["unknown_upz"])].copy()
    if located.UPZ.isna().any():
        raise ValueError("Faltan nombres de UPZ en registros con ubicación conocida.")
    if located.duplicated(["period", "COD_UPZ"]).any():
        raise ValueError("Hay claves mes-UPZ duplicadas en el alcance. Revise antes de sumar.")
    if located.groupby("COD_UPZ").UPZ.nunique().gt(1).any():
        raise ValueError("Una UPZ tiene varios nombres; revise la correspondencia territorial.")
    panel = located.pivot(index="period", columns="COD_UPZ", values="count").sort_index(axis=1)
    panel = panel.reindex(pd.period_range(config["start"], config["end"], freq="M"))
    if panel.empty or panel.isna().any().any():
        raise ValueError("Panel incompleto: una ausencia no equivale automáticamente a cero.")
    total = selected.groupby("period")["count"].sum().reindex(panel.index)
    coverage = panel.sum(axis=1).div(total.where(total > 0))
    if coverage.isna().any():
        raise ValueError("No es posible calcular la cobertura geográfica en todos los meses.")
    names = located.drop_duplicates("COD_UPZ").set_index("COD_UPZ").UPZ.to_dict()
    audit = {
        "source_sha256": sha256(path), "raw_rows": rows,
        "scope_rows": len(selected), "panel_cells": int(panel.size),
        "months": len(panel), "upz_count": len(panel.columns),
        "start": str(panel.index.min()), "end": str(panel.index.max()),
        "all_locality_incidents": int(total.sum()),
        "located_incidents": int(panel.to_numpy().sum()),
        "excluded_unknown_upz_incidents": int(total.sum() - panel.to_numpy().sum()),
        "weighted_geographic_coverage": float(panel.to_numpy().sum() / total.sum()),
        "names": names,
        "warning": "Conteos de registros NUSE, no accidentes únicos ni demanda total de Bogotá."
    }
    return panel.astype(float), coverage, names, audit


def causal_history(panel: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = panel.copy().astype(float)
    flags = pd.DataFrame(False, index=panel.index, columns=panel.columns)
    if config["exclude_suspect"]:
        for date in sorted(pd.Period(p, "M") for p in config["suspect_months"]):
            if date in clean.index:
                past = clean.loc[clean.index < date].tail(3)
                if past.empty:
                    raise ValueError("No hay historia previa para corregir el mes sospechoso.")
                clean.loc[date] = past.mean(axis=0)
                flags.loc[date] = True
    return clean, flags


def make_features(panel: pd.DataFrame, coverage: pd.Series, config: dict,
                  include_next: bool = False) -> pd.DataFrame:
    clean, flags = causal_history(panel, config)
    suspect = set(pd.Period(p, "M") for p in config["suspect_months"]) if config["exclude_suspect"] else set()
    end = panel.index.max() + int(include_next)
    records = []
    for date in pd.period_range(config["first_target"], end, freq="M"):
        days = pd.date_range(date.start_time, date.end_time, freq="D")
        festive = holidays.Colombia(years=date.year)
        workdays = sum(d.weekday() < 5 and d.date() not in festive for d in days)
        previous_total = clean.loc[date - 1].sum()
        for upz in panel.columns:
            row = {"period": str(date), "upz": upz,
                   "y": float(panel.loc[date, upz]) if date in panel.index else np.nan,
                   "eligible": date not in suspect,
                   "month_sin": np.sin(2 * np.pi * date.month / 12),
                   "month_cos": np.cos(2 * np.pi * date.month / 12), "workdays": workdays,
                   "coverage_lag1": coverage.loc[date - 1], "coverage_lag12": coverage.loc[date - 12],
                   "other_upz_lag1": previous_total - clean.loc[date - 1, upz]}
            for lag in range(1, 13):
                row[f"lag{lag}"] = clean.loc[date - lag, upz]
                row[f"raw_lag{lag}"] = panel.loc[date - lag, upz]
                row[f"valid_lag{lag}"] = not flags.loc[date - lag, upz]
            row["ma3"] = np.mean([row[f"lag{i}"] for i in (1, 2, 3)])
            row["imputed_lags"] = sum(not row[f"valid_lag{i}"] for i in range(1, 13))
            records.append(row)
    result = pd.DataFrame(records)
    if result[NUMERIC].isna().any().any():
        raise ValueError("Se encontraron variables incompletas.")
    return result


def split_label(period: str, config: dict) -> str:
    if period <= config["train_end"]:
        return "entrenamiento"
    if period <= config["validation_end"]:
        return "validacion"
    if period <= config["test_end"]:
        return "prueba"
    return "pronostico"
