from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import time
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from nusecast.data import load_panel, make_features, sha256, split_label
from nusecast.metrics import average_seeds, metrics, summarize
from nusecast.models import BASELINES, TUNED, make_model
from nusecast.training import model_grid, validate_training


def save_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def evaluate(frame, periods, family, params, seeds, profile, level, diagnostics, progress=None):
    pieces = []
    for seed in seeds:
        for period in periods:
            train = frame.loc[(frame.period < period) & frame.eligible & frame.y.notna()]
            target = frame.loc[(frame.period == period) & frame.eligible]
            if train.empty or target.empty:
                raise ValueError(f"Partición vacía para {period}.")
            if train.period.max() >= period:
                raise AssertionError("Se intentó entrenar con el mes objetivo o el futuro.")
            model = make_model(family, params, seed, profile, level)
            start = time.perf_counter()
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                model.fit(train)
                prediction = model.predict(target)
                lower, upper = model.intervals(target) if hasattr(model, "intervals") else (np.nan, np.nan)
            if not np.isfinite(prediction).all() or (prediction < 0).any():
                raise ValueError(f"Salida inválida de {family} en {period}.")
            record = {"family": family, "seed": seed, "period": period,
                      "training_end": train.period.max(), "training_rows": len(train),
                      "params": params, "seconds": time.perf_counter() - start,
                      "warnings": sorted(set(str(item.message) for item in captured)),
                      **getattr(model, "diagnostics", {})}
            diagnostics.append(record)
            piece = target[["period", "upz", "y"]].copy()
            piece["family"], piece["seed"] = family, seed
            piece["prediction"], piece["lower"], piece["upper"] = prediction, lower, upper
            pieces.append(piece)
            if progress is not None:
                progress(f"Ajuste: {family} · mes {period} · semilla {seed}")
    return pd.concat(pieces, ignore_index=True)


def run_experiment(root: Path, source: Path, config: dict, profile_name: str = "demo", progress=None) -> Path:
    validate_training(config, profile_name)
    profile = config["profiles"][profile_name]
    if not profile["seeds"] or len(set(profile["seeds"])) != len(profile["seeds"]):
        raise ValueError("Las semillas deben ser una lista no vacía, sin duplicados.")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = root / "outputs" / run_id
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "status.json", {"state": "running", "profile": profile_name})
    try:
        with threadpool_limits(limits=1):
            _run(root, source, config, profile_name, profile, output, run_id, progress)
        save_json(output / "status.json", {"state": "complete", "profile": profile_name})
    except Exception as error:
        save_json(output / "status.json", {"state": "failed", "error": str(error)})
        raise
    except BaseException:
        save_json(output / "status.json", {"state": "interrupted"})
        raise
    return output


def _run(root, source, config, profile_name, profile, output, run_id, progress=None):
    panel, coverage, names, audit = load_panel(source, config)
    frame = make_features(panel, coverage, config, include_next=True)
    frame["split"] = frame.period.map(lambda period: split_label(period, config))
    frame.to_csv(output / "features.csv", index=False)
    panel.rename_axis("period").to_csv(output / "panel.csv")
    coverage.rename("coverage").rename_axis("period").to_csv(output / "coverage.csv")
    save_json(output / "data_audit.json", audit)
    save_json(output / "config.json", config)
    code_files = sorted((root / "src").rglob("*.py"))
    for path in code_files:
        if path.stat().st_size and not path.read_bytes():
            raise OSError(f"Archivo no disponible localmente: {path}. Descárguelo antes de ejecutar.")
    manifest = {"run_id": run_id, "profile": profile_name,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_file": source.name, "source_sha256": audit["source_sha256"],
                "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                "python": platform.python_version(),
                "versions": {name: importlib.metadata.version(name) for name in
                             ["numpy", "pandas", "scipy", "scikit-learn", "holidays", "plotly", "streamlit"]},
                "source_code_sha256": {str(p.relative_to(root)): sha256(p) for p in code_files},
                "protocol": f"Origen móvil mensual expansivo; validación hasta {config['validation_end']}; prueba hasta {config['test_end']}.",
                "forecast_horizon_months": 1,
                "seeds": profile["seeds"], "configs_per_family": profile["candidates"],
                "actual_configs_per_family": {family: len(model_grid(config, family, profile))
                    for family in config.get("models", list(TUNED))},
                "status_note": "Exploratorio: el periodo de prueba ya fue examinado en el trabajo previo."}
    save_json(output / "manifest.json", manifest)
    eligible = frame.loc[frame.eligible & frame.y.notna()]
    validation_periods = sorted(eligible.loc[eligible.split == "validacion", "period"].unique())
    test_periods = sorted(eligible.loc[eligible.split == "prueba", "period"].unique())
    if not validation_periods or not test_periods:
        raise ValueError("Se necesita validación y prueba no vacías.")
    diagnostics, search, val_all, selected = [], [], [], {}
    for family in config.get("models", list(TUNED)):
        print(f"Validación: {family}", flush=True)
        family_scores = []
        for index, params in enumerate(model_grid(config, family, profile)):
            prediction = evaluate(frame, validation_periods, family, params, profile["seeds"],
                                  profile, config["interval_level"], diagnostics, progress)
            prediction["candidate"] = index
            val_all.append(prediction)
            scores = [metrics(rows)["mae"] for _, rows in prediction.groupby("seed")]
            score = {"family": family, "candidate": index, "mae_mean_seeds": float(np.mean(scores)),
                     "mae_std_seeds": float(np.std(scores)), "params": json.dumps(params)}
            search.append(score)
            family_scores.append((score["mae_mean_seeds"], index, params))
        _, index, params = min(family_scores, key=lambda item: (item[0], item[1]))
        selected[family] = {"candidate": index, "params": params}
    for family in BASELINES:
        print(f"Validación: {family}", flush=True)
        prediction = evaluate(frame, validation_periods, family, {}, [profile["seeds"][0]],
                              profile, config["interval_level"], diagnostics, progress)
        prediction["candidate"] = 0
        val_all.append(prediction)
        selected[family] = {"candidate": 0, "params": {}}
    pd.DataFrame(search).to_csv(output / "search_validation.csv", index=False)
    all_val = pd.concat(val_all, ignore_index=True)
    all_val.to_csv(output / "validation_all_candidates.csv", index=False)
    selected_val = pd.concat([all_val.loc[(all_val.family == family) &
                             (all_val.candidate == spec["candidate"])] for family, spec in selected.items()])
    validation_averaged = average_seeds(selected_val)
    validation_summary = summarize(validation_averaged, config["interval_level"])
    winner = str(validation_summary.iloc[0].family)
    save_json(output / "selection.json", {"models": selected, "chosen_by_validation": winner,
                                         "criterion": "MAE de validación; nunca MAE de prueba"})
    validation_summary.to_csv(output / "metrics_validation.csv", index=False)
    validation_averaged.to_csv(output / "predictions_validation.csv", index=False)
    test_all, next_all = [], []
    next_period = str(panel.index.max() + 1)
    for family, spec in selected.items():
        print(f"Prueba mensual y pronóstico: {family}", flush=True)
        seeds = profile["seeds"] if family in TUNED else [profile["seeds"][0]]
        test_all.append(evaluate(frame, test_periods, family, spec["params"], seeds,
                                 profile, config["interval_level"], diagnostics, progress))
        next_all.append(evaluate(frame, [next_period], family, spec["params"], seeds,
                                 profile, config["interval_level"], diagnostics, progress))
    raw_test = pd.concat(test_all, ignore_index=True)
    raw_test.to_csv(output / "predictions_test_seeds.csv", index=False)
    test = average_seeds(raw_test)
    test.to_csv(output / "predictions_test.csv", index=False)
    summarize(test, config["interval_level"]).to_csv(output / "metrics_test.csv", index=False)
    pd.DataFrame([{"family": f, "upz": u, **metrics(rows, config["interval_level"])}
                  for (f, u), rows in test.groupby(["family", "upz"])]).to_csv(output / "metrics_by_upz.csv", index=False)
    pd.DataFrame([{"family": f, "period": p, **metrics(rows, config["interval_level"])}
                  for (f, p), rows in test.groupby(["family", "period"])]).to_csv(output / "metrics_by_month.csv", index=False)
    pd.DataFrame([{"family": f, "seed": s, **metrics(rows, config["interval_level"])}
                  for (f, s), rows in raw_test.groupby(["family", "seed"])]).to_csv(output / "metrics_by_seed.csv", index=False)
    forecast = average_seeds(pd.concat(next_all, ignore_index=True)).drop(columns="y")
    forecast["upz_name"] = forecast.upz.map(names)
    forecast.to_csv(output / "forecast.csv", index=False)
    save_json(output / "fit_diagnostics.json", diagnostics)
    from nusecast.report import build_report
    if progress is not None:
        progress("Preparando las gráficas y el informe descargable…")
    build_report(output)
    print(f"Terminado: {output}", flush=True)
