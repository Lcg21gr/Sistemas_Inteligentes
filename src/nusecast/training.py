import copy
import fcntl
import time
from contextlib import contextmanager

from nusecast.data import load_panel, make_features, split_label, validate_config
from nusecast.models import BASELINES, TUNED, candidates


def model_grid(config, family, profile):
    return config.get("model_grid", {}).get(family, candidates(family, profile["candidates"]))


def bounded(value, low, high, name, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
        raise ValueError(f"{name} debe estar entre {low} y {high}.")
    if integer and not isinstance(value, int):
        raise ValueError(f"{name} debe ser entero.")


def validate_training(config, profile_name):
    validate_config(config)
    selected = config.get("models", list(TUNED))
    if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected) or set(selected) - set(TUNED):
        raise ValueError("Selecciona al menos un modelo avanzado válido, sin duplicados.")
    profile = config["profiles"][profile_name]
    seeds = profile["seeds"]
    if not 1 <= len(seeds) <= 5 or len(set(seeds)) != len(seeds):
        raise ValueError("Usa de una a cinco semillas distintas.")
    for seed in seeds:
        bounded(seed, 0, 2**32 - 1, "Semilla", True)
    for field, low, high in [("candidates", 1, 3), ("mlp_iterations", 1, 2000),
                             ("evolution_generations", 1, 200), ("trees", 1, 300)]:
        bounded(profile[field], low, high, field, True)
    for family in selected:
        grid = model_grid(config, family, profile)
        if not isinstance(grid, list) or not 1 <= len(grid) <= 3:
            raise ValueError("Cada modelo admite de una a tres configuraciones.")
        for params in grid:
            if family == "RN":
                if not 1 <= len(params["hidden"]) <= 3:
                    raise ValueError("La MLP admite de una a tres capas ocultas.")
                for units in params["hidden"]:
                    bounded(units, 1, 128, "Neuronas por capa", True)
                bounded(params["alpha"], 0, 10, "Regularización")
                bounded(params.get("learning_rate", 0.003), 0.00001, 0.1, "Tasa de aprendizaje")
                if params.get("activation", "relu") not in {"relu", "tanh", "logistic"}:
                    raise ValueError("Activación no admitida.")
            elif family == "AM":
                if params["max_depth"] is not None:
                    bounded(params["max_depth"], 1, 30, "Profundidad", True)
                bounded(params["min_samples_leaf"], 1, 50, "Muestras por hoja", True)
            elif family == "ME":
                bounded(params["penalty"], 0, 10, "Penalización evolutiva")
                bounded(params["popsize"], 2, 20, "Multiplicador de población", True)
            elif family == "RI":
                bounded(params["window"], 1, 12, "Ventana histórica", True)
                bounded(params["prior_strength"], 0.01, 20, "Fuerza del prior")
    return config


def validate_source(source, config, profile_name="custom"):
    validate_training(config, profile_name)
    panel, coverage, names, audit = load_panel(source, config)
    if len(panel) > 180 or len(panel.columns) > 150:
        raise ValueError("Este laboratorio admite hasta 180 meses y 150 UPZ por experimento.")
    frame = make_features(panel, coverage, config)
    eligible = frame.loc[frame.eligible].copy()
    eligible["split"] = eligible.period.map(lambda period: split_label(period, config))
    counts = eligible.groupby("split").period.nunique().to_dict()
    for name in ("entrenamiento", "validacion", "prueba"):
        if not counts.get(name):
            raise ValueError(f"La partición {name} no tiene meses utilizables.")
    profile = config["profiles"][profile_name]
    selected = config.get("models", list(TUNED))
    fits = (counts["validacion"] * (len(BASELINES) + len(profile["seeds"]) *
            sum(len(model_grid(config, family, profile)) for family in selected)) +
            (counts["prueba"] + 1) * (len(BASELINES) + len(profile["seeds"]) * len(selected)))
    if fits > 2500:
        raise ValueError(f"La configuración requiere {fits} ajustes. Reduce modelos, semillas o meses (máximo 2500).")
    return {"audit": audit, "partitions": counts, "fits": fits}


@contextmanager
def training_lock(root):
    folder = root / "outputs"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".training.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Ya hay un entrenamiento activo. Espera a que termine antes de iniciar otro.") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def train_interactive(root, source, config, progress):
    from nusecast.experiment import run_experiment
    with training_lock(root):
        snapshot = copy.deepcopy(config)
        progress(0, "Validando los datos antes de entrenar…")
        validated = validate_source(source, snapshot)
        started, completed = time.monotonic(), 0

        def advance(message):
            nonlocal completed
            if time.monotonic() - started > 15 * 60:
                raise ValueError("Se alcanzó el presupuesto de 15 minutos. Reduce el entrenamiento. La interrupción se comprueba entre ajustes.")
            if message.startswith("Ajuste:"):
                completed += 1
            progress(min(0.99, completed / validated["fits"]), message)

        result = run_experiment(root, source, snapshot, "custom", progress=advance)
        progress(1.0, "Experimento guardado correctamente")
        return result
