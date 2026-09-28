from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.special import softmax
from scipy.stats import nbinom
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from nusecast.data import NUMERIC

FAMILIES = {"RN": "Red neuronal MLP", "ME": "Evolución diferencial",
            "RI": "Gamma-Poisson", "AM": "Bosque aleatorio",
            "BASE_LAST": "Último mes", "BASE_MA3": "Promedio de 3 meses",
            "BASE_SEASONAL": "Mismo mes del año anterior", "GLM": "Regresión Poisson"}
TUNED = ("RN", "ME", "RI", "AM")
BASELINES = ("BASE_LAST", "BASE_MA3", "BASE_SEASONAL", "GLM")


def candidates(family: str, count: int) -> list[dict]:
    grids = {
        "RN": [{"hidden": [16], "alpha": 0.01}, {"hidden": [32, 16], "alpha": 0.1},
               {"hidden": [32], "alpha": 1.0}],
        "ME": [{"penalty": 0.01, "popsize": 6}, {"penalty": 0.1, "popsize": 8},
               {"penalty": 0.5, "popsize": 10}],
        "RI": [{"window": 3, "prior_strength": 0.5}, {"window": 6, "prior_strength": 1.0},
               {"window": 12, "prior_strength": 1.0}],
        "AM": [{"max_depth": 5, "min_samples_leaf": 5},
               {"max_depth": 10, "min_samples_leaf": 3},
               {"max_depth": None, "min_samples_leaf": 8}],
    }
    if not 1 <= count <= 3:
        raise ValueError("Hay tres configuraciones predefinidas por familia; candidates debe ser 1–3.")
    return grids[family][:count]


class Baseline:
    def __init__(self, column: str):
        self.column = column

    def fit(self, frame: pd.DataFrame):
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame[self.column].to_numpy(float)


class TabularModel:
    def __init__(self, estimator, target_scale: float = 1.0):
        self.target_scale = target_scale
        self.pipeline = Pipeline([
            ("prepare", ColumnTransformer([
                ("numeric", StandardScaler(), NUMERIC),
                ("territory", OneHotEncoder(handle_unknown="error", sparse_output=False), ["upz"])
            ])), ("model", estimator)
        ])

    def fit(self, frame: pd.DataFrame):
        self.pipeline.fit(frame, frame.y.to_numpy() / self.target_scale)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.maximum(1e-8, self.pipeline.predict(frame) * self.target_scale)


class EvolutionaryBlend:
    columns = ["lag1", "lag2", "lag3", "lag12", "ma3"]

    def __init__(self, params: dict, seed: int, generations: int):
        self.params, self.seed, self.generations = params, seed, generations

    @staticmethod
    def _prediction(weights, matrix, season):
        return np.maximum(1e-8, matrix @ softmax(weights[:5]) + season @ weights[5:])

    def fit(self, frame: pd.DataFrame):
        matrix = frame[self.columns].to_numpy(float) / 100
        season = np.column_stack([np.ones(len(frame)), frame.month_sin, frame.month_cos])
        y = frame.y.to_numpy(float) / 100

        def objective(weights):
            prediction = self._prediction(weights, matrix, season)
            return np.mean(np.abs(y - prediction)) + self.params["penalty"] * np.mean(weights[5:] ** 2)

        result = differential_evolution(
            objective, [(-4, 4)] * 5 + [(-1, 1)] * 3,
            rng=np.random.default_rng(self.seed), maxiter=self.generations,
            popsize=self.params["popsize"], polish=False, tol=0.001, workers=1)
        self.weights = result.x
        self.diagnostics = {"converged": bool(result.success), "generations": int(result.nit),
                            "evaluations": int(result.nfev), "message": str(result.message),
                            "lag_weights": dict(zip(self.columns, softmax(result.x[:5]).tolist()))}
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = frame[self.columns].to_numpy(float) / 100
        season = np.column_stack([np.ones(len(frame)), frame.month_sin, frame.month_cos])
        return self._prediction(self.weights, matrix, season) * 100


class GammaPoisson:
    def __init__(self, params: dict, level: float):
        self.window = params["window"]
        self.strength = params["prior_strength"]
        self.level = level

    def fit(self, frame: pd.DataFrame):
        self.history = frame[["period", "upz", "y"]].copy()
        return self

    def _parameters(self, frame: pd.DataFrame):
        means = []
        for row in frame.itertuples():
            cutoff = str(pd.Period(row.period, "M") - self.window)
            prior = self.history.loc[(self.history.period < cutoff) & (self.history.upz == row.upz), "y"]

            means.append(max(float(prior.mean()), 1e-3) if len(prior) else 1.0)
        raw = frame[[f"raw_lag{i}" for i in range(1, self.window + 1)]].to_numpy(float)
        valid = frame[[f"valid_lag{i}" for i in range(1, self.window + 1)]].to_numpy(bool)
        alpha = np.array(means) * self.strength + np.where(valid, raw, 0).sum(axis=1)
        beta = self.strength + valid.sum(axis=1)
        return alpha, beta

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        alpha, beta = self._parameters(frame)
        return alpha / beta

    def intervals(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        alpha, beta = self._parameters(frame)
        tail = (1 - self.level) / 2
        return nbinom.ppf(tail, alpha, beta / (beta + 1)), nbinom.ppf(1 - tail, alpha, beta / (beta + 1))


def make_model(family: str, params: dict, seed: int, profile: dict, level: float = 0.9):
    if family in BASELINES[:3]:
        return Baseline({"BASE_LAST": "lag1", "BASE_MA3": "ma3", "BASE_SEASONAL": "lag12"}[family])
    if family == "GLM":
        return TabularModel(PoissonRegressor(alpha=1.0, max_iter=500))
    if family == "RN":
        return TabularModel(MLPRegressor(
            hidden_layer_sizes=tuple(params["hidden"]), alpha=params["alpha"], loss="poisson",
            max_iter=profile["mlp_iterations"], random_state=seed, early_stopping=False,
            learning_rate_init=params.get("learning_rate", 0.003),
            activation=params.get("activation", "relu"),
            n_iter_no_change=40, tol=1e-5), target_scale=100)
    if family == "AM":
        return TabularModel(RandomForestRegressor(
            n_estimators=profile["trees"], criterion="poisson", random_state=seed,
            n_jobs=1, **params))
    if family == "ME":
        return EvolutionaryBlend(params, seed, profile["evolution_generations"])
    if family == "RI":
        return GammaPoisson(params, level)
    raise ValueError(f"Familia desconocida: {family}")
