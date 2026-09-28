import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance


def metrics(frame: pd.DataFrame, level: float = 0.9) -> dict:
    y, pred = frame.y.to_numpy(float), frame.prediction.to_numpy(float)
    if not np.isfinite(y).all() or not np.isfinite(pred).all() or (pred < 0).any():
        raise ValueError("Predicciones no válidas para evaluación.")
    residual = pred - y
    result = {"n": len(y), "mae": float(mean_absolute_error(y, pred)),
              "rmse": float(np.sqrt(np.mean(residual ** 2))),
              "poisson_deviance": float(mean_poisson_deviance(y, np.maximum(pred, 1e-8))),
              "bias": float(residual.mean()), "mean_underprediction": float(np.maximum(-residual, 0).mean()),
              "mean_overprediction": float(np.maximum(residual, 0).mean())}
    if "lower" in frame and frame[["lower", "upper"]].notna().all().all():
        lower, upper = frame.lower.to_numpy(), frame.upper.to_numpy()
        if (lower > upper).any():
            raise ValueError("Intervalos invertidos.")
        result.update(coverage=float(np.mean((y >= lower) & (y <= upper))),
                      mean_width=float(np.mean(upper - lower)), nominal_coverage=level,
                      interval_score=float(np.mean(upper - lower +
                        2 / (1 - level) * np.maximum(lower - y, 0) +
                        2 / (1 - level) * np.maximum(y - upper, 0))))
    return result


def average_seeds(predictions: pd.DataFrame) -> pd.DataFrame:

    keys = ["family", "period", "upz"]
    values = ["y", "prediction", "lower", "upper"]
    return predictions.groupby(keys, as_index=False, dropna=False)[values].mean()


def summarize(predictions: pd.DataFrame, level: float) -> pd.DataFrame:
    return pd.DataFrame([{"family": family, **metrics(rows, level)}
                         for family, rows in predictions.groupby("family")]).sort_values("mae")
