import sys, json, time, warnings, itertools
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "../src")
warnings.filterwarnings("ignore")
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from nusecast.data import NUMERIC
from nusecast.models import make_model
from nusecast.metrics import metrics

OUT = Path("res"); OUT.mkdir(exist_ok=True)
fr = pd.read_csv("features.csv")
elig = fr[fr.eligible & fr.y.notna()]
VAL = sorted(elig[elig.split == "validacion"].period.unique())
TEST = sorted(elig[elig.split == "prueba"].period.unique())
LR = [0.0003, 0.003, 0.03]; NEU = [8, 32, 128]; ALPHA = [0.01, 1.0, 10.0]
ITER = 1500

class MLP:
    def __init__(self, lr, neu, alpha, seed, cols=NUMERIC):
        self.cols = list(cols)
        self.p = Pipeline([("prep", ColumnTransformer([("num", StandardScaler(), self.cols),
                   ("upz", OneHotEncoder(sparse_output=False), ["upz"])])),
                   ("mlp", MLPRegressor(hidden_layer_sizes=(neu,), alpha=alpha, loss="poisson",
                    learning_rate_init=lr, max_iter=ITER, random_state=seed, early_stopping=False,
                    n_iter_no_change=40, tol=1e-5))])
    def fit(self, d):
        self.p.fit(d, d.y.values / 100); return self
    def predict(self, d):
        return np.maximum(1e-8, self.p.predict(d) * 100)

def origen_movil(periods, factory):
    out, iters = [], []
    for per in periods:
        tr = elig[elig.period < per]; te = elig[elig.period == per]
        m = factory().fit(tr)
        out.append(te[["period", "upz", "y"]].assign(prediction=m.predict(te)))
        iters.append(m.p[-1].n_iter_)
    return pd.concat(out), iters

def exp1(chunk):
    f = OUT / "exp1.csv"
    done = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["lr", "neuronas", "alpha", "semilla"])
    hechos = set(zip(done.lr, done.neuronas, done.alpha, done.semilla))
    filas = done.to_dict("records"); t0 = time.time()
    for lr, n, a in itertools.product(LR, NEU, ALPHA):
        for s in [0, 1, 2]:
            if (lr, n, a, s) in hechos: continue
            if time.time() - t0 > chunk: pd.DataFrame(filas).to_csv(f, index=False); return False
            t = time.time()
            pred, it = origen_movil(VAL, lambda: MLP(lr, n, a, s))
            m = metrics(pred)
            filas.append({"lr": lr, "neuronas": n, "alpha": a, "semilla": s, "MAE": m["mae"],
                          "RMSE": m["rmse"], "dev_poisson": m["poisson_deviance"], "sesgo": m["bias"],
                          "iter_medias": np.mean(it), "pct_no_converge": np.mean(np.array(it) >= ITER),
                          "seg": time.time() - t})
    pd.DataFrame(filas).to_csv(f, index=False); return True

if __name__ == "__main__":
    print(exp1(float(sys.argv[1])))
