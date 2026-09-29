import sys, time
from exp import *
from nusecast.metrics import metrics
SEL = dict(lr=0.003, neu=128, alpha=1.0)
SIN = [c for c in NUMERIC if c != "other_upz_lag1"]

def run(chunk):
    f = OUT / "exp23_pred.csv"
    prev = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["variante", "semilla", "fase"])
    hechos = set(zip(prev.variante, prev.semilla, prev.fase)); partes = [prev]; t0 = time.time()
    for var, cols in [("con_entorno", NUMERIC), ("sin_entorno", SIN)]:
        for s in range(5):
            for fase, pers in [("validacion", VAL), ("prueba", TEST)]:
                if (var, s, fase) in hechos: continue
                if time.time() - t0 > chunk:
                    pd.concat(partes).to_csv(f, index=False); return False
                p, it = origen_movil(pers, lambda: MLP(SEL["lr"], SEL["neu"], SEL["alpha"], s, cols))
                partes.append(p.assign(variante=var, semilla=s, fase=fase))
    pd.concat(partes).to_csv(f, index=False); return True

def referencias():
    prof = {"mlp_iterations": ITER}
    filas = []
    for fam in ["BASE_LAST", "BASE_MA3", "BASE_SEASONAL", "GLM"]:
        for fase, pers in [("validacion", VAL), ("prueba", TEST)]:
            out = []
            for per in pers:
                tr = elig[elig.period < per]; te = elig[elig.period == per]
                m = make_model(fam, {}, 0, prof).fit(tr)
                out.append(te[["period", "upz", "y"]].assign(prediction=m.predict(te)))
            filas.append(pd.concat(out).assign(variante=fam, semilla=0, fase=fase))
    pd.concat(filas).to_csv(OUT / "referencias_pred.csv", index=False)

if __name__ == "__main__":
    if sys.argv[1] == "ref": referencias(); print("ok")
    else: print(run(float(sys.argv[1])))
