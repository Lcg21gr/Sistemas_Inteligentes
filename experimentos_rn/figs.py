import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
NAVY, GOLD, BLUE, RED, GREY = "#002E4D", "#F1B728", "#376092", "#C0392B", "#8A8F98"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#B0B4BA", "axes.labelcolor": "#333",
                     "xtick.color": "#333", "ytick.color": "#333"})
F = "figuras/"

fr = pd.read_csv("features.csv")
names = {"UPZ113":"Bavaria","UPZ44":"Américas","UPZ45":"Carvajal","UPZ46":"Castilla","UPZ47":"Kennedy Central",
 "UPZ48":"Timiza","UPZ78":"Tintal Norte","UPZ79":"Calandaima","UPZ80":"Corabastos","UPZ81":"Gran Britalia",
 "UPZ82":"Patio Bonito","UPZ83":"Las Margaritas"}
import sys; sys.path.insert(0, "../src")
from pathlib import Path
from nusecast.data import load_config, load_panel
cfg = load_config(Path("../config.json"))
panel, cov, _, _ = load_panel(Path("../data/raw/Numero_incidentes.csv"), cfg)
ts = panel.index.to_timestamp()
fig, ax = plt.subplots(figsize=(11, 4.1))
orden = panel.sum().sort_values(ascending=False).index
for i, u in enumerate(orden):
    hl = u in ("UPZ46", "UPZ78")
    ax.plot(ts, panel[u], lw=2.2 if hl else 0.9, color=(NAVY if u == "UPZ46" else GOLD) if hl else "#B7C4D1", zorder=3 if hl else 2)
ax.text(ts[-1], panel["UPZ46"].iloc[-1] + 12, " Castilla", color=NAVY, fontsize=10, weight="bold", ha="right", bbox=dict(fc="white", ec="none", pad=1, alpha=.85))
ax.text(ts[-1], panel["UPZ78"].iloc[-1] + 20, " Tintal Norte", color="#9A7300", fontsize=10, weight="bold", ha="right", bbox=dict(fc="white", ec="none", pad=1, alpha=.85))
ax.axvspan(pd.Timestamp("2021-01-01"), pd.Timestamp("2023-12-31"), color=NAVY, alpha=.05)
ax.axvspan(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"), color=GOLD, alpha=.15)
ax.axvspan(pd.Timestamp("2025-01-01"), pd.Timestamp("2026-07-31"), color=BLUE, alpha=.12)
top = ax.get_ylim()[1]
for x, t in [("2022-06-15", "Entrenamiento 2021–2023"), ("2024-07-01", "Validación"), ("2025-10-15", "Prueba")]:
    ax.text(pd.Timestamp(x), top * .97, t, ha="center", va="top", fontsize=9.5, color="#333")
ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-08-31"), color="#999", alpha=.18)
ax.text(pd.Timestamp("2020-05-15"), top * .97, "Confina-\nmiento", ha="center", va="top", fontsize=8.5, color="#555")
for p in ["2024-01-15", "2024-04-15"]:
    ax.axvline(pd.Timestamp(p), color=RED, ls="--", lw=1)
ax.text(pd.Timestamp("2024-02-25"), 545, "meses\nanómalos", color=RED, fontsize=8.5, ha="center", bbox=dict(fc="white", ec="none", pad=1))
ax.set_ylabel("Registros 942 por mes"); ax.set_ylim(0, top)
plt.tight_layout(); plt.savefig(F + "panel.png", dpi=200); plt.close()

d = pd.read_csv("res/exp1.csv")
cmap = LinearSegmentedColormap.from_list("m", ["#1E7B4F", "#F7F3D6", "#C0392B"])
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.7))
for ax, a in zip(axs, [0.01, 1.0, 10.0]):
    m = d[d.alpha == a].pivot_table(index="lr", columns="neuronas", values="MAE")
    s = d[d.alpha == a].pivot_table(index="lr", columns="neuronas", values="MAE", aggfunc="std")
    im = ax.imshow(m.values, cmap=cmap, vmin=23, vmax=45, aspect="auto")
    for i in range(3):
        for j in range(3):
            v = m.values[i, j]; best = (a == 1.0 and i == 1 and j == 2)
            ax.text(j, i - .08, f"{v:.1f}".replace(".", ","), ha="center", va="center", fontsize=13,
                    weight="bold", color="white" if v > 40 else "#111")
            ax.text(j, i + .25, f"± {s.values[i, j]:.1f}".replace(".", ","), ha="center", va="center", fontsize=8.5,
                    color="white" if v > 40 else "#444")
            if best:
                ax.add_patch(plt.Rectangle((j - .48, i - .48), .96, .96, fill=False, lw=3, ec=NAVY))
    ax.set_xticks(range(3), ["8", "32", "128"]); ax.set_yticks(range(3), ["0,0003", "0,003", "0,03"] if a == 0.01 else ["", "", ""])
    if a != 0.01: ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Neuronas en la capa oculta")
    ax.set_title(f"α (L2) = {str(a).replace('.', ',')}", weight="bold", color=NAVY)
    for sp in ax.spines.values(): sp.set_visible(False)
axs[0].set_ylabel("Tasa de aprendizaje")
cb = fig.colorbar(im, ax=axs, fraction=.02, pad=.02); cb.set_label("MAE validación (media de 3 semillas)")
plt.savefig(F + "heatmaps.png", dpi=200, bbox_inches="tight"); plt.close()

p = pd.read_csv("res/exp23_pred.csv"); r = pd.read_csv("res/referencias_pred.csv")
pt = p[(p.fase == "prueba") & (p.variante == "con_entorno")]
band = pt.groupby(["period", "upz"]).prediction.agg(["mean", "min", "max"]).reset_index()
ma3 = r[(r.fase == "prueba") & (r.variante == "BASE_MA3")]
fig, axs = plt.subplots(1, 3, figsize=(12, 3.5), sharey=False)
for ax, u in zip(axs, ["UPZ46", "UPZ47", "UPZ78"]):
    h = panel[u]; h = h[h.index >= pd.Period("2023-07", "M")]
    ax.plot(h.index.to_timestamp(), h.values, color="#222", lw=1.6, marker="o", ms=3, label="Real")
    b = band[band.upz == u]; t = pd.PeriodIndex(b.period, freq="M").to_timestamp()
    ax.fill_between(t, b["min"], b["max"], color=BLUE, alpha=.25, lw=0, label="Rango 5 semillas")
    ax.plot(t, b["mean"], color=BLUE, lw=2.2, label="MLP (promedio semillas)")
    mm = ma3[ma3.upz == u]
    ax.plot(pd.PeriodIndex(mm.period, freq="M").to_timestamp(), mm.prediction, color=GOLD, lw=1.6, ls="--", label="Media móvil 3 meses")
    ax.axvline(pd.Timestamp("2025-01-01"), color="#aaa", lw=1)
    ax.set_title(names[u], weight="bold", color=NAVY)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
    ax.tick_params(axis="x", labelsize=9)
axs[0].set_ylabel("Registros por mes")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=10)
plt.tight_layout(rect=(0, .08, 1, 1)); plt.savefig(F + "pronosticos.png", dpi=200); plt.close()
print("ok")
