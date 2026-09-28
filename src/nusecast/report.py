import html
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from nusecast.models import FAMILIES

COLORS = ["#0A4D68", "#EDAA25", "#178A75", "#BD4A54", "#8665A6", "#61758A", "#79A7C7", "#333333"]


def forecast_chart(predictions, upz, family, level=0.9):
    data = predictions.loc[(predictions.upz == upz) & (predictions.family == family)].sort_values("period")
    fig = go.Figure()
    if data.lower.notna().all():
        fig.add_trace(go.Scatter(x=data.period, y=data.upper, mode="lines", line_width=0,
                                 showlegend=False, name="Límite superior"))
        fig.add_trace(go.Scatter(x=data.period, y=data.lower, mode="lines", line_width=0,
                                 fill="tonexty", fillcolor="rgba(23,138,117,0.17)", name=f"Intervalo predictivo {level:.0%}"))
    fig.add_trace(go.Scatter(x=data.period, y=data.y, mode="lines+markers", name="Observado",
                             line=dict(color=COLORS[0], width=3)))
    fig.add_trace(go.Scatter(x=data.period, y=data.prediction, mode="lines+markers", name="Pronóstico a un mes",
                             line=dict(color=COLORS[1], width=3)))
    fig.update_layout(template="plotly_white", title=f"{upz} · {FAMILIES[family]}",
                      xaxis_title="Mes objetivo", yaxis_title="Registros de incidentes", height=430,
                      legend=dict(orientation="h", y=-0.22), margin=dict(t=60, b=90))
    return fig


def build_report(output: Path):
    audit = json.loads((output / "data_audit.json").read_text())
    selection = json.loads((output / "selection.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    config = json.loads((output / "config.json").read_text())
    level = config["interval_level"]
    scores = pd.read_csv(output / "metrics_test.csv")
    pred = pd.read_csv(output / "predictions_test.csv")
    forecast = pd.read_csv(output / "forecast.csv")
    winner = selection["chosen_by_validation"]
    scores["Modelo"] = scores.family.map(FAMILIES)
    bar = px.bar(scores, x="mae", y="Modelo", orientation="h", text="mae",
                 title="Error en prueba · menor es mejor", color_discrete_sequence=[COLORS[0]])
    bar.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    bar.update_layout(template="plotly_white", yaxis=dict(autorange="reversed"), height=430,
                      xaxis_title="MAE · incidentes por UPZ y mes", yaxis_title=None)
    temporal = pred.loc[pred.family == winner].groupby("period", as_index=False)[["y", "prediction"]].sum()
    line = px.line(temporal, x="period", y=["y", "prediction"], markers=True,
                   title=f"Suma de las UPZ con ubicación · {FAMILIES[winner]}", color_discrete_sequence=COLORS)
    line.update_layout(template="plotly_white", yaxis_title="Registros", xaxis_title="Mes", height=430)
    ri_rows = scores.loc[scores.family == "RI"]
    uncertainty = "No se incluyó Gamma-Poisson en esta ejecución; los demás modelos no producen intervalos predictivos."
    if not ri_rows.empty:
        ri = ri_rows.iloc[0]
        uncertainty = (f"Gamma-Poisson: cobertura nominal {level:.0%}; cobertura observada en prueba "
                       f"<b>{ri['coverage']:.1%}</b>; anchura media <b>{ri['mean_width']:.1f}</b>. "
                       "Si la cobertura queda por debajo del objetivo, los intervalos son demasiado optimistas. "
                       "Pertenecen solo a Gamma-Poisson, no a otros modelos ni a la suma de UPZ.")
    upz_figs = [forecast_chart(pred, upz, winner, level) for upz in audit["names"]]
    figures = [bar, line, *upz_figs]
    plots = "\n".join(fig.to_html(full_html=False, include_plotlyjs=True if i == 0 else False)
                       for i, fig in enumerate(figures))
    selected_forecast = forecast.loc[forecast.family == winner, ["period", "upz_name", "prediction"]].copy()
    selected_forecast.columns = ["Mes", "UPZ", "Pronóstico"]
    summary = scores[["Modelo", "mae", "rmse", "bias", "poisson_deviance"]].to_html(index=False, float_format=lambda v: f"{v:.2f}", border=0)
    content = f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>NUSECast · Informe experimental</title>
<style>body{{font:16px system-ui;margin:0;background:#F3F6F8;color:#173248}}main{{max-width:1140px;margin:auto;padding:32px}}
header{{background:#062F46;color:white;padding:32px;border-radius:16px}}h1{{font-size:38px;margin:0}}h2{{margin-top:32px}}
.note{{padding:20px;background:#FFF1CE;border-left:5px solid #EAAA28;margin:20px 0;line-height:1.6}}
.cards{{display:flex;gap:15px;flex-wrap:wrap;margin:22px 0}}.card{{background:white;padding:22px;border-radius:12px;flex:1;min-width:180px}}
.card strong{{display:block;font-size:28px}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #DFE5EB}}
.plotly-graph-div{{border-radius:12px;margin:20px 0}}footer{{font-size:13px;color:#61758A;margin-top:32px}}</style></head><body><main>
<header><p>SISTEMAS INTELIGENTES · LABORATORIO ACADÉMICO</p><h1>NUSECast</h1><p>Pronóstico mensual · {html.escape(config['locality'])} · código {html.escape(config['incident_code'])}</p></header>
<div class="note"><b>Resultado experimental · perfil {html.escape(manifest['profile'])}.</b> Se pronostican registros agregados, no accidentes individuales.
Datos hasta {audit['end']}. La prueba ya se había explorado en el proyecto anterior: esto no es validación prospectiva independiente.
El modelo de referencia se escoge en validación, no buscando el menor error de prueba.</div>
<div class="cards"><div class="card"><strong>{audit['upz_count']}</strong>UPZ incluidas</div>
<div class="card"><strong>{audit['months']}</strong>meses observados</div>
<div class="card"><strong>{audit['weighted_geographic_coverage']:.1%}</strong>conteos con UPZ asignada</div></div>
<h2>1. Qué aprendemos de la comparación</h2><p>Seleccionado por validación: <b>{html.escape(FAMILIES[winner])}</b>.
Es correcto que un modelo sencillo supere a una red: no se fuerza un ganador.</p>{summary}
<h2>2. Observado frente a pronosticado</h2><p>Cada punto se calcula usando únicamente meses anteriores; al avanzar se incorporan observaciones nuevas.
No es un pronóstico simultáneo a 19 meses.</p>{plots}
<h2>3. Incertidumbre: comprobar, no prometer</h2><p>{uncertainty}</p>
<h2>4. Siguiente mes disponible: {html.escape(str(forecast.period.iloc[0]))}</h2>
<p>No es necesariamente el mes siguiente al día de hoy. Son valores esperados, por eso pueden tener decimales.</p>
{selected_forecast.to_html(index=False, float_format=lambda v: f'{v:.1f}', border=0)}
<h2>5. Límites y revisión humana</h2><p>Faltan actualización de datos, sensibilidad a los meses sospechosos,
presupuesto académico acordado y una evaluación futura independiente. No se despachan recursos automáticamente.
Las gráficas, predicciones, parámetros y advertencias están disponibles en los CSV y JSON de esta ejecución.</p>
<footer>NUSECast · {html.escape(manifest['run_id'])} · Informe generado localmente, sin ChatGPT ni API externa.</footer>
</main></body></html>"""
    (output / "informe.html").write_text(content, encoding="utf-8")
