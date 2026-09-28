import base64
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from nusecast.models import FAMILIES
from nusecast.report import forecast_chart
from nusecast.assistant import AssistantError, ask, build_context, load_settings

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="NUSECast · Laboratorio", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>
.block-container {max-width:1440px;padding-top:2rem;}
h1,h2,h3 {color:#062F46;}
[data-testid="stMetric"] {background:#EAF0F5;border-top:4px solid #F4BB24;padding:16px;border-radius:6px;}
[data-baseweb="tab-list"] {gap:20px;border-bottom:1px solid #D5E0E8;}
button[data-baseweb="tab"][aria-selected="true"] {color:#062F46;font-weight:700;}
.nuse-brand {position:relative;margin-bottom:20px;}
.nuse-brand img {width:100%;height:auto;display:block;}
.nuse-brand-title {position:absolute;left:16%;top:27%;color:white;font-size:clamp(20px,3vw,44px);font-weight:700;}
.nuse-brand-title small {display:block;color:#F4BB24;font-size:clamp(10px,1.2vw,17px);font-weight:500;}
@media(max-width:650px){.nuse-brand-title small{display:none;}}
</style>""", unsafe_allow_html=True)
brand = ROOT / "assets" / "javeriana-banner.png"
if brand.is_file():
    encoded = base64.b64encode(brand.read_bytes()).decode("ascii")
    st.markdown(f'<div class="nuse-brand"><img src="data:image/png;base64,{encoded}" alt="Pontificia Universidad Javeriana, Bogotá">'
                '<div class="nuse-brand-title">NUSECast<small>Sistemas Inteligentes · Laboratorio de pronóstico</small></div></div>', unsafe_allow_html=True)
else:
    st.title("NUSECast")
st.caption("Proyecto académico · Registros de incidentes 123 · Kennedy, Bogotá")
settings = load_settings(ROOT)
page = st.radio("Ir a", ["Resultados", "Datos y entrenamiento", "Historial", "Configuración IA"], horizontal=True, key="main_page")
if page == "Datos y entrenamiento":
    from nusecast.workbench import render_training
    render_training(ROOT)
    st.stop()
if page == "Configuración IA":
    from nusecast.workbench import render_assistant_settings
    render_assistant_settings(ROOT)
    st.stop()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


runs = []
unavailable = 0
for status in (ROOT / "outputs").glob("*/status.json"):
    if getattr(status.stat(), "st_flags", 0) & 0x40000000:
        unavailable += 1
        continue
    try:
        if read_json(status).get("state") == "complete":
            runs.append(status.parent)
    except (OSError, ValueError):
        unavailable += 1
runs.sort(reverse=True)
if page == "Historial":
    from nusecast.workbench import render_history
    render_history(runs)
    st.stop()
if unavailable:
    st.sidebar.caption(f"{unavailable} ejecución(es) no disponibles localmente. Descárgalas desde Finder para consultarlas.")
if not runs:
    st.info("Todavía no hay ejecuciones terminadas. Abre Datos y entrenamiento para comenzar.")
    st.stop()

run = st.sidebar.selectbox("Ejecución reproducible", runs, format_func=lambda p: p.name)
audit = read_json(run / "data_audit.json")
config = read_json(run / "config.json")
selection = read_json(run / "selection.json")
manifest = read_json(run / "manifest.json")
scores = pd.read_csv(run / "metrics_test.csv")
validation = pd.read_csv(run / "metrics_validation.csv")
prediction = pd.read_csv(run / "predictions_test.csv")
forecast = pd.read_csv(run / "forecast.csv")
diagnostics = read_json(run / "fit_diagnostics.json")
winner = selection["chosen_by_validation"]

st.sidebar.markdown("**Modo académico, no operativo**")
st.sidebar.write(f"Perfil: {manifest['profile']}")
st.sidebar.write(f"Datos hasta: {audit['end']}")
st.sidebar.write("Pronósticos locales · Asistente OpenAI opcional")
st.sidebar.caption("IA configurada" if settings.ready else "IA desactivada o sin clave")
st.warning("Resultados exploratorios. No autorizan despachos ni decisiones automáticas de emergencia.")
if pd.Period(audit["end"], "M") < pd.Timestamp.now().to_period("M") - 1:
    st.info(f"Datos desactualizados para un pronóstico actual: la fuente termina en {audit['end']}. "
            f"La salida a un mes corresponde a {forecast.period.iloc[0]}.")

columns = st.columns(4)
columns[0].metric("UPZ", audit["upz_count"])
columns[1].metric("Meses históricos", audit["months"])
columns[2].metric("Cobertura UPZ", f"{audit['weighted_geographic_coverage']:.1%}")
columns[3].metric("Modelo · validación", winner)
st.caption(f"Referencia elegida por MAE de validación: {FAMILIES[winner]}.")

comparison, territory, uncertainty, future, protocol, assistant = st.tabs([
    "Comparar modelos", "Ver una UPZ", "Incertidumbre", "Siguiente mes", "Método y trazabilidad", "Asistente IA"])

with comparison:
    st.subheader("¿El modelo mejora las referencias sencillas?")
    st.write("El error se calcula sobre los mismos meses y UPZ. Menor MAE significa menor error medio.")
    display = scores.assign(Modelo=scores.family.map(FAMILIES))
    fig = px.bar(display, x="mae", y="Modelo", orientation="h", text_auto=".1f",
                 color_discrete_sequence=["#062F46"], labels={"mae": "MAE · incidentes por UPZ y mes"})
    fig.update_layout(yaxis=dict(autorange="reversed"), template="plotly_white", height=450)
    st.plotly_chart(fig, width="stretch")
    st.caption("Esta gráfica describe la prueba; no se usa para cambiar el modelo seleccionado en validación.")
    st.dataframe(display[["Modelo", "mae", "rmse", "bias", "poisson_deviance"]].round(3), hide_index=True)
    st.download_button("Descargar comparación CSV", (run / "metrics_test.csv").read_bytes(), "metricas_prueba.csv", "text/csv")

with territory:
    left, right = st.columns(2)
    upz = left.selectbox("Territorio", list(audit["names"]),
                        format_func=lambda u: f"{u} · {audit['names'][u]}")
    families = [family for family in FAMILIES if family in set(scores.family)]
    family = right.selectbox("Modelo", families, index=families.index(winner), format_func=FAMILIES.get)
    chart = forecast_chart(prediction, upz, family, config["interval_level"])
    st.plotly_chart(chart, width="stretch", key="territory-chart")
    st.caption("Cada punto es una predicción a un mes con datos anteriores; no un pronóstico simultáneo de toda la curva.")
    st.dataframe(prediction.loc[(prediction.upz == upz) & (prediction.family == family)].round(2), hide_index=True)

with uncertainty:
    st.subheader("¿Los intervalos contienen lo que realmente ocurrió?")
    if "RI" not in set(scores.family):
        st.info("Esta ejecución no incluye Gamma-Poisson. Selecciona ese modelo al entrenar para evaluar intervalos.")
    else:
        ri = scores.loc[scores.family == "RI"].iloc[0]
        a, b, c = st.columns(3)
        a.metric("Cobertura nominal", f"{config['interval_level']:.0%}")
        b.metric("Cobertura observada", f"{ri.coverage:.1%}")
        c.metric("Anchura media", f"{ri.mean_width:.1f}")
        if ri.coverage < config["interval_level"]:
            st.warning("Los intervalos cubrieron menos casos de los esperados. No deben presentarse como una garantía de seguridad.")
        st.write("Gamma-Poisson representa una tasa de incidentes incierta. Sus intervalos incluyen variación del conteo "
                 "y de la tasa, pero el modelo supone estabilidad en su ventana histórica. Cambios de régimen pueden romper ese supuesto.")
        interval_chart = forecast_chart(prediction, upz, "RI", config["interval_level"])
        st.plotly_chart(interval_chart, width="stretch", key="uncertainty-chart")
        st.caption("Estos intervalos pertenecen a Gamma-Poisson: no se transfieren a MLP, al bosque o a la suma de UPZ.")

with future:
    month = str(forecast.period.iloc[0])
    st.subheader(f"Pronóstico para {month}")
    st.write(f"Modelo elegido por validación: **{FAMILIES[winner]}**. Son valores esperados, no conteos confirmados.")
    chosen = forecast.loc[forecast.family == winner].sort_values("prediction", ascending=False)
    st.plotly_chart(px.bar(chosen, x="upz_name", y="prediction", text_auto=".1f",
                          color_discrete_sequence=["#F4BB24"],
                          labels={"upz_name": "UPZ", "prediction": "Registros esperados"}), width="stretch")
    st.dataframe(chosen[["period", "upz_name", "prediction"]].round(1), hide_index=True)
    st.caption("Alcance: UPZ con ubicación conocida, no el total de Kennedy ni toda Bogotá.")
    st.download_button("Descargar todos los pronósticos", (run / "forecast.csv").read_bytes(), "pronosticos.csv", "text/csv")

with protocol:
    st.subheader("Qué se hizo y qué todavía no podemos concluir")
    st.markdown(f"- Entrenamiento inicial: {config['first_target']} a {config['train_end']}. "
                f"Validación: {pd.Period(config['train_end'], 'M') + 1} a {config['validation_end']}. "
                f"Prueba: {pd.Period(config['validation_end'], 'M') + 1} a {config['test_end']}.\n"
                "- Reentrenamiento mensual, incorporando únicamente observaciones anteriores.\n"
                "- Escalado y selección de parámetros sin usar el mes objetivo.\n"
                "- Las anomalías son hipótesis de calidad del dato; no se afirma una causa comprobada.\n"
                "- El test ya fue explorado en el proyecto previo: se necesita validación futura independiente.\n"
                "- No se ha demostrado reducción de accidentes, tiempos de respuesta ni necesidad de ambulancias.")
    st.write("Tratamiento de meses sospechosos:", "Exclusión de objetivos y corrección causal de rezagos"
             if config["exclude_suspect"] else "Conservados sin corregir (sensibilidad)")
    warned = sum(bool(d["warnings"]) for d in diagnostics)
    capped = sum(d.get("converged") is False for d in diagnostics)
    st.write(f"Ajustes con advertencias: {warned}. Búsquedas evolutivas sin criterio de convergencia alcanzado: {capped}.")
    with st.expander("Parámetros seleccionados y resultados de validación"):
        st.json(selection)
        st.dataframe(validation.round(3), hide_index=True)
    with st.expander("Auditoría del dato y versiones"):
        st.json(audit)
        st.json(manifest)
    with st.expander("Advertencias de los ajustes"):
        st.json([d for d in diagnostics if d["warnings"] or d.get("converged") is False])
    st.download_button("Informe HTML sin conexión", (run / "informe.html").read_bytes(), "NUSECast-informe.html", "text/html")

with assistant:
    st.subheader("Entiende los resultados de NUSECast")
    st.write("Pregunta por el error de la red neuronal, la comparación de modelos o la incertidumbre. "
             "El asistente explica esta ejecución; no modifica los pronósticos.")
    context = build_context(audit, scores, selection, forecast, config["interval_level"], FAMILIES)
    with st.expander("Ver las métricas agregadas que se enviarán"):
        st.json(context)
    st.caption(f"Modelo configurado: {settings.model} · temperatura: {settings.temperature:g} · límite: {settings.max_tokens} tokens. Cada consulta puede generar un cargo en tu cuenta API. "
               "Se envían tu pregunta y este resumen, no el CSV completo. No incluyas datos personales ni claves.")
    if not settings.ready:
        st.info("Activa el asistente en Configuración IA y comprueba que OPENAI_API_KEY esté configurada en .env.")
    with st.form("ask_assistant"):
        question = st.text_area("Tu pregunta", placeholder="¿Cuánto dio el MAE de nuestra red neuronal y qué significa?", max_chars=2000)
        consent = st.checkbox("Enviar mi pregunta y las métricas mostradas a OpenAI")
        submitted = st.form_submit_button("Consultar asistente", disabled=not settings.ready)
    if submitted:
        if not consent:
            st.warning("Confirma el envío antes de consultar.")
        elif st.session_state.get("api_attempts", 0) >= 10:
            st.warning("Se alcanzaron 10 intentos en esta sesión de prueba. No es un límite de gasto de la cuenta.")
        else:
            st.session_state["api_attempts"] = st.session_state.get("api_attempts", 0) + 1
            try:
                with st.spinner("Consultando las métricas…"):
                    answer = ask(load_settings(ROOT), question, context)
                st.session_state["last_answer"] = (run.name, question, answer)
            except AssistantError as error:
                st.error(str(error))
    last = st.session_state.get("last_answer")
    if last and last[0] == run.name:
        st.markdown("**Última consulta:**")
        st.text(last[1])
        st.markdown(last[2])
        st.caption("Respuesta generada por IA. Contrasta las cifras con las tablas del panel.")
