import copy
import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from nusecast.assistant import load_settings, save_settings
from nusecast.data import REQUIRED, load_config, sha256
from nusecast.models import FAMILIES, TUNED
from nusecast.training import factorial_grid, train_interactive, validate_source
from nusecast.uploads import ingest, sheet_names


def config_identity(source, config):
    return hashlib.sha256((str(source) + json.dumps(config, sort_keys=True)).encode()).hexdigest()


def render_assistant_settings(root):
    settings = load_settings(root)
    st.header("Configuración del asistente")
    st.info("Estos ajustes solo cambian las explicaciones de la IA. No entrenan modelos ni modifican el MAE.")
    st.caption("La clave se conserva en .env y no se muestra aquí. Guardar no realiza ninguna consulta de pago.")
    models = ["gpt-4.1-mini", "gpt-4.1"]
    with st.form("assistant_configuration"):
        enabled = st.checkbox("Activar el asistente", value=settings.enabled)
        model = st.selectbox("Modelo de OpenAI", models, index=models.index(settings.model) if settings.model in models else 0)
        temperature = st.slider("Temperatura", 0.0, 2.0, settings.temperature, 0.1,
                                help="Menor: respuestas menos variables. Mayor: más variedad. No garantiza exactitud.")
        tokens = st.number_input("Máximo de tokens de respuesta", 32, 1000, settings.max_tokens, step=50)
        st.caption("Para explicar métricas, puedes empezar con 0,2. Un token no equivale necesariamente a una palabra.")
        submitted = st.form_submit_button("Guardar configuración", type="primary")
    if submitted:
        save_settings(root, {"model": model, "enabled": enabled, "max_tokens": tokens, "temperature": temperature})
        st.success("Configuración guardada. Se aplicará a la siguiente consulta.")
    if not settings.api_key:
        st.warning("Falta la clave de OpenAI en .env. Los pronósticos locales siguen funcionando sin ella.")


def render_training(root):
    st.header("Datos y entrenamiento")
    st.write("1. Elige los datos → 2. Ajusta el experimento → 3. Valida → 4. Entrena")
    base = load_config(root / "config.json")
    previous_import = st.session_state.get("imported_data")
    reusable = previous_import is not None and Path(previous_import[1]).is_file()
    sources = (["Reutilizar último archivo importado"] if reusable else []) + ["Datos actuales de NUSECast", "Cargar CSV o Excel"]
    origin = st.radio("Fuente de datos", sources, horizontal=True)
    source = root / "data" / "raw" / "Numero_incidentes.csv"
    provenance = None
    if origin == "Reutilizar último archivo importado":
        source, provenance = Path(previous_import[1]), previous_import[2]
        st.success(f"Mismo archivo: {provenance['original_name']} · {provenance['rows']:,} registros.")
        st.caption("Puedes ejecutar otra prueba cambiando los hiperparámetros sin volver a cargar el archivo.")
    if origin == "Cargar CSV o Excel":
        st.caption("Solo datos agregados de incidentes, sin información personal. Máximo 120 MB. XLSM: lectura de valores, nunca macros.")
        with st.expander("Columnas requeridas y límites"):
            st.code(", ".join(REQUIRED), language=None)
            st.write("Una fila por mes, UPZ, localidad y tipo de incidente. Encabezados en la primera fila. "
                     "Hasta 2 millones de filas CSV o 250.000 filas Excel. No se calculan fórmulas: Excel debe estar guardado con sus valores calculados.")
            st.download_button("Descargar encabezados CSV", ",".join(REQUIRED) + "\n", "estructura_nusecast.csv", "text/csv")
        uploaded = st.file_uploader("Archivo de incidentes", type=["csv", "xlsx", "xlsm"], max_upload_size=120)
        if uploaded is None:
            return
        content = uploaded.getvalue()
        sheet, sep, encoding = None, ",", "utf-8-sig"
        if Path(uploaded.name).suffix.lower() in {".xlsx", ".xlsm"}:
            try:
                sheets = sheet_names(content)
            except Exception:
                st.error("No se pudo abrir el libro. Revisa el formato y el límite descomprimido de 300 MB.")
                return
            sheet = st.selectbox("Hoja a importar", sheets)
        else:
            a, b = st.columns(2)
            sep = a.selectbox("Separador del CSV", [",", ";", "\t"], format_func=lambda s: {",": "Coma", ";": "Punto y coma", "\t": "Tabulación"}[s])
            encoding = b.selectbox("Codificación", ["utf-8-sig", "latin-1"])
        import_key = hashlib.sha256(content).hexdigest() + str((sheet, sep, encoding))
        if st.button("Importar y previsualizar", type="primary"):
            st.session_state.pop("imported_data", None)
            st.session_state.pop("validated_training", None)
            try:
                with st.spinner("Leyendo y comprobando el archivo…"):
                    imported, metadata = ingest(root, content, uploaded.name, sheet, sep, encoding)
                st.session_state["imported_data"] = (import_key, str(imported), metadata)
            except (ValueError, OSError, TypeError, OverflowError):
                st.error("No se pudo importar: revisa encabezados, valores numéricos, límites y formato. "
                         "Año: 1900–2100; mes: 1–12; cantidad: entero no negativo. No se reemplazó el dataset anterior.")
        imported = st.session_state.get("imported_data")
        if not imported or imported[0] != import_key:
            st.info("Importa esta selección para continuar.")
            return
        source, provenance = Path(imported[1]), imported[2]
        st.success(f"Importación disponible: {provenance['rows']:,} registros.")
    if not source.is_file():
        st.warning("No se encontró el dataset local. Selecciona Cargar CSV o Excel.")
        return
    with st.expander("Vista previa de los datos", expanded=True):
        st.dataframe(pd.read_csv(source, nrows=12, dtype=str), hide_index=True, width="stretch")
    st.subheader("Configuración del experimento")
    st.caption("Alcance del proyecto: Kennedy, tipo 942. Cambiar un control invalida la validación anterior. Las referencias sencillas se incluyen siempre.")
    selected = st.multiselect("Modelos a entrenar", list(TUNED), default=["RN", "RI", "AM"], format_func=FAMILIES.get)
    mode = st.radio("Ajuste de hiperparámetros", ["Búsqueda predefinida en validación", "Configuración manual", "Factorial controlado RN"], horizontal=True)
    config = copy.deepcopy(base)
    config["experiment_name"] = st.text_input("Nombre de esta prueba", "Prueba", max_chars=80,
        help="Ejemplo: Prueba 1 · 8 neuronas. Cada ejecución se guarda por separado aunque repitas el nombre.")
    st.caption("Para comparar versiones: usa Configuración manual, conserva archivo, fechas y semillas, "
               "cambia el factor que quieras estudiar y vuelve a entrenar. Después entra en Historial.")
    factorial = mode == "Factorial controlado RN"
    if factorial:
        selected = ["RN"]
        config["experiment_design"] = "factorial_rn"
        st.info("Este modo evalúa solamente la RN y las referencias. Combina automáticamente los valores de los factores, "
                "manteniendo los mismos datos, fechas y semillas; elige la configuración con validación.")
    config["models"] = selected
    if provenance:
        config["import"] = provenance
    profile = copy.deepcopy(base["profiles"]["demo"])
    a, b, c = st.columns(3)
    first_seed = a.number_input("Semilla inicial", 0, 10000, 42)
    repetitions = b.number_input("Número de semillas", 1, 3, 1)
    profile["seeds"] = list(range(first_seed, first_seed + repetitions))
    profile["candidates"] = c.number_input("Configuraciones por modelo", 1, 3, 2 if mode.startswith("Búsqueda") else 1,
                                           disabled=not mode.startswith("Búsqueda"))
    manual = mode == "Configuración manual"
    grid = {}
    if "RN" in selected:
        with st.expander("Red neuronal · capas, aprendizaje y regularización", expanded=manual or factorial):
            profile["mlp_iterations"] = st.number_input("Iteraciones máximas MLP", 10, 2000, 350, step=10)
            if manual:
                hidden_text = st.text_input("Neuronas por capa oculta", "32,16", help="Ejemplo: 32,16 significa dos capas. Máximo tres capas de 128 neuronas.")
                try:
                    hidden = [int(x.strip()) for x in hidden_text.split(",")]
                except ValueError:
                    st.error("Escribe enteros separados por comas, por ejemplo 32,16.")
                    return
                a, b = st.columns(2)
                rate = a.number_input("Tasa de aprendizaje", 0.00001, 0.1, 0.003, step=0.001, format="%.5f")
                alpha = b.number_input("Regularización L2 (alpha)", 0.0, 10.0, 0.1, step=0.01)
                activation = st.selectbox("Activación oculta", ["relu", "tanh", "logistic"])
                grid["RN"] = [{"hidden": hidden, "alpha": alpha, "learning_rate": rate, "activation": activation}]
            if factorial:
                rates = st.text_input("Tasas a probar (separadas por comas)", "0.0003,0.003,0.03")
                neurons = st.text_input("Neuronas a probar (una capa)", "8,32,128")
                alphas = st.text_input("Valores L2 a probar", "1.0")
                try:
                    grid["RN"] = factorial_grid([float(v.strip()) for v in rates.split(",")],
                        [int(v.strip()) for v in neurons.split(",")], [float(v.strip()) for v in alphas.split(",")])
                except ValueError as error:
                    st.error(f"Revisa los factores (usa punto decimal): {error}")
                    return
                st.write(f"{len(grid['RN'])} combinaciones. Una capa oculta y activación ReLU constantes.")
                st.caption("Tres tasas × tres tamaños = nueve combinaciones. Con tres valores L2 serían 27. "
                           "No confundir neuronas a probar con varias capas de una misma red.")
            st.caption("Más iteraciones no garantizan menor error. El entrenamiento puede terminar antes si deja de mejorar su pérdida; no usa el test para detenerse.")
    if "AM" in selected:
        with st.expander("Bosque aleatorio", expanded=manual):
            profile["trees"] = st.number_input("Número de árboles", 5, 300, 60, step=5)
            if manual:
                depth = st.number_input("Profundidad máxima", 1, 30, 10)
                leaf = st.number_input("Muestras mínimas por hoja", 1, 50, 3)
                grid["AM"] = [{"max_depth": depth, "min_samples_leaf": leaf}]
    if "RI" in selected:
        with st.expander("Incertidumbre · Gamma-Poisson", expanded=manual):
            config["interval_level"] = st.slider("Cobertura nominal del intervalo", 0.5, 0.99, 0.9, 0.01)
            if manual:
                window = st.slider("Meses de historia reciente", 1, 12, 6)
                strength = st.number_input("Fuerza del prior", 0.01, 20.0, 1.0, step=0.1)
                grid["RI"] = [{"window": window, "prior_strength": strength}]
    if "ME" in selected:
        with st.expander("Evolución diferencial", expanded=manual):
            profile["evolution_generations"] = st.number_input("Generaciones máximas", 1, 200, 35)
            if manual:
                penalty = st.number_input("Penalización evolutiva", 0.0, 10.0, 0.1, step=0.01)
                population = st.number_input("Multiplicador de población", 2, 20, 8)
                grid["ME"] = [{"penalty": penalty, "popsize": population}]
    if manual or factorial:
        config["model_grid"] = grid
        profile["candidates"] = 1
    config["profiles"]["custom"] = profile
    with st.expander("Fechas y tratamiento de datos", expanded=bool(provenance)):
        st.caption("Se reservan 12 meses para construir rezagos. Los meses posteriores se separan cronológicamente; nunca al azar.")
        a, b = st.columns(2)
        start = a.text_input("Inicio de la historia (AAAA-MM)", base["start"])
        end = b.text_input("Fin de los datos y de la prueba (AAAA-MM)", base["end"])
        train = a.text_input("Último mes de entrenamiento", base["train_end"])
        val = b.text_input("Último mes de validación", base["validation_end"])
        config["exclude_suspect"] = st.checkbox("Aplicar corrección causal a meses sospechosos", base["exclude_suspect"])
        suspect = st.text_input("Meses sospechosos, separados por comas", ", ".join(base["suspect_months"]))
        try:
            config.update(start=str(pd.Period(start, "M")), end=str(pd.Period(end, "M")),
                          first_target=str(pd.Period(start, "M") + 12), train_end=str(pd.Period(train, "M")),
                          validation_end=str(pd.Period(val, "M")), test_end=str(pd.Period(end, "M")),
                          suspect_months=[str(pd.Period(p.strip(), "M")) for p in suspect.split(",") if p.strip()])
        except (ValueError, TypeError):
            st.error("Revisa las fechas: utiliza AAAA-MM.")
            return
    identity = config_identity(source, config)
    if st.button("Validar datos y configuración", type="primary"):
        st.session_state.pop("validated_training", None)
        try:
            with st.spinner("Validando meses, duplicados y particiones…"):
                checked = validate_source(source, config)
            st.session_state["validated_training"] = (identity, checked)
        except (ValueError, KeyError, OSError, TypeError) as error:
            st.error(f"No se puede entrenar: {error}")
    checked_state = st.session_state.get("validated_training")
    valid = checked_state is not None and checked_state[0] == identity
    if valid:
        checked = checked_state[1]
        a, b, c = st.columns(3)
        a.metric("Meses validados", checked["audit"]["months"])
        b.metric("UPZ", checked["audit"]["upz_count"])
        c.metric("Ajustes previstos", checked["fits"])
        st.success("Datos y particiones válidos. Listo para entrenar localmente.")
        st.write("Meses por partición:", checked["partitions"])
        st.caption(f"Incidentes sin UPZ conocida excluidos: {checked['audit']['excluded_unknown_upz_incidents']:,}.")
    else:
        st.info("Valida la selección actual para habilitar el entrenamiento.")
    st.warning("Escoge hiperparámetros por validación, no repitiendo ajustes para bajar el error de prueba. "
               "Presupuesto: hasta 2500 ajustes y 15 minutos, comprobados entre ajustes. No cierres ni cambies esta pantalla mientras entrena.")
    if st.button("Entrenar y guardar experimento", type="primary", disabled=not valid):
        st.session_state.pop("last_training_run", None)
        bar = st.progress(0.0, text="Preparando…")
        try:
            if sha256(source) != checked_state[1]["audit"]["source_sha256"]:
                raise ValueError("Los datos cambiaron después de validar. Vuelve a validarlos.")
            result = train_interactive(root, source, config, lambda value, message: bar.progress(value, text=message))
            st.session_state["last_training_run"] = str(result)
        except (ValueError, OSError, KeyError) as error:
            st.error(f"No se completó el experimento: {error}")
    if st.session_state.get("last_training_run"):
        result = Path(st.session_state["last_training_run"])
        st.success(f"Experimento disponible: {result.name}. Selecciónalo en Resultados.")
        st.info("Para una segunda o tercera prueba, cambia el nombre y los hiperparámetros, valida y vuelve a entrenar. "
                "El resultado anterior se conserva. En Historial puedes comparar hasta tres pruebas.")
        st.download_button("Descargar configuración del experimento", (result / "config.json").read_bytes(), "configuracion_nusecast.json", "application/json")


def render_history(runs):
    from nusecast.comparison import render_comparison
    render_comparison(runs)
