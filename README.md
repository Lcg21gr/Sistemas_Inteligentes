# NUSECast · Sistemas Inteligentes

Laboratorio académico de pronóstico mensual de registros de incidentes NUSE 123
en Kennedy, Bogotá. Interfaz Streamlit con identidad visual de la Javeriana,
carga de CSV/XLSX/XLSM, ajuste de hiperparámetros y comparación temporal de modelos.
No es una aplicación oficial de la universidad ni un sistema operativo de emergencias.

## Ejecutar localmente

Requiere Python 3.12. En Linux o macOS, desde la raíz del repositorio:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1
```

También puedes abrir `Abrir NUSECast.command` en macOS una vez instalado el entorno.
`requirements.txt` instala el paquete local y aplica las versiones fijadas en
`requirements-lock.txt`. El código de los modelos está en `src/nusecast`.

## Datos y primer experimento

El repositorio no contiene claves, archivos de usuarios ni resultados históricos.
La pantalla inicial sin ejecuciones es normal. Abre **Datos y entrenamiento →
Cargar CSV o Excel**, importa el archivo, ajusta fechas y modelos, valida y entrena.

Columnas obligatorias:

`ANIO, MES, TIPO_INCIDENTE, LOCALIDAD, COD_UPZ, UPZ, CANT_INCIDENTES`

La configuración inicial filtra Kennedy y el tipo 942, con historia desde enero de
2020 hasta julio de 2026. El archivo debe cubrir las fechas seleccionadas y contar
con doce meses iniciales para construir rezagos. No se ejecutan macros de XLSM.

Para usar el comando de experimentos, coloca el CSV autorizado en
`data/raw/Numero_incidentes.csv`, o pasa su ruta explícitamente:

```bash
python -m nusecast.cli run --source /ruta/Numero_incidentes.csv --profile demo
```

Los datos cargados quedan en `data/uploads/`; los experimentos, en `outputs/`.
Ambas carpetas están excluidas de Git. Conserva respaldos fuera del servicio de
despliegue: no se implementa almacenamiento remoto persistente.

## Modelos y evaluación

- RN: red neuronal MLP con pérdida Poisson.
- AM: bosque aleatorio.
- RI: Gamma-Poisson con intervalos predictivos que deben evaluarse, no asumirse fiables.
- ME: combinación de rezagos optimizada por evolución diferencial.
- Referencias: último mes, promedio de tres meses, mismo mes del año anterior y GLM Poisson.

La selección usa validación temporal. La evaluación incluye MAE, RMSE, sesgo y
cobertura de intervalos. No se utiliza ChatGPT para producir los pronósticos.
Los Transformers, LSTM, RAG y sistemas multiagente no están implementados.

Consulta [la guía de uso](docs/GUIA_LABORATORIO.md) y
[el protocolo experimental](docs/PROTOCOLO.md).

## Preparar Streamlit Community Cloud

En **Create app**, selecciona:

| Campo | Valor |
|---|---|
| Repository | `Lcg21gr/Sistemas_Inteligentes` |
| Branch | `main` |
| Main file path | `app.py` |
| Python, en Advanced settings | `3.12` |

No hace falta una clave API para abrir la aplicación ni entrenar modelos locales.
Empieza con acceso restringido al equipo, archivos no sensibles y experimentos
pequeños. Los límites de recursos de Community Cloud pueden impedir entrenamientos
que sí funcionan en el equipo local. El despliegue remoto debe verificarse aparte.

Esta versión es un laboratorio de confianza, no un servicio multiusuario aislado:
las ejecuciones y preferencias se comparten dentro de la instancia, y no hay roles
de administrador ni cuotas globales. No publiques la app abierta con una clave API
configurada: quienes accedan podrían generar cargos. El límite por sesión no es un
presupuesto global. No dependas del disco del servicio como único respaldo.

Guías oficiales: [desplegar](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
y [dependencias](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

## Asistente opcional y secretos

`.env.example` contiene nombres de configuración, sin clave. Para pruebas locales,
crea tu propio `.env` privado a partir del ejemplo. No lo compartas ni lo añadas a Git.
`.streamlit/secrets.toml` también está excluido. Revoca cualquier clave que se haya
compartido anteriormente. El asistente explica métricas y puede generar cargos;
no entrena ni modifica modelos. En esta copia está desactivado por defecto.

## Pruebas

```bash
python -m pip install -c requirements-lock.txt '.[dev]'
python -m pytest -q
```

Las pruebas generan datos sintéticos temporales: no requieren el CSV original
ni hacen llamadas de pago al asistente.

## Alcance

Resultados exploratorios por UPZ y mes; no autorizan despachos ni decisiones
automáticas de emergencia. Comparar errores solo tiene sentido con las mismas
unidades, datos y periodos. El periodo de prueba original ya fue explorado y se
necesita una evaluación futura independiente.
