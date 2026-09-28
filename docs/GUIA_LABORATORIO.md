# NUSECast: cargar datos, entrenar y comparar

Abre http://127.0.0.1:8501/. Se conserva la identidad visual de la Javeriana.

## Datos y entrenamiento

1. Elige **Datos actuales de NUSECast** o **Cargar CSV o Excel**.
2. Para archivos propios, usa CSV, XLSX o XLSM. Selecciona hoja o separador y pulsa **Importar y previsualizar**.
3. Elige modelos y búsqueda predefinida o configuración manual. Los cuatro modelos de referencia siempre se incluyen.
4. Ajusta las fechas si el archivo cubre otro periodo. El proyecto sigue delimitado a Kennedy y código 942.
5. Pulsa **Validar datos y configuración**. Se revisan meses, conteos, duplicados, UPZ, historia suficiente y particiones temporales.
6. Pulsa **Entrenar y guardar experimento**. No cambies la pantalla ni cierres la aplicación mientras entrena. El progreso cuenta ajustes mensuales, no porcentaje de tiempo.
7. Consulta la nueva ejecución en **Resultados**. Las anteriores no se sobrescriben. En **Historial** puedes consultar configuraciones y comparar errores.

Columnas requeridas en la primera fila:

`ANIO, MES, TIPO_INCIDENTE, LOCALIDAD, COD_UPZ, UPZ, CANT_INCIDENTES`

Una fila por mes, UPZ, localidad y tipo. No se suman duplicados ni se completan ausencias con ceros automáticamente. Los registros sin UPZ conocida se excluyen del panel y se contabilizan en la auditoría de cobertura.

Límites: 120 MB por archivo; 2 millones de registros CSV; 250.000 filas Excel; 300 MB de contenido Excel descomprimido; 180 meses y 150 UPZ en el panel. No se ejecutan macros ni fórmulas. Para Excel, guarda el libro con los valores calculados: una fórmula sin valor almacenado se leerá como vacía y puede impedir la validación. XLS antiguo no está habilitado.

## Qué controla cada opción

| Modelo | Controles |
|---|---|
| Red neuronal MLP | Capas ocultas, neuronas, activación, tasa de aprendizaje, regularización L2 e iteraciones máximas |
| Bosque aleatorio | Número de árboles, profundidad y muestras mínimas por hoja |
| Gamma-Poisson | Ventana histórica, fuerza del prior y cobertura nominal |
| Evolución diferencial | Generaciones, multiplicador de población y penalización |

La búsqueda predefinida compara hasta tres configuraciones de cada familia con validación. En manual, se prueba una configuración por familia. Las semillas permiten observar variabilidad y promediar predicciones. Más iteraciones o más complejidad no garantizan mejora.

La partición es temporal. Los primeros doce meses construyen rezagos. Después se separan entrenamiento, validación y prueba. Cada predicción usa únicamente meses anteriores. El mes siguiente se pronostica con toda la historia disponible. Los controles no incorporan redes Transformer ni LSTM: la red actual es una MLP de variables históricas.

Presupuesto de la interfaz: máximo 2500 ajustes y 15 minutos, comprobados entre ajustes. No es un cierre forzado al segundo: un ajuste individual puede terminar después de ese límite. Solo se permite un entrenamiento simultáneo desde la interfaz o el comando oficial. Una interrupción no se presenta como ejecución completa.

Cada ejecución guarda configuraciones, selección, métricas, pronósticos, hashes de datos/código, versiones y advertencias en `outputs/<identificador>/`. Las importaciones normalizadas quedan en `data/uploads/<identificador>/datos.csv`, con metadatos del archivo y hoja originales. El libro original no se altera ni se conserva como nueva copia. La misma configuración sin cambios debe mantener los cálculos originales.

## Comparación responsable

Selecciona hiperparámetros por MAE de validación; no ajustes reiteradamente para reducir el test. El historial avisa si cambian los datos o las filas de evaluación. Un CSV normalizado puede tener distinto hash aun cuando sus filas relevantes sean equivalentes: el aviso es conservador. El test original ya fue examinado; hace falta una evaluación futura independiente para una conclusión confirmatoria.

## Configuración IA

La sección permite activar/desactivar el asistente, elegir GPT-4.1 mini o GPT-4.1, ajustar temperatura entre 0 y 2 y limitar la respuesta a 32–1000 tokens. La disponibilidad depende de la cuenta. La temperatura inicial es 0,2: controla la variación del texto, no su veracidad ni el error del pronóstico.

Guardar configura la siguiente consulta y no hace una llamada de pago. Las preferencias se guardan en `assistant_settings.json`; prevalecen sobre las opciones equivalentes de `.env`. La clave sigue exclusivamente en `.env` o en las variables del proceso y nunca se escribe en ese archivo público. Una configuración pública inválida desactiva el asistente.

El asistente solo envía la pregunta y el resumen visible de métricas cuando confirmas el envío. No manda el archivo cargado, no entrena modelos y no modifica resultados. Se mantienen diez intentos por sesión, sin reintentos automáticos: no es un límite de gasto de la cuenta.

Referencia sobre temperatura y límite de salida: [OpenAI, creación de respuestas](https://developers.openai.com/api/reference/cli/resources/responses/methods/create).
