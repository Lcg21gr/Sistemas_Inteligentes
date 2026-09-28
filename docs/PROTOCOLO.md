# Protocolo y límites metodológicos

## Problema y alcance

Unidad: UPZ–mes. Objetivo: conteo de registros NUSE código 942. Horizonte: un mes.
Fuente recibida: `Numero_incidentes.csv` de datos abiertos de Bogotá, no observaciones individuales georreferenciadas.
Se conserva copia del original y su SHA-256. No se inventan coordenadas ni atributos de clima.

Alcance inicial: Kennedy, enero de 2020 a julio de 2026. UPZ desconocida (`UPZ999`) se excluye del objetivo territorial,
pero su volumen se declara y entra en la medida de cobertura geográfica. La suma de las 12 UPZ no es el total de Kennedy.
El campo 942H no se fusiona en este periodo: en la revisión previa no aparecía después de 2019.
Las celdas faltantes y las claves duplicadas provocan error, no un relleno automático con cero.
No se eliminan duplicados globales del CSV: se validan las claves dentro del alcance elegido.

## Tratamiento causal de enero y abril de 2024

Son meses sospechosos según la revisión previa, **no se ha probado su causa**.
En la configuración principal:

- Se preservan los valores originales y se excluyen como objetivos de ajuste/evaluación.
- Para los rezagos de los modelos puntuales se usa la media de tres meses estrictamente anteriores.
- Las variables de cobertura se usan rezagadas, sin anticipar la cobertura del mes objetivo.
- Gamma-Poisson omite esos conteos en su actualización en lugar de tratar valores imputados como evidencia observada.
- `--keep-suspect` desactiva ambas decisiones para permitir sensibilidad.

Se corrige una fuga comprobada del código previo: promediar el mes anterior y el posterior para imputar un mes
introducía el propio objetivo al construir el rezago de la siguiente predicción.
Las pruebas modifican los objetivos/futuros y comprueban que los predictores del presente no cambien.

## Variables y disponibilidad

Historia: 12 rezagos de conteos, promedio causal de tres meses y suma rezagada de las otras UPZ.
Calendario: seno/coseno del mes y días laborales excluyendo festivos colombianos.
Calidad: cobertura geográfica rezagada 1 y 12 meses y número de rezagos imputados.
Territorio: codificación one-hot para MLP, Random Forest y GLM.
`other_upz_lag1` representa **otras UPZ de Kennedy**, no vecinos geográficos.

Todos tienen acceso a la misma historia hasta t−1, pero no consumen exactamente las mismas variables:

| Método | Variables efectivamente usadas |
|---|---|
| MLP, Random Forest, GLM | Las variables numéricas anteriores + UPZ |
| Evolución diferencial | lag1, lag2, lag3, lag12, media3, seno/coseno e intercepto |
| Gamma-Poisson | Conteos observados recientes válidos y prior por UPZ de historia anterior a esa ventana |
| Baselines | lag1; media3; lag12, respectivamente |

Por tanto es una comparación de **métodos completos**, no una ablación que aísle exclusivamente la arquitectura.
No se atribuye una diferencia a una sola variable sin un experimento adicional controlado.

## Partición y selección

- Primer objetivo: enero de 2021, cuando existen doce rezagos.
- Entrenamiento inicial: 2021–2023 (432 filas en el panel recibido).
- Validación: 2024 (120 filas si se excluyen enero y abril).
- Prueba: enero de 2025–julio de 2026 (228 filas).
- Futuro a un paso: agosto de 2026, sin objetivo observado ni métrica.

**Origen móvil expansivo tanto en validación como en prueba**: para predecir cada mes se vuelve a ajustar con
los objetivos elegibles estrictamente anteriores. En prueba se incorporan meses de prueba anteriores cuando ya son observados;
esto es válido para pronóstico a un mes, pero NO equivale a pronosticar todos los meses de una sola vez.
Se supone que el conteo del mes anterior ya está publicado al emitir el pronóstico; si hay retraso de publicación,
habrá que desplazar los rezagos y reevaluar con fechas reales de disponibilidad.

El escalador y el codificador se ajustan solo sobre el entrenamiento disponible en cada origen.
No se usa validación aleatoria automática de MLP. Su regularización L2 y tamaño se seleccionan temporalmente.
La selección por familia minimiza el MAE medio de las semillas de validación. Los empates conservan la primera configuración.
El modelo de referencia se selecciona comparando MAE de las predicciones promediadas entre semillas en validación,
incluidas las líneas base. Esa selección se guarda antes de abrir la evaluación de prueba.
No hay ajuste ni nueva selección a partir del error de prueba dentro del programa.

Sin embargo, el equipo ya analizó ese periodo en notebooks anteriores: **no es una prueba totalmente virgen**.
No deben usarse los resultados actuales como estimación definitiva del desempeño futuro.

## Modelos

### RN: MLP

MLP global con pérdida Poisson y enlace exponencial (salida positiva). El objetivo se divide por 100 durante
el entrenamiento para facilitar la optimización y se vuelve a unidades originales al predecir.
Se registra cada advertencia de convergencia; alcanzar el máximo de iteraciones no equivale a converger.

### ME: evolución diferencial

Población de vectores con ocho componentes: cinco logits convertidos a pesos positivos que suman uno,
un intercepto y dos coeficientes estacionales. Cada candidato combina lag1, lag2, lag3, lag12 y media3.
Función objetivo: MAE sobre conteos divididos por 100 + penalización L2 de intercepto/estacionalidad.
Mutación y recombinación: valores por defecto documentados de SciPy, estrategia `best1bin`;
sin pulido por gradiente (`polish=False`), un trabajador, semilla explícita.
La longitud de la población es `popsize × 8`; 35 generaciones máximo en demo, 180 en extendido.
Se guardan pesos, generaciones, evaluaciones y estado de convergencia. No se afirma óptimo global probado.

### RI: Gamma-Poisson

Para una UPZ, tasa λ con prior Gamma(α₀, β₀), donde β usa parametrización de **tasa**.
Media previa empírica calculada con objetivos de entrenamiento anteriores a la ventana reciente;
α₀ = media × fuerza, β₀ = fuerza. Así no se duplican conteos entre prior y ventana posterior.
Después de n meses válidos, α = α₀ + suma de conteos, β = β₀ + n.
Pronóstico: α/β. Predictiva: binomial negativa con parámetros SciPy n=α, p=β/(β+1).
Intervalo central según `interval_level`, por defecto 90 %.

Supuestos: tasa aproximadamente constante en la ventana y observaciones condicionalmente Poisson.
La elección del prior es empírica; no se integra incertidumbre de sus hiperparámetros.
Estacionalidad, dependencia temporal, subregistro y cambios de régimen pueden producir infracobertura.
Se reportan cobertura observada, anchura y puntuación de intervalos. No hay garantías de cobertura.

### AM: Random Forest

Ensamble de árboles con criterio Poisson. Profundidad y mínimo de observaciones por hoja limitan sobreajuste;
número de árboles según perfil. No se confunde la variación entre árboles o semillas con incertidumbre predictiva calibrada.

## Presupuesto y métricas

Demo: 2 configuraciones × 1 semilla por familia. Extendido: 3 configuraciones × 5 semillas por familia.
El número de configuraciones/semillas es igual entre las cuatro familias, pero no el coste computacional.
RI es determinista: repetir semillas no añade evidencia independiente. Las baselines y GLM se ejecutan una vez.
Los métodos no tienen el mismo número de parámetros ni la misma función de ajuste.
El presupuesto previo de 30 × 5 **no está implementado** en estos perfiles.

MAE principal; RMSE, devianza Poisson, sesgo firmado, error medio por defecto/exceso como complementos.
Un error por defecto no demuestra demanda de emergencia desatendida: solo indica un conteo subpronosticado.
No se usa MAPE por su mal comportamiento cerca de cero. No se reportan p-valores de superioridad.
El promedio entre semillas produce la predicción final puntual; los CSV por semilla preservan variabilidad del entrenamiento.

## Agente y supervisión

Implementado: herramienta de soporte que recibe histórico, produce pronósticos y deja trazabilidad.
Conceptualmente: meta = reducir error; situado = histórico mensual; autónomo = cálculo reproducible una vez ejecutado;
proactivo = anticipa el siguiente mes. **No se implementa una cualidad social/SMA ni negociación entre agentes**.
No hay tareas programadas, despacho, comunicación con operadores ni decisiones automáticas de personal.
La revisión de cobertura, calidad y advertencias queda en manos humanas.

## Referencias técnicas de implementación

- [MLPRegressor y pérdida Poisson, scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html).
- [RandomForestRegressor, scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html).
- [Evolución diferencial, SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html).
- [Binomial negativa, SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nbinom.html).
- [Validación temporal, scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
- [Streamlit](https://docs.streamlit.io/).
- [Datos abiertos Bogotá: incidentes](https://datosabiertos.bogota.gov.co/dataset/incidentes).

Estas referencias técnicas no reemplazan los artículos científicos exigidos para cada módulo.
