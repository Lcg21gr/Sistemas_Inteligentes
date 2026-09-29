# Experimento sistemático del módulo RN

Esta carpeta se conserva en la raíz del repositorio como material de apoyo.
La aplicación usa su propio motor para las pruebas del usuario y no carga estos
resultados en Historial ni ejecuta estos scripts automáticamente.

Para reproducir los scripts recibidos, instala primero el paquete desde la raíz:
`python -m pip install '.[experiments]'`. Luego entra en `experimentos_rn/` y ejecuta
los comandos siguientes. Se necesita el CSV autorizado en `data/raw/Numero_incidentes.csv`.
Los archivos existentes de `res/` hacen que los scripts omitan casos ya calculados;
para repetir todo, trabaja en una copia separada sin esos resultados, preservando los originales.

1. `python preparar.py` → `features.csv` (mismas variables y partición del repositorio).
2. `python exp.py 100000` → Experimento 1: factorial 3 × 3 × 3 (tasa de aprendizaje × neuronas × α),
   3 semillas, origen móvil en los 10 meses válidos de 2024 = 810 entrenamientos (~15 min). Es reanudable.
3. `python exp23.py ref` y `python exp23.py 100000` → referencias, Experimento 2 (con/sin señal del entorno)
   y Experimento 3 (prueba ene-2025 → jul-2026, 5 semillas).
4. `python figs.py` → figuras de la presentación.

`res/` contiene los resultados ya calculados: cada caso del factorial (`exp1.csv`) y cada predicción
de validación y prueba por semilla (`exp23_pred.csv`, `referencias_pred.csv`).

## Integración y límites

- Los tres CSV y las figuras recibidas se conservaron sin cambiar sus valores.
- Se retiraron únicamente comentarios y docstrings de las copias de los scripts.
- No se recibieron el hash del CSV original ni las versiones de dependencias del cálculo externo.
- El diseño declara 810 ajustes; `exp1.csv` contiene 81 resúmenes, no 810 trazas individuales.
- El mejor MAE de selección es 24,107 (promedio de MAE de tres semillas).
- La evaluación con cinco semillas promedia primero las predicciones: MAE de prueba 29,707
  con señal de otras UPZ y 30,489 sin ella. No son promedios de MAE por semilla.
- Media móvil: 28,686; GLM: 27,019. No se ha demostrado un empate estadístico.
- La señal de otras UPZ ofrece una mejora descriptiva pequeña en estos resultados;
  no constituye una prueba de comportamiento social ni de superioridad confirmada.
- El código recibido oculta advertencias; la aplicación principal sí registra
  advertencias de cada ajuste. No se toma la ausencia de advertencias externas como convergencia.

`referencia_anterior/` contiene configuración, selección, manifiesto y predicciones
de la ejecución propia `20260928T153109Z-12e53a1d`. Es una copia de consulta que no
altera ni reemplaza las ejecuciones originales. Las filas y objetivos evaluados
coinciden con los del estudio recibido, pero eso no certifica su procedencia de entrenamiento.
