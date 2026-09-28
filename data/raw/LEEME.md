# Datos de entrada

La versión local utiliza estos archivos entregados por el equipo. No se incluyen en el repositorio:

- `Numero_incidentes.csv`: fuente principal.
- `TipificacionIncidentes.csv`: diccionario de tipos.
- `Definicion_campos.csv`: diccionario recibido; no describe todos los campos adicionales de UPZ.

`metadata.csv` del envío previo contiene el mismo diccionario que `TipificacionIncidentes.csv`;
no se utiliza como metadato temporal ni como una segunda fuente independiente.

Los CSV no se incluyen en Git por tamaño. Para compartir una entrega académica reproducible,
incluye el CSV autorizado en el ZIP o acuerda una descarga verificable. No incluyas `.venv`.

Cada ejecución guarda SHA-256 del CSV y su auditoría en `outputs/<id>/data_audit.json`.
La fuente termina en julio de 2026; las predicciones posteriores no son datos observados.

Portal de origen indicado por el equipo: https://datosabiertos.bogota.gov.co/dataset/incidentes
