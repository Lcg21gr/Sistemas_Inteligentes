import hashlib
import io
import json
import uuid
import zipfile
from pathlib import Path
from xml.etree.ElementTree import ParseError

import openpyxl
import pandas as pd
from defusedxml.common import DefusedXmlException
from openpyxl.utils.exceptions import InvalidFileException

from nusecast.data import REQUIRED, sha256

MAX_BYTES = 120 * 1024 * 1024
MAX_CSV_ROWS = 2_000_000
MAX_EXCEL_ROWS = 250_000


def open_workbook(content):
    if len(content) > MAX_BYTES:
        raise ValueError("El archivo supera el límite de 120 MB.")
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            entries = archive.infolist()
            if len(entries) > 10000 or sum(item.file_size for item in entries) > 300 * 1024 * 1024:
                raise ValueError("El libro descomprimido excede el límite de seguridad de 300 MB.")
        return openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=True, keep_links=False)
    except (zipfile.BadZipFile, KeyError, OSError, ParseError, DefusedXmlException, InvalidFileException) as error:
        raise ValueError("No se pudo leer el libro Excel. Comprueba que sea XLSX o XLSM válido.") from error


def sheet_names(content):
    workbook = open_workbook(content)
    try:
        return workbook.sheetnames
    finally:
        workbook.close()


def normalize(frame):
    frame.columns = [str(column).strip().upper() for column in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError("Hay encabezados duplicados después de normalizar sus nombres.")
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError("Faltan columnas: " + ", ".join(sorted(missing)))
    frame = frame[REQUIRED].astype("string").copy()
    for column in REQUIRED:
        frame[column] = frame[column].str.strip().replace("", pd.NA)
    for column in ["ANIO", "MES", "TIPO_INCIDENTE", "CANT_INCIDENTES"]:
        frame[column] = frame[column].str.replace(r"^(\d+)\.0+$", r"\1", regex=True)
    frame["LOCALIDAD"] = frame.LOCALIDAD.str.upper()
    if frame[["ANIO", "MES", "TIPO_INCIDENTE", "LOCALIDAD", "CANT_INCIDENTES"]].isna().any().any():
        raise ValueError("Faltan valores en año, mes, tipo, localidad o cantidad.")
    years = pd.to_numeric(frame.ANIO, errors="raise")
    months = pd.to_numeric(frame.MES, errors="raise")
    counts = pd.to_numeric(frame.CANT_INCIDENTES, errors="raise")
    if not (years.between(1900, 2100) & years.mod(1).eq(0)).all():
        raise ValueError("El año debe ser un entero entre 1900 y 2100.")
    if not (months.between(1, 12) & months.mod(1).eq(0)).all():
        raise ValueError("El mes debe ser un entero entre 1 y 12.")
    if not (counts.ge(0) & counts.mod(1).eq(0) & counts.le(1_000_000_000)).all():
        raise ValueError("Los conteos deben ser enteros no negativos, como máximo 1.000.000.000.")
    frame["ANIO"] = years.astype(int).astype(str)
    frame["MES"] = months.astype(int).astype(str)
    frame["CANT_INCIDENTES"] = counts.astype(int).astype(str)
    return frame


def ingest(root: Path, content: bytes, filename: str, sheet=None, separator=",", encoding="utf-8-sig"):
    extension = Path(filename).suffix.lower()
    if extension not in {".csv", ".xlsx", ".xlsm"}:
        raise ValueError("Usa CSV, XLSX o XLSM. El formato XLS antiguo no está habilitado.")
    if not content or len(content) > MAX_BYTES:
        raise ValueError("El archivo está vacío o supera 120 MB.")
    folder = root / "data" / "uploads" / uuid.uuid4().hex
    folder.mkdir(parents=True)
    destination = folder / "datos.csv"
    rows, first = 0, True
    workbook = None
    try:
        if extension == ".csv":
            if separator not in {",", ";", "\t"} or encoding not in {"utf-8-sig", "latin-1"}:
                raise ValueError("Separador o codificación no permitidos.")
            chunks = pd.read_csv(io.BytesIO(content), sep=separator, encoding=encoding,
                                 dtype=str, chunksize=50000)
        else:
            workbook = open_workbook(content)
            if sheet not in workbook.sheetnames:
                raise ValueError("Selecciona una hoja del libro.")
            worksheet = workbook[sheet]
            worksheet.reset_dimensions()
            iterator = worksheet.iter_rows(values_only=True, max_col=101)
            headers = list(next(iterator, ()))
            while headers and headers[-1] is None:
                headers.pop()
            if not headers or len(headers) > 100:
                raise ValueError("La primera fila debe contener los encabezados (máximo 100 columnas).")

            def excel_chunks():
                batch, visited = [], 0
                for values in iterator:
                    visited += 1
                    if visited > MAX_EXCEL_ROWS:
                        raise ValueError("Excel admite hasta 250.000 filas en este laboratorio. Usa CSV para más registros.")
                    if any(value is not None for value in values):
                        if any(value is not None for value in values[len(headers):]):
                            raise ValueError("Hay celdas con datos fuera de los encabezados.")
                        batch.append(values[:len(headers)])
                    if len(batch) == 50000:
                        yield pd.DataFrame(batch, columns=headers)
                        batch = []
                if batch:
                    yield pd.DataFrame(batch, columns=headers)

            chunks = excel_chunks()
        for frame in chunks:
            cleaned = normalize(frame)
            rows += len(cleaned)
            if rows > MAX_CSV_ROWS:
                raise ValueError("El archivo supera 2.000.000 de registros.")
            cleaned.to_csv(destination, mode="w" if first else "a", header=first, index=False)
            first = False
        if not rows:
            raise ValueError("El archivo no tiene registros.")
        metadata = {"original_name": Path(filename).name, "original_sha256": hashlib.sha256(content).hexdigest(),
                    "normalized_sha256": sha256(destination), "rows": rows, "sheet": sheet,
                    "separator": separator if extension == ".csv" else None,
                    "encoding": encoding if extension == ".csv" else None}
        (folder / "import.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return destination, metadata
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        if workbook is not None:
            workbook.close()
