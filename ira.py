"""Lógica de predicción del IRA (sin interfaz). La usa app.py."""
import joblib
import numpy as np
import pandas as pd

CAT_COLS = ['DESC_CAMPUS', 'DESC_ESCUELA', 'DESC_PROGRAMA',
            'ACADEMIC_STATUS_TERM', 'GENERO', 'REGION']


def cargar_paquete(ruta='modelo-final-reg.joblib'):
    return joblib.load(ruta)


def limpiar(df_nuevo, paquete):
    """Aplica la misma limpieza del entrenamiento y deja las columnas en el orden del modelo."""
    d = df_nuevo.copy()
    d.columns = d.columns.str.strip()

    faltan = [c for c in paquete['columnas'] if c not in d.columns]
    if faltan:
        raise ValueError(f"Faltan columnas: {faltan}")

    for c in CAT_COLS:
        d[c] = d[c].astype(str).str.strip()
    for c, mapa in paquete['canon'].items():          # unifica mayúsculas/minúsculas
        d[c] = d[c].str.lower().map(mapa).fillna(d[c])

    num_cols = [c for c in paquete['columnas'] if c not in CAT_COLS]
    for c in num_cols:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    if d[num_cols].isna().any().any():
        malas = d[num_cols].columns[d[num_cols].isna().any()].tolist()
        raise ValueError(f"Valores vacíos o no numéricos en: {malas}")
    d['UBICACION SEMESTRAL'] = d['UBICACION SEMESTRAL'].round().astype(int)
    return d[paquete['columnas']]


def categorias_desconocidas(d_limpio, paquete):
    """Categorías que el modelo no vio al entrenar (el modelo las ignora)."""
    out = {}
    for c in CAT_COLS:
        nuevas = sorted(set(d_limpio[c]) - set(paquete['categorias'][c]))
        if nuevas:
            out[c] = nuevas
    return out


def predecir(df_nuevo, paquete):
    d = limpiar(df_nuevo, paquete)
    return np.clip(paquete['modelo'].predict(d), 0, 1)


# Umbrales de referencia tomados de la distribución del IRA en los datos de entrenamiento
# (percentil 75 = 0.08 y percentil 90 = 0.15). No son umbrales oficiales.
UMBRAL_MEDIO = 0.08
UMBRAL_ALTO = 0.15


def nivel(valor):
    """Nivel de referencia: Bajo (< P75), Medio (P75 a P90) o Alto (>= P90)."""
    if valor < UMBRAL_MEDIO:
        return 'Bajo'
    if valor < UMBRAL_ALTO:
        return 'Medio'
    return 'Alto'
