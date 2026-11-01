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


def percentil(valor, percentiles):
    """Porcentaje de registros del entrenamiento cuyo IRA es menor o igual a `valor`."""
    return float(np.interp(valor, percentiles, np.arange(len(percentiles))))


def avisos(r, ref):
    """Revisa la coherencia de un caso (dict con las 14 variables) con lo observado en los datos.
    Devuelve una lista de (nivel, mensaje): 'error' = combinación imposible en los datos,
    'aviso' = combinación poco frecuente."""
    out = []
    lim = ref['limites']
    if r['CREDITS_APPROVED_CUM_ALL'] < r['CREDITS_APPROVED_TERM']:
        out.append(('error', 'Los créditos aprobados acumulados no pueden ser menores que los aprobados en el periodo.'))
    if r['CREDITS_FAILED_CUM'] < r['CREDITS_FAILED_TERM']:
        out.append(('error', 'Los créditos perdidos acumulados no pueden ser menores que los perdidos en el periodo.'))
    if r['GPA_TERM'] == 0 and r['CREDITS_APPROVED_TERM'] > 0:
        out.append(('error', 'Un promedio del periodo de 0.00 no es compatible con créditos aprobados en el periodo.'))
    if r['GPA_CUM'] == 0 and r['CREDITS_APPROVED_CUM_ALL'] > 0:
        out.append(('error', 'Un promedio acumulado de 0.00 no es compatible con créditos aprobados acumulados.'))

    total = r['CREDITS_APPROVED_TERM'] + r['CREDITS_FAILED_TERM']
    if total > lim['creditos_periodo_p99']:
        out.append(('aviso', f"{total:.0f} créditos en un periodo es poco frecuente: menos del 1% de los registros "
                             f"supera {lim['creditos_periodo_p99']:.0f}."))
    por_sem = r['CREDITS_APPROVED_CUM_ALL'] / max(r['UBICACION SEMESTRAL'], 1)
    if por_sem > lim['creditos_por_semestre_p99']:
        out.append(('aviso', f"{por_sem:.0f} créditos aprobados por semestre cursado es poco frecuente: menos del 1% de los "
                             f"registros supera {lim['creditos_por_semestre_p99']:.0f}."))
    emin = ref['edad_min_por_semestre'].get(str(int(r['UBICACION SEMESTRAL'])))
    if emin is not None and r['EDAD'] < emin:
        out.append(('aviso', f"Una edad de {r['EDAD']:.0f} años en el semestre {int(r['UBICACION SEMESTRAL'])} no se observa "
                             f"en los datos (la edad mínima en ese semestre es {emin:.0f})."))
    return out
