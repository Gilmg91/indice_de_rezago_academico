import json
import sys

import numpy as np
import pandas as pd
import sklearn
import streamlit as st

import ira

st.set_page_config(page_title="Predicción del IRA", page_icon="📊", layout="centered")


@st.cache_resource                       # el modelo se carga una sola vez, no en cada clic
def cargar():
    return ira.cargar_paquete("modelo-final-reg.joblib")


try:
    paquete = cargar()
except Exception as e:             # Streamlit Cloud oculta los errores: aquí se muestran para poder diagnosticar
    st.error("No se pudo cargar el modelo.")
    st.code(f"{type(e).__name__}: {e}")
    st.write(f"Python {sys.version.split()[0]} | scikit-learn {sklearn.__version__} | numpy {np.__version__}")
    st.stop()

@st.cache_data
def cargar_referencias():
    with open("referencias.json", encoding="utf-8") as f:
        return json.load(f)


ref = cargar_referencias()
jer = ref["jerarquia"]                    # campus -> escuela -> programas que existen en los datos
rangos = paquete["rangos"]
cats = paquete["categorias"]


def num_int(col, etiqueta, donde):
    r = rangos[col]
    return donde.number_input(etiqueta, min_value=int(r["min"]), max_value=int(r["max"]),
                              value=int(r["median"]), step=1)


def num_float(col, etiqueta, donde):
    r = rangos[col]
    return donde.number_input(etiqueta, min_value=float(r["min"]), max_value=float(r["max"]),
                              value=float(r["median"]), step=0.01, format="%.2f")


# ---------- Encabezado ----------
st.title("Predicción del índice IRA")
st.write("Ingrese los datos de un estudiante y el modelo estima su **IRA_SCORE**. "
         f"Modelo: {paquete['nombre']} (Random Forest)")

if sklearn.__version__ != paquete["sklearn"]:
    st.warning(f"El modelo se guardó con scikit-learn {paquete['sklearn']} y esta app usa "
               f"{sklearn.__version__}. Ajuste requirements.txt para evitar resultados distintos.")

# ---------- Programa (fuera del formulario: cada lista depende de la anterior) ----------
st.subheader("Programa")
c1, c2 = st.columns(2)
campus = c1.selectbox("Campus", sorted(jer))
escuela = c2.selectbox("Escuela", sorted(jer[campus]), key=f"escuela_{campus}")
programa = st.selectbox("Programa", jer[campus][escuela], key=f"programa_{campus}_{escuela}")

# ---------- Formulario ----------
with st.form("formulario"):
    st.subheader("Perfil")
    semestre = num_int("UBICACION SEMESTRAL", "Ubicación semestral", st)

    c3, c4, c5 = st.columns(3)
    edad = num_int("EDAD", "Edad", c3)
    genero = c4.selectbox("Género", cats["GENERO"])
    region = c5.selectbox("Región de origen", cats["REGION"])

    st.subheader("Desempeño del periodo")
    c6, c7, c8 = st.columns(3)
    gpa_term = num_float("GPA_TERM", "Promedio del periodo (GPA)", c6)
    cred_apr = num_int("CREDITS_APPROVED_TERM", "Créditos aprobados", c7)
    cred_per = num_int("CREDITS_FAILED_TERM", "Créditos perdidos", c8)
    estado = st.selectbox("Estado académico", cats["ACADEMIC_STATUS_TERM"],
                          index=cats["ACADEMIC_STATUS_TERM"].index("Estado academico normal"))

    st.subheader("Histórico acumulado")
    c9, c10, c11 = st.columns(3)
    gpa_cum = num_float("GPA_CUM", "Promedio acumulado", c9)
    cred_apr_cum = num_int("CREDITS_APPROVED_CUM_ALL", "Créditos aprobados acumulados", c10)
    cred_per_cum = num_int("CREDITS_FAILED_CUM", "Créditos perdidos acumulados", c11)

    enviar = st.form_submit_button("Predecir IRA", type="primary")

# ---------- Resultado ----------
if enviar:
    fila = pd.DataFrame([{
        "DESC_CAMPUS": campus, "DESC_ESCUELA": escuela, "DESC_PROGRAMA": programa,
        "UBICACION SEMESTRAL": semestre, "GPA_TERM": gpa_term,
        "CREDITS_APPROVED_TERM": cred_apr, "CREDITS_FAILED_TERM": cred_per,
        "ACADEMIC_STATUS_TERM": estado, "GPA_CUM": gpa_cum,
        "CREDITS_APPROVED_CUM_ALL": cred_apr_cum, "CREDITS_FAILED_CUM": cred_per_cum,
        "EDAD": edad, "GENERO": genero, "REGION": region}])

    # Alertas de coherencia: límites tomados de los datos de entrenamiento (referencias.json)
    alertas = ira.avisos(fila.iloc[0].to_dict(), ref)
    for tipo, texto in alertas:
        (st.error if tipo == "error" else st.warning)(texto)
    if any(tipo == "error" for tipo, _ in alertas):
        st.info("Los datos ingresados son incoherentes: la predicción que sigue no es confiable.")

    valor = float(ira.predecir(fila, paquete)[0])
    nivel = ira.nivel(valor)

    pct = ira.percentil(valor, ref["percentiles_ira"])
    veces = valor / ref["ira_media"]

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("IRA predicho", f"{valor:.3f}")
    m2.metric("Frente al promedio", f"{veces:.1f} veces")
    m3.metric("Supera al", f"{pct:.0f}% de los registros")

    mensaje = f"Nivel de referencia: **{nivel}**"
    {"Bajo": st.success, "Medio": st.warning, "Alto": st.error}[nivel](mensaje)

    st.caption(
        f"El error típico del modelo en datos que no vio al entrenar es de ±{paquete['rmse_test']:.3f} (RMSE). "
        "Niveles de referencia según la distribución del IRA en los datos: Bajo (menos de 0.08), "
        "Medio (0.08 a 0.15) y Alto (0.15 o más); no son umbrales oficiales. "
        "El modelo tiende a subestimar los IRA altos.")
