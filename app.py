# -*- coding: utf-8 -*-
"""
Dashboard Bank Marketing — Predicción de Suscripción a Depósito a Plazo Fijo
Proyecto Integrador UTEC - Grupo 8

Cómo correrlo localmente:
    pip install -r requirements.txt
    streamlit run app.py

El CSV (bank-full.csv, separado por ';') se busca automáticamente en la misma
carpeta que este archivo. Si no lo encuentra, la app te deja subirlo desde
la interfaz.
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

# ------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Bank Marketing",
    layout="wide",
)

NUM_COLS = ["age", "balance", "duration", "campaign", "pdays", "previous", "day"]
CAT_COLS = ["job", "marital", "education", "default", "housing", "loan",
            "contact", "month", "poutcome"]
FEATURE_COLS = NUM_COLS + CAT_COLS
TARGET = "y"

MONTH_ORDER = ["jan", "feb", "mar", "apr", "may", "jun",
               "jul", "aug", "sep", "oct", "nov", "dec"]

# ------------------------------------------------------------------
# PALETA PASTEL — sin rojo ni verde (solo azules, lilas, durazno y amarillo)
# ------------------------------------------------------------------
PASTEL_MAP = {"yes": "#A7C7E7", "no": "#FFD8A8"}   # azul pastel vs. durazno pastel
PASTEL_SCALE = ["#FFD8A8", "#F6C6D8", "#D9C6EC", "#B8C6E8", "#8FB8DE"]  # durazno -> azul
PASTEL_LINE = "#7A6F8A"          # lila grisáceo para líneas de referencia
PASTEL_SINGLE = "#B8C6E8"        # azul lila para barras de una sola serie
DIVERGING_SCALE = "PuOr"         # violeta <-> naranja, sin rojo ni verde

# ------------------------------------------------------------------
# CARGA DE DATOS
# ------------------------------------------------------------------
@st.cache_data
def cargar_csv(path_or_buffer):
    return pd.read_csv(path_or_buffer, sep=";")


def obtener_datos():
    """Busca bank-full.csv junto al script; si no existe, permite subirlo."""
    posibles_rutas = ["bank-full.csv", "data/bank-full.csv"]
    for ruta in posibles_rutas:
        if os.path.exists(ruta):
            return cargar_csv(ruta), f"Cargado automáticamente desde `{ruta}`"

    st.sidebar.warning("No encontré `bank-full.csv` junto a la app.")
    archivo = st.sidebar.file_uploader("Sube el archivo bank-full.csv", type=["csv"])
    if archivo is not None:
        return cargar_csv(archivo), f"Cargado desde el archivo subido: `{archivo.name}`"
    return None, None


df_raw, fuente = obtener_datos()

st.title("Dashboard: Predicción de Suscripción a Depósito a Plazo Fijo")
st.caption("Proyecto Integrador UTEC · Grupo 8 · Bank Marketing (UCI)")

if df_raw is None:
    st.info(
        "⬅️ Sube el archivo **bank-full.csv** (separado por `;`) desde la barra "
        "lateral para activar el dashboard."
    )
    st.stop()

st.sidebar.success(fuente)

# Validación mínima de columnas esperadas
faltantes = [c for c in FEATURE_COLS + [TARGET] if c not in df_raw.columns]
if faltantes:
    st.error(f"Al CSV le faltan columnas esperadas: {faltantes}")
    st.stop()

# ------------------------------------------------------------------
# NAVEGACIÓN
# ------------------------------------------------------------------
seccion = st.sidebar.radio(
    "Navegación",
    ["Dashboard EDA", "Modelo y Métricas", "Predicción de un cliente"],
)

# ==================================================================
# SECCIÓN 1 — DASHBOARD EDA
# ==================================================================
if seccion == "Dashboard EDA":

    st.sidebar.markdown("---")
    st.sidebar.header("Filtros de exploración")

    trabajos_sel = st.sidebar.multiselect(
        "Ocupación / trabajo:",
        sorted(df_raw["job"].unique()),
        default=sorted(df_raw["job"].unique()),
    )
    marital_sel = st.sidebar.multiselect(
        "Estado civil:",
        sorted(df_raw["marital"].unique()),
        default=sorted(df_raw["marital"].unique()),
    )
    educacion_sel = st.sidebar.multiselect(
        "Nivel educativo:",
        sorted(df_raw["education"].unique()),
        default=sorted(df_raw["education"].unique()),
    )
    edad_min, edad_max = int(df_raw["age"].min()), int(df_raw["age"].max())
    rango_edad = st.sidebar.slider("Rango de edad:", edad_min, edad_max, (edad_min, edad_max))

    df = df_raw[
        (df_raw["job"].isin(trabajos_sel))
        & (df_raw["marital"].isin(marital_sel))
        & (df_raw["education"].isin(educacion_sel))
        & (df_raw["age"].between(rango_edad[0], rango_edad[1]))
    ]

    if df.empty:
        st.warning("No hay datos con los filtros seleccionados.")
        st.stop()

    # ---- KPIs ----
    total_clientes = len(df)
    total_yes = int((df["y"] == "yes").sum())
    tasa_conversion = total_yes / total_clientes * 100
    balance_prom = df["balance"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clientes filtrados", f"{total_clientes:,}")
    c2.metric("Suscripciones ('yes')", f"{total_yes:,}")
    c3.metric("Tasa de conversión", f"{tasa_conversion:.2f}%")
    c4.metric("Balance promedio", f"€{balance_prom:,.0f}")

    st.markdown("---")

    # ---- Gráficos categóricos vs y: frecuencia absoluta + porcentaje lado a lado ----
    st.subheader("Suscripción por variable categórica")

    var_cat = st.selectbox(
        "Elegí la variable categórica:",
        CAT_COLS,
        index=CAT_COLS.index("job"),
    )

    col_izq, col_der = st.columns(2)

    with col_izq:
        st.markdown("**Frecuencia absoluta**")
        df_group = df.groupby([var_cat, "y"]).size().reset_index(name="cantidad")
        fig_abs = px.bar(
            df_group, x=var_cat, y="cantidad", color="y", barmode="group",
            labels={var_cat: var_cat, "cantidad": "Número de clientes", "y": "¿Suscribió?"},
            color_discrete_map=PASTEL_MAP,
            category_orders={"month": MONTH_ORDER} if var_cat == "month" else None,
            text="cantidad",
        )
        fig_abs.update_traces(texttemplate="%{text:,}", textposition="outside", cliponaxis=False)
        fig_abs.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig_abs, use_container_width=True)

    with col_der:
        st.markdown("**Porcentaje (tasa de conversión)**")
        rate = (
            df.groupby(var_cat)["y"].apply(lambda x: (x == "yes").mean() * 100)
            .reset_index(name="tasa")
            .sort_values("tasa", ascending=False)
        )
        fig_pct = px.bar(
            rate, x=var_cat, y="tasa",
            labels={var_cat: var_cat, "tasa": "% que suscribió"},
            color="tasa", color_continuous_scale=PASTEL_SCALE,
            text="tasa",
        )
        fig_pct.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
        fig_pct.add_hline(
            y=df["y"].eq("yes").mean() * 100,
            line_dash="dash", line_color=PASTEL_LINE,
            annotation_text="Promedio del grupo filtrado",
        )
        fig_pct.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig_pct, use_container_width=True)

    st.markdown("---")

    # ---- Numéricas vs y ----
    st.subheader("Variables numéricas vs. resultado")
    col_izq2, col_der2 = st.columns(2)
    with col_izq2:
        var_num = st.selectbox("Elegí la variable numérica:", NUM_COLS, index=NUM_COLS.index("duration"))
        fig_box = px.box(
            df, x="y", y=var_num, color="y",
            labels={"y": "¿Suscribió?", var_num: var_num},
            color_discrete_map=PASTEL_MAP,
            points=False,
        )
        st.plotly_chart(fig_box, use_container_width=True)
    with col_der2:
        fig_hist = px.histogram(
            df, x=var_num, color="y", barmode="overlay", opacity=0.7, nbins=40,
            color_discrete_map=PASTEL_MAP,
            labels={"y": "¿Suscribió?"},
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")

    # ---- Correlación ----
    st.subheader("Correlación entre variables numéricas")
    corr = df[NUM_COLS].corr().round(2)
    fig_corr = px.imshow(
        corr, text_auto=True, color_continuous_scale=DIVERGING_SCALE, zmin=-1, zmax=1,
        aspect="auto",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")
    st.subheader("Muestra de datos filtrados")
    st.dataframe(df.head(50), use_container_width=True)

# ==================================================================
# ENTRENAMIENTO DEL MODELO (compartido por secciones 2 y 3)
# ==================================================================
@st.cache_resource
def entrenar_modelos(df_train):
    X = df_train[FEATURE_COLS].copy()
    y = (df_train[TARGET] == "yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocesador = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ]
    )

    modelos = {
        "Regresión Logística": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }

    resultados = {}
    for nombre, modelo in modelos.items():
        pipe = Pipeline([("prep", preprocesador), ("clf", modelo)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        resultados[nombre] = {
            "pipeline": pipe,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "roc_curve": roc_curve(y_test, y_proba),
        }

    return resultados, X_test, y_test


with st.spinner("Entrenando modelos (esto se guarda en caché y solo corre una vez)..."):
    resultados, X_test, y_test = entrenar_modelos(df_raw)

# ==================================================================
# SECCIÓN 2 — MODELO Y MÉTRICAS
# ==================================================================
if seccion == "Modelo y Métricas":

    st.subheader("Comparación de modelos")
    st.markdown(
        "Se entrenaron dos modelos de clasificación con `class_weight='balanced'` "
        "para compensar el desbalance de clases (88.3% no vs. 11.7% sí). "
        "El set de prueba es el 20% de los datos, no visto durante el entrenamiento."
    )

    tabla_metricas = pd.DataFrame({
        nombre: {
            "Accuracy": r["accuracy"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1-score": r["f1"],
            "ROC AUC": r["roc_auc"],
        }
        for nombre, r in resultados.items()
    }).T.round(3)

    st.dataframe(
        tabla_metricas.style.background_gradient(cmap="PuBu", axis=0),
        use_container_width=True,
    )

    mejor_modelo = tabla_metricas["ROC AUC"].idxmax()
    st.success(f"Mejor modelo según ROC AUC: **{mejor_modelo}**")

    st.markdown("---")

    modelo_sel = st.selectbox("Ver detalle del modelo:", list(resultados.keys()))
    r = resultados[modelo_sel]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Matriz de confusión**")
        cm = r["confusion_matrix"]
        fig_cm = px.imshow(
            cm, text_auto=True, color_continuous_scale="Purples",
            x=["Predijo: No", "Predijo: Sí"], y=["Real: No", "Real: Sí"],
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        st.markdown("**Curva ROC**")
        fpr, tpr, _ = r["roc_curve"]
        fig_roc = px.area(
            x=fpr, y=tpr,
            labels={"x": "Tasa de falsos positivos", "y": "Tasa de verdaderos positivos"},
        )
        fig_roc.update_traces(line_color=PASTEL_SINGLE, fillcolor="rgba(184,198,232,0.35)")
        fig_roc.add_shape(type="line", line=dict(dash="dash", color=PASTEL_LINE), x0=0, x1=1, y0=0, y1=1)
        fig_roc.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(fig_roc, use_container_width=True)

    # Importancia de variables (solo para Random Forest, es directo)
    if modelo_sel == "Random Forest":
        st.markdown("**Importancia de variables**")
        pipe = r["pipeline"]
        nombres_ohe = pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(CAT_COLS)
        nombres_features = NUM_COLS + list(nombres_ohe)
        importancias = pipe.named_steps["clf"].feature_importances_
        df_imp = pd.DataFrame({"variable": nombres_features, "importancia": importancias})
        df_imp = df_imp.sort_values("importancia", ascending=False).head(15)
        fig_imp = px.bar(
            df_imp, x="importancia", y="variable", orientation="h",
            text="importancia", color_discrete_sequence=[PASTEL_SINGLE],
        )
        fig_imp.update_traces(texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False)
        fig_imp.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_imp, use_container_width=True)

# ==================================================================
# SECCIÓN 3 — PREDICCIÓN DE UN CLIENTE NUEVO
# ==================================================================
if seccion == "Predicción de un cliente":

    st.subheader("¿Este cliente suscribiría el depósito a plazo?")
    st.markdown("Completá los datos del cliente y el modelo estima la probabilidad de que diga **sí**.")

    modelo_pred_nombre = st.selectbox(
        "Modelo a usar para la predicción:", list(resultados.keys()), index=1
    )
    pipe = resultados[modelo_pred_nombre]["pipeline"]

    with st.form("form_prediccion"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Edad", 18, 100, 40)
            job = st.selectbox("Trabajo", sorted(df_raw["job"].unique()))
            marital = st.selectbox("Estado civil", sorted(df_raw["marital"].unique()))
            education = st.selectbox("Educación", sorted(df_raw["education"].unique()))
            default = st.selectbox("¿Crédito en default?", sorted(df_raw["default"].unique()))
            balance = st.number_input("Balance anual (€)", -10000, 200000, 1000)

        with col2:
            housing = st.selectbox("¿Hipoteca?", sorted(df_raw["housing"].unique()))
            loan = st.selectbox("¿Préstamo personal?", sorted(df_raw["loan"].unique()))
            contact = st.selectbox("Tipo de contacto", sorted(df_raw["contact"].unique()))
            day = st.number_input("Día del mes del contacto", 1, 31, 15)
            month = st.selectbox(
                "Mes del contacto",
                [m for m in MONTH_ORDER if m in df_raw["month"].unique()],
            )
            duration = st.number_input("Duración de la llamada (segundos)", 0, 6000, 200)

        with col3:
            campaign = st.number_input("Contactos en esta campaña", 1, 100, 1)
            pdays = st.number_input("Días desde el último contacto previo (-1 = nunca)", -1, 999, -1)
            previous = st.number_input("Contactos en campañas previas", 0, 100, 0)
            poutcome = st.selectbox("Resultado de campaña previa", sorted(df_raw["poutcome"].unique()))

        enviado = st.form_submit_button("Predecir")

    if enviado:
        cliente = pd.DataFrame([{
            "age": age, "job": job, "marital": marital, "education": education,
            "default": default, "balance": balance, "housing": housing, "loan": loan,
            "contact": contact, "day": day, "month": month, "duration": duration,
            "campaign": campaign, "pdays": pdays, "previous": previous, "poutcome": poutcome,
        }])

        proba = pipe.predict_proba(cliente)[0, 1]
        pred = "SÍ suscribiría" if proba >= 0.5 else "probablemente NO suscribiría"

        st.markdown("---")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Probabilidad de suscripción", f"{proba*100:.1f}%")
        with c2:
            if proba >= 0.5:
                st.success(f"Predicción: **{pred}**")
            else:
                st.warning(f"Predicción: **{pred}**")

        st.progress(min(max(proba, 0.0), 1.0))
