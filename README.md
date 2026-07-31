# 🏦 Dashboard Bank Marketing — Streamlit

App de Streamlit basada en el análisis del Colab `Proyecto_Grupo_8`. Convierte
el EDA y el objetivo del proyecto (predecir si un cliente suscribe un depósito
a plazo fijo) en una app interactiva con tres secciones:

1. **📊 Dashboard EDA** — filtros, KPIs y gráficos interactivos (los mismos
   que ya tenías esbozados en el Colab, más histogramas, boxplots y
   correlación).
2. **🤖 Modelo y Métricas** — entrena Regresión Logística y Random Forest
   (con `class_weight='balanced'` por el desbalance 88.3%/11.7%), y muestra
   accuracy, precision, recall, F1, ROC AUC, matriz de confusión, curva ROC
   e importancia de variables.
3. **🔮 Predicción de un cliente** — formulario para ingresar los datos de
   un cliente nuevo y obtener la probabilidad de que suscriba.

## Cómo correrla

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Datos

Colocá el archivo **`bank-full.csv`** (separado por `;`, el mismo que usás
en el Colab) en la misma carpeta que `app.py`. Si no lo encuentra ahí, la
app te va a mostrar un botón en la barra lateral para subirlo manualmente
— así no dependés de Google Drive ni de montar el Colab.

## Diferencias respecto al boceto original del Colab

- No usa `google.colab.drive` ni `ngrok`: corre como cualquier app de
  Streamlit local o se puede desplegar directo en **Streamlit Community
  Cloud** subiendo este repo (con el CSV incluido o pidiéndolo por upload).
- Agrega la parte de Machine Learning que el Colab todavía no tenía
  desarrollada (el notebook llegaba hasta el EDA): entrenamiento,
  evaluación y un formulario de predicción en vivo.
- El modelo se entrena una sola vez gracias a `@st.cache_resource` y se
  reutiliza entre secciones sin volver a correr desde cero cada vez que
  cambiás un filtro.

## Desplegar en Streamlit Community Cloud (gratis)

1. Subí esta carpeta (`app.py`, `requirements.txt`, y opcionalmente
   `bank-full.csv`) a un repo de GitHub.
2. Entrá a share.streamlit.io, conectá el repo y elegí `app.py` como
   archivo principal.
3. Si no incluiste el CSV en el repo (por tamaño o privacidad), la app te
   va a pedir subirlo la primera vez que entre cualquier usuario a esa
   sesión.
