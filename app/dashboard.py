import streamlit as st
import pandas as pd
import joblib

from app.data_pipeline import load_and_clean
from app.model_pipeline import FEATURE_COLUMNS

# --------------------------------------------
# CONFIG
# --------------------------------------------
st.set_page_config(page_title="Airbnb Dashboard", layout="wide")

st.title("🏡 Dashboard Airbnb — CosmosDB + Streamlit + MLP")


# --------------------------------------------
# LOAD DATA WITH LIMIT (important for CosmosDB)
# --------------------------------------------
@st.cache_data(ttl=120)  # cache for 2 minutes
def get_data(limit):
    return load_and_clean(limit=limit)

st.sidebar.header("⚙️ Configuración")
limit = st.sidebar.slider("Cantidad de registros (para EDA)", 200, 5000, 1000)

df = get_data(limit)


# --------------------------------------------
# KPIs
# --------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total listings", len(df))
col2.metric("Precio promedio", f"{df['price'].mean():.2f}")
col3.metric("Rating prom.", f"{df['review_scores_rating'].mean():.1f}")
col4.metric("Reviews prom.", f"{df['number_of_reviews'].mean():.1f}")


# --------------------------------------------
# TABS
# --------------------------------------------
tab1, tab2 = st.tabs(["EDA", "Predicción"])


# --------------------------------------------
# EDA TAB
# --------------------------------------------
with tab1:
    st.header("Exploración de Datos")

    with st.expander("📈 Histograma de precios"):
        st.bar_chart(df["price"])

    with st.expander("📉 Precio vs Capacidad"):
        st.scatter_chart(df[["accommodates", "price"]])

    with st.expander("🗺 Mapa de listings (si hay lat/lon)"):
        if "latitude" in df.columns and "longitude" in df.columns:
            st.map(df.rename(columns={"latitude": "lat", "longitude": "lon"}))


# --------------------------------------------
# PREDICTION TAB
# --------------------------------------------
with tab2:
    st.header("Predicción usando modelo MLP (.pkl)")

    model = joblib.load("models/airbnb_mlp.pkl")

    st.subheader("📝 Ingresa datos para predecir el precio")

    inputs = {}
    colA, colB = st.columns(2)

    # numéricas
    with colA:
        for col in FEATURE_COLUMNS:
            if df[col].dtype != "object":
                inputs[col] = st.number_input(
                    col, value=float(df[col].median())
                )

    # categóricas
    with colB:
        for col in FEATURE_COLUMNS:
            if df[col].dtype == "object":
                options = sorted(df[col].dropna().unique())
                if len(options) == 0:
                    options = ["unknown"]
                inputs[col] = st.selectbox(col, options=options)

    if st.button("Calcular precio"):
        df_input = pd.DataFrame([inputs])
        predicted = model.predict(df_input)[0]
        st.success(f"💰 Precio estimado: **{predicted:.2f} USD**")
