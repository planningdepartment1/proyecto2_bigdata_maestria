import sys
import os

# Añadir el directorio raíz al PYTHONPATH para evitar errores de import
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

from app.data_pipeline import load_and_clean

# ---------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ---------------------------------------------------------
st.set_page_config(
    page_title="Airbnb Analytics Dashboard",
    layout="wide"
)

CORAL = "#FF5A5F"
GRAY = "#484848"
BLACK = "#000000"

# Estilos corporativos Airbnb
st.markdown(
    f"""
<style>
body {{
    background-color: #FFFFFF;
    color: {BLACK};
    font-family: "Segoe UI", sans-serif;
}}
h1, h2, h3 {{
    color: {BLACK};
    font-weight: 600;
}}
.card {{
    background-color: #F7F7F7;
    padding: 18px;
    border-radius: 6px;
    border: 1px solid #E1E1E1;
}}
</style>
""",
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# CONFIGURACIÓN DEL SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("Configuración de Datos")

limit = st.sidebar.slider(
    "Cantidad de registros a cargar desde CosmosDB",
    min_value=200,
    max_value=5000,
    value=1000,
    step=200
)

@st.cache_data(ttl=120)
def get_data(limit):
    return load_and_clean(limit)

df = get_data(limit)

# ---------------------------------------------------------
# TÍTULO
# ---------------------------------------------------------
st.title("Airbnb Data Analytics Dashboard")

st.markdown("""
Panel analítico profesional basado en datos reales de Airbnb obtenidos desde
Azure Cosmos DB, con exploración visual y un modelo predictivo basado en MLP.
""")

st.markdown("---")

# ---------------------------------------------------------
# MÉTRICAS | KPI
# ---------------------------------------------------------
st.header("Indicadores Generales")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Registros cargados", len(df))
col2.metric("Precio promedio", f"${df['price'].mean():.2f}")
col3.metric("Capacidad promedio", f"{df['accommodates'].mean():.1f}")
col4.metric("Baños promedio", f"{df['bathrooms'].mean():.1f}")

st.markdown("---")

# ---------------------------------------------------------
# TABS PROFESIONALES
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["Exploración de Datos", "Predicción"])

# =========================================================
# TAB 1 — EDA PROFESIONAL
# =========================================================
with tab1:

    # -----------------------------------------------------
    # BOX: Precio según número de baños
    # -----------------------------------------------------
    st.subheader("Precio según número de baños")
    fig1 = px.box(
        df,
        x="bathrooms_int",
        y="price",
        color_discrete_sequence=[CORAL],
        labels={"bathrooms_int": "Número de baños", "price": "Precio (USD)"}
    )
    fig1.update_layout(showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

    # -----------------------------------------------------
    # BOX: Precio según número de habitaciones
    # -----------------------------------------------------
    st.subheader("Precio según número de habitaciones")
    fig2 = px.box(
        df,
        x="bedrooms",
        y="price",
        color_discrete_sequence=[GRAY],
        labels={"bedrooms": "Habitaciones", "price": "Precio (USD)"}
    )
    fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    # -----------------------------------------------------
    # BOX: Precio según capacidad
    # -----------------------------------------------------
    st.subheader("Precio según capacidad del alojamiento")
    fig3 = px.box(
        df,
        x="accommodates",
        y="price",
        color_discrete_sequence=[CORAL],
        labels={"accommodates": "Capacidad (huéspedes)", "price": "Precio (USD)"}
    )
    fig3.update_layout(showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

    # -----------------------------------------------------
    # BAR CHART: Precio promedio por ciudad
    # -----------------------------------------------------
    st.subheader("Precio promedio por ciudad")

    city_avg = (
        df.groupby("city")["price"]
        .mean()
        .reset_index()
        .sort_values("price", ascending=False)
    )

    fig_city = px.bar(
        city_avg,
        x="city",
        y="price",
        color="price",
        color_continuous_scale="Blues",
        labels={"city": "Ciudad", "price": "Precio promedio (USD)"},
    )
    fig_city.update_layout(
        xaxis_tickangle=-45,
        showlegend=False
    )
    st.plotly_chart(fig_city, use_container_width=True)

    # -----------------------------------------------------
    # MAPA PROFESIONAL (Mapbox / OpenStreetMaps)
    # -----------------------------------------------------
    st.subheader("Mapa de propiedades Airbnb")

    if "latitude" in df.columns and "longitude" in df.columns:

        fig_map = px.scatter_mapbox(
            df,
            lat="latitude",
            lon="longitude",
            color="price",
            size="price",
            color_continuous_scale="Blues",
            zoom=1.2,
            height=550,
            hover_name="city",
            hover_data={"price": True},
        )

        fig_map.update_layout(
            mapbox_style="open-street-map",
            margin={"r": 0, "t": 0, "l": 0, "b": 0}
        )

        st.plotly_chart(fig_map, use_container_width=True)

# =========================================================
# TAB 2 — PREDICCIÓN CON MLP
# =========================================================
with tab2:
    st.header("Predicción de Precio usando MLP")

    model_path = "models/airbnb_mlp.pkl"
    if not os.path.exists(model_path):
        st.error("El modelo no está entrenado. Entrénalo antes de usar esta sección.")
    else:
        model = joblib.load(model_path)

        st.subheader("Ingrese las características del alojamiento")

        with st.form("predict_form"):
            colA, colB = st.columns(2)

            # Valores numéricos
            with colA:
                accommodates = st.number_input("Capacidad (huéspedes)", 1, 16, 2)
                bedrooms = st.number_input("Habitaciones", 0, 8, 1)
                beds = st.number_input("Camas", 0, 10, 1)
                bathrooms = st.number_input("Baños", min_value=0.0, step=0.5, value=1.0)
                amenities_count = st.number_input("Amenidades", 0, 50, 5)
                number_of_reviews = st.number_input("Reseñas", 0, 500, 10)

            # Valores categóricos
            with colB:
                property_type = st.selectbox("Tipo de propiedad", sorted(df["property_type"].unique()))
                room_type = st.selectbox("Tipo de habitación", sorted(df["room_type"].unique()))
                cancellation_policy = st.selectbox("Política de cancelación", sorted(df["cancellation_policy"].unique()))
                city = st.selectbox("Ciudad", sorted(df["city"].unique()))
                country = st.selectbox("País", sorted(df["country"].unique()))

            submitted = st.form_submit_button("Calcular precio")

        if submitted:
            X = {
                "accommodates": accommodates,
                "bedrooms": bedrooms,
                "beds": beds,
                "bathrooms": bathrooms,
                "number_of_reviews": number_of_reviews,
                "review_scores_rating": df["review_scores_rating"].mean(),
                "review_scores_accuracy": df["review_scores_accuracy"].mean(),
                "review_scores_cleanliness": df["review_scores_cleanliness"].mean(),
                "review_scores_checkin": df["review_scores_checkin"].mean(),
                "review_scores_communication": df["review_scores_communication"].mean(),
                "review_scores_location": df["review_scores_location"].mean(),
                "review_scores_value": df["review_scores_value"].mean(),
                "availability_365": df["availability_365"].mean(),
                "amenities_count": amenities_count,
                "property_type": property_type,
                "room_type": room_type,
                "cancellation_policy": cancellation_policy,
                "city": city,
                "country": country,
            }

            df_input = pd.DataFrame([X])
            pred = model.predict(df_input)[0]

            st.markdown("### Precio estimado")
            st.markdown(
                f"""
                <div class="card">
                    <h2 style="color:{CORAL}; margin:0;">USD ${pred:,.2f}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
