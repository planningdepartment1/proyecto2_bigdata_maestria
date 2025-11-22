import pandas as pd
import numpy as np

from app.data_pipeline import load_and_clean

# ------------------------------
# 1. Columnas que usará tu modelo
# ------------------------------

FEATURE_COLUMNS = [
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "number_of_reviews",
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
    "availability_365",
    "amenities_count",
    "property_type",
    "room_type",
    "cancellation_policy",
    "city",
    "country"
]

TARGET_COLUMN = "price"

# ------------------------------
# 2. Crear el DF final para ML
# ------------------------------

def get_model_dataframe(limit=None):
    df = load_and_clean(limit)

    # Asegurar columnas
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # Seleccionar features + target
    df_model = df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()

    # Eliminar precios nulos
    df_model = df_model[df_model[TARGET_COLUMN].notna()]

    # Restablecer índice
    df_model.reset_index(drop=True, inplace=True)

    return df_model
