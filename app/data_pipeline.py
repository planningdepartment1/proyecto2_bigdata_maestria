import os
import pandas as pd
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------
# 1. Conexión a Cosmos DB (Mongo API)
# -----------------------------------------------------------

def get_collection():
    uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB", "sample_airbnb")
    coll_name = os.getenv("MONGO_COLLECTION", "listingsAndReviews")

    client = MongoClient(uri)
    db = client[db_name]
    collection = db[coll_name]
    return collection


# -----------------------------------------------------------
# 2. Convertir $numberDecimal y $date RECURSIVAMENTE
# -----------------------------------------------------------

def convert_extended_json(value):
    """Convierte objetos como {'$numberDecimal': '80.00'} o {'$date': '...'}."""
    # $numberDecimal
    if isinstance(value, dict) and "$numberDecimal" in value:
        return float(value["$numberDecimal"])

    # $date
    if isinstance(value, dict) and "$date" in value:
        return pd.to_datetime(value["$date"], errors="coerce")

    # Diccionario anidado
    if isinstance(value, dict):
        return {k: convert_extended_json(v) for k, v in value.items()}

    # Lista
    if isinstance(value, list):
        return [convert_extended_json(v) for v in value]

    # Valor simple
    return value


# -----------------------------------------------------------
# 3. Leer datos desde Cosmos y convertir Extended JSON
# -----------------------------------------------------------

def load_raw_data(limit=None):
    coll = get_collection()

    cursor = coll.find({})
    if limit:
        cursor = cursor.limit(limit)

    docs = list(cursor)

    # Convertir cada documento extendido
    docs = [convert_extended_json(doc) for doc in docs]

    df = pd.json_normalize(docs, max_level=5)
    return df


# -----------------------------------------------------------
# 4. Limpieza final del DataFrame
# -----------------------------------------------------------

def clean_data(df):
    df = df.copy()

    # Renombrar ID
    if "_id" in df.columns:
        df["listing_id"] = df["_id"].astype(str)

    # Lat / Lon (vienen como [lon, lat])
    if "address.location.coordinates" in df.columns:
        df["longitude"] = df["address.location.coordinates"].apply(
            lambda x: x[0] if isinstance(x, list) else np.nan
        )
        df["latitude"] = df["address.location.coordinates"].apply(
            lambda x: x[1] if isinstance(x, list) else np.nan
        )

    # Price ya viene como float por convert_extended_json
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Bathrooms igual
    if "bathrooms" in df.columns:
        df["bathrooms"] = pd.to_numeric(df["bathrooms"], errors="coerce")

    # Reviews score
    score_cols = [
        "review_scores.review_scores_rating",
        "review_scores.review_scores_accuracy",
        "review_scores.review_scores_cleanliness",
        "review_scores.review_scores_checkin",
        "review_scores.review_scores_communication",
        "review_scores.review_scores_location",
        "review_scores.review_scores_value",
    ]

    for col in score_cols:
        if col in df.columns:
            df[col.split(".")[-1]] = pd.to_numeric(df[col], errors="coerce")

    # availability
    if "availability.availability_365" in df.columns:
        df["availability_365"] = df["availability.availability_365"]

    # amenities_count
    df["amenities_count"] = df["amenities"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )

    # Filtrar filas válidas de precio
    df = df[df["price"].notna()]
    df = df[df["price"] > 0]

    return df


# -----------------------------------------------------------
# 5. Función pública usada por test_connection.py
# -----------------------------------------------------------

def load_and_clean(limit=None):
    df_raw = load_raw_data(limit)
    df_clean = clean_data(df_raw)
    return df_clean
