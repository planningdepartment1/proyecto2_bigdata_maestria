import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer   # 👈 IMPORTANTE: imputador

from app.model_pipeline import (
    get_model_dataframe,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)

def train_price_model():
    print("Cargando datos desde Cosmos DB...")
    df = get_model_dataframe()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # detectar numéricas y categóricas
    categorical = X.select_dtypes(include=["object"]).columns
    numeric = X.select_dtypes(include=["int64", "float64"]).columns

    # 🔥 IMPUTADORES AGREGADOS ⚠️
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),  # imputar NA
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),  # imputar NA
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric),
            ("cat", categorical_transformer, categorical),
        ]
    )

    # Red neuronal MLP
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(64, 32),
                #activation="relu",
                #solver="adam",
                #learning_rate_init=0.001,
                #alpha=0.0005,
                max_iter=250,
                random_state=42,
            )),
        ]
    )

    print("Entrenando modelo...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    # guardar modelo
    #os.makedirs("models", exist_ok=True)
    #joblib.dump(model, "models/airbnb_mlp.pkl")

    #print("Modelo guardado en models/airbnb_mlp.pkl")

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np

    model.fit(X_train, y_train)

    # Predicciones para evaluar desempeño
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n=== MÉTRICAS DEL MODELO ===")
    print(f"MAE  (Error Absoluto Medio):     {mae:.4f}")
    print(f"RMSE (Raíz Error Cuadrático):    {rmse:.4f}")
    print(f"R²   (Coeficiente de determinación): {r2:.4f}")

    # guardar modelo
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/airbnb_mlp.pkl")

    print("\nModelo guardado en models/airbnb_mlp.pkl")


if __name__ == "__main__":
    train_price_model()


    
