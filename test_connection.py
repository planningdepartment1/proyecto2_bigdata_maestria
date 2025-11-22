from app.data_pipeline import load_and_clean

df = load_and_clean(limit=10)
print(df.head())
print(f"Total registros cargados: {len(df)}")
