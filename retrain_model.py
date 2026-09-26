import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# --------------------------------------------------
# 1. Load 2026 dataset
# --------------------------------------------------

data_path = "dataset/waste_data.csv"

df = pd.read_csv(data_path)

print("=" * 60)
print("RETRAINING WASTE PREDICTION MODEL")
print("=" * 60)

print(f"Records loaded: {len(df):,}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")


# --------------------------------------------------
# 2. Features and target
# --------------------------------------------------

X = df[
    [
        "Population",
        "Collection_Vehicles",
        "Temperature_C",
        "Rainfall_mm",
        "Area",
        "Waste_Type"
    ]
]

y = df["Waste_Collected_kg"]


# --------------------------------------------------
# 3. Categorical columns
# --------------------------------------------------

categorical_features = [
    "Area",
    "Waste_Type"
]

numeric_features = [
    "Population",
    "Collection_Vehicles",
    "Temperature_C",
    "Rainfall_mm"
]


# --------------------------------------------------
# 4. Preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)


# --------------------------------------------------
# 5. Linear Regression model
# --------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)


# --------------------------------------------------
# 6. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# --------------------------------------------------
# 7. Train
# --------------------------------------------------

print("\nTraining model...")

model.fit(X_train, y_train)


# --------------------------------------------------
# 8. Evaluate
# --------------------------------------------------

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMODEL RESULTS")
print("-" * 40)
print(f"MAE : {mae:.2f} kg")
print(f"R²  : {r2:.4f}")


# --------------------------------------------------
# 9. Save model
# --------------------------------------------------

model_path = "src/waste_prediction_model.pkl"

joblib.dump(model, model_path)

print("\nMODEL SAVED SUCCESSFULLY")
print(f"File: {model_path}")

print("=" * 60)