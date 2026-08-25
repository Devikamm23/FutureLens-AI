import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

DATA_FILE = "data/college_data.csv"
MODEL_FILE = "models/facility_models.pkl"

os.makedirs("models", exist_ok=True)

data = pd.read_csv(DATA_FILE)

features = [
    "students",
    "vehicles",
    "temperature",
    "working_hours"
]

targets = [
    "benches",
    "washrooms",
    "classrooms",
    "labs",
    "library_seats",
    "parking",
    "canteen_seats"
]

X = data[features]

models = {}
scores = {}

for target in targets:

    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    models[target] = model
    scores[target] = round(mae, 2)

joblib.dump(
    {
        "models": models,
        "features": features,
        "scores": scores
    },
    MODEL_FILE
)

print("\n======================================")
print(" FUTURELENS AI MODEL TRAINED")
print("======================================")

for target, score in scores.items():
    print(f"{target:18} MAE: {score}")

print("\nModel saved to:")
print(MODEL_FILE)