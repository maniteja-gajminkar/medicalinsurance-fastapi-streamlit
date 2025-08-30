import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score

def train_model(config):
    df = pd.read_csv(config["data"]["path"])
    X = df.drop("charges", axis=1)
    y = df["charges"]

    categorical = ["sex", "smoker", "region"]
    numerical = ["age", "bmi", "children"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=config["model"]["n_estimators"],
            random_state=config["model"]["random_state"]
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"✅ Model trained successfully")
    print(f"📊 MSE: {mse:.2f}")
    print(f"📈 R² Score: {r2:.2f}")

    os.makedirs(os.path.dirname(config["output"]["model_path"]), exist_ok=True)
    joblib.dump(model, config["output"]["model_path"])
    print(f"💾 Model saved to {config['output']['model_path']}")