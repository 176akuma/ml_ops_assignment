import os
from pathlib import Path
import yaml
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

with open("src/config/params.yaml") as f:
    params = yaml.safe_load(f)
with open("src/config/config.yaml") as f:
    config = yaml.safe_load(f)

csv_path = Path(config["data"]["output_csv"])
df = pd.read_csv(csv_path)

X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["test_size"], random_state=params["random_state"]
)

mlflow.set_experiment(config["mlflow"]["experiment"])
mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

results = []

# Model 1: Linear Regression
with mlflow.start_run(run_name="linreg"):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    preds = lr.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_metric("rmse", float(rmse))
    mlflow.sklearn.log_model(lr, "model")
    results.append(("LinearRegression", rmse, lr))

# Model 2: RandomForest
with mlflow.start_run(run_name="rf"):
    rf = RandomForestRegressor(
        n_estimators=params["rf"]["n_estimators"],
        max_depth=params["rf"]["max_depth"],
        n_jobs=params["rf"]["n_jobs"],
        random_state=params["random_state"],
    )
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_params(params["rf"])
    mlflow.log_metric("rmse", float(rmse))
    mlflow.sklearn.log_model(rf, "model")
    results.append(("RandomForestRegressor", rmse, rf))

# Choose best
best_name, best_rmse, best_model = sorted(results, key=lambda x: x[1])[0]

# Save locally for API
models_dir = Path(config["artifacts"]["models_dir"])
models_dir.mkdir(parents=True, exist_ok=True)
model_path = models_dir / config["artifacts"]["model_file"]
joblib.dump(best_model, model_path)
print(f"Best model: {best_name} RMSE={best_rmse:.4f}; saved to {model_path}")
