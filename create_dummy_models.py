import numpy as np
import pandas as pd
import joblib
import json
import os
from pathlib import Path
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler

# Opret models-mappen hvis den ikke findes
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

print("Genererer dummy-modelartefakter...")

# Definer features, der skal bruges af modellerne
feature_names = [
    "close", "open", "high", "low", "volume", "market_cap", 
    "sma_7", "sma_30", "rsi_14", "macd", "bbands_upper", "bbands_lower"
]

# Gem feature names fil
feature_names_file = models_dir / "feature_names.joblib"
joblib.dump(feature_names, feature_names_file)
print(f"Feature names gemt til {feature_names_file}")

# Gem target columns
target_columns = ['price_target_1d', 'price_target_3d', 'price_target_7d']
target_columns_file = models_dir / "target_columns.joblib"
joblib.dump(target_columns, target_columns_file)
print(f"Target columns gemt til {target_columns_file}")

# Opret en dummy XGBoost-model for hver tidshorisont
horizon_mapping = {
    "1day": "xgboost_model_1d.joblib",
    "3day": "xgboost_model_3d.joblib",
    "7day": "xgboost_model_7d.joblib"
}

for horizon, filename in horizon_mapping.items():
    # Opret en simpel XGBoost regression model
    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=10)
    
    # Træn modellen på dummy data
    X = np.random.rand(100, len(feature_names))
    y = np.random.rand(100) * 1000 + 60000  # Tilfældige prisværdier omkring 60.000
    
    # Tilpas feature_names_in_ attributten
    model.fit(X, y)
    
    # Gem modellen
    model_file = models_dir / filename
    joblib.dump(model, model_file)
    print(f"Dummy model gemt til {model_file}")

# Opret også en generel model
general_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=10)
general_model.fit(X, y) 
joblib.dump(general_model, models_dir / "xgboost_model.joblib")
print(f"Generel dummy model gemt til {models_dir / 'xgboost_model.joblib'}")

# Opret en MinMaxScaler
scaler = MinMaxScaler()
dummy_data = np.random.rand(100, len(feature_names))
scaler.fit(dummy_data)

# Gem scaler
for scaler_name in ["scaler.joblib", "minmax_scaler.joblib"]:
    scaler_file = models_dir / scaler_name
    joblib.dump(scaler, scaler_file)
    print(f"Dummy scaler gemt til {scaler_file}")

# Opret dummy metrics
training_metrics = {
    "accuracy": 0.85,
    "precision": 0.83,
    "recall": 0.82,
    "f1_score": 0.82,
    "timestamp": "2023-04-09T18:20:00"
}
with open(models_dir / "training_metrics.json", "w") as f:
    json.dump(training_metrics, f, indent=2)
print(f"Dummy training metrics gemt til {models_dir / 'training_metrics.json'}")

print("Alle dummy-modelartefakter er genereret!") 