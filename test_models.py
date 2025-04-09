import joblib
from pathlib import Path
import sys

# Print arbejdsmappe
print(f"Current working directory: {Path.cwd()}")

# Definér sti til modeller
models_dir = Path("models")
print(f"Models directory path: {models_dir.absolute()}")
print(f"Models directory exists: {models_dir.exists()}")

# Liste modelfilerne
if models_dir.exists():
    model_files = list(models_dir.glob("*.joblib"))
    print(f"Found {len(model_files)} .joblib files:")
    for file in model_files:
        print(f"  - {file.name}")
else:
    print("Models directory not found!")
    sys.exit(1)

# Definér modelstier
model_paths = {
    "1day": models_dir / "xgboost_model_1d.joblib",
    "3day": models_dir / "xgboost_model_3d.joblib",
    "7day": models_dir / "xgboost_model_7d.joblib"
}
scaler_path = models_dir / "scaler.joblib"
feature_names_path = models_dir / "feature_names.joblib"

# Test indlæsningen af modeller
models = {}
print("\nTesting model loading:")
for horizon, path in model_paths.items():
    print(f"Loading model for {horizon} from {path}...")
    try:
        if path.exists():
            model = joblib.load(path)
            models[horizon] = model
            print(f"✅ Successfully loaded model for {horizon}")
        else:
            print(f"❌ Model file not found: {path}")
    except Exception as e:
        print(f"❌ Error loading model for {horizon}: {str(e)}")

# Test indlæsningen af scaler
print("\nTesting scaler loading:")
try:
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        print(f"✅ Successfully loaded scaler")
    else:
        print(f"❌ Scaler file not found: {scaler_path}")
except Exception as e:
    print(f"❌ Error loading scaler: {str(e)}")

# Test indlæsningen af feature names
print("\nTesting feature names loading:")
try:
    if feature_names_path.exists():
        feature_names = joblib.load(feature_names_path)
        print(f"✅ Successfully loaded feature names: {feature_names}")
    else:
        print(f"❌ Feature names file not found: {feature_names_path}")
except Exception as e:
    print(f"❌ Error loading feature names: {str(e)}")

# Opsummer resultater
print(f"\nSummary:")
print(f"Models loaded: {len(models)}/{len(model_paths)}")
print(f"Models: {list(models.keys())}")
for horizon, model in models.items():
    print(f"  - {horizon}: {type(model).__name__}") 