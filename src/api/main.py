from fastapi import FastAPI, HTTPException
from .models import InputFeatures, PredictionResponse
import joblib
import pandas as pd
import logging
from pathlib import Path
import sys
import numpy as np
from datetime import datetime

# Tilføj project root til Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.monitoring.prediction_store import PredictionStore

# Konfigurer logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration & Model Loading ---
MODELS_DIR = project_root / "models"
MODEL_FILE_PATH = MODELS_DIR / "xgboost_model.joblib"
SCALER_FILE_PATH = MODELS_DIR / "scaler.joblib"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.joblib"

# --- Global Variables for Model ---
model = None
scaler = None
expected_features = None
feature_names = None
prediction_store = None

app = FastAPI(
    title="Cryptocurrency Price Predictor API",
    description="API to predict if the cryptocurrency price will go up the next day.",
    version="0.1.0"
)

@app.on_event("startup")
def load_model():
    """Load the trained model when the API starts."""
    global model, scaler, expected_features, feature_names, prediction_store
    logging.info("--- Loading Model --- ")
    
    try:
        # Initialiser prediction store
        prediction_store = PredictionStore()
        
        if not MODEL_FILE_PATH.exists():
            logging.error(f"Model file not found at {MODEL_FILE_PATH}. Train the model first.")
            model = None
            scaler = None
            expected_features = []
            feature_names = []
            return

        # Load model
        model = joblib.load(MODEL_FILE_PATH)
        if hasattr(model, 'feature_names_in_'):
            expected_features = list(model.feature_names_in_)
            logging.info(f"Model loaded successfully from {MODEL_FILE_PATH}. Expected features: {expected_features}")
        else:
            logging.warning("Model loaded, but feature names attribute not found. Prediction might fail if input order is incorrect.")
            expected_features = []

        # Load scaler
        if not SCALER_FILE_PATH.exists():
            logging.error(f"Scaler file not found at {SCALER_FILE_PATH}. Run preprocessing first.")
            scaler = None
            return
        scaler = joblib.load(SCALER_FILE_PATH)
        logging.info("Scaler loaded successfully.")
        
        # Load feature names
        if not FEATURE_NAMES_PATH.exists():
            logging.error(f"Feature names file not found at {FEATURE_NAMES_PATH}. Train the model first.")
            feature_names = []
            return
        feature_names = joblib.load(FEATURE_NAMES_PATH)
        logging.info(f"Feature names loaded successfully: {feature_names}")

    except Exception as e:
        logging.error(f"Error loading model or scaler: {e}", exc_info=True)
        model = None
        scaler = None
        expected_features = []
        feature_names = []

@app.get("/")
def read_root():
    """Root endpoint providing basic API information."""
    return {"message": "Welcome to the Crypto Price Predictor API. Use the /predict endpoint."}

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: InputFeatures):
    """Make a prediction using the trained model."""
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    try:
        # Konverter input features til dictionary
        feature_dict = features.dict()
        
        # Opret feature array i korrekt rækkefølge
        feature_array = np.array([feature_dict[feature] for feature in expected_features]).reshape(1, -1)
        
        # Skaler features
        scaled_features = scaler.transform(feature_array)
        
        # Lav prædiktion
        prediction = model.predict(scaled_features)[0]
        prediction_proba = model.predict_proba(scaled_features)[0]
        
        # Gem prædiktion
        prediction_store.store_prediction(
            prediction=prediction,
            actual_value=None,  # Vi kender ikke den faktiske værdi endnu
            features_used=feature_dict,
            model_version="1.0"  # Dette bør hentes fra model metadata
        )
        
        return PredictionResponse(
            prediction=prediction,
            confidence=float(prediction_proba[1]),
            timestamp=datetime.now().isoformat(),
            features_used=expected_features
        )
        
    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Check if the API is healthy."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.get("/metrics")
async def get_metrics():
    """Get model performance metrics."""
    if prediction_store is None:
        raise HTTPException(status_code=500, detail="Prediction store not initialized")
        
    try:
        metrics = prediction_store.get_prediction_metrics()
        return metrics
    except Exception as e:
        logging.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Optional: Add endpoint to view loaded model info ---
@app.get("/model_info")
def get_model_info():
    if model is None:
        return {"status": "Model not loaded"}
    return {
        "status": "Model loaded",
        "model_type": str(type(model)),
        "expected_features": expected_features if expected_features else "Order unknown",
        # Add other relevant info if available, e.g., model parameters
        # "params": model.get_params() if hasattr(model, 'get_params') else {}
    }

# To run the API locally: cd src/api && uvicorn main:app --reload --port 8000
