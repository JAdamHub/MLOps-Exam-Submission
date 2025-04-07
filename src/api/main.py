from fastapi import FastAPI, HTTPException
from .models import InputFeatures, PredictionResponse
import joblib
import pandas as pd
import logging
from pathlib import Path
import sys
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration & Model Loading ---
# Determine project root based on script location
# Assumes the script is in src/api
PROJECT_ROOT = Path(__file__).resolve().parents[2] # Corrected: Go up two levels to project root
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILE_PATH = MODELS_DIR / "logistic_regression_model.joblib"
# SCALER_FILE_PATH = MODELS_DIR / "minmax_scaler.joblib" # We don't load scaler here due to simplified input

# --- Global Variables for Model --- (Load once at startup)
model = None
expected_features = None # Store the feature order expected by the model

app = FastAPI(
    title="Cryptocurrency Price Predictor API",
    description="API to predict if the cryptocurrency price will go up the next day.",
    version="0.1.0"
)

@app.on_event("startup")
def load_model():
    """Load the trained model when the API starts."""
    global model, expected_features
    logging.info("--- Loading Model --- ")
    if not MODEL_FILE_PATH.exists():
        logging.error(f"Model file not found at {MODEL_FILE_PATH}. Train the model first.")
        # Depending on deployment, might want to raise an error or exit
        # For now, we log and the predict endpoint will fail gracefully.
        model = None
        expected_features = []
        return

    try:
        model = joblib.load(MODEL_FILE_PATH)
        # Store the feature names used during training (important for order)
        if hasattr(model, 'feature_names_in_'):
            expected_features = list(model.feature_names_in_)
            logging.info(f"Model loaded successfully from {MODEL_FILE_PATH}. Expected features: {expected_features}")
        else:
            # Fallback if feature names aren't stored (less robust)
            logging.warning("Model loaded, but feature names attribute not found. Prediction might fail if input order is incorrect.")
            expected_features = [] # Indicate uncertainty about feature order

        # --- Scaler Loading (Commented Out - Not needed with current simplified input) ---
        # global scaler
        # if not SCALER_FILE_PATH.exists():
        #     logging.error(f"Scaler file not found at {SCALER_FILE_PATH}. Run preprocessing first.")
        #     scaler = None
        #     return
        # scaler = joblib.load(SCALER_FILE_PATH)
        # logging.info("Scaler loaded successfully.")

    except Exception as e:
        logging.error(f"Error loading model or scaler: {e}", exc_info=True)
        model = None
        expected_features = []
        # scaler = None

@app.get("/")
def read_root():
    """Root endpoint providing basic API information."""
    return {"message": "Welcome to the Crypto Price Predictor API. Use the /predict endpoint."}

@app.post("/predict", response_model=PredictionResponse)
def predict(features: InputFeatures):
    """Predicts if the price will go up (1) or not (0)."""
    global model, expected_features
    if model is None:
        logging.error("Model not loaded. Cannot predict.")
        raise HTTPException(status_code=503, detail="Model is not available. Please check server logs.")

    try:
        # Convert Pydantic model to DataFrame
        input_data = pd.DataFrame([features.dict()])
        logging.debug(f"Received input data: \n{input_data}")

        # --- Feature Scaling (Commented Out - Not needed with current simplified input) ---
        # if scaler is None:
        #     logging.error("Scaler not loaded. Cannot preprocess input.")
        #     raise HTTPException(status_code=503, detail="Scaler is not available.")
        # try:
        #     # Ensure columns are in the same order as during scaling fit
        #     numerical_cols = scaler.feature_names_in_ # Get columns scaler was fit on
        #     input_data_scaled = scaler.transform(input_data[numerical_cols])
        #     input_data[numerical_cols] = input_data_scaled
        #     logging.debug("Input data scaled successfully.")
        # except Exception as e:
        #     logging.error(f"Error scaling input data: {e}")
        #     raise HTTPException(status_code=400, detail="Error processing input features for scaling.")

        # Ensure feature order matches the model's expectation
        if expected_features:
            try:
                input_data = input_data[expected_features] # Reorder columns
                logging.debug("Input data columns reordered to match model expectation.")
            except KeyError as e:
                logging.error(f"Missing expected feature in input: {e}")
                raise HTTPException(status_code=400, detail=f"Missing expected feature: {e}. Expected: {expected_features}")
        else:
            logging.warning("Model feature order unknown, predicting with received order. May be inaccurate.")

        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0, 1] # Probability of class 1 (price up)

        logging.info(f"Prediction successful: Class={prediction}, Probability={probability:.4f}")

        return PredictionResponse(prediction=int(prediction), probability=float(probability))

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions directly
        raise http_exc
    except Exception as e:
        logging.error(f"Error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error during prediction: {str(e)}")

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
