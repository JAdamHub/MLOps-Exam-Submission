from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from src.api.models import InputFeatures, PredictionResponse
import joblib
import pandas as pd
import logging
from pathlib import Path
import sys
import numpy as np
from datetime import datetime
import time
import os

# Miljøvariabel til at styre om vi er i produktion eller udvikling
ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")

# Tilføj project root til Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.monitoring.prediction_store import PredictionStore
from src.monitoring.scheduler import ModelUpdateScheduler
from src.monitoring.drift_detector import DriftDetector
from src.monitoring.evaluation import ModelEvaluator

# Konfigurer logging baseret på miljø
if ENVIRONMENT == "production":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger("bitcoin-predictor-api")

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
scheduler = None
drift_detector = None
model_evaluator = None

app = FastAPI(
    title="Cryptocurrency Price Predictor API",
    description="API to predict if the cryptocurrency price will go up the next day.",
    version="0.1.0"
)

# Tilføj CORS-middleware for at tillade cross-origin requests fra Streamlit app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tillad alle origins i dette eksempel (ikke sikker i produktion)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware til at logge requests og timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    logger.info(f"Request processed: {request.method} {request.url.path} - {process_time:.4f}s")
    return response

@app.on_event("startup")
def load_model():
    """Load the trained model when the API starts."""
    global model, scaler, expected_features, feature_names, prediction_store, scheduler, drift_detector, model_evaluator
    logger.info("--- Starting API and loading model --- ")
    
    try:
        # Initialiser prediction store
        prediction_store = PredictionStore()
        
        # Initialiser drift detektor
        drift_detector = DriftDetector()
        
        # Initialiser model evaluator
        model_evaluator = ModelEvaluator()
        
        # Start model update scheduler i production mode
        if ENVIRONMENT == "production":
            logger.info("Starting model update scheduler in production mode")
            scheduler = ModelUpdateScheduler()
            scheduler.start()
        
        # Load model
        if not MODEL_FILE_PATH.exists():
            logger.error(f"Model file not found at {MODEL_FILE_PATH}. Train the model first.")
            model = None
            scaler = None
            expected_features = []
            feature_names = []
            return

        # Load model
        model = joblib.load(MODEL_FILE_PATH)
        if hasattr(model, 'feature_names_in_'):
            expected_features = list(model.feature_names_in_)
            logger.info(f"Model loaded successfully from {MODEL_FILE_PATH}. Expected features: {expected_features}")
        else:
            logger.warning("Model loaded, but feature names attribute not found. Prediction might fail if input order is incorrect.")
            expected_features = []

        # Load scaler
        if not SCALER_FILE_PATH.exists():
            logger.error(f"Scaler file not found at {SCALER_FILE_PATH}. Run preprocessing first.")
            scaler = None
            return
        scaler = joblib.load(SCALER_FILE_PATH)
        logger.info("Scaler loaded successfully.")
        
        # Load feature names
        if not FEATURE_NAMES_PATH.exists():
            logger.error(f"Feature names file not found at {FEATURE_NAMES_PATH}. Train the model first.")
            feature_names = []
            return
        feature_names = joblib.load(FEATURE_NAMES_PATH)
        logger.info(f"Feature names loaded successfully: {feature_names}")

    except Exception as e:
        logger.error(f"Error loading model or starting monitoring: {e}", exc_info=True)
        model = None
        scaler = None
        expected_features = []
        feature_names = []

@app.on_event("shutdown")
async def shutdown_event():
    """Stop the scheduler when the API is shutting down"""
    global scheduler
    
    if scheduler is not None:
        logger.info("Stopping model update scheduler")
        scheduler.stop()
    
    logger.info("API shutting down")

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
        
        # Tjek for data drift hvis drift detector er initialiseret
        if drift_detector is not None and ENVIRONMENT == "production":
            try:
                # Lav et mini-dataset med den nye observation
                mini_df = pd.DataFrame([feature_dict])
                # Kør drift detection i en separat tråd for ikke at blokere API kald
                # Dette vil kun logge resultatet, ikke stoppe prædiktionen
                import threading
                threading.Thread(
                    target=lambda: drift_detector.detect_drift(mini_df),
                    daemon=True
                ).start()
            except Exception as drift_e:
                logger.warning(f"Failed to run drift detection: {drift_e}")
        
        # Skaler features
        scaled_features = scaler.transform(feature_array)
        
        # Lav prædiktion
        prediction = model.predict(scaled_features)[0]
        prediction_proba = model.predict_proba(scaled_features)[0]
        
        # Gem prædiktion i prediction store
        if prediction_store is not None:
            prediction_store.store_prediction(
                prediction=prediction,
                actual_value=None,  # Vi kender ikke den faktiske værdi endnu
                features_used=feature_dict,
                model_version="1.0"  # Dette bør hentes fra model metadata
            )
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(prediction_proba[1]),
            timestamp=datetime.now().isoformat(),
            features_used=expected_features
        )
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
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
        logger.error(f"Error getting metrics: {e}")
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

@app.get("/monitoring/drift")
async def get_drift_status():
    """Get status of drift detection."""
    if drift_detector is None:
        raise HTTPException(status_code=500, detail="Drift detector not initialized")
        
    try:
        # Hent seneste data til drift detection
        latest_data_path = project_root / "data" / "features" / "bitcoin_features.csv"
        if not latest_data_path.exists():
            return {"status": "No recent data available for drift detection"}
            
        latest_data = pd.read_csv(latest_data_path)
        # Brug kun de seneste 7 dages data
        latest_data = latest_data.tail(7)
        
        # Udfør drift detection
        drift_results = drift_detector.detect_drift(latest_data)
        
        # Tilføj anbefaling om genoptræning
        needs_retraining = drift_detector.should_retrain(drift_results)
        drift_results["recommendation"] = {
            "needs_retraining": needs_retraining,
            "reason": "Significant drift detected in data distribution" if needs_retraining else "No significant drift detected"
        }
        
        return drift_results
    
    except Exception as e:
        logger.error(f"Error in drift detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitoring/performance")
async def get_model_performance():
    """Get performance metrics of the model."""
    if model_evaluator is None:
        raise HTTPException(status_code=500, detail="Model evaluator not initialized")
        
    try:
        # Hent model performance metrics
        performance_summary = model_evaluator.get_metrics_summary()
        return performance_summary
    
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/monitoring/retrain")
async def trigger_retraining():
    """Manually trigger model retraining."""
    if scheduler is None:
        raise HTTPException(status_code=500, detail="Scheduler not initialized")
        
    try:
        # Trigger retraining
        import threading
        threading.Thread(
            target=scheduler.retrain_model,
            daemon=True
        ).start()
        
        return {"status": "Retraining initiated in the background"}
    
    except Exception as e:
        logger.error(f"Error triggering retraining: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitoring/predictions")
async def get_prediction_history(days: int = 7):
    """Get history of recent predictions."""
    if prediction_store is None:
        raise HTTPException(status_code=500, detail="Prediction store not initialized")
        
    try:
        # Hent seneste prædiktioner
        recent_predictions = prediction_store.get_recent_predictions(days=days)
        return {"predictions": recent_predictions.to_dict(orient="records")}
        
    except Exception as e:
        logger.error(f"Error getting prediction history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# To run the API locally: cd src/api && uvicorn main:app --reload --port 8000
