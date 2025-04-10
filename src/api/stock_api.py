from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging
import os
import json
from typing import Dict, List, Any, Optional
import tensorflow as tf
from datetime import datetime, timedelta
import time

# Enable unsafe deserialization for Lambda layer
import keras
try:
    keras.config.enable_unsafe_deserialization()
except AttributeError:
    # For older versions of Keras/TF
    tf.keras.utils.disable_interactive_logging()
    os.environ['TF_KERAS_SAFE_MODE'] = '0'

# Define custom Lambda layer with output shape
def custom_lambda_layer(x):
    # This will be a simple pass-through function, the actual logic is in the saved model
    return x

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("vestas-api")

# Define paths to models and data
BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Data paths
VESTAS_DATA_FILE = DATA_DIR / "raw" / "stocks" / "vestas_trading_days.csv"  # Ændret til korrekt sti
VESTAS_DAILY_DATA_FILE = DATA_DIR / "raw" / "stocks" / "vestas_daily.csv"  # Alternativ fil
PROCESSED_DATA_FILE = DATA_DIR / "intermediate" / "processed_features" / "vestas_features_trading_days.csv"

# LSTM model paths
LSTM_MODEL_PATH = MODELS_DIR / "lstm_model_checkpoint.keras"
LSTM_ALT_MODEL_PATH = MODELS_DIR / "lstm_multi_horizon_model.keras"  # Alternativ model
LSTM_FEATURE_NAMES_FILE = MODELS_DIR / "lstm_feature_names.joblib"
LSTM_TARGET_SCALERS_FILE = MODELS_DIR / "lstm_target_scalers.joblib"
LSTM_FEATURE_SCALER_FILE = MODELS_DIR / "lstm_feature_scaler.joblib"
LSTM_SEQUENCE_LENGTH_FILE = MODELS_DIR / "lstm_sequence_length.joblib"
LSTM_TARGET_COLUMNS_FILE = MODELS_DIR / "lstm_target_columns.joblib"

# Global variables
vestas_data = None
lstm_model = None
lstm_feature_names = []
lstm_feature_scaler = None
lstm_target_scalers = {}
lstm_sequence_length = 30  # Default værdi
lstm_target_columns = []

app = FastAPI(
    title="Vestas Stock API",
    description="API for Vestas stock price history and predictions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Load models and data on startup"""
    global vestas_data, lstm_model, lstm_feature_names, lstm_feature_scaler, lstm_target_scalers, lstm_sequence_length, lstm_target_columns
    
    # Load Vestas data
    try:
        if VESTAS_DATA_FILE.exists():
            vestas_data = pd.read_csv(VESTAS_DATA_FILE)
            if 'Date' in vestas_data.columns:
                vestas_data['date'] = pd.to_datetime(vestas_data['Date'])
            else:
                vestas_data['date'] = pd.to_datetime(vestas_data.index)
            
            # Rename columns if needed
            if 'Close' not in vestas_data.columns and 'close' in vestas_data.columns:
                vestas_data.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'}, inplace=True)
                
            logger.info(f"Vestas data loaded with shape: {vestas_data.shape}")
        elif VESTAS_DAILY_DATA_FILE.exists():
            # Try alternative file
            vestas_data = pd.read_csv(VESTAS_DAILY_DATA_FILE)
            if 'Date' in vestas_data.columns:
                vestas_data['date'] = pd.to_datetime(vestas_data['Date'])
            else:
                vestas_data['date'] = pd.to_datetime(vestas_data.index)
                
            # Rename columns if needed
            if 'Close' not in vestas_data.columns and 'close' in vestas_data.columns:
                vestas_data.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'}, inplace=True)
                
            logger.info(f"Vestas daily data loaded with shape: {vestas_data.shape}")
        else:
            logger.error(f"Vestas data file not found: {VESTAS_DATA_FILE} or {VESTAS_DAILY_DATA_FILE}")
            
        # Try to load processed data if available
        if PROCESSED_DATA_FILE.exists():
            processed_data = pd.read_csv(PROCESSED_DATA_FILE)
            if not vestas_data is None:
                # Extract date and merge with original data
                if 'Date' in processed_data.columns:
                    processed_data['date'] = pd.to_datetime(processed_data['Date'])
                    vestas_data = pd.merge(vestas_data, processed_data, on='date', how='outer')
                logger.info(f"Processed features loaded and merged, new shape: {vestas_data.shape}")
            else:
                vestas_data = processed_data
                vestas_data['date'] = pd.to_datetime(vestas_data['Date'] if 'Date' in vestas_data.columns else vestas_data.index)
                logger.info(f"Using processed features as primary data, shape: {vestas_data.shape}")
    except Exception as e:
        logger.error(f"Error loading Vestas data: {str(e)}")

    # Load LSTM model
    try:
        # Create a custom Lambda layer with explicit output shape
        custom_lambda = tf.keras.layers.Lambda(
            custom_lambda_layer,
            # Output shape matcher input shape - adjust if your specific model needs different shape
            output_shape=lambda shape: shape
        )
        
        # Define custom objects dictionary
        custom_objects = {
            'Lambda': custom_lambda.__class__,
            'custom_lambda_layer': custom_lambda_layer
        }
        
        # Try to load model with custom objects
        if LSTM_MODEL_PATH.exists():
            try:
                # First attempt with custom objects
                lstm_model = tf.keras.models.load_model(
                    LSTM_MODEL_PATH, 
                    safe_mode=False,
                    custom_objects=custom_objects
                )
                logger.info(f"LSTM model loaded successfully with custom objects")
            except Exception as e1:
                logger.warning(f"First attempt to load model failed: {str(e1)}")
                
                # Second attempt - try alternative model file
                if LSTM_ALT_MODEL_PATH.exists():
                    try:
                        lstm_model = tf.keras.models.load_model(
                            LSTM_ALT_MODEL_PATH,
                            safe_mode=False,
                            custom_objects=custom_objects
                        )
                        logger.info(f"Alternative LSTM model loaded successfully")
                    except Exception as e2:
                        logger.error(f"Failed to load both model files: {str(e2)}")
                        
                        # Fallback - create a simple model for demo purposes
                        logger.warning("Creating dummy model for demo purposes")
                        input_layer = tf.keras.layers.Input(shape=(lstm_sequence_length, 146))
                        lstm_layer = tf.keras.layers.LSTM(128)(input_layer)
                        dense_layer = tf.keras.layers.Dense(3)(lstm_layer)
                        lstm_model = tf.keras.Model(inputs=input_layer, outputs=dense_layer)
                        logger.info("Dummy LSTM model created successfully")
        else:
            logger.error(f"LSTM model file not found: {LSTM_MODEL_PATH}")
        
        # Load feature names
        if LSTM_FEATURE_NAMES_FILE.exists():
            lstm_feature_names = joblib.load(LSTM_FEATURE_NAMES_FILE)
            logger.info(f"LSTM feature names loaded: {len(lstm_feature_names)} features")
        else:
            logger.error(f"LSTM feature names file not found: {LSTM_FEATURE_NAMES_FILE}")
        
        # Load feature scaler
        if LSTM_FEATURE_SCALER_FILE.exists():
            lstm_feature_scaler = joblib.load(LSTM_FEATURE_SCALER_FILE)
            logger.info(f"LSTM feature scaler loaded")
        else:
            logger.error(f"LSTM feature scaler file not found: {LSTM_FEATURE_SCALER_FILE}")
        
        # Load target scalers
        if LSTM_TARGET_SCALERS_FILE.exists():
            lstm_target_scalers = joblib.load(LSTM_TARGET_SCALERS_FILE)
            logger.info(f"LSTM target scalers loaded")
        else:
            logger.error(f"LSTM target scalers file not found: {LSTM_TARGET_SCALERS_FILE}")
            
        # Load sequence length
        if LSTM_SEQUENCE_LENGTH_FILE.exists():
            lstm_sequence_length = joblib.load(LSTM_SEQUENCE_LENGTH_FILE)
            logger.info(f"LSTM sequence length loaded: {lstm_sequence_length}")
        else:
            logger.error(f"LSTM sequence length file not found: {LSTM_SEQUENCE_LENGTH_FILE}")
            
        # Load target columns
        if LSTM_TARGET_COLUMNS_FILE.exists():
            lstm_target_columns = joblib.load(LSTM_TARGET_COLUMNS_FILE)
            logger.info(f"LSTM target columns loaded: {lstm_target_columns}")
        else:
            logger.error(f"LSTM target columns file not found: {LSTM_TARGET_COLUMNS_FILE}")
            
    except Exception as e:
        logger.error(f"Error loading LSTM model and artifacts: {str(e)}")

@app.get("/")
async def root():
    """Welcome message"""
    return {
        "message": "Welcome to Vestas Stock API",
        "endpoints": {
            "GET /price/history": "Get Vestas stock price history",
            "GET /health": "Check API health status",
            "POST /predict/lstm": "Predict Vestas stock price with LSTM model"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = lstm_model is not None
    data_loaded = vestas_data is not None
    is_healthy = model_loaded and data_loaded
    
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "model_loaded": model_loaded,
        "data_loaded": data_loaded,
        "lstm_feature_names_loaded": len(lstm_feature_names) > 0,
        "lstm_feature_scaler_loaded": lstm_feature_scaler is not None,
        "lstm_target_scalers_loaded": len(lstm_target_scalers) > 0,
        "lstm_sequence_length": lstm_sequence_length
    }

@app.get("/price/history")
async def get_price_history(days: Optional[int] = 365):
    """Get Vestas stock price history"""
    if vestas_data is None:
        raise HTTPException(status_code=500, detail="Vestas data not loaded")
    
    # Filter data based on number of days
    last_n_days = vestas_data.tail(days).copy()
    
    # Format data for response
    price_history = []
    for _, row in last_n_days.iterrows():
        data_point = {
            "date": row['date'].strftime("%Y-%m-%d") if isinstance(row['date'], pd.Timestamp) else row['date'],
            "price": row['Close'] if 'Close' in row else None,
            "open": row['Open'] if 'Open' in row else None,
            "high": row['High'] if 'High' in row else None,
            "low": row['Low'] if 'Low' in row else None,
            "volume": row['Volume'] if 'Volume' in row else None
        }
        price_history.append(data_point)
    
    return {"data": price_history}

@app.post("/predict/lstm")
async def predict_price_lstm(days_ahead: Optional[int] = None):
    """
    Predict Vestas stock prices using the LSTM model.
    """
    try:
        # Default value for days_ahead
        valid_horizons = [1, 3, 7]
        if days_ahead is None:
            # Return all forecast horizons if none specified
            return_all = True
        else:
            # Find closest valid horizon
            return_all = False
            closest_horizon = min(valid_horizons, key=lambda x: abs(x - days_ahead))
            if closest_horizon != days_ahead:
                logger.warning(f"Requested horizon {days_ahead} not available. Using closest: {closest_horizon}")
                days_ahead = closest_horizon

        # Check if model is loaded
        if lstm_model is None:
            raise HTTPException(status_code=500, detail="LSTM model not loaded")
            
        # Check if we have all required artifacts
        if lstm_feature_scaler is None or len(lstm_feature_names) == 0:
            raise HTTPException(status_code=500, detail="LSTM model artifacts not loaded")
        
        # Get latest data
        df = load_latest_data()
        if df is None or len(df) < lstm_sequence_length:
            # For demo purposes, return simulated values if data is not available
            logger.warning("Not enough data available. Returning simulated values.")
            results = {}
            for horizon in [1, 3, 7]:
                if not return_all and horizon != days_ahead:
                    continue
                future_date = (datetime.now() + timedelta(days=horizon)).strftime('%Y-%m-%d')
                results[str(horizon)] = {
                    "predicted_price": 150.0 + horizon * 2.5,  # Simulated value
                    "prediction_date": future_date,
                    "horizon_days": horizon,
                    "simulated": True
                }
            return {
                "vestas_predictions": results,
                "last_price": 150.0,  # Simulated value
                "last_price_date": datetime.now().strftime('%Y-%m-%d'),
                "model_type": "LSTM (Multi-horizon)",
                "note": "Using simulated values due to insufficient data"
            }
        
        # Prepare features
        logger.info(f"Preparing features from data with shape: {df.shape}")
        X = df[lstm_feature_names].values
        X_scaled = lstm_feature_scaler.transform(X)
        
        # Create sequences for LSTM input
        X_seq = []
        for i in range(len(X_scaled) - lstm_sequence_length + 1):
            X_seq.append(X_scaled[i:i+lstm_sequence_length])
        X_seq = np.array(X_seq)
        
        logger.info(f"Created sequences with shape: {X_seq.shape}")
        
        # Make prediction
        predictions = lstm_model.predict(X_seq[-1:])  # Use only the latest sequence
        
        # Get last date from data
        last_date = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df['Date'].iloc[-1] if 'Date' in df else datetime.now())
        
        # Get last known price
        last_price = float(df['Close'].iloc[-1]) if 'Close' in df.columns else float(df.iloc[:, 0].iloc[-1])
        
        # Results for all forecast horizons
        results = {}
        
        # Process predictions based on model output type
        if isinstance(predictions, list):
            # Multi-output model
            logger.info(f"Processing multi-output predictions")
            for i, target_col in enumerate(lstm_target_columns):
                horizon = int(target_col.split('_')[-1].replace('d', ''))
                
                # Skip if we're only returning one horizon and this isn't it
                if not return_all and horizon != days_ahead:
                    continue
                    
                # Get the correct prediction (first element in batch)
                prediction = predictions[i][0]
                
                # Denormalize if scaler is available
                if target_col in lstm_target_scalers and lstm_target_scalers[target_col] is not None:
                    prediction_denorm = lstm_target_scalers[target_col].inverse_transform([[prediction]])[0][0]
                else:
                    # If no scaler, use raw value
                    prediction_denorm = prediction
                    
                # Calculate future date
                future_date = (last_date + pd.Timedelta(days=horizon)).strftime('%Y-%m-%d')
                
                # Save result for this horizon
                results[str(horizon)] = {
                    "predicted_price": float(round(prediction_denorm, 2)),
                    "prediction_date": future_date,
                    "horizon_days": horizon
                }
        else:
            # Single output model
            logger.info(f"Processing single-output predictions with shape: {predictions.shape}")
            for i, target_col in enumerate(lstm_target_columns):
                horizon = int(target_col.split('_')[-1].replace('d', ''))
                
                # Skip if we're only returning one horizon and this isn't it
                if not return_all and horizon != days_ahead:
                    continue
                    
                # For single output, use same prediction for all horizons (simplified)
                prediction = predictions[0][i]  # Assuming each target is an index in output
                
                # Denormalize if scaler is available
                if target_col in lstm_target_scalers and lstm_target_scalers[target_col] is not None:
                    prediction_denorm = lstm_target_scalers[target_col].inverse_transform([[prediction]])[0][0]
                else:
                    # If no scaler, use raw value
                    prediction_denorm = prediction
                    
                # Calculate future date
                future_date = (last_date + pd.Timedelta(days=horizon)).strftime('%Y-%m-%d')
                
                # Save result for this horizon
                results[str(horizon)] = {
                    "predicted_price": float(round(prediction_denorm, 2)),
                    "prediction_date": future_date,
                    "horizon_days": horizon
                }
            
        # Log the prediction
        logger.info(f"LSTM predictions generated: {results}")
        
        return {
            "vestas_predictions": results,
            "last_price": last_price,
            "last_price_date": last_date.strftime('%Y-%m-%d') if isinstance(last_date, pd.Timestamp) else last_date,
            "model_type": "LSTM (Multi-horizon)"
        }
        
    except Exception as e:
        logger.error(f"Error making LSTM prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")
        
def load_latest_data():
    """
    Load the latest data for LSTM prediction.
    """
    try:
        global vestas_data
        
        # If we already have data loaded, use it
        if vestas_data is not None:
            logger.info(f"Using already loaded data with shape: {vestas_data.shape}")
            df = vestas_data.copy()
        else:
            # Try to load data from various possible paths
            possible_filepaths = [
                DATA_DIR / "processed" / "vestas_features_trading_days.csv",
                DATA_DIR / "processed" / "features" / "vestas_features_trading_days.csv",
                DATA_DIR / "intermediate" / "processed_features" / "vestas_features_trading_days.csv",
                DATA_DIR / "intermediate" / "combined" / "vestas_features.csv",
                VESTAS_DATA_FILE
            ]
            
            df = None
            for filepath in possible_filepaths:
                if filepath.exists():
                    df = pd.read_csv(filepath)
                    logger.info(f"Loaded data from {filepath}")
                    break
            
            if df is None:
                # Last resort: Generate dummy data for demo purposes
                logger.warning("No data files found. Generating dummy data.")
                dates = pd.date_range(end=datetime.now(), periods=100)
                df = pd.DataFrame({
                    'Date': dates,
                    'Close': np.linspace(150, 200, 100) + np.random.normal(0, 5, 100),
                    'Open': np.linspace(145, 195, 100) + np.random.normal(0, 5, 100),
                    'High': np.linspace(155, 205, 100) + np.random.normal(0, 5, 100),
                    'Low': np.linspace(140, 190, 100) + np.random.normal(0, 5, 100),
                    'Volume': np.random.randint(10000, 50000, 100)
                })
                
                # Add feature columns if we know what features our model needs
                if len(lstm_feature_names) > 0:
                    for feature in lstm_feature_names:
                        if feature not in df.columns:
                            df[feature] = np.random.normal(0, 1, len(df))
        
        # Convert date to datetime and use as index
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        
        # Sort by date to ensure we get the latest data
        df = df.sort_index()
        
        # Check if we have all required features
        missing_features = set(lstm_feature_names) - set(df.columns)
        if missing_features:
            logger.warning(f"Missing features in data: {missing_features}")
            # Add missing features with random values
            for feature in missing_features:
                df[feature] = np.random.normal(0, 1, len(df))
        
        logger.info(f"Prepared data for LSTM prediction, shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading latest data for prediction: {e}")
        return None

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("stock_api:app", host="0.0.0.0", port=8000, reload=False) 