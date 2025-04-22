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
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input, RepeatVector, Flatten, Activation, Multiply, Lambda, Permute
from tensorflow.keras.optimizers import Adam
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import sqlite3

# Enable unsafe deserialization for Lambda layer
import keras
try:
    keras.config.enable_unsafe_deserialization()
except AttributeError:
    # For older versions of Keras/TF
    tf.keras.utils.disable_interactive_logging()
    os.environ['TF_KERAS_SAFE_MODE'] = '0'

# Define custom Lambda layer with explicit output shape
def custom_lambda_layer(tensor):
    # Based on error message, we know input shape is (None, 30, 128)
    # Return tensor unchanged
    return tensor

def constrain_price_prediction(prediction: float, current_price: float, horizon_days: int) -> float:
    max_daily_pct_change = 0.05
    max_pct_change = max_daily_pct_change * min(horizon_days, 1) + 0.03 * max(0, horizon_days - 1)
    max_prediction = current_price * (1 + max_pct_change)
    min_prediction = current_price * (1 - max_pct_change)
    
    # Set prediction to the bounds
    constrained_prediction = max(min_prediction, min(prediction, max_prediction))
    return constrained_prediction

# Define the function used by the Lambda layer in the saved model
def sum_over_time_axis(x):
    # Sums the input tensor along the time axis (axis=1)
    return K.sum(x, axis=1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("vestas-api")

# Define paths to models and data
BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
DB_FILE = DATA_DIR / "market_data.db" # Corrected path
DB_TABLE_NAME = "market_data"


# LSTM model paths
LSTM_MODEL_PATH = MODELS_DIR / "lstm_model_checkpoint.keras"
LSTM_ALT_MODEL_PATH = MODELS_DIR / "lstm_multi_horizon_model.keras" 
LSTM_FEATURE_NAMES_FILE = MODELS_DIR / "lstm_feature_names.joblib"
LSTM_TARGET_SCALERS_FILE = MODELS_DIR / "lstm_target_scalers.joblib"
LSTM_FEATURE_SCALER_FILE = MODELS_DIR / "lstm_feature_scaler.joblib"
LSTM_SEQUENCE_LENGTH_FILE = MODELS_DIR / "lstm_sequence_length.joblib"
LSTM_TARGET_COLUMNS_FILE = MODELS_DIR / "lstm_target_columns.joblib"
LSTM_FEATURE_MEDIANS_FILE = MODELS_DIR / "lstm_feature_medians.joblib"

# Global variables
vestas_data = None
lstm_model = None
lstm_feature_names = []
lstm_feature_scaler = None
lstm_target_scalers = {}
lstm_sequence_length = 30  # Default value, will be overwritten during loading
lstm_target_columns = []
lstm_feature_medians = {}

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
    global vestas_data, lstm_model, lstm_feature_names, lstm_feature_scaler, lstm_target_scalers, lstm_sequence_length, lstm_target_columns, lstm_feature_medians
    
    try:
        # Load feature names and validate
        if LSTM_FEATURE_NAMES_FILE.exists():
            lstm_feature_names = joblib.load(LSTM_FEATURE_NAMES_FILE)
            logger.info(f"LSTM feature names loaded: {len(lstm_feature_names)} features")
            # Log a sample of feature names for validation
            if lstm_feature_names:
                sample_size = min(5, len(lstm_feature_names))
                logger.info(f"Sample feature names: {lstm_feature_names[:sample_size]}...")
        else:
            logger.error(f"LSTM feature names file not found: {LSTM_FEATURE_NAMES_FILE}")
            lstm_feature_names = [] # Ensure it's a list
    
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
            lstm_target_columns = [] # Ensure it's a list
    except Exception as e:
        logger.error(f"Error loading model metadata: {str(e)}")
        lstm_feature_names = []
        lstm_target_columns = []

    # Load Vestas data with better error handling
    try:
        # Load OHLCV data directly from the database for the /price/history endpoint
        logger.info(f"Loading price history data from database: {DB_FILE}")
        if not DB_FILE.exists():
            raise FileNotFoundError(f"Database file not found at {DB_FILE}")

        conn = sqlite3.connect(DB_FILE)
        # Select only the necessary stock columns
        query = f"SELECT date, stock_open, stock_high, stock_low, stock_close, stock_volume FROM {DB_TABLE_NAME} ORDER BY date"
        df_db = pd.read_sql_query(query, conn)
        conn.close()

        if df_db.empty:
            logger.error("No data returned from database query for price history.")
            vestas_data = None
        else:
            # Parse date and set index
            df_db['date'] = pd.to_datetime(df_db['date'])
            df_db.set_index('date', inplace=True)

            # Rename columns for the endpoint
            rename_map = {
                'stock_open': 'Open',
                'stock_high': 'High',
                'stock_low': 'Low',
                'stock_close': 'Close',
                'stock_volume': 'Volume'
            }
            df_db.rename(columns=rename_map, inplace=True)
            
            # Ensure correct types
            for col in ['Open', 'High', 'Low', 'Close']:
                df_db[col] = pd.to_numeric(df_db[col], errors='coerce')
            if 'Volume' in df_db.columns:
                 df_db['Volume'] = pd.to_numeric(df_db['Volume'], errors='coerce').fillna(0).astype(int)

            # Remove rows where essential price data might be missing after conversion
            df_db.dropna(subset=['Close'], inplace=True)

            vestas_data = df_db
            logger.info(f"Successfully loaded price history from database. Shape: {vestas_data.shape}")
            logger.info(f"Date range: {vestas_data.index.min()} to {vestas_data.index.max()}")

    except Exception as e:
        logger.error(f"Error loading Vestas data: {str(e)}")
        vestas_data = None
        logger.critical("No data loaded for /price/history endpoint. API will return errors.")

    # Load LSTM model
    try:
        # Load metadata first (needed for model potentially)
        if LSTM_FEATURE_NAMES_FILE.exists():
            lstm_feature_names = joblib.load(LSTM_FEATURE_NAMES_FILE)
            logger.info(f"LSTM feature names loaded: {len(lstm_feature_names)} features")
        else:
            logger.error(f"LSTM feature names file not found: {LSTM_FEATURE_NAMES_FILE}")
            lstm_feature_names = [] # Ensure it's a list

        if LSTM_SEQUENCE_LENGTH_FILE.exists():
            lstm_sequence_length = joblib.load(LSTM_SEQUENCE_LENGTH_FILE)
            logger.info(f"LSTM sequence length loaded: {lstm_sequence_length}")
        else:
            logger.error(f"LSTM sequence length file not found: {LSTM_SEQUENCE_LENGTH_FILE}")

        if LSTM_TARGET_COLUMNS_FILE.exists():
            lstm_target_columns = joblib.load(LSTM_TARGET_COLUMNS_FILE)
            logger.info(f"LSTM target columns loaded: {lstm_target_columns}")
        else:
            logger.error(f"LSTM target columns file not found: {LSTM_TARGET_COLUMNS_FILE}")
            lstm_target_columns = [] # Ensure it's a list

        # --- Build model structure and load weights --- 
        global lstm_model # Ensure we modify the global variable
        
        # Define expected input shape based on loaded metadata
        if lstm_sequence_length > 0 and lstm_feature_names:
             input_shape = (lstm_sequence_length, len(lstm_feature_names))
             logger.info(f"Building model structure with input shape: {input_shape}")
             
             # Define horizon keys based on loaded target columns
             horizon_keys = [col.split('_')[-1] for col in lstm_target_columns]
             logger.info(f"Using horizon keys for model outputs: {horizon_keys}")
             
             # Build the empty model structure
             lstm_model = build_seq2seq_model(input_shape, horizon_keys=horizon_keys)
             
             # Define path to weights file (assuming .keras file contains weights)
             weights_path = LSTM_ALT_MODEL_PATH # Path to lstm_multi_horizon_model.keras
             
             if weights_path.exists():
                 try:
                     # Load weights into the structure
                     lstm_model.load_weights(weights_path)
                     logger.info(f"Successfully loaded weights from {weights_path} into model structure.")
                 except Exception as e:
                     logger.error(f"Failed to load weights from {weights_path}: {str(e)}")
                     lstm_model = None # Failed to load weights
             else:
                 logger.error(f"Model weights file not found: {weights_path}")
                 lstm_model = None
        else:
             logger.error("Cannot build model structure: Sequence length or feature names not loaded.")
             lstm_model = None

        # --- Load feature scaler ---
        if LSTM_FEATURE_SCALER_FILE.exists():
            try:
                lstm_feature_scaler = joblib.load(LSTM_FEATURE_SCALER_FILE)
                logger.info(f"LSTM feature scaler loaded ({type(lstm_feature_scaler)})")
                # Validate the scaler type
                expected_scaler_type = "MinMaxScaler"
                scaler_type = type(lstm_feature_scaler).__name__
                if scaler_type != expected_scaler_type:
                    logger.warning(f"Unexpected feature scaler type: {scaler_type}. Expected: {expected_scaler_type}")
                
                # Validate scaler attributes
                if hasattr(lstm_feature_scaler, 'data_min_') and hasattr(lstm_feature_scaler, 'data_max_'):
                    logger.info(f"Feature Scaler Data Min (partial): {lstm_feature_scaler.data_min_[:5]}...")
                    logger.info(f"Feature Scaler Data Max (partial): {lstm_feature_scaler.data_max_[:5]}...")
                    logger.info(f"Feature scale range: [{lstm_feature_scaler.feature_range[0]}, {lstm_feature_scaler.feature_range[1]}]")
                else:
                    logger.warning("Feature scaler missing expected MinMaxScaler attributes. Data scaling may be inconsistent.")
            except Exception as e:
                logger.error(f"Error loading feature scaler: {str(e)}")
                lstm_feature_scaler = None
        else:
            logger.error(f"LSTM feature scaler file not found: {LSTM_FEATURE_SCALER_FILE}")
            lstm_feature_scaler = None

        # --- Load target scalers ---
        if LSTM_TARGET_SCALERS_FILE.exists():
            try:
                lstm_target_scalers = joblib.load(LSTM_TARGET_SCALERS_FILE)
                logger.info(f"LSTM target scalers loaded ({len(lstm_target_scalers)} scalers)")
                
                # Validate each target scaler
                valid_scalers = True
                for target_col, scaler in lstm_target_scalers.items():
                    expected_scaler_type = "MinMaxScaler"
                    scaler_type = type(scaler).__name__
                    if scaler_type != expected_scaler_type:
                        logger.warning(f"Unexpected target scaler type for {target_col}: {scaler_type}. Expected: {expected_scaler_type}")
                        valid_scalers = False
                        continue
                        
                    # Log MinMaxScaler specific attributes
                    if hasattr(scaler, 'data_min_') and hasattr(scaler, 'data_max_'):
                        logger.info(f"Target Scaler {target_col} - Min: {scaler.data_min_}")
                        logger.info(f"Target Scaler {target_col} - Max: {scaler.data_max_}")
                        logger.info(f"Target Scaler {target_col} - Scale range: [{scaler.feature_range[0]}, {scaler.feature_range[1]}]")
                    else:
                        logger.warning(f"Target scaler for {target_col} is missing expected MinMaxScaler attributes")
                        valid_scalers = False
                
                if not valid_scalers:
                    logger.warning("Some target scalers have validation issues. Predictions may be inaccurate.")
            except Exception as e:
                logger.error(f"Error loading target scalers: {str(e)}")
                lstm_target_scalers = {}
        else:
            logger.error(f"LSTM target scalers file not found: {LSTM_TARGET_SCALERS_FILE}")
            lstm_target_scalers = {}

    except Exception as e:
        logger.error(f"Error loading LSTM model and artifacts: {str(e)}")
        # Ensure globals are None if errors occur during loading
        lstm_model = None
        lstm_feature_scaler = None
        lstm_target_scalers = {}
        lstm_feature_names = []
        lstm_target_columns = []
        lstm_feature_medians = {}

    # Load feature medians AFTER other artifacts
    try:
        if LSTM_FEATURE_MEDIANS_FILE.exists():
            lstm_feature_medians = joblib.load(LSTM_FEATURE_MEDIANS_FILE)
            logger.info(f"LSTM feature medians loaded ({len(lstm_feature_medians)} medians)")
            # Log a few median values for verification
            logged_medians = 0
            for k, v in lstm_feature_medians.items():
                if logged_medians < 5:
                    logger.info(f"  Median for '{k}': {v}")
                    logged_medians += 1
                else:
                    break
        else:
            logger.error(f"LSTM feature medians file not found: {LSTM_FEATURE_MEDIANS_FILE}")
            lstm_feature_medians = {} # Ensure it's an empty dict if file not found
    except Exception as e:
        logger.error(f"Error loading LSTM feature medians: {str(e)}")
        lstm_feature_medians = {}

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
        "lstm_sequence_length": lstm_sequence_length,
        "lstm_feature_medians_loaded": len(lstm_feature_medians) > 0
    }

@app.get("/price/history")
async def get_price_history(days: Optional[int] = 6570):
    """Get Vestas stock price history from the data loaded at startup."""
    if vestas_data is None or vestas_data.empty:
        logger.error("Price history data not available.")
        raise HTTPException(status_code=500, detail="Price history data is not loaded or empty.")

    try:
        # Use the globally loaded vestas_data DataFrame
        df = vestas_data.copy() # Work on a copy

        # Filter by days if specified
        if days is not None and days > 0:
            # Ensure index is sorted before tailing
            df = df.sort_index().tail(days)

        if df.empty:
             logger.warning(f"No data available for the specified period ({days} days).")
             return {"data": []}

        # Format data for response
        price_history = []
        for idx, row in df.iterrows():
            try:
                date_str = idx.strftime("%Y-%m-%d")
                data_point = {
                    "date": date_str,
                    # Use .get() with default None for robustness
                    "price": float(row['Close']) if pd.notna(row.get('Close')) else None,
                    "open": float(row['Open']) if pd.notna(row.get('Open')) else None,
                    "high": float(row['High']) if pd.notna(row.get('High')) else None,
                    "low": float(row['Low']) if pd.notna(row.get('Low')) else None,
                    "volume": int(row['Volume']) if pd.notna(row.get('Volume')) else None
                }
                # Filter out entries with no price
                if data_point["price"] is not None:
                    price_history.append(data_point)
            except Exception as e:
                logger.error(f"Error processing row for date {idx}: {e}")
                continue

        logger.info(f"Returning {len(price_history)} price history records.")
        return {"data": price_history}

    except Exception as e:
        logger.error(f"Error generating price history response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating price data: {str(e)}")

@app.post("/predict/lstm")
async def predict_price_lstm(days_ahead: Optional[int] = None):
    """
    Predict Vestas stock prices using the LSTM model.
    """
    try:
        # Default value for days_ahead
        valid_horizons = [1, 3, 7] # Assumes these are the trained horizons
        return_all = days_ahead is None
        if not return_all:
            closest_horizon = min(valid_horizons, key=lambda x: abs(x - days_ahead))
            if closest_horizon != days_ahead:
                logger.warning(f"Requested horizon {days_ahead} not available. Using closest: {closest_horizon}")
                days_ahead = closest_horizon

        # Check if model and necessary artifacts are loaded
        if lstm_model is None: raise HTTPException(status_code=500, detail="LSTM model not loaded")
        if lstm_feature_scaler is None: raise HTTPException(status_code=500, detail="LSTM feature scaler not loaded")
        if not lstm_feature_names: raise HTTPException(status_code=500, detail="LSTM feature names not loaded")
        if not lstm_target_scalers: raise HTTPException(status_code=500, detail="LSTM target scalers not loaded")
        if not lstm_target_columns: raise HTTPException(status_code=500, detail="LSTM target columns not loaded")

        # Get latest feature data using the improved function
        df_features = load_and_prepare_latest_data(required_features=lstm_feature_names, seq_length=lstm_sequence_length)

        if df_features is None: raise HTTPException(status_code=500, detail="Could not load feature data for prediction.")
        if len(df_features) < lstm_sequence_length:
            raise HTTPException(status_code=500, detail=f"Insufficient data ({len(df_features)} rows) for sequence length {lstm_sequence_length}.")

        # Prepare features
        logger.info(f"Preparing features from data with shape: {df_features.shape}")
        missing_features = set(lstm_feature_names) - set(df_features.columns)
        if missing_features:
            raise HTTPException(status_code=500, detail=f"Data preparation failed: Missing features {missing_features}")

        # Extract features in the correct order and scale them
        X = df_features[lstm_feature_names].values
        X_scaled = lstm_feature_scaler.transform(X)

        # Prepare the last sequence for prediction
        last_sequence_scaled = X_scaled[-lstm_sequence_length:]
        X_pred_seq = np.array([last_sequence_scaled]) # Reshape to (1, seq_length, n_features)
        logger.info(f"Created prediction sequence with shape: {X_pred_seq.shape}")

        # Make prediction
        try:
            predictions = lstm_model.predict(X_pred_seq)
            logger.info(f"Model prediction successful. Output type: {type(predictions)}")
            if isinstance(predictions, list): logger.info(f"  Number of outputs: {len(predictions)}")
            elif isinstance(predictions, np.ndarray): logger.info(f"  Output shape: {predictions.shape}")
        except Exception as e:
            logger.error(f"Error during model prediction: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")

        # Get last date and price from feature data (most up-to-date)
        last_date = df_features.index[-1]
        price_column = None
        for candidate in ['Close', 'close', 'price', 'Price', 'stock_close']: # Added stock_close
            if candidate in df_features.columns:
                price_column = candidate
                break
        if price_column is None:
             # Try finding the first column that looks like a price
             potential_price_cols = [col for col in df_features.columns if 'close' in col.lower() or 'price' in col.lower()]
             if potential_price_cols:
                 price_column = potential_price_cols[0]
                 logger.warning(f"Could not find standard price column, using '{price_column}' as fallback.")
             else:
                 logger.error("Could not find any price column in feature data.")
                 raise HTTPException(status_code=500, detail="Cannot determine last price from feature data.")
        last_price = float(df_features[price_column].iloc[-1])

        # Process predictions based on model output type and target columns
        results = {}
        # Determine if model predicts percentage change
        metadata_path = MODELS_DIR / "lstm_model_metadata.json"
        is_percent_change_model = False
        try:
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    is_percent_change_model = metadata.get('is_percent_change_model', False)
                    logger.info(f"Model metadata indicates {'percent-change' if is_percent_change_model else 'direct-price'} prediction.")
        except Exception as e:
            logger.warning(f"Could not read model metadata: {e}. Assuming direct price prediction.")

        # Extract trained horizon keys from target columns
        # Example target_col format: 'pct_change_price_target_7d' or 'price_target_7d'
        trained_horizon_keys = {} # Map horizon (e.g., 7) to target column name
        for col in lstm_target_columns:
            try:
                # Extract number at the end, assuming format like '..._Xd'
                horizon_str = col.split('_')[-1]
                if 'd' in horizon_str:
                    horizon = int(horizon_str.replace('d', ''))
                    trained_horizon_keys[horizon] = col
            except:
                logger.warning(f"Could not parse horizon from target column name: {col}")
        logger.info(f"Extracted trained horizons: {list(trained_horizon_keys.keys())}")

        # Process predictions
        if isinstance(predictions, list) and len(predictions) == len(trained_horizon_keys):
             pred_dict = dict(zip(sorted(trained_horizon_keys.keys()), predictions)) # Map horizon number to prediction array
        elif isinstance(predictions, np.ndarray) and len(predictions.shape) > 1 and predictions.shape[1] == len(trained_horizon_keys):
             # Handle case where model returns single array with multiple outputs
             pred_dict = {h: predictions[:, i] for i, h in enumerate(sorted(trained_horizon_keys.keys()))}
        else:
             logger.error(f"Prediction output structure mismatch. Expected {len(trained_horizon_keys)} outputs, got {type(predictions)}")
             raise HTTPException(status_code=500, detail="Model prediction output format unexpected.")

        for horizon in sorted(trained_horizon_keys.keys()):
            target_col = trained_horizon_keys[horizon] # Get the original target column name

            if not return_all and horizon != days_ahead: continue

            pred_value = float(pred_dict[horizon][0]) # Get scalar prediction for this horizon

            # Denormalize using the correct scaler
            scaler = lstm_target_scalers.get(target_col) # Find scaler by original target name
            if scaler:
                pred_array = np.array([[pred_value]])
                prediction_denorm = scaler.inverse_transform(pred_array)[0][0]
                logger.info(f"[{target_col}] Raw: {pred_value:.4f}, Denormalized: {prediction_denorm:.4f}")
            else:
                prediction_denorm = pred_value
                logger.warning(f"[{target_col}] No scaler found. Using raw predicted value: {prediction_denorm}")

            # Convert percent change back to price if necessary
            if is_percent_change_model:
                price_prediction = last_price * (1 + prediction_denorm / 100)
                logger.info(f"Converted {prediction_denorm:.2f}% change to price: {price_prediction:.2f} (base: {last_price:.2f})")
            else:
                price_prediction = prediction_denorm

            price_prediction = constrain_price_prediction(price_prediction, last_price, horizon)

            # Calculate future date based on trading days
            future_date = get_next_trading_days(last_date, horizon).strftime('%Y-%m-%d')

            results[str(horizon)] = {
                "predicted_price": float(round(price_prediction, 2)),
                "prediction_date": future_date,
                "horizon_days": horizon,
                "trading_days": horizon # Clarify this is trading days
            }

        # Final logging and return
        logger.info(f"LSTM predictions generated: {results}")
        return {
            "request_details": {
                "requested_horizon": days_ahead if not return_all else "all",
            },
            "vestas_predictions": results,
            "last_price": last_price,
            "last_price_date": last_date.strftime('%Y-%m-%d'),
            "model_type": "LSTM (Multi-horizon, Seq2Seq)"
        }

    except HTTPException as http_exc:
         logger.error(f"HTTP Exception during prediction: {http_exc.detail}")
         raise http_exc # Re-raise FastAPI exceptions
    except Exception as e:
        logger.error(f"Error making LSTM prediction: {e}")
        import traceback
        logger.error(traceback.format_exc())

        # Ensure the error message is user-friendly
        detail_message = f"An internal error occurred while making the prediction."
        # You might want to log the original error `e` for debugging purposes but not expose it to the user
        logger.error(f"Internal prediction error details: {str(e)}") 
        raise HTTPException(status_code=500, detail=detail_message)

def load_and_prepare_latest_data(required_features: List[str], seq_length: int):
    """
    Loads the latest raw/preprocessed data from the database,
    performs feature engineering, cleans, and prepares it for scaling and prediction.

    Args:
        required_features: List of feature names the model expects.
        seq_length: The sequence length needed by the model.

    Returns:
        DataFrame with the latest features (unscaled) or None if processing fails.
    """
    # --- Feature Engineering Logic (mirrored from feature_engineering.py for API use) ---
    # Import necessary functions (ensure feature_engineering.py is importable)
    try:
        # Assumes src is in PYTHONPATH or feature_engineering is in the same directory level
        from src.pipeline.feature_engineering import (
            create_features as calculate_all_features,
            calculate_market_features,
            calculate_macro_features
        )
        feature_engineering_available = True
    except ImportError as ie:
        logger.error(f"Could not import feature engineering functions: {ie}. API predictions might fail.")
        feature_engineering_available = False
        # Define dummy functions if import fails, to prevent NameErrors later
        def calculate_all_features(df): return df
        def calculate_market_features(df): return pd.DataFrame(index=df.index)
        def calculate_macro_features(df): return pd.DataFrame(index=df.index)
    # ---------------------------------------------------------------------------------

    if not feature_engineering_available:
        logger.error("Feature engineering module not available. Cannot prepare data.")
        return None

    # 1. Load data from Database
    if not DB_FILE.exists():
        logger.error(f"Database file not found: {DB_FILE}")
        return None
    try:
        conn = sqlite3.connect(DB_FILE)
        # Load enough data: sequence length + extra for feature calculation lookback (e.g., 90 days)
        days_to_load = seq_length + 90
        # Query to get the last N records based on date
        query = f"SELECT * FROM {DB_TABLE_NAME} ORDER BY date DESC LIMIT {days_to_load}"
        df_raw = pd.read_sql_query(query, conn)
        conn.close()

        if df_raw.empty:
            logger.error("No data loaded from database.")
            return None

        # Convert date and set index, sort oldest to newest
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        df_raw.set_index('date', inplace=True)
        df_raw.sort_index(inplace=True)
        logger.info(f"Loaded last {len(df_raw)} records from database {DB_FILE}")

    except Exception as e:
        logger.error(f"Error loading data from database {DB_FILE}: {e}")
        return None

    # 2. Preprocessing (Minimal - mirroring preprocessing.py)
    df_processed = df_raw.copy()
    # Rename stock columns for consistency with feature engineering functions
    rename_map = {
        'stock_open': 'open',
        'stock_high': 'high',
        'stock_low': 'low',
        'stock_close': 'close',
        'stock_volume': 'volume'
    }
    df_processed.rename(columns=rename_map, inplace=True, errors='ignore')

    for col in ['open', 'high', 'low', 'close', 'volume']:
         if col in df_processed.columns:
             df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    logger.info("Minimal preprocessing applied.")

    # 3. Feature Engineering (Using imported functions)
    try:
        logging.info("Applying feature engineering...")
        # Calculate base features, technical indicators etc.
        df_features = calculate_all_features(df_processed)
        if df_features is None:
            raise ValueError("calculate_all_features returned None")

        # Calculate and merge market/macro features
        market_features = calculate_market_features(df_features)
        macro_features = calculate_macro_features(df_features)

        if not market_features.empty:
            df_features = df_features.merge(market_features, left_index=True, right_index=True, how='left')
        if not macro_features.empty:
            df_features = df_features.merge(macro_features, left_index=True, right_index=True, how='left')

        logger.info(f"Feature engineering complete. Shape before cleaning: {df_features.shape}")

    except Exception as e:
        logger.error(f"Error during feature engineering in API: {e}")
        logger.error(traceback.format_exc())
        return None

    # --- 4. Data Cleaning (Handle inf/NaN using loaded medians) ---
    df = df_features # Use df from now on

    # Check for required features *before* extensive cleaning
    missing_model_features = set(required_features) - set(df.columns)
    if missing_model_features:
        logger.error(f"Data is missing features required by the model: {missing_model_features}")
        # Attempt to fill with median if available, otherwise fail
        can_continue = True
        for col in missing_model_features:
            if col in lstm_feature_medians:
                 logger.warning(f"Filling missing required feature '{col}' with its median value.")
                 df[col] = lstm_feature_medians[col]
            else:
                 logger.error(f"Cannot proceed: Missing required feature '{col}' and no median available.")
                 can_continue = False
        if not can_continue:
             return None

    # Select only the features the model needs + necessary columns like 'close'
    # Determine the actual price column name present in the data
    price_column_actual = None
    price_col_candidates = ['close', 'Close'] # Check lowercase first
    for pc in price_col_candidates:
        if pc in df.columns:
            price_column_actual = pc
            break

    cols_to_keep = list(required_features) # Start with model features
    if price_column_actual and price_column_actual not in cols_to_keep:
        cols_to_keep.append(price_column_actual) # Ensure price column is kept

    # Check if all cols_to_keep actually exist in df.columns before filtering
    existent_cols_to_keep = [col for col in cols_to_keep if col in df.columns]
    missing_cols_in_df = set(cols_to_keep) - set(existent_cols_to_keep)
    if missing_cols_in_df:
        logger.warning(f"Columns specified to keep but not found in DataFrame: {missing_cols_in_df}")

    if not existent_cols_to_keep:
         logger.error("No required columns found in the dataframe after feature engineering.")
         return None

    df = df[existent_cols_to_keep].copy() # Work with only necessary, existing columns
    logger.info(f"Filtered DataFrame to required features + price. Shape: {df.shape}, Columns: {df.columns.tolist()}")


    # Handle infinities first (within required features)
    for col in required_features:
         # Check if column exists before processing
         if col in df.columns and df[col].isin([np.inf, -np.inf]).any():
              inf_count = df[col].isin([np.inf, -np.inf]).sum()
              logger.warning(f"Feature '{col}' contains {inf_count} inf values. Replacing with NaN.")
              df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    # Handle NaNs using preloaded medians (from training)
    if not lstm_feature_medians:
         logger.warning("Feature medians not loaded. Cannot impute NaNs accurately. Dropping rows with NaNs as fallback.")
         df.dropna(subset=required_features, inplace=True)
    else:
         features_to_check_nan = [f for f in required_features if f in df.columns] # Only check existing columns
         for col in features_to_check_nan:
              if df[col].isna().any():
                   nan_count = df[col].isna().sum()
                   if col in lstm_feature_medians:
                       median_val = lstm_feature_medians[col]
                       # Ensure median_val is a compatible type (float/int)
                       if pd.api.types.is_numeric_dtype(df[col].dtype):
                            try:
                                median_val = float(median_val) # Convert just in case
                                logger.warning(f"Feature '{col}' contains {nan_count} NaNs. Filling with PRELOADED median ({median_val:.4f}).")
                                df[col].fillna(median_val, inplace=True)
                            except (ValueError, TypeError) as fill_err:
                                logger.error(f"Could not fill NaNs for '{col}' with median {median_val}: {fill_err}. Dropping rows.")
                                df.dropna(subset=[col], inplace=True)
                       else:
                            logger.warning(f"Feature '{col}' is not numeric, skipping NaN fill with median.")

                   else:
                       # Fallback if median for this specific feature wasn't loaded
                       logger.error(f"Feature '{col}' contains {nan_count} NaNs, BUT median not preloaded! Dropping rows.")
                       df.dropna(subset=[col], inplace=True) # Drop rows missing this essential median

    # Final check for NaNs after imputation
    final_features_to_check = [f for f in required_features if f in df.columns]
    if final_features_to_check:
        remaining_nans = df[final_features_to_check].isna().sum().sum()
        if remaining_nans > 0:
             logger.error(f"{remaining_nans} NaNs remain in required features after imputation. Prediction might fail or be inaccurate.")
             # For now, just log and continue, model might handle it or fail.

    # Ensure we have enough data for the sequence
    if len(df) < seq_length:
        logger.error(f"Insufficient data ({len(df)} rows) after processing for sequence length {seq_length}.")
        return None

    logger.info(f"Data loaded and prepared successfully. Final shape for scaling: {df.shape}")
    return df # Return the unscaled features DataFrame

# --- LSTM Model Definition ---
def build_seq2seq_model(input_shape, horizon_keys=['1d', '3d', '7d']):
    """
    Builds an advanced seq2seq model with encoder-decoder architecture.
    """
    logging.info(f"Building seq2seq model structure with input shape {input_shape} for inference.")
    encoder_inputs = Input(shape=input_shape, name='encoder_input')
    encoder_lstm1 = Bidirectional(LSTM(128, return_sequences=True), name='encoder_bilstm')
    encoder_output1 = encoder_lstm1(encoder_inputs)
    encoder_output1 = Dropout(0.25)(encoder_output1)
    encoder_lstm2 = LSTM(128, return_sequences=True, return_state=True, name='encoder_lstm')
    encoder_output2, state_h, state_c = encoder_lstm2(encoder_output1)

    attention = Dense(1, activation='tanh')(encoder_output2)
    attention = Flatten()(attention)
    attention_weights = Activation('softmax')(attention)
    attention_weights = RepeatVector(128)(attention_weights)
    attention_weights = Permute([2, 1])(attention_weights)
    context_vector = Multiply()([encoder_output2, attention_weights])
    # Use the globally defined function for Lambda layer consistency if needed, or standard Keras sum
    context_vector = Lambda(sum_over_time_axis, name='lambda_sum')(context_vector) # Use the named function
    # context_vector = Lambda(lambda x: K.sum(x, axis=1), name='lambda_sum')(context_vector) # Alternative direct sum

    encoder_states = [state_h, state_c]
    outputs = []
    output_names = [f'output_{h}' for h in horizon_keys]

    for h in horizon_keys:
        decoder_lstm = LSTM(128, name=f'decoder_lstm_{h}')
        decoder_input = RepeatVector(1)(context_vector)
        decoder_output = decoder_lstm(decoder_input, initial_state=encoder_states)
        decoder_dense1 = Dense(64, activation='relu', name=f'decoder_dense1_{h}')
        decoder_dropout1 = Dropout(0.2)
        decoder_dense2 = Dense(32, activation='relu', name=f'decoder_dense2_{h}')
        decoder_dropout2 = Dropout(0.2)
        decoder_output_layer = Dense(1, name=f'output_{h}') # Ensure name matches trained model

        dense_out1 = decoder_dense1(decoder_output)
        dense_out1 = decoder_dropout1(dense_out1)
        dense_out2 = decoder_dense2(dense_out1)
        dense_out2 = decoder_dropout2(dense_out2)
        final_output = decoder_output_layer(dense_out2)
        outputs.append(final_output)

    model = Model(inputs=encoder_inputs, outputs=outputs)
    logging.info(f"Model structure built successfully with {len(outputs)} outputs.")
    # model.summary(print_fn=logging.info) # Optional: Log summary
    return model

def get_next_trading_days(start_date, num_days):
    """
    Calculates the next N trading days from a given start date using US holidays as proxy.
    """
    if isinstance(start_date, str): start_date = pd.to_datetime(start_date)
    dk_holidays = USFederalHolidayCalendar()
    business_days = CustomBusinessDay(calendar=dk_holidays)
    future_date = start_date + business_days * num_days
    return future_date

if __name__ == "__main__":
    # Ensure the script can run independently for testing if needed
    # Load environment variables if running directly
    dotenv_path = Path(__file__).resolve().parents[2] / ".env"
    if dotenv_path.exists(): load_dotenv(dotenv_path=dotenv_path)

    uvicorn.run("stock_api:app", host="0.0.0.0", port=8000, reload=False) # Use reload=False for production/stable testing 