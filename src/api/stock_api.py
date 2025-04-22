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
RAW_DATA_DIR = DATA_DIR / "raw"
DB_FILE = RAW_DATA_DIR / "market_data.db"
DB_TABLE_NAME = "market_data"

# Data paths
VESTAS_DATA_FILE = DATA_DIR / "raw" / "stocks" / "vestas_macro_combined_trading_days.csv"  # Changed to combined file
VESTAS_DAILY_DATA_FILE = DATA_DIR / "raw" / "stocks" / "vestas_daily.csv"  # Alternative file
# Define the primary feature file path used for training and prediction
PROCESSED_FEATURES_FILE = DATA_DIR / "features" / "vestas_features_trading_days.csv" # Match training script path

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
                
                # Validate if feature file exists and contains required features
                if PROCESSED_FEATURES_FILE.exists():
                    try:
                        # Just read the column names without loading all data
                        feature_df = pd.read_csv(PROCESSED_FEATURES_FILE, nrows=1)
                        available_features = feature_df.columns.tolist()
                        
                        # Check for missing features
                        missing_features = set(lstm_feature_names) - set(available_features)
                        if missing_features:
                            logger.error(f"Feature file is missing {len(missing_features)} required features: {list(missing_features)[:5]}...")
                        else:
                            logger.info(f"Feature file contains all {len(lstm_feature_names)} required features")
                    except Exception as e:
                        logger.error(f"Error validating feature file: {str(e)}")
                else:
                    logger.error(f"Feature file for prediction not found: {PROCESSED_FEATURES_FILE}")
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
        # Try loading vestas_daily.csv first (robust method)
        if VESTAS_DAILY_DATA_FILE.exists():
            try:
                # Try to load with explicit date parsing
                vestas_data = pd.read_csv(VESTAS_DAILY_DATA_FILE, parse_dates=[0], index_col=0)
                vestas_data.index.name = 'date'
                
                # Verify we have valid dates
                if vestas_data.index.isna().any() or (vestas_data.index.year < 1980).any():
                    logger.warning("Some invalid dates in daily data, trying to fix...")
                    # Get valid rows
                    valid_rows = ~vestas_data.index.isna() & (vestas_data.index.year >= 1980)
                    if valid_rows.any():
                        vestas_data = vestas_data.loc[valid_rows]
                        logger.info(f"Filtered to {len(vestas_data)} valid rows")
                    else:
                        logger.error("No valid dates in daily data")
                        vestas_data = None
                
                if vestas_data is not None and len(vestas_data) > 0:
                    # Rename columns to ensure consistent naming
                    if 'Close' not in vestas_data.columns and 'close' in vestas_data.columns:
                        vestas_data.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'}, inplace=True)
                    
                    logger.info(f"Successfully loaded daily data with shape: {vestas_data.shape}")
                    logger.info(f"Date range: {vestas_data.index.min()} to {vestas_data.index.max()}")
                else:
                    logger.error("Failed to load valid daily data")
            except Exception as e:
                logger.error(f"Error loading daily data: {str(e)}")
                vestas_data = None
        else:
            logger.warning(f"Daily data file not found: {VESTAS_DAILY_DATA_FILE}")
            vestas_data = None
        
        # Try using features data as fallback if daily data failed
        if (vestas_data is None or len(vestas_data) == 0) and PROCESSED_FEATURES_FILE.exists():
            try:
                logger.info("Trying to use features file as fallback for price history")
                
                # Use the load_latest_data function for robust date parsing
                feature_data = load_latest_data(required_features=lstm_feature_names[:5])  # Only need a few features
                
                if feature_data is not None and len(feature_data) > 0:
                    # Extract key price columns for price history endpoint
                    price_cols = ['close', 'open', 'high', 'low', 'volume']
                    price_cols_cap = ['Close', 'Open', 'High', 'Low', 'Volume']
                    
                    vestas_data = pd.DataFrame(index=feature_data.index)
                    
                    # Try to find price columns in either lowercase or uppercase
                    for i, col in enumerate(price_cols):
                        if col in feature_data.columns:
                            vestas_data[price_cols_cap[i]] = feature_data[col]
                        elif price_cols_cap[i] in feature_data.columns:
                            vestas_data[price_cols_cap[i]] = feature_data[price_cols_cap[i]]
                    
                    # Make sure we have at least Close/close column
                    if 'Close' not in vestas_data.columns:
                        logger.error("Could not find price columns in features file")
                        vestas_data = None
                    else:
                        logger.info(f"Successfully created price history from features with shape: {vestas_data.shape}")
                        
                        # Save this as a new daily data file for future use
                        try:
                            vestas_data.to_csv(VESTAS_DAILY_DATA_FILE)
                            logger.info(f"Saved extracted price data to {VESTAS_DAILY_DATA_FILE}")
                        except Exception as e:
                            logger.error(f"Failed to save extracted price data: {str(e)}")
                else:
                    logger.error("Failed to load features file as fallback")
            except Exception as e:
                logger.error(f"Error using features file as fallback: {str(e)}")
        
        # Log final data status
        if vestas_data is None:
            logger.critical("No data loaded for /price/history endpoint. API will return errors.")
        else:
            logger.info(f"Final data for price history endpoint: {vestas_data.shape} rows")

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
        df_features = load_latest_data(required_features=lstm_feature_names)

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
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

def load_latest_data(required_features: List[str]):
    """
    Load the latest feature data for LSTM prediction.
    Prioritizes loading the specific feature file used in training.
    Handles NaN/inf values using preloaded medians.

    Returns:
        DataFrame with the latest data or None if loading fails.
    """
    source_path = PROCESSED_FEATURES_FILE
    if not source_path.exists():
        logger.error(f"Required feature file not found: {source_path}")
        return None

    try:
        # Load data, ensuring date parsing
        df = pd.read_csv(source_path, index_col=0, parse_dates=True)
        df.sort_index(inplace=True)
        logger.info(f"Loaded feature data from {source_path}. Shape: {df.shape}")

        # --- Data Cleaning ---
        # Check for required features *before* extensive cleaning
        missing_features = set(required_features) - set(df.columns)
        if missing_features:
            logger.error(f"Loaded data is missing required features: {missing_features}")
            return None

        # Handle infinities first
        for col in required_features:
             if df[col].isin([np.inf, -np.inf]).any():
                  inf_count = df[col].isin([np.inf, -np.inf]).sum()
                  logger.warning(f"Feature '{col}' contains {inf_count} inf values. Replacing with NaN.")
                  df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        # Handle NaNs using preloaded medians
        if not lstm_feature_medians:
             logger.warning("Feature medians not loaded. Cannot impute NaNs accurately.")
             # Fallback: fill with column median from the loaded data itself
             df[required_features] = df[required_features].fillna(df[required_features].median())
        else:
             for col in required_features:
                  if df[col].isna().any():
                       nan_count = df[col].isna().sum()
                       if col in lstm_feature_medians:
                           median_val = lstm_feature_medians[col]
                           logger.warning(f"Feature '{col}' contains {nan_count} NaNs. Filling with PRELOADED median ({median_val:.4f}).")
                           df[col].fillna(median_val, inplace=True)
                       else:
                           # Fallback if median for this specific feature wasn't loaded
                           fallback_median = df[col].median()
                           logger.error(f"Feature '{col}' contains {nan_count} NaNs, BUT median not preloaded! Using fallback median ({fallback_median:.4f}).")
                           df[col].fillna(fallback_median, inplace=True)

        # Final check for NaNs after imputation
        remaining_nans = df[required_features].isna().sum().sum()
        if remaining_nans > 0:
             logger.error(f"{remaining_nans} NaNs remain after imputation. Prediction might fail.")
             # Optional: Fill remaining with 0 or raise error
             df[required_features] = df[required_features].fillna(0)

        logger.info(f"Data loaded and cleaned successfully. Final shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Error loading or processing feature file {source_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

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