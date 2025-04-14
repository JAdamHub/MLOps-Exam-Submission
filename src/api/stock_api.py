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
async def get_price_history(days: Optional[int] = 7300):
    """Get Vestas stock price history"""
    try:
        # Use existing data or return an error if not available
        if vestas_data is not None and len(vestas_data) > 0:
            logger.info("Using existing data for API response")
            # Deep copy to avoid modifying original data
            df = vestas_data.copy()
            
            # Ensure dates are sorted if we're working with a DataFrame with a DatetimeIndex
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.sort_index()
                # Filter to the desired number of days
                if len(df) > days:
                    df = df.tail(days)
                    
                # Format data for response
                price_history = []
                
                for idx, row in df.iterrows():
                    try:
                        # Convert date to string from the index
                        date_str = idx.strftime("%Y-%m-%d") if isinstance(idx, pd.Timestamp) else str(idx)
                        
                        data_point = {
                            "date": date_str,
                            "price": float(row['Close']) if 'Close' in row and pd.notna(row['Close']) else float(row['close']) if 'close' in row and pd.notna(row['close']) else None,
                            "open": float(row['Open']) if 'Open' in row and pd.notna(row['Open']) else float(row['open']) if 'open' in row and pd.notna(row['open']) else None,
                            "high": float(row['High']) if 'High' in row and pd.notna(row['High']) else float(row['high']) if 'high' in row and pd.notna(row['high']) else None,
                            "low": float(row['Low']) if 'Low' in row and pd.notna(row['Low']) else float(row['low']) if 'low' in row and pd.notna(row['low']) else None,
                            "volume": int(row['Volume']) if 'Volume' in row and pd.notna(row['Volume']) else int(row['volume']) if 'volume' in row and pd.notna(row['volume']) else None
                        }
                        price_history.append(data_point)
                    except Exception as e:
                        logger.error(f"Error processing row: {e}")
                        continue
                
                logger.info(f"Returning {len(price_history)} price history records")
                return {"data": price_history}
            
            # Legacy support for DataFrame with a 'date' column
            elif 'date' in df.columns:
                # Ensure dates are sorted
                df = df.sort_values('date')
                
                # Filter to the desired number of days
                if len(df) > days:
                    df = df.tail(days)
                    
                # Format data for response
                price_history = []
                for _, row in df.iterrows():
                    try:
                        # Convert date to string
                        date_str = row['date'].strftime("%Y-%m-%d") if isinstance(row['date'], pd.Timestamp) else str(row['date'])
                        
                        data_point = {
                            "date": date_str,
                            "price": float(row['Close']) if 'Close' in row and pd.notna(row['Close']) else float(row['close']) if 'close' in row and pd.notna(row['close']) else None,
                            "open": float(row['Open']) if 'Open' in row and pd.notna(row['Open']) else float(row['open']) if 'open' in row and pd.notna(row['open']) else None,
                            "high": float(row['High']) if 'High' in row and pd.notna(row['High']) else float(row['high']) if 'high' in row and pd.notna(row['high']) else None,
                            "low": float(row['Low']) if 'Low' in row and pd.notna(row['Low']) else float(row['low']) if 'low' in row and pd.notna(row['low']) else None,
                            "volume": int(row['Volume']) if 'Volume' in row and pd.notna(row['Volume']) else int(row['volume']) if 'volume' in row and pd.notna(row['volume']) else None
                        }
                        price_history.append(data_point)
                    except Exception as e:
                        logger.error(f"Error processing row: {e}")
                        continue
                
                logger.info(f"Returning {len(price_history)} price history records")
                return {"data": price_history}
            else:
                logger.error("Data format not recognized - no date column or DatetimeIndex")
                raise HTTPException(status_code=500, detail="Data format not recognized")
        else:
            logger.error("No valid data found for price history")
            
            # Try to load data directly using the more robust load_latest_data function
            try:
                logger.info("Attempting to load data directly from features file as fallback")
                required_features = ['close', 'open', 'high', 'low', 'volume']
                fallback_data = load_latest_data(required_features=required_features)
                
                if fallback_data is not None and len(fallback_data) > 0:
                    # Format data for response
                    price_history = []
                    fallback_data = fallback_data.sort_index().tail(days)
                    
                    for idx, row in fallback_data.iterrows():
                        try:
                            # Convert date to string from the index
                            date_str = idx.strftime("%Y-%m-%d") if isinstance(idx, pd.Timestamp) else str(idx)
                            
                            data_point = {
                                "date": date_str,
                                "price": float(row['close']) if 'close' in row and pd.notna(row['close']) else None,
                                "open": float(row['open']) if 'open' in row and pd.notna(row['open']) else None,
                                "high": float(row['high']) if 'high' in row and pd.notna(row['high']) else None,
                                "low": float(row['low']) if 'low' in row and pd.notna(row['low']) else None,
                                "volume": int(row['volume']) if 'volume' in row and pd.notna(row['volume']) else None
                            }
                            price_history.append(data_point)
                        except Exception as e:
                            logger.error(f"Error processing fallback row: {e}")
                            continue
                    
                    if price_history:
                        logger.info(f"Returning {len(price_history)} price history records from direct load")
                        return {"data": price_history}
            except Exception as e:
                logger.error(f"Failed to load data directly: {e}")
            
            # If all else fails, raise an exception
            raise HTTPException(status_code=500, detail="No valid price history data available")
    except Exception as e:
        logger.error(f"Error generating price history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating price data: {str(e)}")

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
        if lstm_feature_scaler is None or not lstm_feature_names:
             logger.error("LSTM feature scaler or feature names not loaded.")
             raise HTTPException(status_code=500, detail="LSTM model artifacts (scaler/features) not loaded")
        if not lstm_target_scalers or not lstm_target_columns:
             logger.error("LSTM target scalers or target columns not loaded.")
             raise HTTPException(status_code=500, detail="LSTM model artifacts (target info) not loaded")
        
        # Get latest data using the improved function
        df = load_latest_data(required_features=lstm_feature_names)

        # Check if data loading was successful and sufficient
        if df is None:
             logger.error("Failed to load data for prediction.")
             raise HTTPException(status_code=500, detail="Could not load data for prediction.")
        if len(df) < lstm_sequence_length:
            logger.error(f"Not enough data available ({len(df)} rows) for sequence length {lstm_sequence_length}. Required at least {lstm_sequence_length} rows.")
            raise HTTPException(status_code=500, detail=f"Insufficient data for prediction. Need at least {lstm_sequence_length} rows, but only have {len(df)}.")
        
        # Prepare features
        logger.info(f"Preparing features from data with shape: {df.shape}")
        logger.info(f"Using features: {lstm_feature_names}")

        # Ensure all required features are present (double-check after load_latest_data)
        missing_features = set(lstm_feature_names) - set(df.columns)
        if missing_features:
            logger.error(f"FATAL: Missing required features after data loading: {missing_features}")
            raise HTTPException(status_code=500, detail=f"Data preparation failed: Missing features {missing_features}")

        # Extract features in the correct order and scale them
        X = df[lstm_feature_names].values # Use the exact order from training
        X_scaled = lstm_feature_scaler.transform(X)

        # Create sequences for LSTM input
        # Need at least lstm_sequence_length rows to make one sequence
        if len(X_scaled) < lstm_sequence_length:
             logger.error(f"Data length ({len(X_scaled)}) is less than sequence length ({lstm_sequence_length}) after scaling.")
             raise HTTPException(status_code=500, detail="Insufficient data length for LSTM sequence.")

        # Prepare the last sequence for prediction
        last_sequence_scaled = X_scaled[-lstm_sequence_length:]
        X_pred_seq = np.array([last_sequence_scaled]) # Reshape to (1, seq_length, n_features)

        logger.info(f"Created prediction sequence with shape: {X_pred_seq.shape}")

        # Make prediction
        try:
            # Use .predict() for TensorFlow models
            predictions = lstm_model.predict(X_pred_seq)
            logger.info(f"Model prediction successful. Output type: {type(predictions)}")
            if isinstance(predictions, list):
                 logger.info(f"  Number of outputs: {len(predictions)}")
                 for i, p in enumerate(predictions):
                     logger.info(f"  Output {i} shape: {p.shape}")
            elif isinstance(predictions, np.ndarray):
                 logger.info(f"  Output shape: {predictions.shape}")

        except Exception as e:
            logger.error(f"Error during model prediction: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")
        
        # Get last date from data
        last_date = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df['Date'].iloc[-1] if 'Date' in df else datetime.now())
        
        # Get last known price
        price_column = None
        for candidate in ['Close', 'close', 'price', 'Price']:
            if candidate in df.columns:
                price_column = candidate
                break
        
        if price_column is None:
            logger.warning("Could not find price column in data, using first column as fallback")
            last_price = float(df.iloc[:, 0].iloc[-1])
        else:
            last_price = float(df[price_column].iloc[-1])
        
        # Results for all forecast horizons
        results = {}
        
        # Check if this is a percent-change model or direct price prediction model
        # We look for metadata to determine this
        metadata_path = MODELS_DIR / "lstm_model_metadata.json"
        is_percent_change_model = False
        
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    is_percent_change_model = metadata.get('is_percent_change_model', False)
                    logger.info(f"Model is {'percent-change' if is_percent_change_model else 'direct-price'} prediction type")
            except Exception as e:
                logger.warning(f"Could not read model metadata: {e}. Assuming direct price prediction model.")

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
                pred_value = float(predictions[i][0][0])  # Ensure we're getting a scalar
                
                # Denormalize if scaler is available
                if target_col in lstm_target_scalers and lstm_target_scalers[target_col] is not None:
                    # Reshape to 2D for scaler and ensure we have the right shape
                    pred_array = np.array([[pred_value]])
                    # Ensure the shape is correct for inverse_transform
                    if pred_array.shape[1] != 1:
                        pred_array = pred_array.reshape(-1, 1)

                    scaler_instance = lstm_target_scalers[target_col]
                    logger.info(f"[{target_col}] Raw predicted value (scaled): {pred_value}")
                    logger.info(f"[{target_col}] Using scaler type: {type(scaler_instance)}")
                    if hasattr(scaler_instance, 'data_min_'): 
                         logger.info(f"[{target_col}] Scaler data_min_: {scaler_instance.data_min_}")
                    if hasattr(scaler_instance, 'data_max_'): 
                         logger.info(f"[{target_col}] Scaler data_max_: {scaler_instance.data_max_}")
                    if hasattr(scaler_instance, 'data_range_'): 
                         logger.info(f"[{target_col}] Scaler data_range_: {scaler_instance.data_range_}")

                    # Log scaler info for debugging
                    logger.info(f"Using scaler for {target_col}: {type(lstm_target_scalers[target_col])}")
                    if hasattr(lstm_target_scalers[target_col], 'scale_'):
                        logger.info(f"Scale: {lstm_target_scalers[target_col].scale_}")
                    if hasattr(lstm_target_scalers[target_col], 'min_'):
                        logger.info(f"Min: {lstm_target_scalers[target_col].min_}")
                    if hasattr(lstm_target_scalers[target_col], 'data_min_'):
                        logger.info(f"Data min: {lstm_target_scalers[target_col].data_min_}")
                    if hasattr(lstm_target_scalers[target_col], 'data_max_'):
                        logger.info(f"Data max: {lstm_target_scalers[target_col].data_max_}")
                    logger.info(f"Input shape for inverse_transform: {pred_array.shape}")
                    # Denormalize
                    prediction_denorm = lstm_target_scalers[target_col].inverse_transform(pred_array)[0][0]
                    logger.info(f"Denormalized prediction: {prediction_denorm}")
                else:
                    # If no scaler, use raw value
                    prediction_denorm = pred_value
                    logger.warning(f"[{target_col}] No scaler found. Using raw predicted value: {prediction_denorm}") 
                
                # If model predicts percent change, convert to absolute price
                if is_percent_change_model:
                    # prediction_denorm is now the percent change
                    logger.info(f"Converting percent change {prediction_denorm:.2f}% to price with base price {last_price:.2f}")
                    
                    # Convert percent to absolute price change
                    price_prediction = last_price * (1 + prediction_denorm/100)
                    logger.info(f"Percent-based price prediction: {price_prediction:.2f}")
                else:
                    # Model predicts price directly
                    price_prediction = prediction_denorm
                price_prediction = constrain_price_prediction(price_prediction, last_price, horizon)
                    
                # Calculate future date based on trading days, not calendar days
                future_date = get_next_trading_days(last_date, horizon).strftime('%Y-%m-%d')
                
                # Save result for this horizon
                results[str(horizon)] = {
                    "predicted_price": float(round(price_prediction, 2)),
                    "prediction_date": future_date,
                    "horizon_days": horizon,
                    "trading_days": horizon  # Clarify that this is trading days
                }
        else:
            # Single output model
            logger.info(f"Processing single-output predictions with shape: {predictions.shape}")
            
            # Check if output is 3D [batch, seq_len, features] or 2D [batch, features]
            if len(predictions.shape) == 3:
                # If 3D, take the last time step
                pred_values = predictions[0, -1, :]
            else:
                # If 2D, use as is
                pred_values = predictions[0]
                
            # Ensure we have the right number of outputs
            if len(pred_values) >= len(lstm_target_columns):
                for i, target_col in enumerate(lstm_target_columns):
                    horizon = int(target_col.split('_')[-1].replace('d', ''))
                    
                    # Skip if we're only returning one horizon and this isn't it
                    if not return_all and horizon != days_ahead:
                        continue
                        
                    # For single output, use same prediction for all horizons (simplified)
                    pred_value = float(pred_values[i])  # Ensure we're getting a scalar
                    
                    # Denormalize if scaler is available
                    if target_col in lstm_target_scalers and lstm_target_scalers[target_col] is not None:
                        # Reshape to 2D for scaler and ensure we have the right shape
                        pred_array = np.array([[pred_value]])
                        # Ensure the shape is correct for inverse_transform
                        if pred_array.shape[1] != 1:
                            pred_array = pred_array.reshape(-1, 1)

                        scaler_instance = lstm_target_scalers[target_col]
                        logger.info(f"[{target_col}] Raw predicted value (scaled): {pred_value}")
                        logger.info(f"[{target_col}] Using scaler type: {type(scaler_instance)}")
                        if hasattr(scaler_instance, 'data_min_'): 
                             logger.info(f"[{target_col}] Scaler data_min_: {scaler_instance.data_min_}")
                        if hasattr(scaler_instance, 'data_max_'): 
                             logger.info(f"[{target_col}] Scaler data_max_: {scaler_instance.data_max_}")
                        if hasattr(scaler_instance, 'data_range_'): 
                             logger.info(f"[{target_col}] Scaler data_range_: {scaler_instance.data_range_}")

                        # Log scaler info for debugging
                        logger.info(f"Using scaler for {target_col}: {type(lstm_target_scalers[target_col])}")
                        if hasattr(lstm_target_scalers[target_col], 'scale_'):
                            logger.info(f"Scale: {lstm_target_scalers[target_col].scale_}")
                        if hasattr(lstm_target_scalers[target_col], 'min_'):
                            logger.info(f"Min: {lstm_target_scalers[target_col].min_}")
                        if hasattr(lstm_target_scalers[target_col], 'data_min_'):
                            logger.info(f"Data min: {lstm_target_scalers[target_col].data_min_}")
                        if hasattr(lstm_target_scalers[target_col], 'data_max_'):
                            logger.info(f"Data max: {lstm_target_scalers[target_col].data_max_}")
                        logger.info(f"Input shape for inverse_transform: {pred_array.shape}")
                        # Denormalize
                        prediction_denorm = lstm_target_scalers[target_col].inverse_transform(pred_array)[0][0]
                        logger.info(f"Denormalized prediction: {prediction_denorm}")
                    else:
                        # If no scaler, use raw value
                        prediction_denorm = pred_value
                        logger.warning(f"[{target_col}] No scaler found. Using raw predicted value: {prediction_denorm}") 
                        
                    # If model predicts percent change, convert to absolute price
                    if is_percent_change_model:
                        # prediction_denorm is now the percent change
                        logger.info(f"Converting percent change {prediction_denorm:.2f}% to price with base price {last_price:.2f}")
                        
                        # Convert percent to absolute price change
                        price_prediction = last_price * (1 + prediction_denorm/100)
                        logger.info(f"Percent-based price prediction: {price_prediction:.2f}")
                    else:
                        # Model predicts price directly
                        price_prediction = prediction_denorm
                    price_prediction = constrain_price_prediction(price_prediction, last_price, horizon)
                        
                    # Calculate future date based on trading days, not calendar days
                    future_date = get_next_trading_days(last_date, horizon).strftime('%Y-%m-%d')
                    
                    # Save result for this horizon
                    results[str(horizon)] = {
                        "predicted_price": float(round(price_prediction, 2)),
                        "prediction_date": future_date,
                        "horizon_days": horizon,
                        "trading_days": horizon  # Clarify that this is trading days
                    }
            else:
                # If output doesn't match expected number of targets
                logger.error(f"Model output size {len(pred_values)} doesn't match target columns {len(lstm_target_columns)}")
                raise HTTPException(status_code=500, detail=f"Model output size {len(pred_values)} doesn't match expected target columns {len(lstm_target_columns)}")
            
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
        
def load_latest_data(required_features: List[str]):
    """
    Load the latest feature data for LSTM prediction.
    Prioritizes loading the same feature file used in training.
    Checks for required features and avoids adding random data.

    Returns:
        DataFrame with the latest data or None if loading fails or features are missing.
    """
    # --- Load the specific feature file used in training --- 
    source_path = PROCESSED_FEATURES_FILE # Use the globally defined correct path

    if not source_path.exists():
        logger.error(f"Required feature file for prediction not found: {source_path}")
        return None

    try:
        # Try different approaches to read the file with proper date parsing
        try:
            # First attempt: Standard CSV read with default options
            df = pd.read_csv(source_path)
            logger.info(f"Successfully loaded feature data from: {source_path}. Shape: {df.shape}")
        except Exception as e:
            logger.error(f"Standard read of {source_path} failed: {e}")
            try:
                # Second attempt: Try with explicit parameters
                df = pd.read_csv(source_path, parse_dates=['date'], infer_datetime_format=True)
                logger.info(f"Loaded data with explicit date parsing. Shape: {df.shape}")
            except Exception as e2:
                logger.error(f"Explicit date parsing failed: {e2}")
                try:
                    # Third attempt: Try reading first using string types, then convert date manually
                    df = pd.read_csv(source_path, dtype={'date': str})
                    logger.info(f"Loaded data with string date column. Shape: {df.shape}")
                except Exception as e3:
                    logger.error(f"All data loading attempts failed: {e3}")
                    return None
    except Exception as e:
        logger.error(f"Error loading feature file {source_path}: {e}")
        return None

    # --- Post-loading checks and preparation ---
    if df is None:
        logger.error("Dataframe is None after attempting to load file.")
        return None

    # Try harder to ensure date column is datetime and set as index
    date_col_found = False
    
    # First look for standard date columns
    date_columns = ['Date', 'date', 'datetime', 'time', 'timestamp']
    
    for date_col in date_columns:
        if date_col in df.columns:
            try:
                # Try various date formats
                try:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                except:
                    # Try ISO format explicitly
                    df[date_col] = pd.to_datetime(df[date_col], format='%Y-%m-%d', errors='coerce')
                
                # Verify successful date conversion
                logger.info(f"{date_col} column converted to datetime. Sample values: {df[date_col].head(3).tolist()}")
                
                # Fix bad dates by looking for invalid values and filtering
                invalid_dates = df[date_col].isna() | (df[date_col].dt.year < 1980)
                if invalid_dates.any():
                    logger.warning(f"Removing {invalid_dates.sum()} rows with invalid dates")
                    df = df[~invalid_dates]
                
                df = df.set_index(date_col)
                date_col_found = True
                logger.info(f"Set {date_col} as index. Date range: {df.index.min()} to {df.index.max()}")
                break
            except Exception as e:
                logger.error(f"Error converting '{date_col}' column to datetime: {e}")
    
    # If no date column, try using the first column as a date
    if not date_col_found and len(df.columns) > 0:
        first_col = df.columns[0]
        try:
            logger.info(f"Trying to use first column '{first_col}' as date")
            df[first_col] = pd.to_datetime(df[first_col], errors='coerce')
            
            # Filter invalid dates
            invalid_dates = df[first_col].isna() | (df[first_col].dt.year < 1980)
            if invalid_dates.any():
                logger.warning(f"Removing {invalid_dates.sum()} rows with invalid dates from first column")
                df = df[~invalid_dates]
                
            df = df.set_index(first_col)
            date_col_found = True
            logger.info(f"Used first column as index. Date range: {df.index.min()} to {df.index.max()}")
        except Exception as e:
            logger.error(f"Error using first column as date: {e}")

    # Last resort: create an artificial date index
    if not date_col_found:
        logger.warning("No date column found. Creating artificial date index.")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=len(df))
        df.index = pd.date_range(start=start_date, periods=len(df))
        logger.info(f"Created artificial date index. Range: {df.index.min()} to {df.index.max()}")

    # Sort by date
    df = df.sort_index()
    logger.info(f"Data sorted by date. Date range: {df.index.min()} to {df.index.max()}")

    # Check for required features *before* returning
    missing_features = set(required_features) - set(df.columns)
    if missing_features:
        logger.error(f"Loaded data from {source_path} is missing required features: {missing_features}")
        return None

    # Handle potential infinities and NaNs in features (use median from training if possible, else log warning)
    for col in required_features:
         if df[col].isin([np.inf, -np.inf]).any():
              inf_count = df[col].isin([np.inf, -np.inf]).sum()
              logger.warning(f"Feature '{col}' contains {inf_count} infinite values. Replacing with NaN.")
              df[col] = df[col].replace([np.inf, -np.inf], np.nan)
         if df[col].isna().any():
              nan_count = df[col].isna().sum()
              if col in lstm_feature_medians:
                  median_val = lstm_feature_medians[col]
                  logger.warning(f"Feature '{col}' contains {nan_count} NaN values. Filling with PRELOADED median ({median_val}).")
                  df[col] = df[col].fillna(median_val)
              else:
                  # Fallback if median for this specific feature wasn't loaded (should not happen ideally)
                  fallback_median = df[col].median()
                  logger.error(f"Feature '{col}' contains {nan_count} NaN values, BUT median not found in preloaded medians! Using fallback median ({fallback_median}).")
                  df[col] = df[col].fillna(fallback_median)

    logger.info(f"Data loaded successfully from {source_path}. Final shape for prediction preparation: {df.shape}")
    return df

# --- LSTM Model Definition ---
def build_seq2seq_model(input_shape, horizon_keys=['1d', '3d', '7d']):
    """
    Builds an advanced seq2seq model with encoder-decoder architecture for multiple time horizons.
    Implements a basic attention mechanism for better forecasting.
    
    Args:
        input_shape: Tuple with (seq_length, n_features)
        horizon_keys: List of time horizons to predict (matches output names)
        
    Returns:
        Keras model (not compiled, as it's not necessary for inference)
    """
    logging.info(f"Building seq2seq model structure with input shape {input_shape} for inference.")
    
    # Input layer
    encoder_inputs = Input(shape=input_shape, name='encoder_input')
    
    # Encoder LSTM layers
    encoder_lstm1 = Bidirectional(LSTM(128, return_sequences=True), name='encoder_bilstm')
    encoder_output1 = encoder_lstm1(encoder_inputs)
    encoder_output1 = Dropout(0.25)(encoder_output1) # Dropout is also used during inference
    
    encoder_lstm2 = LSTM(128, return_sequences=True, return_state=True, name='encoder_lstm')
    encoder_output2, state_h, state_c = encoder_lstm2(encoder_output1)
    
    # Attention mechanism
    attention = Dense(1, activation='tanh')(encoder_output2)
    attention = Flatten()(attention)
    attention_weights = Activation('softmax')(attention)
    attention_weights = RepeatVector(128)(attention_weights)
    attention_weights = Permute([2, 1])(attention_weights)
    
    # Apply attention to encoder output
    context_vector = Multiply()([encoder_output2, attention_weights])
    # Use the defined function for the Lambda layer
    context_vector = Lambda(sum_over_time_axis, name='lambda_sum')(context_vector)
    
    # Save encoder states for decoder initialization
    encoder_states = [state_h, state_c]
    
    # Outputs for each horizon
    outputs = [] # Use a list to maintain order
    output_names = [f'output_{h}' for h in horizon_keys]
    
    for h in horizon_keys:
        # Decoder with attention
        # Ensure layer names match the saved model exactly
        decoder_lstm = LSTM(128, name=f'decoder_lstm_{h}') 
        # Repeat context vector for decoder input
        decoder_input = RepeatVector(1)(context_vector)
        decoder_output = decoder_lstm(decoder_input, initial_state=encoder_states)
        
        # Dense layers for forecast
        decoder_dense1 = Dense(64, activation='relu', name=f'decoder_dense1_{h}')
        decoder_dropout1 = Dropout(0.2) # Dropout is also used during inference
        decoder_dense2 = Dense(32, activation='relu', name=f'decoder_dense2_{h}')
        decoder_dropout2 = Dropout(0.2) # Dropout is also used during inference
        decoder_output_layer = Dense(1, name=f'output_{h}') # Ensure name matches
        
        dense_out1 = decoder_dense1(decoder_output)
        dense_out1 = decoder_dropout1(dense_out1)
        dense_out2 = decoder_dense2(dense_out1)
        dense_out2 = decoder_dropout2(dense_out2)
        final_output = decoder_output_layer(dense_out2)
        outputs.append(final_output)
    
    # Build model
    model = Model(inputs=encoder_inputs, outputs=outputs)
    
    logging.info(f"Model structure built successfully with {len(outputs)} outputs.")
    # model.summary(print_fn=logging.info) # Optional: Log summary for verification
    return model

def is_trading_day(date):
    """
    Checks if a given date is a trading day (not weekend or holiday).
    
    Args:
        date: Date to check (datetime)
        
    Returns:
        bool: True if trading day, False otherwise
    """
    # Skip weekends
    if date.weekday() >= 5:  # 5=Saturday, 6=Sunday
        return False
    
    # Simplified holiday check (should be expanded for production)
    # Here we just use US holidays as an approximation
    us_holidays = USFederalHolidayCalendar()
    holidays = us_holidays.holidays(start=date, end=date)
    return len(holidays) == 0

def get_next_trading_days(start_date, num_days):
    """
    Calculates the next N trading days from a given start date.
    
    Args:
        start_date: Starting date (datetime)
        num_days: Number of trading days to advance
        
    Returns:
        datetime: Date that is num_days trading days after start_date
    """
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    
    # Define custom business day that excludes weekends and holidays
    dk_holidays = USFederalHolidayCalendar()  # Approximation, should use Danish calendar
    business_days = CustomBusinessDay(calendar=dk_holidays)
    
    # Get the date that is num_days trading days ahead
    future_date = start_date + business_days * num_days
    
    return future_date

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("stock_api:app", host="0.0.0.0", port=8000, reload=False) 