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
VESTAS_DATA_FILE = DATA_DIR / "raw" / "stocks" / "vestas_macro_combined_trading_days.csv"  # Ændret til den kombinerede fil
VESTAS_DAILY_DATA_FILE = DATA_DIR / "raw" / "stocks" / "vestas_daily.csv"  # Alternativ fil
# Define the primary feature file path used for training and prediction
PROCESSED_FEATURES_FILE = DATA_DIR / "features" / "vestas_features_trading_days.csv" # Match training script path

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
lstm_sequence_length = 30  # Default værdi, overskrives ved indlæsning
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
            # Identificer datokolonnen korrekt - første kolonne er en ikke-navngivet datofelt
            vestas_data = pd.read_csv(VESTAS_DATA_FILE, parse_dates=['Unnamed: 0'])
            
            # Omdøb den første kolonne til 'date' for konsistent navngivning
            if 'Unnamed: 0' in vestas_data.columns:
                vestas_data.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
                logger.info(f"Renamed first column to 'date'")
            
            # Sørg for at vi har en dato-kolonne
            if 'Date' in vestas_data.columns:
                vestas_data['date'] = pd.to_datetime(vestas_data['Date'], errors='coerce')
                logger.info(f"Date range in data: {vestas_data['date'].min()} to {vestas_data['date'].max()}")
            elif 'date' not in vestas_data.columns:
                # Hvis vi ikke har en dato-kolonne, prøv at bruge indekset
                vestas_data['date'] = pd.to_datetime(vestas_data.index, errors='coerce')
                logger.info(f"Created date from index, range: {vestas_data['date'].min()} to {vestas_data['date'].max()}")
            
            # Tjek om datoekolonnen er korrekt formateret
            if 'date' in vestas_data.columns:
                # Log datointervallet
                logger.info(f"Date range in data: {vestas_data['date'].min()} to {vestas_data['date'].max()}")
                
                # Handle bad dates by creating a dummy date range if needed
                if vestas_data['date'].isna().all() or (pd.notna(vestas_data['date'].min()) and vestas_data['date'].min().year < 1980):
                    logger.warning("Invalid dates detected, creating synthetic date range")
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=len(vestas_data))
                    vestas_data['date'] = pd.date_range(start=start_date, periods=len(vestas_data))
            
            # Rename columns if needed
            if 'Close' not in vestas_data.columns and 'close' in vestas_data.columns:
                vestas_data.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'}, inplace=True)
                
            logger.info(f"Vestas data loaded with shape: {vestas_data.shape}")
            
            # Debug: Log nogle af datoerne for at bekræfte at de er korrekte
            logger.info(f"First 5 dates: {vestas_data['date'].head(5).tolist()}")
            logger.info(f"Last 5 dates: {vestas_data['date'].tail(5).tolist()}")
        elif VESTAS_DAILY_DATA_FILE.exists():
            # Try alternative file
            vestas_data = pd.read_csv(VESTAS_DAILY_DATA_FILE)
            # Ensure date column exists and is properly formatted
            if 'Date' in vestas_data.columns:
                # Try to convert the Date column to datetime
                vestas_data['date'] = pd.to_datetime(vestas_data['Date'], errors='coerce')
                logger.info(f"Date range in data: {vestas_data['date'].min()} to {vestas_data['date'].max()}")
            elif 'date' in vestas_data.columns:
                # If date column already exists, ensure it's datetime
                vestas_data['date'] = pd.to_datetime(vestas_data['date'], errors='coerce')
                logger.info(f"Date range in data: {vestas_data['date'].min()} to {vestas_data['date'].max()}")
            else:
                # Create date from index if no date column
                vestas_data['date'] = pd.to_datetime(vestas_data.index, errors='coerce')
                logger.info(f"Created date from index, range: {vestas_data['date'].min()} to {vestas_data['date'].max()}")
            
            # Handle bad dates by creating a dummy date range if needed
            if vestas_data['date'].isna().all() or (vestas_data['date'].min().year < 1980):
                logger.warning("Invalid dates detected, creating synthetic date range")
                end_date = datetime.now()
                start_date = end_date - timedelta(days=len(vestas_data))
                vestas_data['date'] = pd.date_range(start=start_date, periods=len(vestas_data))
                
            # Rename columns if needed
            if 'Close' not in vestas_data.columns and 'close' in vestas_data.columns:
                vestas_data.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'}, inplace=True)
                
            logger.info(f"Vestas daily data loaded with shape: {vestas_data.shape}")
        else:
            logger.error(f"Vestas data file not found: {VESTAS_DATA_FILE} or {VESTAS_DAILY_DATA_FILE}")
            
        # Try to load processed data if available (This part might be less relevant now, focus is on PROCESSED_FEATURES_FILE)
        if not PROCESSED_FEATURES_FILE.exists():
            logger.warning(f"Core feature file for predictions not found at startup: {PROCESSED_FEATURES_FILE}")

        # If vestas_data is still None, we might log a warning, but prediction relies on PROCESSED_FEATURES_FILE
        # This initial data loading primarily serves the /price/history endpoint.
        # The /predict/lstm endpoint uses load_latest_data which specifically loads PROCESSED_FEATURES_FILE.
        if vestas_data is None:
            logger.warning("No raw/daily data loaded for /price/history endpoint. It might rely on synthetic data.")

        # Remove merging logic based on intermediate file during startup
        '''
        if PROCESSED_FEATURES_FILE.exists():
            processed_data = pd.read_csv(PROCESSED_FEATURES_FILE)
            if not vestas_data is None:
                # Extract date and merge with original data
                if 'Date' in processed_data.columns:
                    processed_data['date'] = pd.to_datetime(processed_data['Date'], errors='coerce')
                    vestas_data = pd.merge(vestas_data, processed_data, on='date', how='outer')
                logger.info(f"Processed features loaded and merged, new shape: {vestas_data.shape}")
            else:
                vestas_data = processed_data
                vestas_data['date'] = pd.to_datetime(vestas_data['Date'] if 'Date' in vestas_data.columns else vestas_data.index, errors='coerce')
                # Handle bad dates by creating a dummy date range if needed
                if vestas_data['date'].isna().all() or (vestas_data['date'].min().year < 1980):
                    logger.warning("Invalid dates detected in processed data, creating synthetic date range")
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=len(vestas_data))
                    vestas_data['date'] = pd.date_range(start=start_date, periods=len(vestas_data))
                logger.info(f"Using processed features as primary data, shape: {vestas_data.shape}")
        '''

        # Generate synthetic data if no valid data is loaded for /price/history
        if vestas_data is None or len(vestas_data) == 0:
            logger.warning("No valid data loaded, generating synthetic dataset")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)  # One year of data
            dates = pd.date_range(start=start_date, end=end_date)
            
            # Create synthetic price data
            base_price = 150
            prices = np.linspace(base_price, base_price*1.2, len(dates)) + np.random.normal(0, 5, len(dates))
            
            vestas_data = pd.DataFrame({
                'date': dates,
                'Close': prices,
                'Open': prices * 0.99,
                'High': prices * 1.01,
                'Low': prices * 0.98,
                'Volume': np.random.randint(10000, 50000, len(dates))
            })
            logger.info(f"Synthetic data generated with shape: {vestas_data.shape}")
    except Exception as e:
        logger.error(f"Error loading Vestas data: {str(e)}")
        # Generate emergency synthetic data
        logger.warning("Generating emergency synthetic data due to data loading error")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # One year of data
        dates = pd.date_range(start=start_date, end=end_date)
        
        # Create synthetic price data
        base_price = 150
        prices = np.linspace(base_price, base_price*1.2, len(dates)) + np.random.normal(0, 5, len(dates))
        
        vestas_data = pd.DataFrame({
            'date': dates,
            'Close': prices,
            'Open': prices * 0.99,
            'High': prices * 1.01,
            'Low': prices * 0.98,
            'Volume': np.random.randint(10000, 50000, len(dates))
        })
        logger.info(f"Emergency synthetic data generated with shape: {vestas_data.shape}")

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
            lstm_feature_scaler = joblib.load(LSTM_FEATURE_SCALER_FILE)
            logger.info(f"LSTM feature scaler loaded ({type(lstm_feature_scaler)})")
            # Log StandardScaler specific attributes
            if hasattr(lstm_feature_scaler, 'mean_') and hasattr(lstm_feature_scaler, 'scale_'):
                 logger.info(f"  Feature Scaler Mean (partial): {lstm_feature_scaler.mean_[:5]}...") # Log first 5 means
                 logger.info(f"  Feature Scaler Scale (partial): {lstm_feature_scaler.scale_[:5]}...") # Log first 5 scales
            else:
                 logger.warning("Loaded feature scaler might not be a StandardScaler or lacks expected attributes.")
        else:
            logger.error(f"LSTM feature scaler file not found: {LSTM_FEATURE_SCALER_FILE}")

        # --- Load target scalers ---
        if LSTM_TARGET_SCALERS_FILE.exists():
            lstm_target_scalers = joblib.load(LSTM_TARGET_SCALERS_FILE)
            logger.info(f"LSTM target scalers loaded ({len(lstm_target_scalers)} scalers)")
            # Log scaler info for debugging
            for target_col, scaler in lstm_target_scalers.items():
                logger.info(f"Scaler for {target_col}: {type(scaler)}")
                # Log StandardScaler specific attributes
                if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
                    logger.info(f"  Target Scaler Mean: {scaler.mean_}")
                    logger.info(f"  Target Scaler Scale: {scaler.scale_}")
                else:
                    logger.warning(f"Loaded target scaler for {target_col} might not be a StandardScaler or lacks expected attributes.")
        else:
            logger.error(f"LSTM target scalers file not found: {LSTM_TARGET_SCALERS_FILE}")

    except Exception as e:
        logger.error(f"Error loading LSTM model and artifacts: {str(e)}")
        # Ensure globals are None if errors occur during loading
        lstm_model = None
        lstm_feature_scaler = None
        lstm_target_scalers = {}
        lstm_feature_names = []
        lstm_target_columns = []

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
    try:
        # Brug enten de eksisterende data eller generer nye hvis nødvendigt
        if vestas_data is not None and 'date' in vestas_data.columns and not vestas_data['date'].isna().all():
            logger.info("Using existing data for API response")
            # Deep copy for at undgå at ændre originale data
            df = vestas_data.copy()
            
            # Sørg for at datoer er sorteret
            df = df.sort_values('date')
            
            # Filtrer til det ønskede antal dage
            if len(df) > days:
                df = df.tail(days)
                
            # Format data for response
            price_history = []
            for _, row in df.iterrows():
                try:
                    # Konverter dato til string
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
                    
            return {"data": price_history}
        else:
            logger.warning("No valid data found, generating synthetic data")
            # Generer helt nye syntetiske data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            dates = pd.date_range(start=start_date, end=end_date)
            
            # Create synthetic price data with some realism
            base_price = 150
            # Add randomness with a slight upward trend
            noise = np.random.normal(0, 5, len(dates))
            trend = np.linspace(0, 10, len(dates))
            seasonal = 5 * np.sin(np.linspace(0, 4*np.pi, len(dates)))  # seasonal pattern
            
            prices = base_price + trend + seasonal + noise
            
            # Create synthetic dataframe with valid dates
            df = pd.DataFrame({
                'date': dates,
                'Close': prices,
                'Open': prices * 0.99 + np.random.normal(0, 1, len(dates)),
                'High': prices * 1.02 + np.random.normal(0, 1, len(dates)),
                'Low': prices * 0.98 + np.random.normal(0, 1, len(dates)),
                'Volume': np.random.randint(100000, 500000, len(dates))
            })
            
            # Take only the requested number of days
            if len(df) > days:
                df = df.tail(days)
            
            # Format data for response
            price_history = []
            for _, row in df.iterrows():
                data_point = {
                    "date": row['date'].strftime("%Y-%m-%d"),
                    "price": float(row['Close']),
                    "open": float(row['Open']),
                    "high": float(row['High']),
                    "low": float(row['Low']),
                    "volume": int(row['Volume'])
                }
                price_history.append(data_point)
            
            return {"data": price_history}
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
            logger.warning(f"Not enough data available ({len(df)} rows) for sequence length {lstm_sequence_length}. Required at least {lstm_sequence_length} rows.")
            # For demo purposes, return simulated values if data is not available
            # (Keep existing simulation logic for now)
            logger.warning("Returning simulated values due to insufficient data.")
            results = {}
            
            # Get current date in proper format
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            for horizon in [1, 3, 7]:
                if not return_all and horizon != days_ahead:
                    continue
                    
                # Ensure future date is properly formatted
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
                "last_price_date": current_date,
                "model_type": "LSTM (Multi-horizon)",
                "note": "Using simulated values due to insufficient data"
            }
        
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
            # (Keep existing simulation logic for error case)
            logger.warning("Returning simulated values due to model prediction error.")
            results = {}
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            for horizon in [1, 3, 7]:
                if not return_all and horizon != days_ahead:
                    continue
                    
                # Ensure future date is properly formatted
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
                "last_price_date": current_date,
                "model_type": "LSTM (Multi-horizon)",
                "note": "Using simulated values due to model prediction error"
            }
        
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
                pred_value = float(predictions[i][0][0])  # Ensure we're getting a scalar
                
                # Denormalize if scaler is available
                if target_col in lstm_target_scalers and lstm_target_scalers[target_col] is not None:
                    # Reshape to 2D for scaler and ensure we have the right shape
                    pred_array = np.array([[pred_value]])
                    # Ensure the shape is correct for inverse_transform
                    if pred_array.shape[1] != 1:
                        pred_array = pred_array.reshape(-1, 1)
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
                        
                    # Calculate future date
                    future_date = (last_date + pd.Timedelta(days=horizon)).strftime('%Y-%m-%d')
                    
                    # Save result for this horizon
                    results[str(horizon)] = {
                        "predicted_price": float(round(prediction_denorm, 2)),
                        "prediction_date": future_date,
                        "horizon_days": horizon
                    }
            else:
                # If output doesn't match expected number of targets
                logger.warning(f"Model output size {len(pred_values)} doesn't match target columns {len(lstm_target_columns)}")
                # For demo purposes, return simulated values
                for i, target_col in enumerate(lstm_target_columns):
                    horizon = int(target_col.split('_')[-1].replace('d', ''))
                    
                    # Skip if we're only returning one horizon and this isn't it
                    if not return_all and horizon != days_ahead:
                        continue
                        
                    future_date = (last_date + pd.Timedelta(days=horizon)).strftime('%Y-%m-%d')
                    results[str(horizon)] = {
                        "predicted_price": float(round(last_price * (1 + horizon * 0.01), 2)),
                        "prediction_date": future_date,
                        "horizon_days": horizon,
                        "simulated": True
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
        df = pd.read_csv(source_path)
        logger.info(f"Successfully loaded feature data from: {source_path}. Shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error loading feature file {source_path}: {e}")
        return None

    # --- Post-loading checks and preparation ---
    if df is None:
        # This case should technically not be reached if file loading fails above,
        # but added as a safeguard.
        logger.error("Dataframe is None after attempting to load file.")
        return None

    # Ensure date column is datetime and set as index
    date_col_found = False
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        if df['Date'].isna().any():
            logger.warning("Some dates in 'Date' column could not be parsed. Dropping these rows.")
            df = df.dropna(subset=['Date'])
        df = df.set_index('Date')
        date_col_found = True
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if df['date'].isna().any():
            logger.warning("Some dates in 'date' column could not be parsed. Dropping these rows.")
            df = df.dropna(subset=['date'])
        df = df.set_index('date')
        date_col_found = True

    if not date_col_found:
         logger.warning("No 'Date' or 'date' column found. Trying to use index if it's DatetimeIndex.")
         if not isinstance(df.index, pd.DatetimeIndex):
              logger.error("Data does not have a usable date column or DatetimeIndex.")
              return None # Cannot proceed without proper time ordering

    # Sort by date
    df = df.sort_index()

    # Check for required features *before* returning
    missing_features = set(required_features) - set(df.columns)
    if missing_features:
        logger.error(f"Loaded data from {source_path} is missing required features: {missing_features}")
        return None

    # Handle potential infinities and NaNs in features (use median from training if possible, else log warning)
    for col in required_features:
         if df[col].isin([np.inf, -np.inf]).any():
              logger.warning(f"Feature '{col}' contains infinite values. Replacing with NaN.")
              df[col] = df[col].replace([np.inf, -np.inf], np.nan)
         if df[col].isna().any():
              median_val = df[col].median() # Calculate median from the loaded data as fallback
              logger.warning(f"Feature '{col}' contains NaN values. Filling with median ({median_val}).")
              df[col] = df[col].fillna(median_val)

    logger.info(f"Data loaded successfully from {source_path}. Final shape for prediction preparation: {df.shape}")
    return df

# --- LSTM Model Definition (Copied from training script) ---
def build_seq2seq_model(input_shape, horizon_keys=['1d', '3d', '7d']):
    """
    Bygger en avanceret seq2seq model med encoder-decoder arkitektur til flere tidshorisonter.
    Implementerer basal attention mekanisme for bedre forecasting.
    
    Args:
        input_shape: Tuple med (seq_length, n_features)
        horizon_keys: Liste med tidshorisonter der skal forudsiges (matches til output navne)
        
    Returns:
        Keras model (ikke kompileret, da det ikke er nødvendigt for inference)
    """
    logging.info(f"Building seq2seq model structure with input shape {input_shape} for inference.")
    
    # Input lag
    encoder_inputs = Input(shape=input_shape, name='encoder_input')
    
    # Encoder LSTM-lag
    encoder_lstm1 = Bidirectional(LSTM(128, return_sequences=True), name='encoder_bilstm')
    encoder_output1 = encoder_lstm1(encoder_inputs)
    encoder_output1 = Dropout(0.25)(encoder_output1) # Dropout anvendes også under inference
    
    encoder_lstm2 = LSTM(128, return_sequences=True, return_state=True, name='encoder_lstm')
    encoder_output2, state_h, state_c = encoder_lstm2(encoder_output1)
    
    # Attention mekanisme
    attention = Dense(1, activation='tanh')(encoder_output2)
    attention = Flatten()(attention)
    attention_weights = Activation('softmax')(attention)
    attention_weights = RepeatVector(128)(attention_weights)
    attention_weights = Permute([2, 1])(attention_weights)
    
    # Anvend attention på encoder output
    context_vector = Multiply()([encoder_output2, attention_weights])
    # Use the defined function for the Lambda layer
    context_vector = Lambda(sum_over_time_axis, name='lambda_sum')(context_vector)
    
    # Gemmer encoder states til decoder initialisation
    encoder_states = [state_h, state_c]
    
    # Outputs for hver horisont
    outputs = [] # Use a list to maintain order
    output_names = [f'output_{h}' for h in horizon_keys]
    
    for h in horizon_keys:
        # Decoder med attention
        # Ensure layer names match the saved model exactly
        decoder_lstm = LSTM(128, name=f'decoder_lstm_{h}') 
        # Repeat context vector for decoder input
        decoder_input = RepeatVector(1)(context_vector)
        decoder_output = decoder_lstm(decoder_input, initial_state=encoder_states)
        
        # Tætte lag for prognose
        decoder_dense1 = Dense(64, activation='relu', name=f'decoder_dense1_{h}')
        decoder_dropout1 = Dropout(0.2) # Dropout anvendes også under inference
        decoder_dense2 = Dense(32, activation='relu', name=f'decoder_dense2_{h}')
        decoder_dropout2 = Dropout(0.2) # Dropout anvendes også under inference
        decoder_output_layer = Dense(1, name=f'output_{h}') # Ensure name matches
        
        dense_out1 = decoder_dense1(decoder_output)
        dense_out1 = decoder_dropout1(dense_out1)
        dense_out2 = decoder_dense2(dense_out1)
        dense_out2 = decoder_dropout2(dense_out2)
        final_output = decoder_output_layer(dense_out2)
        outputs.append(final_output)
    
    # Bygger model
    model = Model(inputs=encoder_inputs, outputs=outputs)
    
    logging.info(f"Model structure built successfully with {len(outputs)} outputs.")
    # model.summary(print_fn=logging.info) # Optional: Log summary for verification
    return model

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("stock_api:app", host="0.0.0.0", port=8000, reload=False) 