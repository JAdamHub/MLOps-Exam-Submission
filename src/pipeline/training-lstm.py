import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import json
import time
import os

# Seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_FEATURES_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
TF_LOGS_DIR = MODELS_DIR / "tf_logs"

# Forecast horizons
FORECAST_HORIZONS = [1, 3, 7]  # Predict price for 1, 3, and 7 days ahead

# Sequence length for LSTM (number of past days to consider)
SEQUENCE_LENGTH = 20

# Training parameters
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.2

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TF_LOGS_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load feature data for training."""
    try:
        input_file = PROCESSED_FEATURES_DIR / "bitcoin_features_trading_days.csv"
        df = pd.read_csv(input_file)
        logging.info(f"Feature data loaded successfully from {input_file}, shape: {df.shape}")
        
        # Check if target columns exist
        target_columns = [f'price_target_{horizon}d' for horizon in FORECAST_HORIZONS]
        missing_targets = [col for col in target_columns if col not in df.columns]
        
        if missing_targets:
            logging.error(f"Missing target columns: {missing_targets}")
            raise ValueError(f"Target columns {missing_targets} not found in data")
        
        return df
    except Exception as e:
        logging.error(f"Error loading feature data: {e}")
        return None

def prepare_data(df):
    """Prepare data for LSTM training."""
    # Remove timestamp and ID columns
    df = df.copy()
    
    # Drop timestamp columns since they can't be converted to floats
    timestamp_cols = [col for col in df.columns if 'timestamp' in col.lower() or 'unnamed' in col.lower()]
    df = df.drop(columns=timestamp_cols, errors='ignore')
    
    # Handle infinity and NaN values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Impute missing values with median
    for col in df.columns:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    
    # Identify target columns
    target_columns = [f'price_target_{horizon}d' for horizon in FORECAST_HORIZONS]
    
    # Separate features and targets
    feature_columns = [col for col in df.columns if col not in target_columns]
    features_df = df[feature_columns]
    targets_df = df[target_columns]
    
    # Scale features using MinMaxScaler (better for LSTM)
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    # Fit and transform features
    scaled_features = scaler_X.fit_transform(features_df)
    
    # Fit and transform targets
    scaled_targets = scaler_y.fit_transform(targets_df)
    
    logging.info(f"Prepared {scaled_features.shape[1]} features and {scaled_targets.shape[1]} targets for training")
    
    return scaled_features, scaled_targets, feature_columns, target_columns, scaler_X, scaler_y

def create_sequences(X, y, sequence_length=SEQUENCE_LENGTH):
    """Create sequences for LSTM training (sliding window approach)."""
    X_seq, y_seq = [], []
    
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length])
    
    return np.array(X_seq), np.array(y_seq)

def build_lstm_model(input_shape, output_size):
    """Build LSTM model architecture."""
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        Dropout(DROPOUT_RATE),
        LSTM(64, return_sequences=False),
        Dropout(DROPOUT_RATE),
        Dense(32, activation='relu'),
        Dense(output_size)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='mse'
    )
    
    return model

def train_model(X, y, feature_names, target_columns):
    """Train LSTM model for multiple horizons simultaneously."""
    logging.info("Creating sequences for LSTM training...")
    X_seq, y_seq = create_sequences(X, y)
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_seq, y_seq, test_size=0.2, shuffle=False
    )
    
    logging.info(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")
    
    # Define input shape for LSTM
    input_shape = (X_train.shape[1], X_train.shape[2])
    output_size = y_train.shape[1]
    
    # Build model
    logging.info("Building LSTM model...")
    model = build_lstm_model(input_shape, output_size)
    model.summary(print_fn=logging.info)
    
    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        ),
        ModelCheckpoint(
            filepath=str(MODELS_DIR / 'lstm_model_best.h5'),
            monitor='val_loss',
            save_best_only=True
        )
    ]
    
    # Train model
    logging.info("Starting model training...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    val_loss = model.evaluate(X_val, y_val, verbose=0)
    logging.info(f"Validation loss: {val_loss:.4f}")
    
    # Make predictions and calculate metrics for each horizon
    y_pred = model.predict(X_val)
    
    # Calculate metrics for each target
    metrics = {}
    for i, target_col in enumerate(target_columns):
        y_true_i = y_val[:, i]
        y_pred_i = y_pred[:, i]
        
        # Calculate metrics
        mse = np.mean((y_true_i - y_pred_i) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true_i - y_pred_i))
        
        # Calculate R-squared (if you need similar metrics to XGBoost)
        ss_total = np.sum((y_true_i - np.mean(y_true_i)) ** 2)
        ss_residual = np.sum((y_true_i - y_pred_i) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        
        metrics[target_col] = {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        }
        
        logging.info(f"Metrics for {target_col} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    # Feature importance is not directly available for LSTM models like it is for XGBoost
    # We could use alternative methods (permutation importance, SHAP values, etc.)
    # For now, we'll just return an empty dictionary
    feature_importances = {}
    
    return model, metrics, feature_importances, history.history

def save_model(model, scaler_X, scaler_y, metrics, feature_importances, feature_names, target_columns):
    """Save model, scalers, and metrics."""
    try:
        # Save TensorFlow model
        tf_model_path = MODELS_DIR / "lstm_model"
        model.save(tf_model_path)
        logging.info(f"TensorFlow model saved to {tf_model_path}")
        
        # Save scalers
        scaler_X_path = MODELS_DIR / "scaler_features.joblib"
        scaler_y_path = MODELS_DIR / "scaler_targets.joblib"
        joblib.dump(scaler_X, scaler_X_path)
        joblib.dump(scaler_y, scaler_y_path)
        logging.info(f"Scalers saved successfully")
        
        # Save feature names
        feature_names_path = MODELS_DIR / "feature_names.joblib"
        joblib.dump(feature_names, feature_names_path)
        logging.info(f"Feature names saved successfully")
        
        # Save target columns
        target_columns_path = MODELS_DIR / "target_columns.joblib"
        joblib.dump(target_columns, target_columns_path)
        logging.info(f"Target columns saved successfully")
        
        # Save sequence length
        sequence_length_path = MODELS_DIR / "sequence_length.joblib"
        joblib.dump(SEQUENCE_LENGTH, sequence_length_path)
        logging.info(f"Sequence length saved successfully")
        
        # Save metrics and model configuration
        metrics_dict = {
            'metrics': metrics,
            'model_config': {
                'type': 'LSTM',
                'sequence_length': SEQUENCE_LENGTH,
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'dropout_rate': DROPOUT_RATE
            }
        }
        
        metrics_path = MODELS_DIR / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        logging.info(f"Metrics saved successfully")
        
        return True
    except Exception as e:
        logging.error(f"Error saving model artifacts: {e}")
        return False

def main():
    """Main function to orchestrate the training process for LSTM model"""
    start_time = time.time()
    logging.info("==== Starting LSTM model training process ====")
    
    # Check if TensorFlow can access GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logging.info(f"Training with GPU: {gpus}")
        # Set memory growth to avoid OOM errors
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        logging.info("No GPU detected, training with CPU")
    
    # Load data
    df = load_data()
    if df is None:
        logging.error("Failed to load data. Exiting training process.")
        return None
    
    # Preprocess data
    X, y, feature_columns, target_columns, scaler_X, scaler_y = prepare_data(df)
    logging.info(f"Data prepared with {X.shape[1]} features, {y.shape[1]} targets, and {X.shape[0]} samples")
    
    # Train LSTM model
    model, metrics, feature_importances, history = train_model(X, y, feature_columns, target_columns)
    
    # Save model and metrics
    save_success = save_model(model, scaler_X, scaler_y, metrics, feature_importances, feature_columns, target_columns)
    if save_success:
        logging.info("Model and metrics saved successfully")
    else:
        logging.warning("Failed to save model or metrics")
    
    logging.info(f"==== Model training completed in {(time.time() - start_time) / 60:.2f} minutes ====")
    
    return model, metrics, history

if __name__ == "__main__":
    main()
