import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
import json
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input, RepeatVector, TimeDistributed, Flatten, Activation, Multiply, Lambda, Permute
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime
import traceback
import pickle
import os
from tensorflow.keras import backend as K

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_FEATURES_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = MODELS_DIR / "figures"

# Forecast horizons
FORECAST_HORIZONS = [1, 3, 7]  # Predict the price 1, 3, and 7 days ahead

# LSTM-specific parameters
SEQUENCE_LENGTH = 10  # Number of days to use as input sequence
BATCH_SIZE = 8  # Smaller batch size for better generalization
NUM_EPOCHS = 200  # More epochs with early stopping

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load feature data for training."""
    try:
        input_file = PROCESSED_FEATURES_DIR / "vestas_features_trading_days.csv"
        df = pd.read_csv(input_file)
        logging.info(f"Feature data loaded successfully from {input_file}, shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading feature data: {e}")
        return None

def prepare_data(df):
    """Prepare data for LSTM training including a proper train/val/test split."""
    # Remove timestamp and ID columns
    df = df.copy()
    
    # Check if the index contains dates and ensure it doesn't become a feature
    if isinstance(df.index, pd.DatetimeIndex) or df.index.name == 'date':
        logging.info("DataFrame has DatetimeIndex - saving index separately from features")
        # Convert index to a separate column if necessary
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        # Keep the index
    elif 'date' in df.columns:
        logging.info("Converting 'date' column to index")
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    
    # Drop timestamp columns since they can't be converted to floats
    timestamp_cols = [col for col in df.columns if 'timestamp' in col.lower() or 'unnamed' in col.lower() or 'date' == col.lower()]
    df = df.drop(columns=timestamp_cols, errors='ignore')
    
    # Handle infinity and NaN values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Find columns with NaN values and log them
    columns_with_nan = df.columns[df.isna().any()].tolist()
    if columns_with_nan:
        logging.info(f"Columns with NaN values: {columns_with_nan}")
        logging.info(f"NaN count: {df[columns_with_nan].isna().sum()}")
        
    # Impute missing values with median
    for col in df.columns:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    
    # Identify target columns based on names
    target_columns = [f'price_target_{horizon}d' for horizon in FORECAST_HORIZONS]
    missing_targets = [col for col in target_columns if col not in df.columns]
    
    if missing_targets:
        logging.error(f"Missing target columns: {missing_targets}")
        raise ValueError(f"Target columns {missing_targets} not found in data")
    
    # Separate features and targets
    feature_columns = [col for col in df.columns if col not in target_columns]
    
    # Remove non-numeric columns from feature_columns
    numeric_feature_columns = []
    for col in feature_columns:
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_feature_columns.append(col)
        else:
            logging.info(f"Dropping non-numeric feature column: {col}")
    
    logging.info(f"Using {len(numeric_feature_columns)} numeric features out of {len(feature_columns)} total features")
    feature_columns = numeric_feature_columns
    
    # Handle infinite values and extreme numbers
    for col in feature_columns:
        # Replace inf and -inf with NaN
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        # Replace NaN with median
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    
    # Log columns with extreme values
    for col in feature_columns:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val > 1e10 or min_val < -1e10:
            logging.warning(f"Feature {col} has extreme values: min={min_val}, max={max_val}")
            
            # Limit extreme values
            df[col] = df[col].clip(-1e10, 1e10)
    
    # --- NEW CODE: Convert target to percentage change ---
    # Save the original target values for later conversion back
    original_target_values = {}
    for target_col in target_columns:
        original_target_values[target_col] = df[target_col].values.copy()
    
    # Find 'Close' or similar column for the current price
    price_column = None
    for candidate in ['Close', 'close', 'price', 'Price']:
        if candidate in df.columns:
            price_column = candidate
            break
    
    if price_column is None:
        raise ValueError("Could not find price column in the dataframe (Close, close, price, Price)")
    
    # Convert targets to percentage changes
    for target_col in target_columns:
        horizon = int(target_col.split('_')[-1].replace('d', ''))
        # Calculate percentage change from current price to target price
        df[f'pct_change_{target_col}'] = ((df[target_col] / df[price_column]) - 1) * 100
        logging.info(f"Converted {target_col} to percent change: Mean = {df[f'pct_change_{target_col}'].mean():.2f}%, Std = {df[f'pct_change_{target_col}'].std():.2f}%")
    
    # Update target_columns to use the new percentage columns
    percent_target_columns = [f'pct_change_{col}' for col in target_columns]
    logging.info(f"New target columns with percentage changes: {percent_target_columns}")
    # --- END OF NEW CODE ---
    
    X = df[feature_columns].values
    
    # Create dictionary of targets for different horizons
    y_dict = {}
    for i, target_col in enumerate(target_columns):
        # Use the percentage change as target
        y_dict[target_col] = df[percent_target_columns[i]].values
    
    # Scale features using MinMaxScaler (better for LSTM than StandardScaler)
    feature_scaler = MinMaxScaler()
    X_scaled = feature_scaler.fit_transform(X)
    
    # Scale targets (important for LSTM)
    target_scalers = {}
    for target_col in target_columns:
        target_scaler = MinMaxScaler()
        y_values = y_dict[target_col].reshape(-1, 1)
        y_dict[target_col] = target_scaler.fit_transform(y_values).flatten()
        target_scalers[target_col] = target_scaler
    
    # Split data into training (64%), validation (16%), and test (20%) sets
    # This follows the approach of the inspiration model
    train_size = int(len(X_scaled) * 0.8)
    train_val_data = X_scaled[:train_size]
    test_data = X_scaled[train_size:]
    
    train_val_size = int(len(train_val_data) * 0.8)
    train_data = train_val_data[:train_val_size]
    val_data = train_val_data[train_val_size:]
    
    # Split targets with same indices
    y_train = {}
    y_val = {}
    y_test = {}
    
    for target_col in target_columns:
        target_vals = y_dict[target_col]
        y_train[target_col] = target_vals[:train_val_size]
        y_val[target_col] = target_vals[train_val_size:train_size]
        y_test[target_col] = target_vals[train_size:]
    
    logging.info(f"Data splits: train={train_data.shape}, validation={val_data.shape}, test={test_data.shape}")
    logging.info(f"Prepared {X_scaled.shape[1]} features and {len(target_columns)} targets for training")
    
    data_splits = {
        'train': (train_data, y_train),
        'val': (val_data, y_val),
        'test': (test_data, y_test)
    }
    
    # Save the original price values to be able to convert back
    data_splits['original_prices'] = {
        'price_column': price_column,
        'last_price_train': df[price_column].iloc[train_val_size-1],
        'last_price_val': df[price_column].iloc[train_size-1],
        'last_price_test': df[price_column].iloc[-1],
        'test_prices': df[price_column].iloc[train_size:].values
    }
    
    return data_splits, feature_columns, target_columns, feature_scaler, target_scalers

def create_sequences(data, target_dict=None, seq_length=10):
    """
    Create sequential data for the seq2seq LSTM model.
    
    Args:
        data: DataFrame or numpy array with input features
        target_dict: Dictionary with targets for each time horizon
        seq_length: Length of the input sequence
    
    Returns:
        X_seq: 3D numpy array with sequential input data [samples, seq_length, features]
        y_seq_dict: Dictionary with 2D numpy arrays for each time horizon
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    n_samples = data.shape[0] - seq_length
    
    if n_samples <= 0:
        raise ValueError(f"Seq_length {seq_length} is longer than data with {data.shape[0]} samples!")
    
    # Create input sequences (X)
    X_seq = np.zeros((n_samples, seq_length, data.shape[1]))
    
    for i in range(n_samples):
        X_seq[i] = data[i:i+seq_length]
    
    # Create target sequences (y) if target_dict is provided
    if target_dict is not None:
        y_seq_dict = {}
        
        for horizon, target in target_dict.items():
            if isinstance(target, pd.DataFrame) or isinstance(target, pd.Series):
                target = target.values
                
            # For sequence-to-one prediction
            if target.ndim == 1:
                target = target.reshape(-1, 1)
                
            # Follow the same structure as the input sequences
            y_seq = np.zeros((n_samples, 1))  # 1-dimensional output for each horizon
            
            for i in range(n_samples):
                # Target is the value that comes after the sequence
                # for the corresponding horizon
                y_seq[i] = target[i+seq_length-1]
                
            y_seq_dict[horizon] = y_seq
            
        return X_seq, y_seq_dict
    
    return X_seq

def build_multi_horizon_lstm_model(input_shape, output_dim=1):
    """
    Builds an LSTM model for predicting multiple time horizons.
    
    Args:
        input_shape: Tuple with (seq_length, n_features)
        output_dim: Number of outputs (typically 1 for each time horizon)
        
    Returns:
        Compiled Keras model
    """
    # Input layer
    input_layer = Input(shape=input_shape)
    
    # LSTM layers with dropout to avoid overfitting
    lstm1 = LSTM(64, return_sequences=True)(input_layer)
    dropout1 = Dropout(0.2)(lstm1)
    
    lstm2 = LSTM(32)(dropout1)
    dropout2 = Dropout(0.2)(lstm2)
    
    # Output layers for each time horizon
    outputs = []
    for h in ['1d', '3d', '7d']:
        dense = Dense(16, activation='relu')(dropout2)
        output = Dense(output_dim, name=f'output_{h}')(dense)
        outputs.append(output)
    
    # Build model
    model = Model(inputs=input_layer, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    logging.info(f"Model built with input shape {input_shape} and {len(outputs)} outputs")
    return model

def build_seq2seq_model(input_shape, horizon_keys=['1d', '3d', '7d']):
    """
    Builds an advanced seq2seq model with encoder-decoder architecture for multiple time horizons.
    Implements a basic attention mechanism for better forecasting.
    
    Args:
        input_shape: Tuple with (seq_length, n_features)
        horizon_keys: List of time horizons to predict
        
    Returns:
        Compiled Keras model
    """
    logging.info(f"Building advanced seq2seq model with input shape {input_shape} and {len(horizon_keys)} horizons")
    
    # Input layer
    encoder_inputs = Input(shape=input_shape, name='encoder_input')
    
    # Encoder LSTM layers
    encoder_lstm1 = Bidirectional(LSTM(128, return_sequences=True), name='encoder_bilstm')
    encoder_output1 = encoder_lstm1(encoder_inputs)
    encoder_output1 = Dropout(0.25)(encoder_output1)
    
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
    context_vector = Lambda(lambda x: K.sum(x, axis=1))(context_vector)
    
    # Save encoder states for decoder initialization
    encoder_states = [state_h, state_c]
    
    # Outputs for each horizon
    outputs = {}
    for h in horizon_keys:
        # Decoder with attention
        decoder_lstm = LSTM(128, name=f'decoder_lstm_{h}')
        decoder_output = decoder_lstm(RepeatVector(1)(context_vector), initial_state=encoder_states)
        
        # Dense layers for prediction
        decoder_dense1 = Dense(64, activation='relu', name=f'decoder_dense1_{h}')
        decoder_dropout1 = Dropout(0.2)
        decoder_dense2 = Dense(32, activation='relu', name=f'decoder_dense2_{h}')
        decoder_dropout2 = Dropout(0.2)
        decoder_output_layer = Dense(1, name=f'output_{h}')
        
        dense_out1 = decoder_dense1(decoder_output)
        dense_out1 = decoder_dropout1(dense_out1)
        dense_out2 = decoder_dense2(dense_out1)
        dense_out2 = decoder_dropout2(dense_out2)
        outputs[f'output_{h}'] = decoder_output_layer(dense_out2)
    
    # Build model
    model = Model(inputs=encoder_inputs, outputs=list(outputs.values()))
    
    # Define loss function and metrics for each output
    losses = {output_name: 'mse' for output_name in outputs.keys()}
    loss_weights = {output_name: 1.0 for output_name in outputs.keys()}
    metrics = {output_name: 'mae' for output_name in outputs.keys()}
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics
    )
    
    model.summary(print_fn=logging.info)
    return model

def create_sequences(data, seq_length):
    """
    Generates sequential input-output pairs for the LSTM model.
    
    Args:
        data: DataFrame with features and targets
        seq_length: Length of input sequences
        
    Returns:
        X: Array with input sequences [samples, seq_length, features]
        y: Array with targets [samples, 1]
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i+seq_length].values)
        y.append(data.iloc[i+seq_length].values)
    
    return np.array(X), np.array(y)

def prepare_sequences_for_seq2seq(X_data, y_data_dict, seq_length, horizon_keys):
    """
    Prepares sequences of data for the seq2seq model.
    
    Args:
        X_data: DataFrame with features
        y_data_dict: Dictionary with targets for each time horizon
        seq_length: Length of sequences
        horizon_keys: List of keys for time horizons
        
    Returns:
        X_sequences: Array with input sequences
        y_sequences_dict: Dictionary with target sequences for each time horizon
    """
    # Create empty list to store sequences
    sequences = []
    for i in range(len(X_data) - seq_length + 1):
        sequences.append(X_data.iloc[i:i+seq_length].values)
    
    X_sequences = np.array(sequences)
    
    # Create target dictionary
    y_sequences_dict = {}
    for h in horizon_keys:
        if h in y_data_dict:
            # Take targets from index seq_length-1 and forward to match X_sequences
            y_seq = y_data_dict[h][seq_length-1:]
            y_sequences_dict[h] = np.array(y_seq)
    
    # Log information about data shapes
    logging.info(f"Sequence data prepared: X shape: {X_sequences.shape}, " + 
               f"target shapes: {', '.join([f'{k}: {v.shape}' for k, v in y_sequences_dict.items()])}")
    
    return X_sequences, y_sequences_dict

def hyperparameter_tuning(data_splits, target_col):
    """
    Runs a simple hyperparameter tuning for the LSTM model.
    """
    logging.info("Starting hyperparameter tuning for LSTM...")
    
    # Get data
    train_data, y_train = data_splits['train']
    val_data, y_val = data_splits['val']
    
    # Target for 1-day horizon
    y_train_target = y_train[target_col]
    y_val_target = y_val[target_col]
    
    # Create sequences for LSTM
    X_train_seq, y_train_seq = create_sequences(train_data, y_train_target)
    X_val_seq, y_val_seq = create_sequences(val_data, y_val_target)
    
    # Parameter grid for LSTM
    param_grid = {
        'lstm_units': [64, 128, 250],
        'dropout_rate': [0.2, 0.3, 0.4],
        'learning_rate': [0.001, 0.0005, 0.0001],
        'batch_size': [8, 16, 32]
    }
    
    best_params = None
    best_val_loss = float('inf')
    
    # Manual grid search for selected combinations
    # We reduce the search space to save time but choose the most likely winners
    search_combinations = [
        {'lstm_units': 250, 'dropout_rate': 0.3, 'learning_rate': 0.0005, 'batch_size': 8},
        {'lstm_units': 128, 'dropout_rate': 0.3, 'learning_rate': 0.0005, 'batch_size': 16},
        {'lstm_units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.001, 'batch_size': 32}
    ]
    
    for params in search_combinations:
        lstm_units = params['lstm_units']
        dropout_rate = params['dropout_rate']
        learning_rate = params['learning_rate']
        batch_size = params['batch_size']
        
        logging.info(f"Testing LSTM with units={lstm_units}, dropout={dropout_rate}, "
                     f"lr={learning_rate}, batch_size={batch_size}")
        
        # Build model
        model = Sequential([
            Bidirectional(LSTM(units=lstm_units, return_sequences=True), 
                          input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
            Dropout(dropout_rate),
            Bidirectional(LSTM(units=lstm_units//2)),
            Dropout(dropout_rate),
            Dense(units=64, activation='relu'),
            Dropout(dropout_rate/2),
            Dense(units=1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse'
        )
        
        # Callbacks for early stopping
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        ]
        
        # Train model
        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=50,  # Reduced for hyperparameter tuning
            batch_size=batch_size,
            validation_data=(X_val_seq, y_val_seq),
            callbacks=callbacks,
            verbose=0
        )
        
        # Evaluate on validation set
        val_loss = model.evaluate(X_val_seq, y_val_seq, verbose=0)
        
        logging.info(f"Validation loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params
    
    logging.info(f"Best parameters found: {best_params} with validation loss: {best_val_loss:.4f}")
    return best_params

def evaluate_model(model, X_test_seq, y_test, target_scaler, horizon):
    """Evaluates the model with test data."""
    # Predict on test data
    predictions = model.predict(X_test_seq)
    
    # Reshape predictions to 1D
    if isinstance(predictions, list):
        # If we have multiple outputs, find the right one based on the horizon
        pred_index = FORECAST_HORIZONS.index(horizon)
        predictions = predictions[pred_index]

    # Reshape to 2D for inverse transform
    predictions = predictions.reshape(-1, 1)
    
    # Convert y_test to 2D if it's 1D
    if y_test.ndim == 1:
        y_test_2d = y_test.reshape(-1, 1)
    else:
        y_test_2d = y_test
    
    # Denormalize predictions and actual values
    predictions_denorm = target_scaler.inverse_transform(predictions)
    y_test_denorm = target_scaler.inverse_transform(y_test_2d)
    
    # Calculate error metrics
    mse = mean_squared_error(y_test_denorm, predictions_denorm)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_denorm, predictions_denorm)
    
    # R^2 score (higher is better, max is 1.0)
    r2 = r2_score(y_test_denorm, predictions_denorm)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_test_denorm - predictions_denorm) / y_test_denorm)) * 100
    
    # If R2 is negative (worse than average), set to 0
    r2 = max(0, r2)
    
    logging.info(f"{horizon}-day horizon forecast metrics:")
    logging.info(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    logging.info(f"R^2 Score: {r2:.4f}, MAPE: {mape:.2f}%")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'predictions': predictions_denorm,
        'actuals': y_test_denorm
    }

def train_model(model, X_train, y_train_dict, X_val, y_val_dict, 
               batch_size=32, epochs=100, patience=20,
               model_save_path=None):
    """
    Trains a seq2seq LSTM model with early stopping.
    
    Args:
        model: Compiled seq2seq model
        X_train: Training data (samples, seq_length, features)
        y_train_dict: Dictionary with targets for each horizon
        X_val: Validation data
        y_val_dict: Dictionary with validation targets for each horizon
        batch_size: Batch size
        epochs: Maximum number of epochs
        patience: Patience for early stopping
        model_save_path: Path to save the best model
        
    Returns:
        Trained model and training history
    """
    logging.info(f"Training seq2seq model with {len(X_train)} training samples and {len(X_val)} validation samples")
    
    # Convert y_train_dict and y_val_dict to the format model.fit expects
    # For the seq2seq model we use {output_name: target_data}
    y_train_formatted = {f'output_{h}': y_train_dict[h] for h in y_train_dict.keys()}
    y_val_formatted = {f'output_{h}': y_val_dict[h] for h in y_val_dict.keys()}
    
    # Set up callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            verbose=1,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=int(patience/2),
            verbose=1,
            min_lr=1e-6
        )
    ]
    
    # Add ModelCheckpoint if a path is provided
    if model_save_path:
        callbacks.append(
            ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_loss',
                verbose=1,
                save_best_only=True,
                save_weights_only=False
            )
        )
    
    # Train model
    history = model.fit(
        X_train,
        y_train_formatted,
        validation_data=(X_val, y_val_formatted),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2
    )
    
    # Log training results
    final_epoch = len(history.history['loss'])
    final_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    logging.info(f"Training completed after {final_epoch} epochs")
    logging.info(f"Final loss: {final_loss:.4f}, val_loss: {final_val_loss:.4f}")
    
    # Log individual horizon metrics
    for h in y_train_dict.keys():
        output_name = f'output_{h}'
        if f'{output_name}_mae' in history.history:
            val_mae = history.history[f'val_{output_name}_mae'][-1]
            logging.info(f"Horizon {h} val_mae: {val_mae:.4f}")
    
    return model, history

def evaluate_multi_horizon_model(model, X_test, y_test_dict, horizon_keys, scaler=None):
    """
    Evaluates the model on test data for all horizons.
    """
    # Get predictions for each horizon
    predictions = model.predict(X_test)
    
    # Ensure predictions is a list
    if not isinstance(predictions, list):
        predictions = [predictions]
    
    evaluation_results = {}
    
    # Extract original test prices for converting percentage changes back to actual prices
    if hasattr(model, 'data_splits') and 'original_prices' in model.data_splits:
        original_prices = model.data_splits['original_prices']
        test_prices = original_prices['test_prices']
        price_column = original_prices['price_column']
        is_percent_change_model = True
        logging.info("Evaluating percent change model - will convert back to prices")
    else:
        is_percent_change_model = False
    
    for i, h in enumerate(horizon_keys):
        target_key = f'price_target_{h}'
        if target_key not in y_test_dict:
            target_key = list(y_test_dict.keys())[i]  # Fallback to order
        
        y_true = y_test_dict[target_key]
        
        # Reshape for inverse_transform
        y_pred = predictions[i].reshape(-1, 1)
        y_true = y_true.reshape(-1, 1)
        
        # Get scaler for this horizon
        if scaler is not None and isinstance(scaler, dict):
            horizon_scaler = scaler.get(target_key)
        else:
            horizon_scaler = scaler
            
        if horizon_scaler is not None:
            # Denormalize
            y_pred_denorm = horizon_scaler.inverse_transform(y_pred)
            y_true_denorm = horizon_scaler.inverse_transform(y_true)
            
            # --- NEW CODE: Convert percentage change back to price values ---
            if is_percent_change_model:
                # The predicted percentage change needs to be converted back to a price
                # Note: If we forecast many days ahead, we need to use the correct base price
                horizon_days = int(h.replace('d', ''))
                
                # Adjust vectors to match lengths (we can only use common data points)
                min_len = min(len(y_pred_denorm), len(test_prices) - horizon_days)
                
                # Create arrays for the converted prices
                actual_prices = np.zeros(min_len)
                predicted_prices = np.zeros(min_len)
                
                for j in range(min_len):
                    # For actual values we use future prices
                    actual_prices[j] = test_prices[j + horizon_days]
                    
                    # For predicted values we apply the percentage change to the current price
                    base_price = test_prices[j]
                    percent_change = y_pred_denorm[j][0]  # Predicted percentage change
                    predicted_prices[j] = base_price * (1 + percent_change/100)
                
                # Replace the denormalized values with actual price values
                y_pred_denorm = predicted_prices.reshape(-1, 1)
                y_true_denorm = actual_prices.reshape(-1, 1)
                
                logging.info(f"Converted {h}-horizon back to prices: Base mean={np.mean(test_prices[:min_len]):.2f}, Forecast mean={np.mean(predicted_prices):.2f}")
            # --- END OF NEW CODE ---
        else:
            # If no scaler, use as is
            y_pred_denorm = y_pred
            y_true_denorm = y_true
            
        # Calculate metrics
        mse = mean_squared_error(y_true_denorm, y_pred_denorm)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_denorm, y_pred_denorm)
        
        # Avoid division by zero in MAPE calculation
        mape = np.mean(np.abs((y_true_denorm - y_pred_denorm) / np.maximum(0.0001, np.abs(y_true_denorm)))) * 100
        
        # R² score
        r2 = r2_score(y_true_denorm, y_pred_denorm)
        
        # Log results
        logging.info(f"Metrics for {h} horizon:")
        logging.info(f"  RMSE: {rmse:.4f}")
        logging.info(f"  MAE: {mae:.4f}")
        logging.info(f"  MAPE: {mape:.2f}%")
        logging.info(f"  R²: {r2:.4f}")
        
        evaluation_results[h] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
        
    return evaluation_results

def save_multi_horizon_model(model, feature_scaler, target_scalers, metrics, feature_names, target_columns, seq_length, history=None):
    """
    Saves the LSTM model along with its scaler and other metadata.
    """
    # Create date-timestamped path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODELS_DIR / f"lstm_multi_horizon_model.keras"
    meta_path = MODELS_DIR / f"lstm_model_metadata.json"
    
    # Save the model
    try:
        model.save(model_path)
        logging.info(f"Model saved to {model_path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        traceback.print_exc()
    
    # Save scalers
    joblib.dump(feature_scaler, MODELS_DIR / "lstm_feature_scaler.joblib")
    joblib.dump(target_scalers, MODELS_DIR / "lstm_target_scalers.joblib")
    
    # Save feature and target names
    joblib.dump(feature_names, MODELS_DIR / "lstm_feature_names.joblib")
    joblib.dump(target_columns, MODELS_DIR / "lstm_target_columns.joblib")
    
    # Save sequence length
    joblib.dump(seq_length, MODELS_DIR / "lstm_sequence_length.joblib")
    
    # Save feature medians for NaN handling during inference
    try:
        if isinstance(feature_names, list) and len(feature_names) > 0:
            # Load original data to calculate medians
            df = load_data()
            if df is not None:
                feature_medians = {}
                for feature in feature_names:
                    if feature in df.columns:
                        feature_medians[feature] = float(df[feature].median())
                
                # Save medians
                if feature_medians:
                    joblib.dump(feature_medians, MODELS_DIR / "lstm_feature_medians.joblib")
                    logging.info(f"Feature medians saved for {len(feature_medians)} features")
    except Exception as e:
        logging.error(f"Error saving feature medians: {e}")
    
    # Save metadata as JSON
    metadata = {
        "model_type": "LSTM Multi-Horizon",
        "created_at": timestamp,
        "sequence_length": seq_length,
        "feature_count": len(feature_names),
        "target_columns": target_columns,
        "metrics": metrics,
        "is_percent_change_model": True  # Added to mark that this model uses percentage changes
    }
    
    # Add training history if available
    if history is not None and hasattr(history, 'history'):
        # Convert numpy arrays to lists for JSON serialization
        hist_dict = {}
        for key, values in history.history.items():
            if isinstance(values, np.ndarray):
                hist_dict[key] = values.tolist()
            else:
                hist_dict[key] = list(values)
        
        metadata["training_history"] = hist_dict
        
        # Plot training history and save as image
        try:
            plt.figure(figsize=(12, 8))
            
            # Plot loss
            plt.subplot(2, 1, 1)
            for key in history.history.keys():
                if 'loss' in key and 'val' not in key:
                    plt.plot(history.history[key], label=key)
            for key in history.history.keys():
                if 'loss' in key and 'val' in key:
                    plt.plot(history.history[key], label=key, linestyle='--')
            
            plt.title('Model Loss During Training')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend()
            
            # Plot metrics (MAE)
            plt.subplot(2, 1, 2)
            for key in history.history.keys():
                if 'mae' in key and 'val' not in key:
                    plt.plot(history.history[key], label=key)
            for key in history.history.keys():
                if 'mae' in key and 'val' in key:
                    plt.plot(history.history[key], label=key, linestyle='--')
            
            plt.title('Model MAE During Training')
            plt.ylabel('MAE')
            plt.xlabel('Epoch')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"training_history_{timestamp}.png")
            plt.close()
        except Exception as e:
            logging.error(f"Error plotting training history: {e}")
    
    # Save metadata
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logging.info(f"Model metadata saved to {meta_path}")
    
    return model_path

def train_seq2seq_model(X_train, y_train_dict, X_val, y_val_dict, 
                        seq_length=30, horizon_keys=['1d', '3d', '7d'], 
                        epochs=100, batch_size=32):
    """
    Trains a seq2seq model with output for multiple time horizons.
    
    Args:
        X_train: Training data, features
        y_train_dict: Dictionary with target for each horizon for training data
        X_val: Validation data, features
        y_val_dict: Dictionary with target for each horizon for validation data
        seq_length: Length of input sequences
        horizon_keys: List of time horizons to predict
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Trained model, scaler and training history
    """
    logging.info(f"Training seq2seq model with {len(horizon_keys)} outputs: {horizon_keys}")
    
    # Prepare data as sequences
    X_train_seq, y_train_seq_dict = prepare_sequences_for_seq2seq(
        X_train, y_train_dict, seq_length, horizon_keys)
    X_val_seq, y_val_seq_dict = prepare_sequences_for_seq2seq(
        X_val, y_val_dict, seq_length, horizon_keys)
    
    # Create model
    model = build_seq2seq_model(
        input_shape=(seq_length, X_train.shape[1]), 
        horizon_keys=horizon_keys
    )
    
    # Restructure target dictionaries to the format keras expects
    train_targets = {f'output_{h}': y_train_seq_dict[h] for h in horizon_keys if h in y_train_seq_dict}
    val_targets = {f'output_{h}': y_val_seq_dict[h] for h in horizon_keys if h in y_val_seq_dict}
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Reduce learning rate callback
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.0001,
        verbose=1
    )
    
    # Model checkpoint callback
    model_checkpoint = ModelCheckpoint(
        filepath=str(MODELS_DIR / "lstm_model_checkpoint.keras"),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Train model
    logging.info(f"Starting training with {epochs} epochs and batch size {batch_size}")
    history = model.fit(
        X_train_seq,
        train_targets,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val_seq, val_targets),
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=1
    )
    
    # Log training results
    logging.info(f"Training completed after {len(history.history['loss'])} epochs")
    
    # Log validation loss for each horizon
    for h in y_train_dict.keys():
        output_name = f'output_{h}'
        if f'{output_name}_mae' in history.history:
            val_mae = history.history[f'val_{output_name}_mae'][-1]
            logging.info(f"Horizon {h} val_mae: {val_mae:.4f}")
    
    return model, history

def evaluate_seq2seq_model(model, X_test, y_test_dict, seq_length=30, horizon_keys=['1d', '3d', '7d']):
    """
    Evaluates a trained seq2seq model on test data and visualizes the results.
    
    Args:
        model: Trained seq2seq model
        X_test: Test data features
        y_test_dict: Dictionary with target values for each horizon
        seq_length: Length of input sequences
        horizon_keys: List of horizons to predict
        
    Returns:
        DataFrame with results and metrics
    """
    logging.info("Evaluating seq2seq model on test data...")
    
    # Prepare sequences for evaluation
    X_test_seq, y_test_seq_dict = prepare_sequences_for_seq2seq(
        X_test, y_test_dict, seq_length, horizon_keys)
    
    # Log information about data shapes
    logging.info(f"Test sequences: X shape={X_test_seq.shape}")
    for h in horizon_keys:
        if h in y_test_seq_dict:
            logging.info(f"Test targets for horizon {h}: shape={y_test_seq_dict[h].shape}")
    
    # Restructure target dictionaries to format that fits Keras
    test_targets = {f'output_{h}': y_test_seq_dict[h] for h in horizon_keys if h in y_test_seq_dict}
    
    # Evaluate model on test data
    test_metrics = model.evaluate(X_test_seq, test_targets, verbose=1)
    logging.info(f"Test metrics: {test_metrics}")
    
    # Make predictions
    predictions = model.predict(X_test_seq)
    
    # Prepare a DataFrame to store results
    results = pd.DataFrame()
    
    # For each horizon, calculate metrics and visualize results
    for i, horizon in enumerate(horizon_keys):
        if horizon in y_test_seq_dict:
            # Get actual and predicted values
            y_true = y_test_seq_dict[horizon]
            
            # Handle different output formats from model.predict
            if isinstance(predictions, dict):
                y_pred = predictions[f'output_{horizon}']
            else:
                # If predictions is a list of arrays (for multi-output models)
                y_pred = predictions[i] if i < len(predictions) else None
                
            if y_pred is None:
                logging.warning(f"Could not find predictions for horizon {horizon}")
                continue
                
            # Calculate metrics
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # Log results
            logging.info(f"Horizon {horizon} - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            
            # Save results
            horizon_results = pd.DataFrame({
                'Horizon': horizon,
                'MSE': mse,
                'RMSE': rmse, 
                'MAE': mae,
                'R2': r2
            }, index=[0])
            
            results = pd.concat([results, horizon_results], ignore_index=True)
            
            # Visualize predictions
            plt.figure(figsize=(12, 6))
            plt.plot(y_true[:100], label='Actual', color='blue')
            plt.plot(y_pred[:100], label='Predicted', color='red', linestyle='--')
            plt.title(f'Seq2Seq Model - Predictions for {horizon} horizon')
            plt.xlabel('Samples')
            plt.ylabel('Normalized Price Change')
            plt.legend()
            
            # Save figure
            fig_dir = Path('plots')
            fig_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(fig_dir / f'seq2seq_predictions_{horizon}.png')
            plt.close()
    
    # Save results to CSV
    results_dir = Path('results')
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results.to_csv(results_dir / f'seq2seq_evaluation_results_{timestamp}.csv', index=False)
    logging.info(f"Evaluation saved to results/seq2seq_evaluation_results_{timestamp}.csv")
    
    return results

def main():
    """
    Trains, evaluates and saves a sequence-to-sequence LSTM model to predict the Vestas stock price
    for multiple time horizons simultaneously (1, 3, and 7 days).
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logging.info("Starting LSTM model training process...")
    
    # Create directories if they don't exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    try:
        # Load and prepare data
        logging.info("Loading data...")
        features_file = PROCESSED_FEATURES_DIR / "vestas_features_trading_days.csv"
        df = pd.read_csv(features_file)
        
        # Convert date to index if it exists
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
        
        # Define target columns - one for each forecast horizon
        forecast_horizons = [1, 3, 7]  # 1, 3, and 7 days ahead
        target_columns = [f'price_target_{h}d' for h in forecast_horizons]
        
        # Check if target columns exist
        for col in target_columns:
            if col not in df.columns:
                raise ValueError(f"Target column {col} does not exist in data")
        
        logging.info(f"Using the following target columns: {target_columns}")
        
        # Remove rows with NaN in target columns
        df = df.dropna(subset=target_columns)
        logging.info(f"Dataset after removing NaN values: {df.shape}")
        
        # Split data into training, validation, and test
        train_size = 0.7
        val_size = 0.15
        
        # Sort by date
        df = df.sort_index()
        
        # Define training, validation, and test indices
        n = len(df)
        train_idx = int(n * train_size)
        val_idx = train_idx + int(n * val_size)
        
        df_train = df.iloc[:train_idx]
        df_val = df.iloc[train_idx:val_idx]
        df_test = df.iloc[val_idx:]
        
        logging.info(f"Training set size: {len(df_train)}")
        logging.info(f"Validation set size: {len(df_val)}")
        logging.info(f"Test set size: {len(df_test)}")
        
        # Select features (all columns except target columns)
        feature_columns = [col for col in df.columns if col not in target_columns]
        
        # Remove non-numeric columns from feature_columns
        numeric_feature_columns = []
        for col in feature_columns:
            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_feature_columns.append(col)
            else:
                logging.info(f"Dropping non-numeric feature column: {col}")
        
        logging.info(f"Using {len(numeric_feature_columns)} numeric features out of {len(feature_columns)} total features")
        feature_columns = numeric_feature_columns
        
        # Handle infinite values and extreme numbers
        for col in feature_columns:
            # Replace inf and -inf with NaN
            df_train[col] = df_train[col].replace([np.inf, -np.inf], np.nan)
            df_val[col] = df_val[col].replace([np.inf, -np.inf], np.nan)
            df_test[col] = df_test[col].replace([np.inf, -np.inf], np.nan)
            
            # Replace NaN with median
            median_val = df_train[col].median()
            df_train[col] = df_train[col].fillna(median_val)
            df_val[col] = df_val[col].fillna(median_val)
            df_test[col] = df_test[col].fillna(median_val)
        
        # Log columns with extreme values
        for col in feature_columns:
            min_val = df_train[col].min()
            max_val = df_train[col].max()
            if max_val > 1e10 or min_val < -1e10:
                logging.warning(f"Feature {col} has extreme values: min={min_val}, max={max_val}")
                
                # Limit extreme values
                df_train[col] = df_train[col].clip(-1e10, 1e10)
                df_val[col] = df_val[col].clip(-1e10, 1e10)
                df_test[col] = df_test[col].clip(-1e10, 1e10)
                
        # Scale features
        feature_scaler = MinMaxScaler()
        X_train = feature_scaler.fit_transform(df_train[feature_columns])
        X_val = feature_scaler.transform(df_val[feature_columns])
        X_test = feature_scaler.transform(df_test[feature_columns])
        
        # ---> CALCULATE AND SAVE MEDIANS FROM THE TRAINING SET <---
        logging.info("Calculating medians from training data...")
        feature_medians = df_train[feature_columns].median().to_dict()
        medians_path = MODELS_DIR / 'lstm_feature_medians.joblib'
        joblib.dump(feature_medians, medians_path)
        logging.info(f"Feature medians saved to {medians_path}")
        # ---> DONE WITH MEDIANS <---

        # Scale targets (one scaler for each target)
        target_scalers = {}
        y_train_dict = {}
        y_val_dict = {}
        y_test_dict = {}
        
        for target_col in target_columns:
            scaler = MinMaxScaler()
            
            # Scale each target
            y_train_dict[target_col] = scaler.fit_transform(df_train[[target_col]]).flatten()
            y_val_dict[target_col] = scaler.transform(df_val[[target_col]]).flatten()
            y_test_dict[target_col] = scaler.transform(df_test[[target_col]]).flatten()
            
            # Save scaler
            target_scalers[target_col] = scaler
        
        # Define sequence length (number of days history)
        seq_length = 30
        
        # Hyperparameters
        epochs = 100
        batch_size = 32
        patience = 15
        
        # Train seq2seq model
        logging.info(f"Starting training of seq2seq LSTM model with sequence length {seq_length}...")
        model, history = train_seq2seq_model(
            pd.DataFrame(X_train, columns=feature_columns, index=df_train.index), 
            {f'{h}d': y_train_dict[f'price_target_{h}d'] for h in [1, 3, 7]},
            pd.DataFrame(X_val, columns=feature_columns, index=df_val.index), 
            {f'{h}d': y_val_dict[f'price_target_{h}d'] for h in [1, 3, 7]},
            seq_length=seq_length,
            horizon_keys=['1d', '3d', '7d'],
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Evaluate seq2seq model
        results = evaluate_seq2seq_model(
            model,
            pd.DataFrame(X_test, columns=feature_columns, index=df_test.index),
            {f'{h}d': y_test_dict[f'price_target_{h}d'] for h in [1, 3, 7]},
            seq_length=seq_length,
            horizon_keys=['1d', '3d', '7d']
        )
        
        # Plot training history
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Seq2Seq Model Training History')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('plots/seq2seq_training_history.png')
        
        logging.info("Seq2Seq model training and evaluation completed")
        
        # Save model and helper files for API
        logging.info("Saving model and helper files for API...")
        
        # Create models directory if it doesn't exist
        models_dir = Path('models')
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model.save(models_dir / 'lstm_multi_horizon_model.keras')
        logging.info("Model saved as 'lstm_multi_horizon_model.keras'")
        
        # Save feature scaler
        joblib.dump(feature_scaler, models_dir / 'lstm_feature_scaler.joblib')
        logging.info("Feature scaler saved as 'lstm_feature_scaler.joblib'")
        
        # Save target scalers
        joblib.dump(target_scalers, models_dir / 'lstm_target_scalers.joblib')
        logging.info("Target scalers saved as 'lstm_target_scalers.joblib'")
        
        # Save feature columns
        joblib.dump(feature_columns, models_dir / 'lstm_feature_names.joblib')
        logging.info("Feature columns saved as 'lstm_feature_names.joblib'")
        
        # Save sequence length
        joblib.dump(seq_length, models_dir / 'lstm_sequence_length.joblib')
        logging.info("Sequence length saved as 'lstm_sequence_length.joblib'")
        
        # ---> SAVE MEDIAN FILENAME <---
        # Save feature medians (added here for consistency)
        joblib.dump(feature_medians, models_dir / 'lstm_feature_medians.joblib')
        logging.info("Feature medians saved as 'lstm_feature_medians.joblib'")
        # ---> DONE WITH MEDIAN FILENAME <---

        # Save target columns
        joblib.dump(target_columns, models_dir / 'lstm_target_columns.joblib')
        logging.info("Target columns saved as 'lstm_target_columns.joblib'")
        
        logging.info("All necessary files for API saved in models directory")
        
        return True
        
    except Exception as e:
        logging.error(f"Error in LSTM model training: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
