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
import sys

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

def main(df_features: pd.DataFrame):
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
        # logging.info("Loading data...")
        # features_file = PROCESSED_FEATURES_DIR / "vestas_features_trading_days.csv"
        # df = pd.read_csv(features_file)
        if df_features is None or df_features.empty:
            logging.error("Halting training: No feature data received.")
            return False
        df = df_features.copy() # Work on a copy

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
        
        # Handle infinite values and extreme numbers
        for col in target_columns:
            # Replace inf and -inf with NaN
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            # We will handle NaNs globally next
            
            # Limit extreme values - Do this *after* handling NaNs
            # median_val = df_train[col].median()
            # df_train[col] = df_train[col].fillna(median_val)
            # df_val[col] = df_val[col].fillna(median_val)
            # df_test[col] = df_test[col].fillna(median_val)
        
        # Ensure no NaNs remain after inf handling (should ideally be cleaned in feature eng.)
        if df[col].isna().any():
            logging.error(f"NaNs detected in column '{col}' in training script after inf handling. Data should be cleaned in feature_engineering.py.")
            # Option: fill with 0 or median as a fallback, but indicates upstream issue
            # df[col].fillna(0, inplace=True)
            # return False # Or halt pipeline
        
        # Log columns with extreme values
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val > 1e10 or min_val < -1e10:
            logging.warning(f"Feature {col} has extreme values: min={min_val}, max={max_val}")
            # Limit extreme values
            df[col] = df[col].clip(-1e10, 1e10)

        # Check if data is empty after dropping NaNs
        if df.empty:
            logging.error("Halting training: DataFrame is empty after dropping NaN values.")
            return False

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
        
        # save feature scaler
        joblib.dump(feature_scaler, models_dir / 'lstm_feature_scaler.joblib')
        logging.info("Feature scaler saved as 'lstm_feature_scaler.joblib'")
        
        # save target scalers
        joblib.dump(target_scalers, models_dir / 'lstm_target_scalers.joblib')
        logging.info("Target scalers saved as 'lstm_target_scalers.joblib'")
        
        # Save feature columns
        joblib.dump(feature_columns, models_dir / 'lstm_feature_names.joblib')
        logging.info("Feature columns saved as 'lstm_feature_names.joblib'")
        
        # Save sequence length
        joblib.dump(seq_length, models_dir / 'lstm_sequence_length.joblib')
        logging.info("Sequence length saved as 'lstm_sequence_length.joblib'")
        
        # save feature medians
        joblib.dump(feature_medians, models_dir / 'lstm_feature_medians.joblib')
        logging.info("Feature medians saved as 'lstm_feature_medians.joblib'")

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
    # Add placeholder logic for running standalone if needed
    logging.warning("This script is intended to be run as part of the pipeline.")
    # Example: Load features data manually if needed for testing
    # project_root = Path(__file__).resolve().parents[2]
    # features_file = project_root / "data" / "features" / "vestas_features_trading_days.csv"
    # if features_file.exists():
    #     df_features = pd.read_csv(features_file)
    #     main(df_features)
    # else:
    #     logging.error("Feature file not found for standalone run.")
    #     sys.exit(1)
    sys.exit(0) # Indicate success if run standalone without error
