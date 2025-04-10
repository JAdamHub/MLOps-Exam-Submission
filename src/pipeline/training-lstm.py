import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
import json
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime
import traceback

# Konfigurer logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_FEATURES_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = MODELS_DIR / "figures"

# Forecast horizons
FORECAST_HORIZONS = [1, 3, 7]  # Forudsig prisen 1, 3 og 7 dage frem

# LSTM-specifikke parametre
SEQUENCE_LENGTH = 10  # Antal dage at bruge som input sekvens
BATCH_SIZE = 8  # Mindre batch size for bedre generalisering
NUM_EPOCHS = 200  # Flere epochs med early stopping

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
    
    # Tjek om indekset indeholder datoer og sørg for at den ikke bliver en feature
    if isinstance(df.index, pd.DatetimeIndex) or df.index.name == 'date':
        logging.info("DataFrame har DatetimeIndex - gemmer indeks separat fra features")
        # Konverter indeks til en separat kolonne, hvis det er nødvendigt
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        # Behold indekset
    elif 'date' in df.columns:
        logging.info("Konverterer 'date' kolonne til indeks")
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
    
    # Identificer target kolonner baseret på navne
    target_columns = [f'price_target_{horizon}d' for horizon in FORECAST_HORIZONS]
    missing_targets = [col for col in target_columns if col not in df.columns]
    
    if missing_targets:
        logging.error(f"Missing target columns: {missing_targets}")
        raise ValueError(f"Target columns {missing_targets} not found in data")
    
    # Separate features and targets
    feature_columns = [col for col in df.columns if col not in target_columns]
    X = df[feature_columns].values
    
    # Create dictionary of targets for different horizons
    y_dict = {}
    for target_col in target_columns:
        y_dict[target_col] = df[target_col].values
    
    # Scale features using MinMaxScaler (bedre for LSTM end StandardScaler)
    feature_scaler = MinMaxScaler()
    X_scaled = feature_scaler.fit_transform(X)
    
    # Scale targets (vigtigt for LSTM)
    target_scalers = {}
    for target_col in target_columns:
        target_scaler = MinMaxScaler()
        y_dict[target_col] = target_scaler.fit_transform(y_dict[target_col].reshape(-1, 1)).flatten()
        target_scalers[target_col] = target_scaler
    
    # Split data into training (64%), validation (16%), and test (20%) sets
    # Dette følger inspirationsmodellens approach
    train_size = int(len(X_scaled) * 0.8)
    train_val_data = X_scaled[:train_size]
    test_data = X_scaled[train_size:]
    
    train_val_size = int(len(train_val_data) * 0.8)
    train_data = train_val_data[:train_val_size]
    val_data = train_val_data[train_val_size:]
    
    # Split targets med samme indekser
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
    
    return data_splits, feature_columns, target_columns, feature_scaler, target_scalers

def create_sequences(X, y_dict, seq_length):
    """
    Opret sekvenser af data for LSTM træning.
    X er input features, y_dict er et dictionary med targets for hver horisont.
    """
    X_seq, y_seq_dict = [], {k: [] for k in y_dict.keys()}
    
    for i in range(len(X) - seq_length):
        # Input sequence
        X_seq.append(X[i:i+seq_length])
        
        # Output sequence/value for each target
        for k, y in y_dict.items():
            y_seq_dict[k].append(y[i+seq_length])
    
    # Convert to numpy arrays
    X_seq = np.array(X_seq)
    y_seq = [np.array(y_seq_dict[k]) for k in sorted(y_seq_dict.keys())]
    
    return X_seq, y_seq

def build_multi_horizon_lstm_model(seq_length, n_features, n_outputs, lstm_units=64, dropout_rate=0.2, dense_units=32, learning_rate=0.001):
    """
    Bygger en multi-output LSTM model til at forudsige flere horisonter samtidigt.
    """
    # Input layer
    inputs = tf.keras.layers.Input(shape=(seq_length, n_features))
    
    # LSTM layers
    x = tf.keras.layers.LSTM(lstm_units, return_sequences=True)(inputs)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.LSTM(lstm_units)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    # Common dense layer
    x = tf.keras.layers.Dense(dense_units, activation='relu')(x)
    
    # Output layers - one for each forecast horizon
    outputs = []
    for i in range(n_outputs):
        output_name = f'output_{i+1}'
        outputs.append(tf.keras.layers.Dense(1, name=output_name)(x))
    
    # Create model
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse'
    )
    
    return model

def hyperparameter_tuning(data_splits, target_col):
    """Perform hyperparameter tuning for LSTM models."""
    logging.info("Starting hyperparameter tuning for LSTM...")
    
    # Hent data
    train_data, y_train = data_splits['train']
    val_data, y_val = data_splits['val']
    
    # Target for 1-dags horisont
    y_train_target = y_train[target_col]
    y_val_target = y_val[target_col]
    
    # Opret sekvenser til LSTM
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
    # Vi reducerer søgespacetn for at spare tid men vælger de mest sandsynlige vindere
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
            epochs=50,  # Reduceret for hyperparameter tuning
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
    """Evaluér model på testdata og returner detaljerede metrics."""
    # Forudsig på testdata
    y_pred = model.predict(X_test_seq).flatten()
    
    # Hvis dataene var skaleret, denormalisér dem
    if target_scaler is not None:
        y_test_denorm = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_denorm = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    else:
        y_test_denorm = y_test
        y_pred_denorm = y_pred
    
    # Beregn metrics
    mae = mean_absolute_error(y_test_denorm, y_pred_denorm)
    mse = mean_squared_error(y_test_denorm, y_pred_denorm)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_denorm, y_pred_denorm)
    
    # Vis metrics
    logging.info(f"\nTest Set Metrics for {horizon}-day horizon:")
    logging.info(f"MAE: {mae:.2f}")
    logging.info(f"MSE: {mse:.2f}")
    logging.info(f"RMSE: {rmse:.2f}")
    logging.info(f"R² Score: {r2:.4f}")
    
    # Lav visualisering
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_denorm, label='Actual')
    plt.plot(y_pred_denorm, label='Predicted')
    plt.title(f'LSTM Predictions vs Actuals - {horizon}-day Horizon (Vestas)')
    plt.xlabel('Time')
    plt.ylabel('Vestas Stock Price')
    plt.legend()
    plt.grid(True)
    
    # Gem visualisering
    fig_path = FIGURES_DIR / f"lstm_prediction_{horizon}d.png"
    plt.savefig(fig_path)
    plt.close()
    logging.info(f"Saved prediction visualization to {fig_path}")
    
    return {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2),
        'actuals': y_test_denorm.tolist(),
        'predictions': y_pred_denorm.tolist()
    }

def train_model(X_train, y_train_dict, X_val, y_val_dict, feature_scaler, target_scalers, 
                seq_length, epochs=50, batch_size=32, patience=10):
    """
    Træner en enkelt multi-horisont LSTM model til at forudsige alle horisonter samtidigt.
    """
    # Setup
    logging.info("Preparing data for LSTM training...")
    
    # Create sequences for training and validation
    X_train_seq, y_train_seq = create_sequences(X_train, y_train_dict, seq_length)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val_dict, seq_length)
    
    logging.info(f"Training data shape: {X_train_seq.shape}, {[y.shape for y in y_train_seq]}")
    logging.info(f"Validation data shape: {X_val_seq.shape}, {[y.shape for y in y_val_seq]}")
    
    # Build model
    logging.info("Building multi-horizon LSTM model...")
    horizon_keys = sorted(y_train_dict.keys())
    n_features = X_train.shape[1]
    
    # Hyperparameters
    lstm_units = 64
    dropout_rate = 0.2
    dense_units = 32
    learning_rate = 0.001
    
    model = build_multi_horizon_lstm_model(
        seq_length=seq_length,
        n_features=n_features, 
        n_outputs=len(horizon_keys),
        lstm_units=lstm_units, 
        dropout_rate=dropout_rate,
        dense_units=dense_units,
        learning_rate=learning_rate
    )
    
    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    
    # Model checkpoint
    checkpoint_path = MODELS_DIR / "lstm_model_checkpoint.keras"
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    )
    
    # Tensorboard
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, 
        histogram_freq=1
    )
    
    # Train model
    logging.info(f"Training LSTM model for {epochs} epochs with batch size {batch_size}...")
    history = model.fit(
        X_train_seq, 
        y_train_seq,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val_seq, y_val_seq),
        callbacks=[early_stopping, model_checkpoint, tensorboard_callback],
        verbose=1
    )
    
    return model, history, horizon_keys

def evaluate_multi_horizon_model(model, X_test, y_test_dict, feature_scaler, target_scalers, seq_length):
    """
    Evaluerer multi-horisont LSTM modellen på test data.
    """
    logging.info("Evaluating LSTM model on test data...")
    
    # Prepare test data
    X_test_seq, y_test_seq = create_sequences(X_test, y_test_dict, seq_length)
    
    # Evaluate model
    logging.info(f"Test data shape: {X_test_seq.shape}, {[y.shape for y in y_test_seq]}")
    test_loss = model.evaluate(X_test_seq, y_test_seq, verbose=0)
    logging.info(f"Test loss: {test_loss}")
    
    # Make predictions for each horizon
    predictions = model.predict(X_test_seq)
    
    # Calculate metrics for each horizon
    metrics = {}
    horizon_keys = sorted(y_test_dict.keys())
    
    for i, horizon_key in enumerate(horizon_keys):
        # Get target scaler for this horizon
        target_scaler = target_scalers[horizon_key]
        
        # Get actual and predicted values
        y_true = y_test_seq[i]
        y_pred = predictions[i]
        
        # Inverse transform if needed (not needed if we're predicting scaled values directly)
        y_true_inv = target_scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred_inv = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mae = mean_absolute_error(y_true_inv, y_pred_inv)
        mse = mean_squared_error(y_true_inv, y_pred_inv)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_inv, y_pred_inv)
        
        # Store metrics
        metrics[horizon_key] = {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'r2': float(r2)
        }
        
        logging.info(f"Metrics for {horizon_key}:")
        logging.info(f"  MAE: {mae:.4f}")
        logging.info(f"  MSE: {mse:.4f}")
        logging.info(f"  RMSE: {rmse:.4f}")
        logging.info(f"  R²: {r2:.4f}")
    
    return metrics

def save_multi_horizon_model(model, feature_scaler, target_scalers, metrics, feature_names, target_columns, seq_length, history=None):
    """
    Gemmer multi-horisont LSTM model, scalers og metrics.
    """
    logging.info("Saving LSTM model artifacts...")
    
    # Create directory if it doesn't exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = MODELS_DIR / "lstm_multi_horizon_model.keras"
    model.save(str(model_path))
    logging.info(f"Model saved to {model_path}")
    
    # Save feature scaler
    feature_scaler_path = MODELS_DIR / "lstm_feature_scaler.joblib"
    joblib.dump(feature_scaler, feature_scaler_path)
    
    # Save target scalers
    target_scalers_path = MODELS_DIR / "lstm_target_scalers.joblib"
    joblib.dump(target_scalers, target_scalers_path)
    
    # Save feature names
    feature_names_path = MODELS_DIR / "lstm_feature_names.joblib"
    joblib.dump(feature_names, feature_names_path)
    
    # Save target columns
    target_columns_path = MODELS_DIR / "lstm_target_columns.joblib"
    joblib.dump(target_columns, target_columns_path)
    
    # Save sequence length
    seq_length_path = MODELS_DIR / "lstm_sequence_length.joblib"
    joblib.dump(seq_length, seq_length_path)
    
    # Save metrics
    metrics_path = MODELS_DIR / "lstm_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save training history if available
    if history:
        history_dict = {
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss']
        }
        
        # Save outputs for each horizon
        for i, target_col in enumerate(target_columns):
            output_name = f'output_{i+1}'
            if f'{output_name}_loss' in history.history:
                history_dict[f'{target_col}_loss'] = history.history[f'{output_name}_loss']
                history_dict[f'{target_col}_val_loss'] = history.history[f'{output_name}_val_loss']
        
        history_path = MODELS_DIR / "lstm_training_history.json"
        with open(history_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_history = {k: [float(val) for val in v] for k, v in history_dict.items()}
            json.dump(serializable_history, f, indent=4)
    
    logging.info("LSTM model artifacts saved successfully")
    
def main():
    """
    Main function to orchestrate the training process for a multi-horizon LSTM model
    """
    start_time = time.time()
    logging.info("==== Starting Multi-Horizon LSTM model training process for Vestas stock prediction ====")
    
    # Tjek om TensorFlow GPU er tilgængeligt
    gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
    logging.info(f"GPU available for training: {gpu_available}")
    
    # Load data
    df = load_data()
    if df is None:
        logging.error("Failed to load data. Exiting training process.")
        return None
    
    # Preprocess data
    data_splits, feature_columns, target_columns, feature_scaler, target_scalers = prepare_data(df)
    
    # Sørg for at der er nok data til sekvenser
    train_data = data_splits['train'][0]
    if train_data.shape[0] <= SEQUENCE_LENGTH:
        logging.error(f"Not enough data points for sequence length {SEQUENCE_LENGTH}. Need more than {SEQUENCE_LENGTH} samples.")
        return None
    
    # Perform hyperparameter tuning
    best_params = hyperparameter_tuning(data_splits, target_columns[0])  # Tune on first target
    logging.info(f"Hyperparameter tuning complete")
    
    # Train multi-horizon model
    model, history, test_metrics = train_model(
        data_splits['train'][0], data_splits['train'][1],
        data_splits['val'][0], data_splits['val'][1],
        feature_scaler, target_scalers, SEQUENCE_LENGTH
    )
    
    # Save model and metrics
    save_success = save_multi_horizon_model(
        model, feature_scaler, target_scalers, 
        test_metrics, feature_columns, target_columns, SEQUENCE_LENGTH, history
    )
    
    if save_success:
        logging.info("Multi-horizon LSTM model and metrics saved successfully")
    else:
        logging.warning("Failed to save multi-horizon LSTM model or metrics")
    
    logging.info(f"==== Multi-horizon LSTM model training completed in {(time.time() - start_time) / 60:.2f} minutes ====")
    
    return model, history, test_metrics

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create directories if they don't exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    try:
        # Load and prepare data
        logging.info("Loading data...")
        features_file = PROCESSED_FEATURES_DIR / "vestas_features_trading_days.csv"
        df = pd.read_csv(features_file)
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
        
        # Define target columns - one for each forecast horizon
        forecast_horizons = [1, 3, 7]  # 1, 3, and 7 days ahead
        target_columns = [f'target_close_{h}d' for h in forecast_horizons]
        
        # Check if target columns exist
        for col in target_columns:
            if col not in df.columns:
                raise ValueError(f"Target column {col} not found in data")
        
        # Remove rows with NaN in target columns
        df = df.dropna(subset=target_columns)
        
        # Split data into train, validation, and test sets
        train_size = 0.7
        val_size = 0.15
        
        # Sort by date
        df = df.sort_index()
        
        # Define train, validation, and test indices
        n = len(df)
        train_idx = int(n * train_size)
        val_idx = train_idx + int(n * val_size)
        
        df_train = df.iloc[:train_idx]
        df_val = df.iloc[train_idx:val_idx]
        df_test = df.iloc[val_idx:]
        
        logging.info(f"Train size: {len(df_train)}")
        logging.info(f"Validation size: {len(df_val)}")
        logging.info(f"Test size: {len(df_test)}")
        
        # Select features (all columns except target columns)
        feature_columns = [col for col in df.columns if col not in target_columns]
        
        # Scale features
        feature_scaler = StandardScaler()
        X_train = feature_scaler.fit_transform(df_train[feature_columns])
        X_val = feature_scaler.transform(df_val[feature_columns])
        X_test = feature_scaler.transform(df_test[feature_columns])
        
        # Scale targets (one scaler for each target)
        target_scalers = {}
        y_train_dict = {}
        y_val_dict = {}
        y_test_dict = {}
        
        for target_col in target_columns:
            scaler = StandardScaler()
            
            # Scale each target
            y_train_dict[target_col] = scaler.fit_transform(df_train[[target_col]]).flatten()
            y_val_dict[target_col] = scaler.transform(df_val[[target_col]]).flatten()
            y_test_dict[target_col] = scaler.transform(df_test[[target_col]]).flatten()
            
            # Store scaler
            target_scalers[target_col] = scaler
        
        # Define sequence length
        seq_length = 30  # 30 days of historic data
        
        # Hyperparameters
        epochs = 100
        batch_size = 32
        patience = 10
        
        # Train model
        model, history, horizon_keys = train_model(
            X_train, y_train_dict, X_val, y_val_dict,
            feature_scaler, target_scalers, seq_length,
            epochs=epochs, batch_size=batch_size, patience=patience
        )
        
        # Evaluate model
        metrics = evaluate_multi_horizon_model(
            model, X_test, y_test_dict,
            feature_scaler, target_scalers, seq_length
        )
        
        # Save model and artifacts
        save_multi_horizon_model(
            model, feature_scaler, target_scalers, 
            metrics, feature_columns, target_columns, 
            seq_length, history
        )
        
        logging.info("LSTM model training completed successfully")
        
    except Exception as e:
        logging.error(f"Error in LSTM model training: {e}")
        traceback.print_exc()
