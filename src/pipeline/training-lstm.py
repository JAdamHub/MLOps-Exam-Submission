import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
import json
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

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

def create_sequences(X, y, seq_length=SEQUENCE_LENGTH):
    """Create sequences for LSTM input."""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

def build_lstm_model(input_shape, lstm_units=250, dropout_rate=0.3):
    """Build a deeper LSTM model architecture."""
    model = Sequential([
        # First LSTM layer with return sequences for stacking
        Bidirectional(LSTM(units=lstm_units, return_sequences=True), input_shape=input_shape),
        Dropout(dropout_rate),
        
        # Second LSTM layer
        Bidirectional(LSTM(units=lstm_units//2, return_sequences=False)),
        Dropout(dropout_rate),
        
        # Dense layers for better representation
        Dense(units=64, activation='relu'),
        Dropout(dropout_rate/2),
        
        # Output layer
        Dense(units=1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005),  # Lavere learning rate for bedre konvergens
        loss='mse'
    )
    
    # Vis model summary
    model.summary()
    
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

def train_model(data_splits, feature_columns, target_columns, target_scalers, best_params=None):
    """Train LSTM models for multiple horizons."""
    logging.info("Starting LSTM model training for multiple forecast horizons...")
    
    # Default parameters if not provided
    if best_params is None:
        best_params = {
            'lstm_units': 250,
            'dropout_rate': 0.3,
            'learning_rate': 0.0005,
            'batch_size': 8
        }
    
    # Dictionary to store models and metrics
    models = {}
    metrics = {}
    histories = {}
    test_metrics = {}
    
    # For each forecast horizon
    for target_col in target_columns:
        horizon = target_col.split('_')[-1].replace('d', '')  # Extract horizon from column name
        logging.info(f"Training LSTM model for {target_col} (horizon: {horizon} days)")
        
        # Hent data for training, validation og test
        train_data, y_train = data_splits['train']
        val_data, y_val = data_splits['val']
        test_data, y_test = data_splits['test']
        
        # Hent target for denne horisont
        y_train_target = y_train[target_col]
        y_val_target = y_val[target_col]
        y_test_target = y_test[target_col]
        
        # Create sequences for LSTM
        X_train_seq, y_train_seq = create_sequences(train_data, y_train_target)
        X_val_seq, y_val_seq = create_sequences(val_data, y_val_target)
        X_test_seq, y_test_seq = create_sequences(test_data, y_test_target)
        
        logging.info(f"Training set: {X_train_seq.shape}, Validation set: {X_val_seq.shape}, Test set: {X_test_seq.shape}")
        
        # Create callbacks
        checkpoint_path = MODELS_DIR / f"lstm_checkpoint_{horizon}d.h5"
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001),
            ModelCheckpoint(filepath=str(checkpoint_path), save_best_only=True, monitor='val_loss')
        ]
        
        # Build model
        model = build_lstm_model(
            input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),
            lstm_units=best_params['lstm_units'],
            dropout_rate=best_params['dropout_rate']
        )
        
        # Train model
        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=NUM_EPOCHS,  # Flere epochs med early stopping
            batch_size=best_params['batch_size'],
            validation_data=(X_val_seq, y_val_seq),
            callbacks=callbacks,
            verbose=1
        )
        
        # Load best model weights
        if checkpoint_path.exists():
            model.load_weights(str(checkpoint_path))
            logging.info(f"Loaded best model weights from checkpoint")
        
        # Evaluate on validation set
        val_metrics = {}
        y_val_pred = model.predict(X_val_seq).flatten()
        
        # Evaluate on test set
        test_results = evaluate_model(
            model, X_test_seq, y_test_seq, 
            target_scalers[target_col], horizon
        )
        
        # Store model and metrics
        models[target_col] = model
        metrics[target_col] = {
            'val_loss': history.history['val_loss'],
            'train_loss': history.history['loss']
        }
        histories[target_col] = history.history
        test_metrics[target_col] = test_results
    
    return models, metrics, histories, test_metrics

def save_model(models, feature_scaler, target_scalers, metrics, histories, test_metrics, feature_columns, target_columns):
    """Save LSTM models, scalers, and metrics."""
    try:
        # Save models in TensorFlow format
        for target_col in target_columns:
            horizon = target_col.split('_')[-1].replace('d', '')
            model_path = MODELS_DIR / f"lstm_model_{horizon}d.keras"
            models[target_col].save(model_path)
            logging.info(f"LSTM model for {horizon}-day forecast saved to {model_path}")
        
        # Save feature scaler
        feature_scaler_path = MODELS_DIR / "lstm_feature_scaler.joblib"
        joblib.dump(feature_scaler, feature_scaler_path)
        
        # Save target scalers
        target_scalers_path = MODELS_DIR / "lstm_target_scalers.joblib"
        joblib.dump(target_scalers, target_scalers_path)
        
        # Save feature names and target columns
        feature_names_path = MODELS_DIR / "lstm_feature_names.joblib"
        joblib.dump(feature_columns, feature_names_path)
        
        target_columns_path = MODELS_DIR / "lstm_target_columns.joblib"
        joblib.dump(target_columns, target_columns_path)
        
        # Save sequence length
        sequence_length_path = MODELS_DIR / "lstm_sequence_length.joblib"
        joblib.dump(SEQUENCE_LENGTH, sequence_length_path)
        
        # Create metrics dict including test metrics
        metrics_dict = {
            'training_metrics': {target: {
                'final_val_loss': float(metrics[target]['val_loss'][-1]),
                'final_train_loss': float(metrics[target]['train_loss'][-1]),
                'best_val_loss': float(min(metrics[target]['val_loss'])),
                'best_epoch': metrics[target]['val_loss'].index(min(metrics[target]['val_loss'])) + 1
            } for target in target_columns},
            'test_metrics': test_metrics,
            'training_history': {target: {
                'loss': [float(x) for x in histories[target]['loss']],
                'val_loss': [float(x) for x in histories[target]['val_loss']]
            } for target in target_columns}
        }
        
        # Plot training history
        for target_col in target_columns:
            horizon = target_col.split('_')[-1].replace('d', '')
            plt.figure(figsize=(10, 6))
            plt.plot(histories[target_col]['loss'], label='Training Loss')
            plt.plot(histories[target_col]['val_loss'], label='Validation Loss')
            plt.title(f'LSTM Training History - {horizon}-day Horizon (Vestas)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss (MSE)')
            plt.legend()
            plt.grid(True)
            
            # Gem plot
            history_path = FIGURES_DIR / f"lstm_history_{horizon}d.png"
            plt.savefig(history_path)
            plt.close()
            logging.info(f"Saved training history plot to {history_path}")
        
        # Save metrics as JSON
        metrics_path = MODELS_DIR / "lstm_training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        
        logging.info(f"All LSTM model artifacts saved successfully")
        return True
    except Exception as e:
        logging.error(f"Error saving LSTM model artifacts: {e}")
        return False

def main():
    """
    Main function to orchestrate the training process for LSTM models
    """
    start_time = time.time()
    logging.info("==== Starting LSTM model training process for Vestas stock price prediction ====")
    
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
    
    # Train models for each forecast horizon
    models, metrics, histories, test_metrics = train_model(
        data_splits, feature_columns, target_columns, target_scalers, best_params
    )
    
    # Save models and metrics
    save_success = save_model(
        models, feature_scaler, target_scalers, 
        metrics, histories, test_metrics, 
        feature_columns, target_columns
    )
    
    if save_success:
        logging.info("LSTM models and metrics saved successfully")
    else:
        logging.warning("Failed to save LSTM models or metrics")
    
    logging.info(f"==== LSTM model training completed in {(time.time() - start_time) / 60:.2f} minutes ====")
    
    return models, metrics, test_metrics

if __name__ == "__main__":
    main()
