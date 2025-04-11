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
    
    # Fjern ikke-numeriske kolonner fra feature_columns
    numeric_feature_columns = []
    for col in feature_columns:
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_feature_columns.append(col)
        else:
            logging.info(f"Dropping non-numeric feature column: {col}")
    
    logging.info(f"Using {len(numeric_feature_columns)} numeric features out of {len(feature_columns)} total features")
    feature_columns = numeric_feature_columns
    
    # Håndter uendelige værdier og ekstreme tal
    for col in feature_columns:
        # Erstat inf og -inf med NaN
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        # Erstat NaN med median
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    
    # Log kolonner med ekstreme værdier
    for col in feature_columns:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val > 1e10 or min_val < -1e10:
            logging.warning(f"Feature {col} har ekstreme værdier: min={min_val}, max={max_val}")
            
            # Begræns ekstreme værdier
            df[col] = df[col].clip(-1e10, 1e10)
    
    # --- NY KODE: Konverter target til procentvis ændring ---
    # Gem de oprindelige målværdier til senere konvertering tilbage
    original_target_values = {}
    for target_col in target_columns:
        original_target_values[target_col] = df[target_col].values.copy()
    
    # Find 'Close' eller lignende kolonne for den aktuelle pris
    price_column = None
    for candidate in ['Close', 'close', 'price', 'Price']:
        if candidate in df.columns:
            price_column = candidate
            break
    
    if price_column is None:
        raise ValueError("Kunne ikke finde priskolonne i dataframen (Close, close, price, Price)")
    
    # Konverter targets til procentvise ændringer
    for target_col in target_columns:
        horizon = int(target_col.split('_')[-1].replace('d', ''))
        # Beregn procentvis ændring fra nuværende pris til target pris
        df[f'pct_change_{target_col}'] = ((df[target_col] / df[price_column]) - 1) * 100
        logging.info(f"Konverteret {target_col} til procent ændring: Middel = {df[f'pct_change_{target_col}'].mean():.2f}%, Std = {df[f'pct_change_{target_col}'].std():.2f}%")
    
    # Opdater target_columns til at bruge de nye procentvise kolonner
    percent_target_columns = [f'pct_change_{col}' for col in target_columns]
    logging.info(f"Nye target kolonner med procentvise ændringer: {percent_target_columns}")
    # --- SLUT PÅ NY KODE ---
    
    X = df[feature_columns].values
    
    # Create dictionary of targets for different horizons
    y_dict = {}
    for i, target_col in enumerate(target_columns):
        # Brug den procentvise ændring som target
        y_dict[target_col] = df[percent_target_columns[i]].values
    
    # Scale features using MinMaxScaler (bedre for LSTM end StandardScaler)
    feature_scaler = MinMaxScaler()
    X_scaled = feature_scaler.fit_transform(X)
    
    # Scale targets (vigtigt for LSTM)
    target_scalers = {}
    for target_col in target_columns:
        target_scaler = MinMaxScaler()
        y_values = y_dict[target_col].reshape(-1, 1)
        y_dict[target_col] = target_scaler.fit_transform(y_values).flatten()
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
    
    # Gem de oprindelige prisværdier for at kunne konvertere tilbage
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
    Opret sekventielle data til seq2seq LSTM modellen.
    
    Args:
        data: DataFrame eller numpy array med input features
        target_dict: Dictionary med targets for hver tidshorisont
        seq_length: Længde af input sekvensen
    
    Returns:
        X_seq: 3D numpy array med sekventielle input data [samples, seq_length, features]
        y_seq_dict: Dictionary med 2D numpy arrays for hver tidshorisont
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    n_samples = data.shape[0] - seq_length
    
    if n_samples <= 0:
        raise ValueError(f"Seq_length {seq_length} er længere end data med {data.shape[0]} samples!")
    
    # Opret input sekvenser (X)
    X_seq = np.zeros((n_samples, seq_length, data.shape[1]))
    
    for i in range(n_samples):
        X_seq[i] = data[i:i+seq_length]
    
    # Opret target sekvenser (y) hvis target_dict er givet
    if target_dict is not None:
        y_seq_dict = {}
        
        for horizon, target in target_dict.items():
            if isinstance(target, pd.DataFrame) or isinstance(target, pd.Series):
                target = target.values
                
            # For sequence-to-one prediction
            if target.ndim == 1:
                target = target.reshape(-1, 1)
                
            # Følg samme struktur som input sekvenserne
            y_seq = np.zeros((n_samples, 1))  # 1-dimensional output for hver horisont
            
            for i in range(n_samples):
                # Target er den værdi der kommer efter sekvensen
                # for den pågældende horisont
                y_seq[i] = target[i+seq_length-1]
                
            y_seq_dict[horizon] = y_seq
            
        return X_seq, y_seq_dict
    
    return X_seq

def build_multi_horizon_lstm_model(input_shape, output_dim=1):
    """
    Bygger en LSTM model til forudsigelse af flere tidshorisonter.
    
    Args:
        input_shape: Tuple med (seq_length, n_features)
        output_dim: Antal outputs (typisk 1 for hver tidshorisont)
        
    Returns:
        Kompileret Keras model
    """
    # Input lag
    input_layer = Input(shape=input_shape)
    
    # LSTM lag med dropout for at undgå overfitting
    lstm1 = LSTM(64, return_sequences=True)(input_layer)
    dropout1 = Dropout(0.2)(lstm1)
    
    lstm2 = LSTM(32)(dropout1)
    dropout2 = Dropout(0.2)(lstm2)
    
    # Output lag for hver tidshorisont
    outputs = []
    for h in ['1d', '3d', '7d']:
        dense = Dense(16, activation='relu')(dropout2)
        output = Dense(output_dim, name=f'output_{h}')(dense)
        outputs.append(output)
    
    # Bygger model
    model = Model(inputs=input_layer, outputs=outputs)
    
    # Kompilerer model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    logging.info(f"Model bygget med input shape {input_shape} og {len(outputs)} outputs")
    return model

def build_seq2seq_model(input_shape, horizon_keys=['1d', '3d', '7d']):
    """
    Bygger en avanceret seq2seq model med encoder-decoder arkitektur til flere tidshorisonter.
    Implementerer basal attention mekanisme for bedre forecasting.
    
    Args:
        input_shape: Tuple med (seq_length, n_features)
        horizon_keys: Liste med tidshorisonter der skal forudsiges
        
    Returns:
        Kompileret Keras model
    """
    logging.info(f"Bygger avanceret seq2seq model med input shape {input_shape} og {len(horizon_keys)} horisonter")
    
    # Input lag
    encoder_inputs = Input(shape=input_shape, name='encoder_input')
    
    # Encoder LSTM-lag
    encoder_lstm1 = Bidirectional(LSTM(128, return_sequences=True), name='encoder_bilstm')
    encoder_output1 = encoder_lstm1(encoder_inputs)
    encoder_output1 = Dropout(0.25)(encoder_output1)
    
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
    context_vector = Lambda(lambda x: K.sum(x, axis=1))(context_vector)
    
    # Gemmer encoder states til decoder initialisation
    encoder_states = [state_h, state_c]
    
    # Outputs for hver horisont
    outputs = {}
    for h in horizon_keys:
        # Decoder med attention
        decoder_lstm = LSTM(128, name=f'decoder_lstm_{h}')
        decoder_output = decoder_lstm(RepeatVector(1)(context_vector), initial_state=encoder_states)
        
        # Tætte lag for prognose
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
    
    # Bygger model
    model = Model(inputs=encoder_inputs, outputs=list(outputs.values()))
    
    # Definerer loss function og metrics for hver output
    losses = {output_name: 'mse' for output_name in outputs.keys()}
    loss_weights = {output_name: 1.0 for output_name in outputs.keys()}
    metrics = {output_name: 'mae' for output_name in outputs.keys()}
    
    # Kompilerer model
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
    Genererer sekventielle input-output par for LSTM modellen.
    
    Args:
        data: DataFrame med features og targets
        seq_length: Længde af input sekvenser
        
    Returns:
        X: Array med input sekvenser [samples, seq_length, features]
        y: Array med targets [samples, 1]
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i+seq_length].values)
        y.append(data.iloc[i+seq_length].values)
    
    return np.array(X), np.array(y)

def prepare_sequences_for_seq2seq(X_data, y_data_dict, seq_length, horizon_keys):
    """
    Forbereder sekvenser af data til seq2seq modellen.
    
    Args:
        X_data: DataFrame med features
        y_data_dict: Dictionary med targets for hver tidshorisont
        seq_length: Længde af sekvenser
        horizon_keys: Liste med nøgler for tidshorisonter
        
    Returns:
        X_sequences: Array med input sekvenser
        y_sequences_dict: Dictionary med target sekvenser for hver tidshorisont
    """
    # Opret tom liste til at gemme sekvenser
    sequences = []
    for i in range(len(X_data) - seq_length + 1):
        sequences.append(X_data.iloc[i:i+seq_length].values)
    
    X_sequences = np.array(sequences)
    
    # Opret target dictionary
    y_sequences_dict = {}
    for h in horizon_keys:
        if h in y_data_dict:
            # Tag targets fra index seq_length-1 og frem for at matche X_sequences
            y_seq = y_data_dict[h][seq_length-1:]
            y_sequences_dict[h] = np.array(y_seq)
    
    # Log information om data shapes
    logging.info(f"Sekvensdata forberedt: X shape: {X_sequences.shape}, " + 
               f"target shapes: {', '.join([f'{k}: {v.shape}' for k, v in y_sequences_dict.items()])}")
    
    return X_sequences, y_sequences_dict

def hyperparameter_tuning(data_splits, target_col):
    """
    Kører en simpel hyperparameter tuning for LSTM modellen.
    """
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
    """Evaluerer modellen med test data."""
    # Forudsig på testdata
    predictions = model.predict(X_test_seq)
    
    # Omform forudsigelser til 1D
    if isinstance(predictions, list):
        # Hvis vi har multiple outputs, find det rette baseret på horisonten
        pred_index = FORECAST_HORIZONS.index(horizon)
        predictions = predictions[pred_index]

    # Reshape til 2D for inverse transform
    predictions = predictions.reshape(-1, 1)
    
    # Konverter y_test til 2D hvis det er 1D
    if y_test.ndim == 1:
        y_test_2d = y_test.reshape(-1, 1)
    else:
        y_test_2d = y_test
    
    # Denormalisér forudsigelser og faktiske værdier
    predictions_denorm = target_scaler.inverse_transform(predictions)
    y_test_denorm = target_scaler.inverse_transform(y_test_2d)
    
    # --- NY KODE: Konverter procentvis ændring tilbage til prisværdier ---
    # Dette kræver at vi kender den aktuelle pris for hver testdag
    # Vi kan bruge de gemte testpriser fra prepare_data
    # Dette vil blive implementeret i evaluate_multi_horizon_model
    # --- SLUT PÅ NY KODE ---
    
    # Beregn fejlmål
    mse = mean_squared_error(y_test_denorm, predictions_denorm)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_denorm, predictions_denorm)
    
    # R^2 score (højere er bedre, max er 1.0)
    r2 = r2_score(y_test_denorm, predictions_denorm)
    
    # Beregn MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_test_denorm - predictions_denorm) / y_test_denorm)) * 100
    
    # Hvis R2 er negativ (værre end gennesnit), sæt til 0
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
    Træner en seq2seq LSTM model med early stopping.
    
    Args:
        model: Kompileret seq2seq model
        X_train: Træningsdata (samples, seq_length, features)
        y_train_dict: Dictionary med targets for hver horisont
        X_val: Valideringsdata
        y_val_dict: Dictionary med valideringsmål for hver horisont
        batch_size: Batch størrelse
        epochs: Maksimalt antal epoker
        patience: Tålmodighed for early stopping
        model_save_path: Sti til at gemme den bedste model
        
    Returns:
        Trænet model og træningshistorik
    """
    logging.info(f"Træner seq2seq model med {len(X_train)} træningssamples og {len(X_val)} valideringssamples")
    
    # Konverter y_train_dict og y_val_dict til formatet model.fit forventer
    # For seq2seq modellen bruger vi {output_navn: target_data}
    y_train_formatted = {f'output_{h}': y_train_dict[h] for h in y_train_dict.keys()}
    y_val_formatted = {f'output_{h}': y_val_dict[h] for h in y_val_dict.keys()}
    
    # Opsæt callbacks
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
    
    # Tilføj ModelCheckpoint hvis en sti er angivet
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
    
    # Træn modellen
    history = model.fit(
        X_train,
        y_train_formatted,
        validation_data=(X_val, y_val_formatted),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2
    )
    
    # Log træningsresultater
    final_epoch = len(history.history['loss'])
    final_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    logging.info(f"Træning afsluttet efter {final_epoch} epoker")
    logging.info(f"Final loss: {final_loss:.4f}, val_loss: {final_val_loss:.4f}")
    
    # Log individuelt horisont-metrics
    for h in y_train_dict.keys():
        output_name = f'output_{h}'
        if f'{output_name}_mae' in history.history:
            val_mae = history.history[f'val_{output_name}_mae'][-1]
            logging.info(f"Horisont {h} val_mae: {val_mae:.4f}")
    
    return model, history

def evaluate_multi_horizon_model(model, X_test, y_test_dict, horizon_keys, scaler=None):
    """
    Evaluerer modellen på testdata for alle horisonter.
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
            target_key = list(y_test_dict.keys())[i]  # Fallback til rækkefølge
        
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
            
            # --- NY KODE: Konverter procentvis ændring tilbage til prisværdier ---
            if is_percent_change_model:
                # Den forudsagte procentvise ændring skal konverteres tilbage til en pris
                # Bemærk: Hvis vi forecastet mange dage frem, skal vi bruge den korrekte basispris
                horizon_days = int(h.replace('d', ''))
                
                # Juster vektorer for at matche længder (vi kan kun bruge fælles datapunkter)
                min_len = min(len(y_pred_denorm), len(test_prices) - horizon_days)
                
                # Oprette arrays til de konverterede priser
                actual_prices = np.zeros(min_len)
                predicted_prices = np.zeros(min_len)
                
                for j in range(min_len):
                    # For de faktiske værdier bruger vi fremtidige priser
                    actual_prices[j] = test_prices[j + horizon_days]
                    
                    # For de forudsagte værdier anvender vi den procentvise ændring på nuværende pris
                    base_price = test_prices[j]
                    percent_change = y_pred_denorm[j][0]  # Forudsagt procentændring
                    predicted_prices[j] = base_price * (1 + percent_change/100)
                
                # Erstat de denormaliserede værdier med faktiske prisværdier
                y_pred_denorm = predicted_prices.reshape(-1, 1)
                y_true_denorm = actual_prices.reshape(-1, 1)
                
                logging.info(f"Konverteret {h}-horisont tilbage til priser: Basis mean={np.mean(test_prices[:min_len]):.2f}, Forecast mean={np.mean(predicted_prices):.2f}")
            # --- SLUT PÅ NY KODE ---
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
    Gemmer LSTM modellen sammen med dens scaler og andre metadata.
    """
    # Opret dato-timestamped path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODELS_DIR / f"lstm_multi_horizon_model.keras"
    meta_path = MODELS_DIR / f"lstm_model_metadata.json"
    
    # Gem modellen
    try:
        model.save(model_path)
        logging.info(f"Model gemt til {model_path}")
    except Exception as e:
        logging.error(f"Fejl ved gemning af model: {e}")
        traceback.print_exc()
    
    # Gem scalers
    joblib.dump(feature_scaler, MODELS_DIR / "lstm_feature_scaler.joblib")
    joblib.dump(target_scalers, MODELS_DIR / "lstm_target_scalers.joblib")
    
    # Gem feature og target navne
    joblib.dump(feature_names, MODELS_DIR / "lstm_feature_names.joblib")
    joblib.dump(target_columns, MODELS_DIR / "lstm_target_columns.joblib")
    
    # Gem sequence length
    joblib.dump(seq_length, MODELS_DIR / "lstm_sequence_length.joblib")
    
    # Gem feature medians til NaN-håndtering ved inference
    try:
        if isinstance(feature_names, list) and len(feature_names) > 0:
            # Load original data to calculate medians
            df = load_data()
            if df is not None:
                feature_medians = {}
                for feature in feature_names:
                    if feature in df.columns:
                        feature_medians[feature] = float(df[feature].median())
                
                # Gem medianer
                if feature_medians:
                    joblib.dump(feature_medians, MODELS_DIR / "lstm_feature_medians.joblib")
                    logging.info(f"Feature medians gemt for {len(feature_medians)} features")
    except Exception as e:
        logging.error(f"Fejl ved gemning af feature medians: {e}")
    
    # Gem metadata som JSON
    metadata = {
        "model_type": "LSTM Multi-Horizon",
        "created_at": timestamp,
        "sequence_length": seq_length,
        "feature_count": len(feature_names),
        "target_columns": target_columns,
        "metrics": metrics,
        "is_percent_change_model": True  # Tilføjet for at markere at denne model bruger procentvise ændringer
    }
    
    # Tilføj training history hvis tilgængelig
    if history is not None and hasattr(history, 'history'):
        # Konverter numpy arrays til lister for JSON serialisering
        hist_dict = {}
        for key, values in history.history.items():
            if isinstance(values, np.ndarray):
                hist_dict[key] = values.tolist()
            else:
                hist_dict[key] = list(values)
        
        metadata["training_history"] = hist_dict
        
        # Plot training history og gem som billede
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
            logging.error(f"Fejl ved plotting af training history: {e}")
    
    # Gem metadata
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logging.info(f"Model metadata gemt til {meta_path}")
    
    return model_path

def train_seq2seq_model(X_train, y_train_dict, X_val, y_val_dict, 
                        seq_length=30, horizon_keys=['1d', '3d', '7d'], 
                        epochs=100, batch_size=32):
    """
    Træner en seq2seq model med output for flere tidshorisonter.
    
    Args:
        X_train: Træningsdata, features
        y_train_dict: Dictionary med target for hver horisont for træningsdata
        X_val: Valideringsdata, features
        y_val_dict: Dictionary med target for hver horisont for valideringsdata
        seq_length: Længde af input sekvenser
        horizon_keys: Liste med tidshorisonter der skal forudsiges
        epochs: Antal trænings-epoker
        batch_size: Batch størrelse til træning
        
    Returns:
        Trænet model, scaler og træningshistorik
    """
    logging.info(f"Træner seq2seq model med {len(horizon_keys)} outputs: {horizon_keys}")
    
    # Forbered data som sekvenser
    X_train_seq, y_train_seq_dict = prepare_sequences_for_seq2seq(
        X_train, y_train_dict, seq_length, horizon_keys)
    X_val_seq, y_val_seq_dict = prepare_sequences_for_seq2seq(
        X_val, y_val_dict, seq_length, horizon_keys)
    
    # Opret model
    model = build_seq2seq_model(
        input_shape=(seq_length, X_train.shape[1]), 
        horizon_keys=horizon_keys
    )
    
    # Omstrukturerer target dictionaries til formatet keras forventer
    train_targets = {f'output_{h}': y_train_seq_dict[h] for h in horizon_keys if h in y_train_seq_dict}
    val_targets = {f'output_{h}': y_val_seq_dict[h] for h in horizon_keys if h in y_val_seq_dict}
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Reducer learning rate callback
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
    
    # Træn model
    logging.info(f"Starter træning med {epochs} epochs og batch size {batch_size}")
    history = model.fit(
        X_train_seq,
        train_targets,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val_seq, val_targets),
        callbacks=[early_stopping, reduce_lr, model_checkpoint],
        verbose=1
    )
    
    # Log træningsresultater
    logging.info(f"Træning afsluttet efter {len(history.history['loss'])} epochs")
    
    # Log validering loss for hver horisont
    for h in y_train_dict.keys():
        output_name = f'output_{h}'
        if f'{output_name}_mae' in history.history:
            val_mae = history.history[f'val_{output_name}_mae'][-1]
            logging.info(f"Horisont {h} val_mae: {val_mae:.4f}")
    
    return model, history

def evaluate_seq2seq_model(model, X_test, y_test_dict, seq_length=30, horizon_keys=['1d', '3d', '7d']):
    """
    Evaluerer en trænet seq2seq model på testdata og visualiserer resultaterne.
    
    Args:
        model: Trænet seq2seq model
        X_test: Testdata features
        y_test_dict: Dictionary med target værdier for hver horisont
        seq_length: Længde af input sekvenser
        horizon_keys: Liste med horisonter der skal forudsiges
        
    Returns:
        DataFrame med resultater og metrics
    """
    logging.info("Evaluerer seq2seq model på testdata...")
    
    # Forbered sekvenser til evaluering
    X_test_seq, y_test_seq_dict = prepare_sequences_for_seq2seq(
        X_test, y_test_dict, seq_length, horizon_keys)
    
    # Log information om data shapes
    logging.info(f"Test sekvenser: X shape={X_test_seq.shape}")
    for h in horizon_keys:
        if h in y_test_seq_dict:
            logging.info(f"Test targets for horisont {h}: shape={y_test_seq_dict[h].shape}")
    
    # Omstrukturér target dictionaries til format der passer til Keras
    test_targets = {f'output_{h}': y_test_seq_dict[h] for h in horizon_keys if h in y_test_seq_dict}
    
    # Evaluér model på testdata
    test_metrics = model.evaluate(X_test_seq, test_targets, verbose=1)
    logging.info(f"Test metrics: {test_metrics}")
    
    # Lav forudsigelser
    predictions = model.predict(X_test_seq)
    
    # Forbered en DataFrame til at gemme resultater
    results = pd.DataFrame()
    
    # For hver horisont, beregn metrics og visualiser resultater
    for i, horizon in enumerate(horizon_keys):
        if horizon in y_test_seq_dict:
            # Hent faktiske og forudsagte værdier
            y_true = y_test_seq_dict[horizon]
            
            # Håndter forskellige output formater fra model.predict
            if isinstance(predictions, dict):
                y_pred = predictions[f'output_{horizon}']
            else:
                # Hvis predictions er en liste af arrays (for multi-output modeller)
                y_pred = predictions[i] if i < len(predictions) else None
                
            if y_pred is None:
                logging.warning(f"Kunne ikke finde forudsigelser for horisont {horizon}")
                continue
                
            # Beregn metrics
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            # Log resultater
            logging.info(f"Horisont {horizon} - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            
            # Gem resultater
            horizon_results = pd.DataFrame({
                'Horizon': horizon,
                'MSE': mse,
                'RMSE': rmse, 
                'MAE': mae,
                'R2': r2
            }, index=[0])
            
            results = pd.concat([results, horizon_results], ignore_index=True)
            
            # Visualiser forudsigelser
            plt.figure(figsize=(12, 6))
            plt.plot(y_true[:100], label='Faktisk', color='blue')
            plt.plot(y_pred[:100], label='Forudsagt', color='red', linestyle='--')
            plt.title(f'Seq2Seq Model - Forudsigelser for {horizon} horisont')
            plt.xlabel('Samples')
            plt.ylabel('Normaliseret Prisændring')
            plt.legend()
            
            # Gem figur
            fig_dir = Path('plots')
            fig_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(fig_dir / f'seq2seq_predictions_{horizon}.png')
            plt.close()
    
    # Gem resultater til CSV
    results_dir = Path('results')
    results_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_dir / 'seq2seq_evaluation_results.csv', index=False)
    logging.info(f"Evaluering gemt til results/seq2seq_evaluation_results.csv")
    
    return results

def main():
    """
    Træner, evaluerer og gemmer en sequence-to-sequence LSTM model til at forudsige Vestas-aktiekursen
    for flere tidshorisonter samtidigt (1, 3 og 7 dage).
    """
    # Konfigurér logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logging.info("Starting LSTM model training process...")
    
    # Opret mapper hvis de ikke eksisterer
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Sæt random seeds for reproducerbarhed
    np.random.seed(42)
    tf.random.set_seed(42)
    
    try:
        # Indlæs og forbered data
        logging.info("Indlæser data...")
        features_file = PROCESSED_FEATURES_DIR / "vestas_features_trading_days.csv"
        df = pd.read_csv(features_file)
        
        # Konverter dato til index hvis den findes
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
        
        # Definer target kolonner - en for hver forecast horisont
        forecast_horizons = [1, 3, 7]  # 1, 3 og 7 dage frem
        target_columns = [f'price_target_{h}d' for h in forecast_horizons]
        
        # Tjek om target kolonnerne findes
        for col in target_columns:
            if col not in df.columns:
                raise ValueError(f"Målkolonne {col} findes ikke i data")
        
        logging.info(f"Anvender følgende målkolonner: {target_columns}")
        
        # Fjern rækker med NaN i target kolonnerne
        df = df.dropna(subset=target_columns)
        logging.info(f"Datasæt efter fjernelse af NaN-værdier: {df.shape}")
        
        # Del data i træning, validering og test
        train_size = 0.7
        val_size = 0.15
        
        # Sortér efter dato
        df = df.sort_index()
        
        # Definer træning, validering og test indeks
        n = len(df)
        train_idx = int(n * train_size)
        val_idx = train_idx + int(n * val_size)
        
        df_train = df.iloc[:train_idx]
        df_val = df.iloc[train_idx:val_idx]
        df_test = df.iloc[val_idx:]
        
        logging.info(f"Træningssæt størrelse: {len(df_train)}")
        logging.info(f"Valideringssæt størrelse: {len(df_val)}")
        logging.info(f"Testsæt størrelse: {len(df_test)}")
        
        # Vælg features (alle kolonner undtagen target kolonner)
        feature_columns = [col for col in df.columns if col not in target_columns]
        
        # Fjern ikke-numeriske kolonner fra feature_columns
        numeric_feature_columns = []
        for col in feature_columns:
            # Tjek om kolonnen er numerisk
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_feature_columns.append(col)
            else:
                logging.info(f"Dropper ikke-numerisk feature kolonne: {col}")
        
        logging.info(f"Bruger {len(numeric_feature_columns)} numeriske features ud af {len(feature_columns)} totale features")
        feature_columns = numeric_feature_columns
        
        # Håndter uendelige værdier og ekstreme tal
        for col in feature_columns:
            # Erstat inf og -inf med NaN
            df_train[col] = df_train[col].replace([np.inf, -np.inf], np.nan)
            df_val[col] = df_val[col].replace([np.inf, -np.inf], np.nan)
            df_test[col] = df_test[col].replace([np.inf, -np.inf], np.nan)
            
            # Erstat NaN med median
            median_val = df_train[col].median()
            df_train[col] = df_train[col].fillna(median_val)
            df_val[col] = df_val[col].fillna(median_val)
            df_test[col] = df_test[col].fillna(median_val)
        
        # Log kolonner med ekstreme værdier
        for col in feature_columns:
            min_val = df_train[col].min()
            max_val = df_train[col].max()
            if max_val > 1e10 or min_val < -1e10:
                logging.warning(f"Feature {col} har ekstreme værdier: min={min_val}, max={max_val}")
                
                # Begræns ekstreme værdier
                df_train[col] = df_train[col].clip(-1e10, 1e10)
                df_val[col] = df_val[col].clip(-1e10, 1e10)
                df_test[col] = df_test[col].clip(-1e10, 1e10)
                
        # Skaler features
        feature_scaler = MinMaxScaler()
        X_train = feature_scaler.fit_transform(df_train[feature_columns])
        X_val = feature_scaler.transform(df_val[feature_columns])
        X_test = feature_scaler.transform(df_test[feature_columns])
        
        # ---> BEREGN OG GEM MEDIANER FRA TRÆNINGSSÆTTET <---
        logging.info("Calculating medians from training data...")
        feature_medians = df_train[feature_columns].median().to_dict()
        medians_path = MODELS_DIR / 'lstm_feature_medians.joblib'
        joblib.dump(feature_medians, medians_path)
        logging.info(f"Feature medians saved to {medians_path}")
        # ---> FÆRDIG MED MEDIANER <---

        # Skaler targets (en scaler for hvert target)
        target_scalers = {}
        y_train_dict = {}
        y_val_dict = {}
        y_test_dict = {}
        
        for target_col in target_columns:
            scaler = MinMaxScaler()
            
            # Skaler hvert target
            y_train_dict[target_col] = scaler.fit_transform(df_train[[target_col]]).flatten()
            y_val_dict[target_col] = scaler.transform(df_val[[target_col]]).flatten()
            y_test_dict[target_col] = scaler.transform(df_test[[target_col]]).flatten()
            
            # Gem scaler
            target_scalers[target_col] = scaler
        
        # Definer sekvens længde (antal dage historik)
        seq_length = 30
        
        # Hyperparametre
        epochs = 100
        batch_size = 32
        patience = 15
        
        # Træn seq2seq model
        logging.info(f"Starter træning af seq2seq LSTM model med sekvens længde {seq_length}...")
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
        
        # Evaluér seq2seq model
        results = evaluate_seq2seq_model(
            model,
            pd.DataFrame(X_test, columns=feature_columns, index=df_test.index),
            {f'{h}d': y_test_dict[f'price_target_{h}d'] for h in [1, 3, 7]},
            seq_length=seq_length,
            horizon_keys=['1d', '3d', '7d']
        )
        
        # Plot træningshistorik
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Seq2Seq Model Training History')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('plots/seq2seq_training_history.png')
        
        logging.info("Seq2Seq model træning og evaluering gennemført")
        
        # Gem model og hjælpefiler til API
        logging.info("Gemmer model og hjælpefiler til API...")
        
        # Opret models-mappe hvis den ikke findes
        models_dir = Path('models')
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Gem model
        model.save(models_dir / 'lstm_multi_horizon_model.keras')
        logging.info("Model gemt som 'lstm_multi_horizon_model.keras'")
        
        # Gem feature scaler
        joblib.dump(feature_scaler, models_dir / 'lstm_feature_scaler.joblib')
        logging.info("Feature scaler gemt som 'lstm_feature_scaler.joblib'")
        
        # Gem target scalers
        joblib.dump(target_scalers, models_dir / 'lstm_target_scalers.joblib')
        logging.info("Target scalers gemt som 'lstm_target_scalers.joblib'")
        
        # Gem feature kolonner
        joblib.dump(feature_columns, models_dir / 'lstm_feature_names.joblib')
        logging.info("Feature kolonner gemt som 'lstm_feature_names.joblib'")
        
        # Gem sequence length
        joblib.dump(seq_length, models_dir / 'lstm_sequence_length.joblib')
        logging.info("Sequence length gemt som 'lstm_sequence_length.joblib'")
        
        # ---> GEM MEDIAN FILNAVN <---
        # Gem feature medians (tilføjet her for konsistens)
        joblib.dump(feature_medians, models_dir / 'lstm_feature_medians.joblib')
        logging.info("Feature medians saved as 'lstm_feature_medians.joblib'")
        # ---> FÆRDIG MED MEDIAN FILNAVN <---

        # Gem target kolonner
        joblib.dump(target_columns, models_dir / 'lstm_target_columns.joblib')
        logging.info("Target kolonner gemt som 'lstm_target_columns.joblib'")
        
        logging.info("Alle nødvendige filer til API gemt i models-mappen")
        
        return True
        
    except Exception as e:
        logging.error(f"Fejl i LSTM model træning: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
