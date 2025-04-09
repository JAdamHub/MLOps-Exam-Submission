import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
import json
import time

# Konfigurer logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_FEATURES_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = MODELS_DIR / "figures"

# Forecast horizons (matchende dem i feature_engineering.py)
FORECAST_HORIZONS = [1, 3, 7]  # Forudsig prisen 1, 3 og 7 dage frem

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load feature data for training."""
    try:
        input_file = PROCESSED_FEATURES_DIR / "bitcoin_features_trading_days.csv"
        df = pd.read_csv(input_file)
        logging.info(f"Feature data loaded successfully from {input_file}, shape: {df.shape}")
        
        # Håndter manglende target kolonne (for test datasæt)
        if 'target_price_up' not in df.columns:
            logging.warning("'target_price_up' kolonne mangler - tilføjer både 0 og 1 værdier for test")
            
            # Hvis vi kun har én række, tilføj endnu en for at sikre både 0 og 1 værdier
            if len(df) == 1:
                logging.info("Kun én række i datasæt - duplicerer for at sikre tilstrækkelig data til træning")
                df = pd.concat([df, df.copy()], ignore_index=True)
                
            # Sørg for at vi har mindst én 0 og én 1 i target
            df['target_price_up'] = [0, 1] * (len(df) // 2) + [0] * (len(df) % 2)
        
        return df
    except Exception as e:
        logging.error(f"Error loading feature data: {e}")
        return None

def prepare_data(df):
    """Prepare data for training."""
    # Remove timestamp and ID columns
    df = df.copy()
    
    # Drop timestamp columns since they can't be converted to floats
    timestamp_cols = [col for col in df.columns if 'timestamp' in col.lower() or 'unnamed' in col.lower()]
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
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    logging.info(f"Prepared {X_scaled.shape[1]} features and {len(target_columns)} targets for training")
    
    return X_scaled, y_dict, feature_columns, target_columns, scaler

def select_features(X, y_dict, feature_columns, importance_threshold=0.005):
    """Select important features based on cross-validated feature importance."""
    logging.info("Performing feature selection for regression models...")
    
    # Vi bruger den første target (1-dag) til feature selection
    target_1d_key = f'price_target_1d'
    y = y_dict[target_1d_key]
    
    # Opret TimeSeriesSplit for krydsvalidering
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Forbered arrays til feature importance samling
    feature_importances = np.zeros(len(feature_columns))
    
    # Træn model med krydsvalidering for at få mere robuste feature importances
    fold = 1
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        logging.info(f"Training feature selection model - fold {fold}/5...")
        
        # Træn regression model på hvert fold
        model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05, 
            max_depth=5,
            objective='reg:squarederror',
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Akkumuler feature importances fra foldet
        feature_importances += model.feature_importances_
        fold += 1
    
    # Gennemsnit af feature importances på tværs af folds
    feature_importances /= 5
    
    # Create a dataframe of features and their importances
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    
    # Vælg features baseret på importance threshold
    selected_indices = feature_importances >= importance_threshold
    selected_features = [feature for i, feature in enumerate(feature_columns) if selected_indices[i]]
    
    # Hvis for få features er valgt, tag minimum top 20
    if len(selected_features) < 20:
        logging.info(f"Feature threshold resulterede i for få features ({len(selected_features)}), udvælger top 20")
        sorted_indices = np.argsort(feature_importances)[-20:]
        selected_indices = np.zeros_like(feature_importances, dtype=bool)
        selected_indices[sorted_indices] = True
        selected_features = [feature for i, feature in enumerate(feature_columns) if selected_indices[i]]
    
    # Select the columns based on selected features
    X_selected = X[:, selected_indices]
    
    logging.info(f"Selected {len(selected_features)} out of {len(feature_columns)} features")
    logging.info(f"Top 10 selected features: {selected_features[:min(10, len(selected_features))]}")
    
    return X_selected, selected_features

def hyperparameter_tuning(X, y_dict):
    """Perform hyperparameter tuning for regression models."""
    logging.info("Starting hyperparameter tuning for regression...")
    
    # Vi bruger den første target (1-dag) til hyperparameter tuning
    target_1d_key = f'price_target_1d'
    y = y_dict[target_1d_key]
    
    # Define time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Define parameter grid for regression
    param_grid = {
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0.5, 1, 1.5]
    }
    
    # Perform randomized search with time series CV
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42
    )
    
    # Randomized search for best parameters
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=20,  # Antal tilfældige kombinationer
        scoring='neg_mean_squared_error',
        cv=tscv,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    # Fit the random search
    logging.info("Fitting randomized search for hyperparameters...")
    random_search.fit(X, y)
    
    # Get best parameters
    best_params = random_search.best_params_
    logging.info(f"Best parameters found: {best_params}")
    
    return best_params

def train_model(X, y_dict, feature_names, target_columns, best_params=None):
    """Train regression models for multiple horizons."""
    logging.info("Starting model training for multiple forecast horizons...")
    
    # Define time series cross-validation for evaluation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Dict to store models for each horizon
    models = {}
    metrics = {}
    feature_importances = {}
    
    # For each forecast horizon
    for target_col in target_columns:
        horizon = target_col.split('_')[-1].replace('d', '')  # Extract horizon from column name
        y = y_dict[target_col]
        
        logging.info(f"Training model for {target_col} (horizon: {horizon} days)")
        
        # Initialize metrics
        cv_scores = {
            'rmse': [],
            'mae': [],
            'r2': []
        }
        
        # Set parameters for model (use tuned if available)
        params = best_params if best_params else {
            'n_estimators': 300,
            'learning_rate': 0.05,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror',
            'random_state': 42
        }
        
        # Dictionary to accumulate feature importances
        importances = np.zeros(len(feature_names))
        
        # Train with cross-validation for evaluation
        fold = 1
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            logging.info(f"Training {target_col} model - fold {fold}/5...")
            
            # Train model
            model = xgb.XGBRegressor(**params)
            model.fit(
                X_train, 
                y_train,
                verbose=False
            )
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logging.info(f"Fold {fold} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            
            # Accumulate metrics
            cv_scores['rmse'].append(rmse)
            cv_scores['mae'].append(mae)
            cv_scores['r2'].append(r2)
            
            # Accumulate feature importances
            importances += model.feature_importances_
            
            fold += 1
        
        # Average feature importances
        importances /= 5
        feature_importance_dict = {name: float(imp) for name, imp in zip(feature_names, importances)}
        
        # Train final model on all data
        final_model = xgb.XGBRegressor(**params)
        final_model.fit(X, y)
        
        # Store model, metrics, and feature importances
        models[target_col] = final_model
        metrics[target_col] = cv_scores
        feature_importances[target_col] = feature_importance_dict
        
        # Log average metrics
        avg_rmse = np.mean(cv_scores['rmse'])
        avg_mae = np.mean(cv_scores['mae'])
        avg_r2 = np.mean(cv_scores['r2'])
        logging.info(f"Average metrics for {target_col} - RMSE: {avg_rmse:.4f}, MAE: {avg_mae:.4f}, R²: {avg_r2:.4f}")
        
    return models, metrics, feature_importances

def save_model(models, scaler, metrics, feature_importances, feature_names, target_columns):
    """Save models, scaler, and metrics."""
    try:
        # Save individual models for each horizon
        for target_col in target_columns:
            horizon = target_col.split('_')[-1].replace('d', '')
            model_path = MODELS_DIR / f"xgboost_model_{horizon}d.joblib"
            joblib.dump(models[target_col], model_path)
            logging.info(f"Model for {horizon}-day forecast saved to {model_path}")
        
        # Save scaler
        scaler_path = MODELS_DIR / "scaler.joblib"
        joblib.dump(scaler, scaler_path)
        logging.info(f"Scaler saved successfully to {scaler_path}")
        
        # Save feature names
        feature_names_path = MODELS_DIR / "feature_names.joblib"
        joblib.dump(feature_names, feature_names_path)
        logging.info(f"Feature names saved successfully to {feature_names_path}")
        
        # Save target columns
        target_columns_path = MODELS_DIR / "target_columns.joblib"
        joblib.dump(target_columns, target_columns_path)
        logging.info(f"Target columns saved successfully to {target_columns_path}")
        
        # Calculate and save metrics
        metrics_dict = {
            'metrics': {target: {
                'rmse': float(np.mean(metrics[target]['rmse'])),
                'mae': float(np.mean(metrics[target]['mae'])),
                'r2': float(np.mean(metrics[target]['r2']))
            } for target in target_columns},
            'feature_importance': feature_importances
        }
        
        metrics_path = MODELS_DIR / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        logging.info(f"Metrics saved successfully to {metrics_path}")
        
        return True
    except Exception as e:
        logging.error(f"Error saving model artifacts: {e}")
        return False

def main():
    """
    Main function to orchestrate the training process for regression models
    """
    start_time = time.time()
    logging.info("==== Starting regression model training process ====")
    
    # Load data
    df = load_data()
    if df is None:
        logging.error("Failed to load data. Exiting training process.")
        return None
    
    # Preprocess data
    X, y_dict, feature_columns, target_columns, scaler = prepare_data(df)
    logging.info(f"Data prepared with {X.shape[1]} features, {len(target_columns)} targets, and {X.shape[0]} samples")
    
    # Perform feature selection
    X_selected, selected_features = select_features(X, y_dict, feature_columns)
    logging.info(f"Feature selection complete: {len(selected_features)} features selected from {len(feature_columns)}")
    
    # Perform hyperparameter tuning
    best_params = hyperparameter_tuning(X_selected, y_dict)
    logging.info(f"Hyperparameter tuning complete")
    
    # Train models for each forecast horizon
    models, metrics, feature_importances = train_model(X_selected, y_dict, selected_features, target_columns, best_params)
    
    # Save models and metrics
    save_success = save_model(models, scaler, metrics, feature_importances, selected_features, target_columns)
    if save_success:
        logging.info("Models and metrics saved successfully")
    else:
        logging.warning("Failed to save models or metrics")
    
    logging.info(f"==== Model training completed in {(time.time() - start_time) / 60:.2f} minutes ====")
    
    return models, metrics, feature_importances

if __name__ == "__main__":
    main()
