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
import matplotlib.pyplot as plt

# Konfigurer logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_FEATURES_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = MODELS_DIR / "figures"

# Forecast horizons (matchende dem i feature_engineering.py)
FORECAST_HORIZONS = [1, 3, 7]  # Forudsig prisen 1, 3 og 7 dage frem

# Mere kompleks træning
CV_FOLDS = 10  # Øget fra 5 til 10 folds
N_ESTIMATORS_DEFAULT = 2000  # Øget fra 300 til 2000 træer
HYPERPARAMETER_ITERATIONS = 50  # Øget fra 20 til 50 kombinationer

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def get_default_params():
    """Return default parameters for XGBoost model."""
    return {
        'n_estimators': N_ESTIMATORS_DEFAULT,
        'learning_rate': 0.01,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'reg:squarederror',
        'random_state': 42,
        'n_jobs': -1,  # Parallel processing
        'eval_metric': ['rmse', 'mae'],
        'early_stopping_rounds': 50
    }

# Tilføj ny funktion til at generere flere features
def generate_additional_features(df):
    """Generate more complex features for better model training."""
    try:
        # Kontroller om de nødvendige kolonner findes
        required_columns = ['close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logging.warning(f"Missing required columns for feature generation: {missing_columns}")
            logging.info(f"Available columns: {df.columns.tolist()[:10]}...")
            return df
        
        # Basis features
        if 'close' in df.columns:
            # Undgå at overtræde kolonner der allerede kan være lavet i feature_engineering.py
            if 'price_volatility_30d' not in df.columns:
                df['price_volatility_30d'] = df['close'].rolling(30).std() / df['close'].rolling(30).mean()
            
            if 'volume_price_ratio' not in df.columns and 'volume' in df.columns:
                df['volume_price_ratio'] = df['volume'] / df['close']
            
            if 'price_momentum' not in df.columns:
                df['price_momentum'] = df['close'] / df['close'].shift(7) - 1
            
            # Ekstra tekniske indikatorer
            # EMA - Exponential Moving Average
            if 'price_ema_5' not in df.columns:
                df['price_ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
            
            if 'price_ema_20' not in df.columns:
                df['price_ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
            
            if 'price_ema_ratio' not in df.columns:
                df['price_ema_ratio'] = df['price_ema_5'] / df['price_ema_20']
        
        # Handle missing values 
        df = df.replace([np.inf, -np.inf], np.nan)
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        logging.info(f"Additional features generated successfully. Now with {len(df.columns)} columns.")
        return df
        
    except Exception as e:
        logging.error(f"Error in generate_additional_features: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return df

def load_data():
    """Load feature data for training."""
    try:
        input_file = PROCESSED_FEATURES_DIR / "vestas_features_trading_days.csv"
        df = pd.read_csv(input_file)
        logging.info(f"Feature data loaded successfully from {input_file}, shape: {df.shape}")
        
        # Håndter manglende target kolonne (for test datasæt)
        for horizon in FORECAST_HORIZONS:
            target_col = f'price_target_{horizon}d'
            if target_col not in df.columns:
                logging.warning(f"'{target_col}' kolonne mangler - tjek data")
        
        # Tilføj flere features med ny funktion
        df = generate_additional_features(df)
        logging.info(f"Udvidet feature engineering udført. Nye shape: {df.shape}")
        
        return df
    except Exception as e:
        logging.error(f"Error loading feature data: {e}")
        return None

def prepare_data(df):
    """Prepare data for training."""
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
    else:
        # Hvis der er indeks som ikke er dato, men numerisk (0, 1, 2, ...) 
        logging.info("DataFrame har numerisk indeks")

    # Drop indeks-relaterede og ID kolonner
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
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    logging.info(f"Prepared {X_scaled.shape[1]} features and {len(target_columns)} targets for training")
    
    return X_scaled, y_dict, feature_columns, target_columns, scaler

def select_features(X, y_dict, feature_columns, importance_threshold=0.003):  # Sænket tærskel for at inkludere flere features
    """Select important features based on cross-validated feature importance."""
    logging.info("Performing feature selection for regression models...")
    
    # Vi bruger den første target (1-dag) til feature selection
    target_1d_key = f'price_target_1d'
    y = y_dict[target_1d_key]
    
    # Opret TimeSeriesSplit for krydsvalidering
    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)  # Øget antal folds
    
    # Forbered arrays til feature importance samling
    feature_importances = np.zeros(len(feature_columns))
    
    # Træn model med krydsvalidering for at få mere robuste feature importances
    fold = 1
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        logging.info(f"Training feature selection model - fold {fold}/{CV_FOLDS}...")
        
        # Træn regression model på hvert fold - med flere træer og kompleksitet
        model = xgb.XGBRegressor(
            n_estimators=1000,  # Øget antal træer for bedre feature selection
            learning_rate=0.01,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.5,
            reg_lambda=1.5,
            objective='reg:squarederror',
            tree_method='hist',  # Hurtigere træningsalgoritme
            n_jobs=-1,  # Brug alle CPU-kerner
            eval_metric=['rmse']  # Tilføjet eval_metric til model instans
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100
        )
        
        # Akkumuler feature importances fra foldet
        feature_importances += model.feature_importances_
        fold += 1
    
    # Gennemsnit af feature importances på tværs af folds
    feature_importances /= CV_FOLDS
    
    # Create a dataframe of features and their importances
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    
    # Visualiser feature importance
    plt.figure(figsize=(12, 10))
    plt.barh(feature_importance_df['Feature'].head(30), feature_importance_df['Importance'].head(30))
    plt.xlabel('Feature Importance')
    plt.title('Top 30 Most Important Features')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance_selection.png")
    plt.close()
    
    # Vælg features baseret på importance threshold
    selected_indices = feature_importances >= importance_threshold
    selected_features = [feature for i, feature in enumerate(feature_columns) if selected_indices[i]]
    
    # Hvis for få features er valgt, tag minimum top 30 (i stedet for 20)
    if len(selected_features) < 30:
        logging.info(f"Feature threshold resulterede i for få features ({len(selected_features)}), udvælger top 30")
        sorted_indices = np.argsort(feature_importances)[-30:]
        selected_indices = np.zeros_like(feature_importances, dtype=bool)
        selected_indices[sorted_indices] = True
        selected_features = [feature for i, feature in enumerate(feature_columns) if selected_indices[i]]
    
    # Select the columns based on selected features
    X_selected = X[:, selected_indices]
    
    logging.info(f"Selected {len(selected_features)} out of {len(feature_columns)} features")
    logging.info(f"Top 15 selected features: {selected_features[:min(15, len(selected_features))]}")
    
    return X_selected, selected_features

def hyperparameter_tuning(X, y_dict):
    """Perform hyperparameter tuning for regression models."""
    logging.info("Starting hyperparameter tuning for regression...")
    
    # Vi bruger den første target (1-dag) til hyperparameter tuning
    target_1d_key = f'price_target_1d'
    y = y_dict[target_1d_key]
    
    # Define time series cross-validation - med flere folds
    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
    
    # Define parameter grid for regression - udvidet med flere parametre og værdier
    param_grid = {
        'n_estimators': [500, 1000, 2000, 3000, 5000],
        'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
        'max_depth': [2, 3, 4, 5, 6, 7, 8, 9],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bylevel': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 2, 3, 5, 7, 10],
        'gamma': [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
        'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
        'reg_lambda': [0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0],
        'tree_method': ['hist', 'approx', 'gpu_hist']
    }
    
    logging.info(f"Hyperparameter grid contains {len(param_grid)} parameters with a total search space of:")
    for param, values in param_grid.items():
        logging.info(f"  - {param}: {len(values)} values = {values}")
    
    # Perform randomized search with time series CV
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1  # Brug alle CPU-kerner
    )
    
    # Øget antal iterationer for mere grundig søgning
    n_iterations = HYPERPARAMETER_ITERATIONS * 3  # Tredobbelt antal søgninger
    logging.info(f"Performing randomized search with {n_iterations} iterations")
    
    # Randomized search for best parameters - med flere kombinationer
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iterations,
        scoring='neg_mean_squared_error',
        cv=tscv,
        verbose=3,  # Meget højere verbosity for løbende opdateringer
        random_state=42,
        n_jobs=-1
    )
    
    # Fit the random search
    logging.info("Fitting randomized search for hyperparameters...")
    search_start = time.time()
    random_search.fit(X, y)
    search_duration = (time.time() - search_start) / 60
    
    # Get best parameters
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    logging.info(f"Best parameters found in {search_duration:.2f} minutes:")
    for param, value in best_params.items():
        logging.info(f"  - {param}: {value}")
    logging.info(f"Best validation score (neg MSE): {best_score:.6f}")
    
    # Gem de 10 bedste parameterkombinationer til reference
    results = pd.DataFrame(random_search.cv_results_)
    top_results = results.sort_values('rank_test_score').head(10)
    logging.info("Top 10 hyperparameter combinations:")
    for i, (_, row) in enumerate(top_results.iterrows(), 1):
        score = row['mean_test_score']
        params = row['params']
        logging.info(f"Rank {i}: Score={score:.6f}, Params={params}")
    
    # Visualisér hvordan de bedste parameters præsterer
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(top_results)), -top_results['mean_test_score'])
    plt.xlabel('Model Rank')
    plt.ylabel('Mean Squared Error')
    plt.title('Top 10 Hyperparameter Combinations')
    plt.xticks(range(len(top_results)), [f'Rank {i+1}' for i in range(len(top_results))])
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "hyperparameter_tuning_results.png")
    plt.close()
    
    # Gem alle parameterresultater til senere analyse
    results_file = MODELS_DIR / "hyperparameter_search_results.csv"
    results.to_csv(results_file)
    logging.info(f"Full hyperparameter search results saved to {results_file}")
    
    return best_params

def train_model(X, y_dict, feature_names, target_columns, best_params=None):
    """Train regression models for multiple horizons."""
    logging.info("Starting model training for multiple forecast horizons...")
    
    # Define time series cross-validation for evaluation - med flere folds
    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
    
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
        params = best_params.copy() if best_params else {
            'n_estimators': N_ESTIMATORS_DEFAULT,  # Øget antal træer
            'learning_rate': 0.01,  # Lavere learning rate for flere iterations
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror',
            'random_state': 42,
            'n_jobs': -1  # Parallel processing
        }
        
        # Log parameters being used
        logging.info(f"Training with parameters:")
        for param, value in params.items():
            logging.info(f"  - {param}: {value}")
        
        # Dictionary to accumulate feature importances
        importances = np.zeros(len(feature_names))
        
        # For at gemme træningskurver
        train_losses = []
        val_losses = []
        
        # Train with cross-validation for evaluation
        fold = 1
        for train_idx, test_idx in tscv.split(X):
            try:
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Yderligere split for tidlig standsning validering
                train_size = int(0.8 * len(X_train))
                X_train_inner, X_val_inner = X_train[:train_size], X_train[train_size:]
                y_train_inner, y_val_inner = y_train[:train_size], y_train[train_size:]
                
                logging.info(f"Training {target_col} model - fold {fold}/{CV_FOLDS}...")
                logging.info(f"Train set size: {X_train_inner.shape[0]}, Validation set size: {X_val_inner.shape[0]}, Test set size: {X_test.shape[0]}")
                
                # Monitor progress carefully to prevent overfitting
                eval_set = [
                    (X_train_inner, y_train_inner),  # Train set for training curve
                    (X_val_inner, y_val_inner),      # Validation set for early stopping
                    (X_test, y_test)                 # Test set for final evaluation
                ]
                
                # Træn modellen med tidsmåling
                start_time = time.time()
                # Tilføj eval_metric til model params i stedet for fit
                if 'eval_metric' not in params:
                    params_with_metric = params.copy()
                    params_with_metric['eval_metric'] = ['rmse', 'mae']
                    params_with_metric['early_stopping_rounds'] = 50  # Tilføjet her i stedet
                else:
                    params_with_metric = params
                    if 'early_stopping_rounds' not in params_with_metric:
                        params_with_metric['early_stopping_rounds'] = 50
                    
                model = xgb.XGBRegressor(**params_with_metric)
                
                model.fit(
                    X_train_inner, 
                    y_train_inner,
                    eval_set=eval_set,
                    verbose=100  # Fjernet early_stopping_rounds herfra
                )
                training_time = time.time() - start_time
                
                # Gem eval resultater
                results = model.evals_result()
                best_iteration = model.best_iteration if hasattr(model, 'best_iteration') else model.get_booster().best_iteration
                logging.info(f"Best iteration found: {best_iteration}")
                
                if results:
                    # Extract training and validation metrics
                    try:
                        if len(results) >= 2:
                            train_key = list(results.keys())[0]
                            val_key = list(results.keys())[1]
                            if 'rmse' in results[train_key]:
                                train_losses.extend(results[train_key]['rmse'])
                                val_losses.extend(results[val_key]['rmse'])
                                
                                # Log final training/validation metrics
                                final_train_rmse = results[train_key]['rmse'][-1]
                                final_val_rmse = results[val_key]['rmse'][-1]
                                logging.info(f"Final training RMSE: {final_train_rmse:.4f}, validation RMSE: {final_val_rmse:.4f}")
                                
                                # Check for potential overfitting
                                if final_val_rmse > 1.2 * final_train_rmse:
                                    logging.warning(f"Potential overfitting detected: validation RMSE is {final_val_rmse/final_train_rmse:.2f}x training RMSE")
                    except Exception as e:
                        logging.warning(f"Error extracting training curves: {e}")
                
                logging.info(f"Fold {fold} training time: {training_time:.2f} seconds, {best_iteration} iterations")
                
                # Make predictions on test set
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
                if hasattr(model, 'feature_importances_'):
                    importances += model.feature_importances_
                
                # Additional diagnostics - residual analysis
                residuals = y_test - y_pred
                mean_residual = np.mean(residuals)
                std_residual = np.std(residuals)
                
                logging.info(f"Residual analysis - Mean: {mean_residual:.4f}, Std: {std_residual:.4f}")
                
                # Generate residual plot for this fold
                plt.figure(figsize=(10, 6))
                plt.scatter(y_pred, residuals)
                plt.axhline(y=0, color='r', linestyle='-')
                plt.xlabel('Predicted Values')
                plt.ylabel('Residuals')
                plt.title(f'Residual Plot - {horizon}-day Forecast - Fold {fold}')
                plt.savefig(FIGURES_DIR / f"residuals_{horizon}d_fold{fold}.png")
                plt.close()
                
                fold += 1
                
            except Exception as e:
                logging.error(f"Error in fold {fold}: {e}")
                import traceback
                logging.error(traceback.format_exc())
                # Continue with next fold
                fold += 1
                continue
        
        # Average feature importances
        importances /= max(1, fold - 1)  # Guard against division by zero
        feature_importance_dict = {name: float(imp) for name, imp in zip(feature_names, importances)}
        
        # Visualiser feature importance for denne model
        if len(feature_names) > 0:
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            plt.figure(figsize=(12, 8))
            plt.barh(importance_df['Feature'].head(20), importance_df['Importance'].head(20))
            plt.xlabel('Feature Importance')
            plt.title(f'Top 20 Features for {horizon}-day Forecast')
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"feature_importance_{horizon}d.png")
            plt.close()
        
        # Visualiser træningskurver hvis tilgængelige
        if train_losses and val_losses:
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Training Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Iterations')
            plt.ylabel('RMSE')
            plt.title(f'Learning Curves for {horizon}-day Forecast')
            plt.legend()
            plt.grid(True)
            plt.savefig(FIGURES_DIR / f"learning_curve_{horizon}d.png")
            plt.close()
        
        # Train final model on entire dataset
        final_params = params.copy()  # Brug de samme params som blev brugt i cross-validation
        if final_params is None:
            # Use default params if none provided
            final_params = get_default_params()
        
        # Logging
        logging.info(f"Training final model with params: {final_params}")
        
        # Tilføj eval_metric til model params i stedet for fit
        if 'eval_metric' not in final_params:
            final_params_with_metric = final_params.copy()
            final_params_with_metric['eval_metric'] = ['rmse', 'mae']
        else:
            final_params_with_metric = final_params
        
        # Create model
        final_model = xgb.XGBRegressor(**final_params_with_metric)
        
        try:
            # Train final model
            final_model.fit(
                X_train, 
                y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=100
            )
            
            # Save model
            models[target_col] = final_model
            metrics[target_col] = cv_scores
            feature_importances[target_col] = feature_importance_dict
            
            # Log average metrics
            avg_rmse = np.mean(cv_scores['rmse'])
            avg_mae = np.mean(cv_scores['mae'])
            avg_r2 = np.mean(cv_scores['r2'])
            logging.info(f"Average metrics for {target_col} - RMSE: {avg_rmse:.4f}, MAE: {avg_mae:.4f}, R²: {avg_r2:.4f}")
            
        except Exception as e:
            logging.error(f"Error training final model: {e}")
            import traceback
            logging.error(traceback.format_exc())
            final_model = None
        
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
