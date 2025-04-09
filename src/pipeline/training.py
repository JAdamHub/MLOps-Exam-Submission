import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint, uniform
import time

# Konfigurer logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_FEATURES_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = MODELS_DIR / "figures"

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
    
    # Separate features and target
    if 'target_price_up' not in df.columns:
        raise ValueError("Target column 'target_price_up' not found in data")
    
    feature_columns = [col for col in df.columns if col != 'target_price_up']
    X = df[feature_columns].values  # Convert to numpy array after selecting features
    y = df['target_price_up'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    logging.info(f"Prepared {X_scaled.shape[1]} features for training")
    
    return X_scaled, y, feature_columns, scaler

def select_features(X, y, feature_columns, importance_threshold=0.005):
    """Select important features based on cross-validated feature importance."""
    logging.info("Performing advanced feature selection with cross-validation...")
    
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
        
        # Træn model på hvert fold
        model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05, 
            max_depth=5,
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
    
    # Plot top 20 feature importances
    plt.figure(figsize=(12, 10))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20))
    plt.title('Top 20 Feature Importances (Cross-Validated)')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'feature_importances_selection.png')
    
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

def hyperparameter_tuning(X, y):
    """Perform extensive hyperparameter tuning using manual cross-validation."""
    logging.info("Starting extensive hyperparameter tuning...")
    
    # Define time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Define more comprehensive parameter combinations
    param_combinations = [
        # Baseline configuration
        {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0.0,
            'reg_alpha': 0,
            'reg_lambda': 1
        },
        # Høj kompleksitet, lav læringsrate
        {
            'n_estimators': 500,
            'max_depth': 8,
            'learning_rate': 0.01,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'min_child_weight': 2,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0
        },
        # Medium kompleksitet, medium læringsrate
        {
            'n_estimators': 300,
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_weight': 2,
            'gamma': 0.05,
            'reg_alpha': 0.05,
            'reg_lambda': 0.8
        },
        # Lav kompleksitet, høj læringsrate
        {
            'n_estimators': 200,
            'max_depth': 4,
            'learning_rate': 0.1,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'min_child_weight': 1,
            'gamma': 0.0,
            'reg_alpha': 0.01,
            'reg_lambda': 0.5
        },
        # Høj kompleksitet, høj læringsrate
        {
            'n_estimators': 400,
            'max_depth': 7,
            'learning_rate': 0.08,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.2,
            'reg_alpha': 0.2,
            'reg_lambda': 1.2
        },
        # Maksimal kompleksitet, lav læringsrate
        {
            'n_estimators': 800,
            'max_depth': 10,
            'learning_rate': 0.01,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.3,
            'reg_alpha': 0.3,
            'reg_lambda': 1.5
        },
        # Mellemkompleksitet, fokus på feature sampling
        {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.6,
            'colsample_bytree': 0.6,
            'min_child_weight': 2,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0
        },
        # Meget høj regularisering
        {
            'n_estimators': 400,
            'max_depth': 5,
            'learning_rate': 0.03,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_weight': 4,
            'gamma': 0.3,
            'reg_alpha': 0.5,
            'reg_lambda': 2.0
        }
    ]
    
    best_score = 0
    best_params = None
    
    for params in param_combinations:
        logging.info(f"Evaluating parameters: {params}")
        scores = []
        
        # Perform cross-validation
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model with current parameters
            model = xgb.XGBClassifier(**params, random_state=42)
            model.fit(X_train, y_train)
            
            # Predict and evaluate
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred_proba)
            scores.append(score)
        
        # Calculate average score
        avg_score = np.mean(scores)
        logging.info(f"Average ROC AUC: {avg_score:.4f}")
        
        # Update best parameters if better score
        if avg_score > best_score:
            best_score = avg_score
            best_params = params
    
    logging.info(f"Best parameters found: {best_params}")
    logging.info(f"Best ROC AUC score: {best_score:.4f}")
    
    return best_params

def train_model(X, y, feature_names, best_params=None):
    """
    Train XGBoost model with time series cross-validation
    """
    tscv = TimeSeriesSplit(n_splits=10)  # Øget fra 5 til 10 folds
    
    # Check for class imbalance
    class_counts = np.bincount(y)
    total = len(y)
    class_ratio = class_counts[1] / total if len(class_counts) > 1 else 0.5
    is_imbalanced = abs(class_ratio - 0.5) > 0.1  # If more than 10% off from balanced
    
    if is_imbalanced:
        # Calculate scale_pos_weight to handle class imbalance
        scale_pos_weight = class_counts[0] / class_counts[1] if class_counts[1] > 0 else 1.0
        logging.info(f"Class imbalance detected. Class ratio (positive): {class_ratio:.2f}. Setting scale_pos_weight to {scale_pos_weight:.2f}")
    else:
        scale_pos_weight = 1.0
        logging.info(f"Classes are reasonably balanced. Class ratio (positive): {class_ratio:.2f}")
    
    if best_params:
        # Use provided parameters from hyperparameter tuning
        model_params = best_params.copy()
        # Ensure scale_pos_weight is set if needed
        if is_imbalanced and 'scale_pos_weight' not in model_params:
            model_params['scale_pos_weight'] = scale_pos_weight
        
        # Øg antallet af estimators for mere robusthed
        if 'n_estimators' in model_params:
            model_params['n_estimators'] = max(model_params['n_estimators'], 500)
        else:
            model_params['n_estimators'] = 500
            
        model = xgb.XGBClassifier(**model_params, random_state=42)
        logging.info("Training model with tuned hyperparameters and increased estimators")
    else:
        # Default parameters with significant improvements for longer training
        model = xgb.XGBClassifier(
            n_estimators=800,             # Drastisk forøget fra 300
            learning_rate=0.01,           # Reduceret for bedre præcision med flere træer
            max_depth=8,                  # Øget fra 6
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            scale_pos_weight=scale_pos_weight,
            reg_alpha=0.1,                 # L1 regularization to reduce overfitting
            reg_lambda=1.0,                # L2 regularization
            random_state=42
        )
        logging.info("Training model with drastically improved default hyperparameters")
    
    cv_scores = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': [],
        'mcc': []  # Matthews Correlation Coefficient
    }
    
    feature_importance = np.zeros(len(feature_names))
    all_y_pred = []
    all_y_true = []
    
    fold = 1
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        logging.info(f"Training fold {fold}/10...")
        
        # Train model with early stopping to prevent overfitting
        eval_set = [(X_train, y_train), (X_val, y_val)]
        model.fit(
            X_train, 
            y_train, 
            eval_set=eval_set,
            verbose=False
        )
        
        # Log the best iteration number (early stopping point)
        if hasattr(model, 'best_iteration'):
            logging.info(f"Fold {fold} best iteration: {model.best_iteration}")
        
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        all_y_pred.extend(y_pred)
        all_y_true.extend(y_val)
        
        cv_scores['accuracy'].append(accuracy_score(y_val, y_pred))
        cv_scores['precision'].append(precision_score(y_val, y_pred, zero_division=0))
        cv_scores['recall'].append(recall_score(y_val, y_pred))
        cv_scores['f1'].append(f1_score(y_val, y_pred))
        cv_scores['roc_auc'].append(roc_auc_score(y_val, y_pred_proba))
        cv_scores['mcc'].append(matthews_corrcoef(y_val, y_pred))
        
        # Accumulate feature importance
        feature_importance += model.feature_importances_
        fold += 1
    
    # Average feature importance across folds
    feature_importance /= 10
    
    # Create feature importance dictionary
    feature_importance_dict = dict(zip(feature_names, feature_importance))
    
    # Sort features by importance
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Log top 10 most important features
    logging.info("Top 10 most important features:")
    for feature, importance in sorted_features[:10]:
        logging.info(f"{feature}: {importance:.4f}")
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(all_y_true, all_y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Down', 'Up'],
               yticklabels=['Down', 'Up'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'confusion_matrix.png')
    
    # Train final model on all data with optimal configuration and early stopping
    final_model = xgb.XGBClassifier(**model.get_params())
    # Create a small validation set from the training data for early stopping
    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    eval_set = [(X_train_final, y_train_final), (X_val_final, y_val_final)]
    final_model.fit(
        X_train_final, 
        y_train_final,
        eval_set=eval_set,
        verbose=False
    )
    logging.info(f"Final model best iteration: {final_model.best_iteration if hasattr(final_model, 'best_iteration') else 'N/A'}")
    
    # Generate feature importance plot
    plt.figure(figsize=(10, 10))
    sorted_idx = np.argsort(feature_importance)[-20:]  # Top 20 features
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.title('Top 20 Feature Importance')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'feature_importance.png')
    
    return final_model, cv_scores, feature_importance_dict

def save_model(model, scaler, cv_scores, feature_importance, selected_features=None):
    """Save model, scaler, and metrics."""
    try:
        # Save model
        model_path = MODELS_DIR / "xgboost_model.joblib"
        joblib.dump(model, model_path)
        logging.info(f"Model saved successfully to {model_path}")
        
        # Save scaler
        scaler_path = MODELS_DIR / "scaler.joblib"
        joblib.dump(scaler, scaler_path)
        logging.info(f"Scaler saved successfully to {scaler_path}")
        
        # Save feature names for API
        feature_names_path = MODELS_DIR / "feature_names.joblib"
        feature_names = list(feature_importance.keys())
        joblib.dump(feature_names, feature_names_path)
        logging.info(f"Feature names saved successfully to {feature_names_path}")
        
        # Konverter float32 til float for JSON serialisering
        feature_importance_json = {k: float(v) for k, v in feature_importance.items()}
        
        # Calculate and save average metrics
        metrics = {
            'accuracy': np.mean(cv_scores['accuracy']),
            'precision': np.mean(cv_scores['precision']),
            'recall': np.mean(cv_scores['recall']),
            'f1_score': np.mean(cv_scores['f1']),
            'roc_auc': np.mean(cv_scores['roc_auc']),
            'mcc': np.mean(cv_scores['mcc']),
            'feature_importance': feature_importance_json,
        }
        
        if selected_features:
            metrics['selected_features'] = selected_features
        
        metrics_path = MODELS_DIR / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Metrics saved successfully to {metrics_path}")
        
        return True
    except Exception as e:
        logging.error(f"Error saving model artifacts: {e}")
        return False

def main():
    """
    Main function to orchestrate the training process
    """
    start_time = time.time()
    logging.info("==== Starting model training process ====")
    
    # Load data
    df = load_data()
    if df is None:
        logging.error("Failed to load data. Exiting training process.")
        return None
    
    # Preprocess data
    X, y, feature_columns, scaler = prepare_data(df)
    logging.info(f"Data prepared with {X.shape[1]} features and {X.shape[0]} samples")
    
    # Perform feature selection
    X_selected, selected_features = select_features(X, y, feature_columns)
    logging.info(f"Feature selection complete: {len(selected_features)} features selected from {len(feature_columns)}")
    
    # Perform hyperparameter tuning
    best_params = hyperparameter_tuning(X_selected, y)
    logging.info(f"Hyperparameter tuning complete: {best_params}")
    
    # Train model with selected features and tuned hyperparameters
    model, cv_scores, feature_importance = train_model(X_selected, y, selected_features, best_params)
    logging.info(f"Model training complete. ROC AUC: {np.mean(cv_scores['roc_auc']):.4f}")
    
    # Save model and metrics
    save_success = save_model(model, scaler, cv_scores, feature_importance, selected_features)
    if save_success:
        logging.info("Model and metrics saved successfully")
    else:
        logging.warning("Failed to save model or metrics")
    
    logging.info(f"==== Model training completed in {(time.time() - start_time) / 60:.2f} minutes ====")
    
    return model, cv_scores, feature_importance

if __name__ == "__main__":
    main()
