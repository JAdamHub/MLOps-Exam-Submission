import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DATA_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"

def load_data():
    """Load feature data for training."""
    try:
        input_file = FEATURES_DATA_DIR / "bitcoin_usd_365d_features.csv"
        df = pd.read_csv(input_file)
        logging.info(f"Feature data loaded successfully from {input_file}")
        return df
    except Exception as e:
        logging.error(f"Error loading feature data: {e}")
        return None

def prepare_data(df):
    """Prepare data for training."""
    # Separate features and target
    feature_columns = [col for col in df.columns if col not in ['timestamp', 'target_price_up']]
    X = df[feature_columns]  # Keep as DataFrame to preserve column names
    y = df['target_price_up'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create a new DataFrame with scaled values and original column names
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Add feature names to scaler
    scaler.feature_names_in_ = X.columns
    
    return X_scaled_df.values, y, feature_columns, scaler

def train_model(X, y, feature_columns):
    """
    Train XGBoost model with time series cross-validation
    """
    tscv = TimeSeriesSplit(n_splits=5)
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        scale_pos_weight=1,
        random_state=42
    )
    
    cv_scores = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': []
    }
    
    feature_importance = np.zeros(len(feature_columns))
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        cv_scores['accuracy'].append(accuracy_score(y_val, y_pred))
        cv_scores['precision'].append(precision_score(y_val, y_pred))
        cv_scores['recall'].append(recall_score(y_val, y_pred))
        cv_scores['f1'].append(f1_score(y_val, y_pred))
        cv_scores['roc_auc'].append(roc_auc_score(y_val, y_pred_proba))
        
        # Accumulate feature importance
        feature_importance += model.feature_importances_
    
    # Average feature importance across folds
    feature_importance /= 5
    
    # Create feature importance dictionary
    feature_importance_dict = dict(zip(feature_columns, feature_importance))
    
    # Sort features by importance
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Log top 10 most important features
    logging.info("Top 10 most important features:")
    for feature, importance in sorted_features[:10]:
        logging.info(f"{feature}: {importance:.4f}")
    
    # Train final model on all data
    model.fit(X, y)
    
    return model, cv_scores, feature_importance_dict

def save_model(model, scaler, cv_scores, feature_importance):
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
        
        # Save feature names
        feature_names_path = MODELS_DIR / "feature_names.joblib"
        joblib.dump(list(feature_importance.keys()), feature_names_path)
        logging.info(f"Feature names saved successfully to {feature_names_path}")
        
        # Calculate and save average metrics
        metrics = {
            'accuracy': np.mean(cv_scores['accuracy']),
            'precision': np.mean(cv_scores['precision']),
            'recall': np.mean(cv_scores['recall']),
            'f1_score': np.mean(cv_scores['f1']),
            'roc_auc': np.mean(cv_scores['roc_auc']),
            'feature_importance': feature_importance
        }
        
        metrics_path = MODELS_DIR / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Metrics saved successfully to {metrics_path}")
        
        return True
    except Exception as e:
        logging.error(f"Error saving model artifacts: {e}")
        return False

def main():
    """Main function to run the model training process."""
    logging.info("--- Starting Model Training ---")
    
    try:
        # Load data
        df = load_data()
        if df is None:
            return
        
        # Prepare data
        X, y, feature_columns, scaler = prepare_data(df)
        
        # Train model
        model, cv_scores, feature_importance = train_model(X, y, feature_columns)
        
        # Save model and metrics
        if save_model(model, scaler, cv_scores, feature_importance):
            logging.info("--- Model Training Completed Successfully ---")
        else:
            logging.error("--- Model Training Failed (Save Error) ---")
            
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise

if __name__ == "__main__":
    main()
