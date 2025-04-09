import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint, uniform

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = MODELS_DIR / "figures"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load feature data for training."""
    try:
        input_file = PROCESSED_FEATURES_DIR / "bitcoin_features.csv"
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
    X = df[feature_columns].values  # Convert to numpy array after selecting features
    y = df['target_price_up'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, feature_columns, scaler

def select_features(X, y, feature_columns, importance_threshold=0.01):
    """Select important features based on a preliminary model."""
    logging.info("Performing feature selection...")
    
    # Train a preliminary model to get feature importances
    model = xgb.XGBClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create a dataframe of features and their importances
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Plot feature importances
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20))
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'feature_importances.png')
    
    # Select features using SelectFromModel
    selection = SelectFromModel(model, threshold=importance_threshold, prefit=True)
    X_selected = selection.transform(X)
    
    # Get selected feature names
    selected_indices = selection.get_support()
    selected_features = [feature for i, feature in enumerate(feature_columns) if selected_indices[i]]
    
    logging.info(f"Selected {len(selected_features)} out of {len(feature_columns)} features")
    logging.info(f"Top 10 selected features: {selected_features[:10]}")
    
    return X_selected, selected_features

def hyperparameter_tuning(X, y):
    """Perform hyperparameter tuning using RandomizedSearchCV."""
    logging.info("Starting hyperparameter tuning...")
    
    # Define time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Define parameter distributions for random search
    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'gamma': uniform(0, 0.5),
        'min_child_weight': randint(1, 6),
        'scale_pos_weight': uniform(0.8, 0.4)
    }
    
    # Initialize the model
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    
    # RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter=20,  # Number of parameter settings sampled
        scoring='roc_auc',
        cv=tscv,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    # Fit on data
    random_search.fit(X, y)
    
    # Get best parameters
    best_params = random_search.best_params_
    logging.info(f"Best parameters found: {best_params}")
    
    # Get best score
    best_score = random_search.best_score_
    logging.info(f"Best ROC AUC score: {best_score:.4f}")
    
    # Return the best parameters
    return best_params

def train_model(X, y, feature_names, best_params=None):
    """
    Train XGBoost model with time series cross-validation
    """
    tscv = TimeSeriesSplit(n_splits=5)
    
    if best_params:
        model = xgb.XGBClassifier(**best_params, random_state=42)
        logging.info("Training model with tuned hyperparameters")
    else:
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
        logging.info("Training model with default hyperparameters")
    
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
        
        logging.info(f"Training fold {fold}/5...")
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
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
    feature_importance /= 5
    
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
    
    # Train final model on all data
    final_model = xgb.XGBClassifier(**model.get_params())
    final_model.fit(X, y)
    
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
        
        # Calculate and save average metrics
        metrics = {
            'accuracy': np.mean(cv_scores['accuracy']),
            'precision': np.mean(cv_scores['precision']),
            'recall': np.mean(cv_scores['recall']),
            'f1_score': np.mean(cv_scores['f1']),
            'roc_auc': np.mean(cv_scores['roc_auc']),
            'mcc': np.mean(cv_scores['mcc']),
            'feature_importance': feature_importance,
        }
        
        if selected_features:
            metrics['selected_features'] = selected_features
        
        metrics_path = MODELS_DIR / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Metrics saved successfully to {metrics_path}")
        
        # Plot ROC curve (requires storing predictions which we aren't doing here)
        # This would be a good addition in future iterations
        
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
        
        # Perform feature selection
        X_selected, selected_features = select_features(X, y, feature_columns)
        
        # Perform hyperparameter tuning
        best_params = hyperparameter_tuning(X_selected, y)
        
        # Train model with selected features and tuned hyperparameters
        model, cv_scores, feature_importance = train_model(X_selected, y, selected_features, best_params)
        
        # Save model and metrics
        if save_model(model, scaler, cv_scores, feature_importance, selected_features):
            logging.info("--- Model Training Completed Successfully ---")
        else:
            logging.error("--- Model Training Failed (Save Error) ---")
            
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise

if __name__ == "__main__":
    main()
