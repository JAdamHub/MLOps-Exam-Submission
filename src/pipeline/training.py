import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier # Alternative model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import joblib
import logging
import sys
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Determine project root based on script location
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DATA_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"

# Input file from feature engineering step
INPUT_FILENAME = "bitcoin_usd_365d_features.csv"
# Output files
MODEL_FILENAME = "logistic_regression_model.joblib"
METRICS_FILENAME = "training_metrics.json"

TARGET_COLUMN = 'target_price_up'
TEST_SIZE = 0.2 # Use last 20% of data for testing in chronological split
RANDOM_STATE = 42 # For reproducibility if using non-chronological splits (but we prioritize chronological)

# Ensure output directory exists
MODELS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE_PATH = FEATURES_DATA_DIR / INPUT_FILENAME
MODEL_FILE_PATH = MODELS_DIR / MODEL_FILENAME
METRICS_FILE_PATH = MODELS_DIR / METRICS_FILENAME # Store metrics alongside the model

def load_data(filepath: Path) -> pd.DataFrame | None:
    """Loads feature data from a CSV file."""
    if not filepath.exists():
        logging.error(f"Input file not found: {filepath}")
        return None
    try:
        df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
        logging.info(f"Feature data loaded successfully from {filepath}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {filepath}: {e}")
        return None

def split_data_chronological(df: pd.DataFrame, target_col: str, test_size: float) -> tuple | None:
    """Splits data chronologically into training and testing sets."""
    if target_col not in df.columns:
        logging.error(f"Target column '{target_col}' not found.")
        return None

    try:
        split_index = int(len(df) * (1 - test_size))
        if split_index <= 0 or split_index >= len(df):
             logging.error(f"Invalid split index calculated: {split_index}. Check data size and test_size.")
             return None

        df_train = df.iloc[:split_index]
        df_test = df.iloc[split_index:]

        X_train = df_train.drop(columns=[target_col])
        y_train = df_train[target_col]
        X_test = df_test.drop(columns=[target_col])
        y_test = df_test[target_col]

        logging.info(f"Data split chronologically: Train set size={len(X_train)}, Test set size={len(X_test)}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error splitting data: {e}")
        return None

def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    """Trains a Logistic Regression model."""
    try:
        # --- Placeholder for Hyperparameter Tuning & Cross-Validation ---
        # For production, use GridSearchCV or RandomizedSearchCV with TimeSeriesSplit
        # Example (commented out):
        # tscv = TimeSeriesSplit(n_splits=5)
        # param_grid = {'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear']}
        # model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
        # grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='roc_auc')
        # grid_search.fit(X_train, y_train)
        # best_model = grid_search.best_estimator_
        # logging.info(f"Best parameters found: {grid_search.best_params_}")

        # Simple model training for now
        model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000) # Increased max_iter
        model.fit(X_train, y_train)
        logging.info("Model training completed.")
        return model
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        return None

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict | None:
    """Evaluates the model and returns classification metrics."""
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] # Probability for the positive class

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': auc
        }
        logging.info(f"Model Evaluation Metrics:\n{json.dumps(metrics, indent=2)}")

        # Optional: Print full classification report
        report = classification_report(y_test, y_pred, zero_division=0)
        logging.info(f"Classification Report:\n{report}")

        return metrics
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        return None

def save_model(model, filepath: Path):
    """Saves the trained model object."""
    try:
        joblib.dump(model, filepath)
        logging.info(f"Model saved successfully to {filepath}")
    except Exception as e:
        logging.error(f"Error saving model to {filepath}: {e}")
        sys.exit(1)

def save_metrics(metrics: dict, filepath: Path):
    """Saves the evaluation metrics to a JSON file."""
    try:
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Metrics saved successfully to {filepath}")
    except Exception as e:
        logging.error(f"Error saving metrics to {filepath}: {e}")
        # Don't necessarily exit if metrics saving fails, model might still be useful

def main():
    """Main function to run the model training process."""
    logging.info("--- Starting Model Training ---")

    # Load data
    features_df = load_data(INPUT_FILE_PATH)
    if features_df is None:
        logging.error("Halting training due to load error.")
        sys.exit(1)

    # Split data
    split_result = split_data_chronological(features_df, TARGET_COLUMN, TEST_SIZE)
    if split_result is None:
         logging.error("Halting training due to data split error.")
         sys.exit(1)
    X_train, X_test, y_train, y_test = split_result

    # Train model
    model = train_model(X_train, y_train)
    if model is None:
        logging.error("Halting training due to model training error.")
        sys.exit(1)

    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    if metrics is None:
        logging.warning("Model evaluation failed, but proceeding to save model.")
        # Decide if you want to exit here based on requirements

    # Save model
    save_model(model, MODEL_FILE_PATH)

    # Save metrics (if evaluation was successful)
    if metrics:
        save_metrics(metrics, METRICS_FILE_PATH)

    logging.info("--- Model Training Completed Successfully (or with evaluation warnings) ---")

if __name__ == "__main__":
    main()
