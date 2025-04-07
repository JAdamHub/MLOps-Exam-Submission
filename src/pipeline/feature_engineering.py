import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Determine project root based on script location
# Assumes the script is in src/pipeline
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_DATA_DIR = PROJECT_ROOT / "data" / "features"

# Input file from preprocessing step
INPUT_FILENAME = "bitcoin_usd_365d_processed.csv"
# Output file
OUTPUT_FILENAME = "bitcoin_usd_365d_features.csv"

# Feature Engineering Parameters
PRICE_COLUMN = 'price' # Assuming 'price' is the column name after preprocessing
LAG_PERIODS = [1, 3, 7] # Lag periods in days
SMA_WINDOWS = [7, 30] # Simple Moving Average windows in days
VOLATILITY_WINDOW = 14 # Window for rolling standard deviation (volatility)
TARGET_SHIFT = -1 # Predict next day's price movement

# Ensure output directory exists
FEATURES_DATA_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE_PATH = PROCESSED_DATA_DIR / INPUT_FILENAME
OUTPUT_FILE_PATH = FEATURES_DATA_DIR / OUTPUT_FILENAME

def load_data(filepath: Path) -> pd.DataFrame | None:
    """Loads data from a CSV file."""
    if not filepath.exists():
        logging.error(f"Input file not found: {filepath}")
        return None
    try:
        df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
        logging.info(f"Processed data loaded successfully from {filepath}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {filepath}: {e}")
        return None

def create_features(df: pd.DataFrame) -> pd.DataFrame | None:
    """Generates features for the model."""
    if PRICE_COLUMN not in df.columns:
        logging.error(f"Required column '{PRICE_COLUMN}' not found in the DataFrame.")
        return None

    try:
        logging.info("Starting feature engineering...")
        features_df = df.copy()

        # 1. Lagged Features
        for lag in LAG_PERIODS:
            features_df[f'{PRICE_COLUMN}_lag_{lag}'] = df[PRICE_COLUMN].shift(lag)
            logging.debug(f"Created lag feature: {PRICE_COLUMN}_lag_{lag}")

        # 2. Moving Averages
        for window in SMA_WINDOWS:
            features_df[f'{PRICE_COLUMN}_sma_{window}'] = df[PRICE_COLUMN].rolling(window=window, min_periods=1).mean()
            logging.debug(f"Created SMA feature: {PRICE_COLUMN}_sma_{window}")

        # 3. Volatility
        features_df[f'{PRICE_COLUMN}_volatility_{VOLATILITY_WINDOW}'] = df[PRICE_COLUMN].rolling(window=VOLATILITY_WINDOW, min_periods=1).std()
        logging.debug(f"Created volatility feature: {PRICE_COLUMN}_volatility_{VOLATILITY_WINDOW}")

        # 4. Time-based Features
        features_df['day_of_week'] = df.index.dayofweek
        features_df['month'] = df.index.month
        features_df['year'] = df.index.year # Keep year if useful for longer trends
        logging.debug("Created time-based features: day_of_week, month, year")

        # 5. Target Variable: Price up (1) or down/same (0) tomorrow?
        features_df['target_price_next_day'] = df[PRICE_COLUMN].shift(TARGET_SHIFT)
        features_df['target_price_up'] = (features_df['target_price_next_day'] > df[PRICE_COLUMN]).astype(int)
        features_df.drop(columns=['target_price_next_day'], inplace=True) # Drop the intermediate column
        logging.debug("Created target variable: target_price_up")

        # Drop rows with NaNs introduced by lags/rolling windows/target shift
        initial_rows = len(features_df)
        features_df.dropna(inplace=True)
        final_rows = len(features_df)
        logging.info(f"Dropped {initial_rows - final_rows} rows due to NaN values from feature creation.")

        if features_df.empty:
            logging.error("DataFrame is empty after feature engineering and NaN removal.")
            return None

        logging.info("Feature engineering completed.")
        return features_df

    except Exception as e:
        logging.error(f"Error during feature engineering: {e}", exc_info=True)
        return None

def save_features(df: pd.DataFrame, filepath: Path):
    """Saves the features DataFrame to a CSV file."""
    try:
        df.to_csv(filepath)
        logging.info(f"Features data saved successfully to {filepath}")
    except Exception as e:
        logging.error(f"Error saving features data to {filepath}: {e}")
        sys.exit(1)

def main():
    """Main function to run the feature engineering process."""
    logging.info("--- Starting Feature Engineering ---")

    # Load preprocessed data
    processed_df = load_data(INPUT_FILE_PATH)
    if processed_df is None:
        logging.error("Halting feature engineering due to load error.")
        sys.exit(1)

    # Create features
    features_df = create_features(processed_df)
    if features_df is None:
        logging.error("Halting feature engineering due to processing error.")
        sys.exit(1)

    # Save features data
    save_features(features_df, OUTPUT_FILE_PATH)

    logging.info("--- Feature Engineering Completed Successfully ---")

if __name__ == "__main__":
    main()
