import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Determine project root based on script location
# Assumes the script is in src/pipeline
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERMEDIATE_COMBINED_DIR = PROJECT_ROOT / "data" / "intermediate" / "combined"  # Updated path
INTERMEDIATE_PREPROCESSED_DIR = PROJECT_ROOT / "data" / "intermediate" / "preprocessed"  # Updated path
MODELS_DIR = PROJECT_ROOT / "models"

# Input file from combined data step
INPUT_FILENAME = "bitcoin_macro_combined.csv"
# Output files
OUTPUT_FILENAME = "bitcoin_macro_preprocessed.csv"  # Updated filename
SCALER_FILENAME = "minmax_scaler.joblib"

# Ensure output directories exist
INTERMEDIATE_PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE_PATH = INTERMEDIATE_COMBINED_DIR / INPUT_FILENAME
OUTPUT_FILE_PATH = INTERMEDIATE_PREPROCESSED_DIR / OUTPUT_FILENAME
SCALER_FILE_PATH = MODELS_DIR / SCALER_FILENAME

def load_data(filepath: Path) -> pd.DataFrame | None:
    """Loads data from a CSV file."""
    if not filepath.exists():
        logging.error(f"Input file not found: {filepath}")
        return None
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Data loaded successfully from {filepath}")
        logging.info(f"Data shape: {df.shape}, columns: {df.columns.tolist()}")
        
        # Check for empty values
        if df.empty:
            logging.error("Data file is empty")
            return None
            
        # Check for missing values
        missing_count = df.isna().sum().sum()
        if missing_count > 0:
            logging.warning(f"Found {missing_count} missing values in input data")
            
        # Convert timestamp to datetime if it exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    except Exception as e:
        logging.error(f"Error loading data from {filepath}: {e}")
        return None

def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame | None, MinMaxScaler | None]:
    """Handles missing values and scales numerical features."""
    try:
        # Ensure we have a copy of the dataframe
        df = df.copy()
        logging.info(f"Starting preprocessing with data shape: {df.shape}")
        
        # Determine if timestamp should be the index
        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)
            logging.info("Set timestamp as index")
            
        # Handle missing values - Forward fill is common for time series
        original_rows = len(df)
        df.ffill(inplace=True) # Forward fill
        df.bfill(inplace=True) # Back fill any remaining NaNs at the beginning
        
        # If there are still NaN values after fill, replace them with 0 for numeric columns
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            df[col] = df[col].fillna(0)
            
        # Check again for NaN values
        nan_rows = df.isna().any(axis=1).sum()
        if nan_rows > 0:
            logging.warning(f"There are still {nan_rows} rows with NaN values. Removing these rows.")
            df.dropna(inplace=True)
            if df.empty:
                logging.error("DataFrame is empty after handling NaNs.")
                return None, None

        logging.info(f"Missing values handled. Original rows: {original_rows}, Final rows: {len(df)}")

        # Scaling numerical features
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        logging.info(f"Numerical columns to scale: {numerical_cols.tolist()}")
        
        if numerical_cols.empty:
            logging.warning("No numerical columns found to scale.")
            return df, None # Return dataframe without scaling if no numerical cols

        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(df[numerical_cols])
        df[numerical_cols] = df_scaled # Update original df with scaled values

        logging.info(f"Numerical features scaled using MinMaxScaler")
        return df, scaler
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None, None

def save_processed_data(df: pd.DataFrame, filepath: Path):
    """Saves the processed DataFrame to a CSV file."""
    try:
        # Save with index (timestamp)
        df.to_csv(filepath)
        logging.info(f"Processed data saved successfully to {filepath}")
    except Exception as e:
        logging.error(f"Error saving processed data to {filepath}: {e}")
        sys.exit(1)

def save_scaler(scaler: MinMaxScaler, filepath: Path):
    """Saves the fitted scaler object."""
    try:
        joblib.dump(scaler, filepath)
        logging.info(f"Scaler saved successfully to {filepath}")
    except Exception as e:
        logging.error(f"Error saving scaler to {filepath}: {e}")
        sys.exit(1)

def main():
    """Main function to run the data preprocessing process."""
    logging.info("--- Starting Data Preprocessing ---")

    # Load data
    raw_df = load_data(INPUT_FILE_PATH)
    if raw_df is None:
        logging.error("Halting preprocessing due to load error.")
        sys.exit(1)

    # Preprocess data
    processed_df, scaler = preprocess_data(raw_df) # Use copy to avoid modifying original df in place if needed elsewhere
    if processed_df is None:
        logging.error("Halting preprocessing due to processing error.")
        sys.exit(1)

    # Save processed data
    save_processed_data(processed_df, OUTPUT_FILE_PATH)

    # Save scaler only if it was created
    if scaler:
        save_scaler(scaler, SCALER_FILE_PATH)
    else:
        logging.warning("Scaler was not created during preprocessing, skipping save.")

    logging.info("--- Data Preprocessing Completed Successfully ---")

if __name__ == "__main__":
    main()
