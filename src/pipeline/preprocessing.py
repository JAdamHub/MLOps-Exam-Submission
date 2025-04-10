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
INPUT_FILENAME = "vestas_macro_combined_trading_days.csv"
# Output files
OUTPUT_FILENAME = "vestas_macro_preprocessed_trading_days.csv"  # Opdateret filnavn til Vestas
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
        # Indlæs data med første kolonne som indeks (dato)
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        logging.info(f"Data loaded successfully from {filepath}")
        logging.info(f"Data shape: {df.shape}, columns: {len(df.columns)}")
        logging.info(f"Sample columns: {df.columns[:5].tolist()}...")
        
        # Check for empty values
        if df.empty:
            logging.error("Data file is empty")
            return None
            
        # Check for missing values
        missing_count = df.isna().sum().sum()
        if missing_count > 0:
            logging.warning(f"Found {missing_count} missing values in input data")
            
        return df
    except Exception as e:
        logging.error(f"Error loading data from {filepath}: {e}")
        return None

def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame | None, MinMaxScaler | None]:
    """
    Preprocess data: handle missing values, create features, and scale
    
    Args:
        df: Raw Vestas data with macroeconomic indicators
        
    Returns:
        Preprocessed data and scaler
    """
    try:
        logging.info(f"Starting preprocessing with data shape: {df.shape}")
        
        # Ensure we have a copy to avoid warnings
        df = df.copy()
        
        # Add percentage change column if not already present
        if 'pct_change' not in df.columns and 'close' in df.columns:
            df['pct_change'] = df['close'].pct_change() * 100
            # Ret FutureWarning ved at bruge assignment i stedet for inplace
            df['pct_change'] = df['pct_change'].fillna(0)
            logging.info("Added percent change column for close price")
            
        # Add volatility (standard deviation of returns over rolling window)
        if 'close' in df.columns and 'volatility_14d' not in df.columns:
            df['volatility_14d'] = df['close'].pct_change().rolling(window=14).std() * 100
            # Ret FutureWarning ved at bruge assignment i stedet for inplace
            df['volatility_14d'] = df['volatility_14d'].fillna(df['volatility_14d'].mean())
            logging.info("Added 14-day volatility column")
        
        # Handle missing values - first check if there are any
        missing_before = df.isna().sum().sum()
        logging.info(f"Original missing values: {missing_before}")
        
        if missing_before > 0:
            logging.info("Handling missing values using forward fill followed by backward fill")
            # Ret FutureWarning - undgå inplace=True
            df = df.ffill().bfill()
            
        missing_after = df.isna().sum().sum()
        logging.info(f"Missing values handled. Original missing: {missing_before}, Remaining: {missing_after}")
        logging.info(f"Data shape after handling missing values: {df.shape}")

        # Apply scaling to numerical columns only
        # We should not scale date/categorical columns
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        
        if not numerical_cols.empty:
            # Initializer scaler and fit it to the data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df[numerical_cols])
            
            # Erstat kolonner i stedet for at bruge inplace=True
            for i, col in enumerate(numerical_cols):
                df[col] = scaled_data[:, i]
                
            logging.info(f"Scaled {len(numerical_cols)} numerical columns: {list(numerical_cols)}")
            return df, scaler
        else:
            logging.warning("No numerical columns found to scale.")
            scaler = None
            return df, scaler
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None, None

def save_processed_data(df: pd.DataFrame, filepath: Path):
    """Saves the processed DataFrame to a CSV file."""
    try:
        # Save with date index
        df.to_csv(filepath)
        logging.info(f"Processed data saved successfully to {filepath}")
        logging.info(f"Final data shape: {df.shape}")
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
    logging.info("--- Starting Vestas Data Preprocessing ---")

    # Load data
    raw_df = load_data(INPUT_FILE_PATH)
    if raw_df is None:
        logging.error("Halting preprocessing due to load error.")
        sys.exit(1)

    # Preprocess data
    processed_df, scaler = preprocess_data(raw_df)
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

    logging.info("--- Vestas Data Preprocessing Completed Successfully ---")

if __name__ == "__main__":
    main()
