import pandas as pd
import logging
import sys
from pathlib import Path
import sqlite3

# configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# determine project root based on script location
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
INTERMEDIATE_PREPROCESSED_DIR = PROJECT_ROOT / "data" / "intermediate" / "preprocessed"

# Database file from the collection step
DB_FILE = RAW_DATA_DIR / "stocks" / "market_data.db"
TABLE_NAME = "market_data"

# ensure output directories exist
# INTERMEDIATE_PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def load_data(db_path: Path, table: str) -> pd.DataFrame | None:
    """Loads combined data from the SQLite database."""
    if not db_path.exists():
        logging.error(f"Database file not found: {db_path}")
        return None
    try:
        conn = sqlite3.connect(db_path)
        # Read the entire table
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        conn.close()

        # Ensure 'date' column exists and parse it
        if 'date' not in df.columns:
             logging.error(f"'date' column not found in table {table}")
             return None

        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True) # Ensure data is sorted by date

        logging.info(f"Data loaded successfully from database {db_path}, table {table}")
        logging.info(f"Data shape: {df.shape}, columns: {len(df.columns)}")
        logging.info(f"Sample columns: {df.columns[:5].tolist()}...")

        # check for empty values
        if df.empty:
            logging.error("Loaded data frame is empty")
            return None

        # check for missing values - log them, but don't handle here yet
        missing_count = df.isna().sum().sum()
        if missing_count > 0:
            logging.warning(f"Found {missing_count} missing values in loaded data (will be handled later)")

        # Rename stock columns back to original simple names for consistency downstream
        rename_map = {
            'stock_open': 'open',
            'stock_high': 'high',
            'stock_low': 'low',
            'stock_close': 'close',
            'stock_volume': 'volume'
        }
        df.rename(columns=rename_map, inplace=True, errors='ignore')
        logging.info(f"Renamed stock columns. Current columns: {df.columns.tolist()[:10]}...")

        return df
    except Exception as e:
        logging.error(f"Error loading data from database {db_path}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Basic preprocessing: ensures data consistency.
    More advanced preprocessing like scaling and NaN filling happens later.

    args:
        df: combined vestas data with macroeconomic indicators read from DB
    returns:
        preprocessed dataframe (or None if error)
    """
    try:
        logging.info(f"Starting basic preprocessing with data shape: {df.shape}")

        # ensure we have a copy to avoid warnings
        df_processed = df.copy()

        # --- Add any minimal preprocessing steps here if needed ---
        # Example: Ensure correct data types (though read_sql should handle much of this)
        for col in ['open', 'high', 'low', 'close', 'volume']:
             if col in df_processed.columns:
                 df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

        logging.info("Basic preprocessing complete. NaN handling and scaling will occur later.")

        return df_processed

    except Exception as e:
        logging.error(f"error during basic preprocessing: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None

# def save_processed_data(df: pd.DataFrame, filepath: Path):
#     """saves the processed dataframe to a csv file."""
#     if df is None:
#         logging.error("Cannot save processed data: DataFrame is None.")
#         return False # Indicate failure
#     try:
#         # save with date index
#         df.to_csv(filepath)
#         logging.info(f"Processed data saved successfully to {filepath}")
#         logging.info(f"Final data shape for next step: {df.shape}")
#         return True # Indicate success
#     except Exception as e:
#         logging.error(f"error saving processed data to {filepath}: {e}")
#         return False # Indicate failure

def main():
    """main function to run the basic data preprocessing step."""
    logging.info("--- starting basic data preprocessing step --- ")

    # load data from database
    raw_df = load_data(DB_FILE, TABLE_NAME)
    if raw_df is None:
        logging.error("halting preprocessing due to data load error from database.")
        return None # Return None on error

    # basic preprocess data (minimal changes now)
    processed_df = preprocess_data(raw_df)
    if processed_df is None:
        logging.error("halting preprocessing due to basic processing error.")
        return None # Return None on error

    # save processed data to intermediate file (still needed for feature engineering step)
    # if save_processed_data(processed_df, OUTPUT_FILE_PATH):
    #     logging.info("--- basic data preprocessing step completed successfully --- ")
    #     return True # indicate success
    # else:
    #     logging.error("--- basic data preprocessing step failed during save ---\")
    #     return False # Indicate failure
    logging.info("--- basic data preprocessing step completed successfully --- ")
    return processed_df # Return the DataFrame

if __name__ == "__main__":
    result = main()
    if result is None: # Check if main returned None
        sys.exit(1)
