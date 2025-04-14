import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import logging
import sys
from pathlib import Path

# configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# determine project root based on script location
# assumes the script is in src/pipeline
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERMEDIATE_COMBINED_DIR = PROJECT_ROOT / "data" / "intermediate" / "combined"
INTERMEDIATE_PREPROCESSED_DIR = PROJECT_ROOT / "data" / "intermediate" / "preprocessed" 
MODELS_DIR = PROJECT_ROOT / "models"

# input file from combined data step
INPUT_FILENAME = "vestas_macro_combined_trading_days.csv"
# output files
OUTPUT_FILENAME = "vestas_macro_preprocessed_trading_days.csv"
SCALER_FILENAME = "minmax_scaler.joblib"

# ensure output directories exist
INTERMEDIATE_PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE_PATH = INTERMEDIATE_COMBINED_DIR / INPUT_FILENAME
OUTPUT_FILE_PATH = INTERMEDIATE_PREPROCESSED_DIR / OUTPUT_FILENAME
SCALER_FILE_PATH = MODELS_DIR / SCALER_FILENAME

def load_data(filepath: Path) -> pd.DataFrame | None:
    """loads data from a csv file."""
    if not filepath.exists():
        logging.error(f"input file not found: {filepath}")
        return None
    try:
        # load data with first column as index (date)
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        logging.info(f"data loaded successfully from {filepath}")
        logging.info(f"data shape: {df.shape}, columns: {len(df.columns)}")
        logging.info(f"sample columns: {df.columns[:5].tolist()}...")
        
        # check for empty values
        if df.empty:
            logging.error("data file is empty")
            return None
            
        # check for missing values
        missing_count = df.isna().sum().sum()
        if missing_count > 0:
            logging.warning(f"found {missing_count} missing values in input data")
            
        return df
    except Exception as e:
        logging.error(f"error loading data from {filepath}: {e}")
        return None

def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame | None, MinMaxScaler | None]:
    """
    preprocess data: handle missing values, create features, and scale
    
    args:
        df: raw vestas data with macroeconomic indicators
        
    returns:
        preprocessed data and scaler
    """
    try:
        logging.info(f"starting preprocessing with data shape: {df.shape}")
        
        # ensure we have a copy to avoid warnings
        df = df.copy()
        
        # add percentage change column if not already present
        if 'pct_change' not in df.columns and 'close' in df.columns:
            df['pct_change'] = df['close'].pct_change() * 100
            # fix futurewarning by using assignment instead of inplace
            df['pct_change'] = df['pct_change'].fillna(0)
            logging.info("added percent change column for close price")
            
        # add volatility (standard deviation of returns over rolling window)
        if 'close' in df.columns and 'volatility_14d' not in df.columns:
            df['volatility_14d'] = df['close'].pct_change().rolling(window=14).std() * 100
            # fix futurewarning by using assignment instead of inplace
            df['volatility_14d'] = df['volatility_14d'].fillna(df['volatility_14d'].mean())
            logging.info("added 14-day volatility column")
        
        # handle missing values - first check if there are any
        missing_before = df.isna().sum().sum()
        logging.info(f"original missing values: {missing_before}")
        
        if missing_before > 0:
            logging.info("handling missing values using forward fill followed by backward fill")
            # fix futurewarning - avoid inplace=true
            df = df.ffill().bfill()
            
        missing_after = df.isna().sum().sum()
        logging.info(f"missing values handled. original missing: {missing_before}, remaining: {missing_after}")
        logging.info(f"data shape after handling missing values: {df.shape}")

        # apply scaling to numerical columns only
        # we should not scale date/categorical columns
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        
        if not numerical_cols.empty:
            # initialize scaler and fit it to the data
            scaler = MinMaxScaler()
            # scaled_data = scaler.fit_transform(df[numerical_cols]) # commented out

            # replace columns instead of using inplace=true
            # for i, col in enumerate(numerical_cols): # commented out
            #     df[col] = scaled_data[:, i] # commented out

            # logging.info(f"scaled {len(numerical_cols)} numerical columns: {list(numerical_cols)}") # commented out
            logging.info(f"skipping minmaxscaler application, returning original numerical data.") # added log
            return df, scaler # still return scaler, but df is unscaled
        else:
            logging.warning("no numerical columns found to scale.")
            scaler = None
            return df, scaler
    except Exception as e:
        logging.error(f"error during preprocessing: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None, None

def save_processed_data(df: pd.DataFrame, filepath: Path):
    """saves the processed dataframe to a csv file."""
    try:
        # save with date index
        df.to_csv(filepath)
        logging.info(f"processed data saved successfully to {filepath}")
        logging.info(f"final data shape: {df.shape}")
    except Exception as e:
        logging.error(f"error saving processed data to {filepath}: {e}")
        sys.exit(1)

def save_scaler(scaler: MinMaxScaler, filepath: Path):
    """saves the fitted scaler object."""
    try:
        joblib.dump(scaler, filepath)
        logging.info(f"scaler saved successfully to {filepath}")
    except Exception as e:
        logging.error(f"error saving scaler to {filepath}: {e}")
        sys.exit(1)

def main():
    """main function to run the data preprocessing process."""
    logging.info("--- starting vestas data preprocessing ---")

    # load data
    raw_df = load_data(INPUT_FILE_PATH)
    if raw_df is None:
        logging.error("halting preprocessing due to load error.")
        sys.exit(1)

    # preprocess data
    processed_df, scaler = preprocess_data(raw_df)
    if processed_df is None:
        logging.error("halting preprocessing due to processing error.")
        sys.exit(1)

    # save processed data
    save_processed_data(processed_df, OUTPUT_FILE_PATH)

    # save scaler only if it was created
    if scaler:
        save_scaler(scaler, SCALER_FILE_PATH)
    else:
        logging.warning("scaler was not created during preprocessing, skipping save.")

    logging.info("--- vestas data preprocessing completed successfully ---")
    return True # indicate success

if __name__ == "__main__":
    if not main():
        sys.exit(1)
