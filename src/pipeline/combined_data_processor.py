import pandas as pd
import logging
from pathlib import Path
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Determine project root based on script location
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
MACRO_DATA_DIR = PROJECT_ROOT / "data" / "macro"
COMBINED_DATA_DIR = PROJECT_ROOT / "data" / "combined"

# Input files
BITCOIN_FILENAME = "bitcoin_usd_365d_raw.csv"
MACRO_FILENAME = "macro_economic_data.csv"

# Output file
OUTPUT_FILENAME = "bitcoin_macro_combined.csv"

# Ensure output directory exists
COMBINED_DATA_DIR.mkdir(parents=True, exist_ok=True)

def load_bitcoin_data():
    """Load Bitcoin data from raw directory"""
    try:
        bitcoin_path = RAW_DATA_DIR / BITCOIN_FILENAME
        if not bitcoin_path.exists():
            logging.error(f"Bitcoin data file not found: {bitcoin_path}")
            return None
            
        df = pd.read_csv(bitcoin_path)
        # Ensure timestamp is in datetime format
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        logging.info(f"Bitcoin data loaded successfully from {bitcoin_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading Bitcoin data: {e}")
        return None

def load_macro_data():
    """Load macroeconomic data"""
    try:
        macro_path = MACRO_DATA_DIR / MACRO_FILENAME
        if not macro_path.exists():
            logging.error(f"Macro data file not found: {macro_path}")
            return None
            
        df = pd.read_csv(macro_path)
        # Ensure date column is in datetime format
        df.index = pd.to_datetime(df.index)
        logging.info(f"Macroeconomic data loaded successfully from {macro_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading macroeconomic data: {e}")
        return None

def combine_datasets(bitcoin_df, macro_df):
    """Combine Bitcoin and macroeconomic datasets"""
    try:
        if bitcoin_df is None or macro_df is None:
            logging.error("Cannot combine datasets: one or both datasets are missing")
            return None
            
        # Set bitcoin_df index to timestamp for merging
        bitcoin_df.set_index('timestamp', inplace=True)
        
        # Resample macro data to daily frequency if needed
        if macro_df.index.freq != 'D':
            macro_df = macro_df.resample('D').ffill()
        
        # Merge datasets on date
        combined_df = pd.merge(
            bitcoin_df, 
            macro_df,
            left_index=True, 
            right_index=True, 
            how='left'
        )
        
        # Forward fill missing macro values (for weekends/holidays)
        combined_df.ffill(inplace=True)
        
        logging.info(f"Datasets combined successfully. Shape: {combined_df.shape}")
        return combined_df
    except Exception as e:
        logging.error(f"Error combining datasets: {e}")
        return None

def save_combined_data(df):
    """Save combined dataset to file"""
    try:
        output_path = COMBINED_DATA_DIR / OUTPUT_FILENAME
        df.to_csv(output_path)
        logging.info(f"Combined data saved successfully to {output_path}")
        return True
    except Exception as e:
        logging.error(f"Error saving combined data: {e}")
        return False

def main():
    """Main function to run the data combination process"""
    logging.info("--- Starting Data Combination Process ---")
    
    # Load Bitcoin data
    bitcoin_df = load_bitcoin_data()
    if bitcoin_df is None:
        logging.error("Halting process due to Bitcoin data load error")
        sys.exit(1)
    
    # Load macroeconomic data
    macro_df = load_macro_data()
    if macro_df is None:
        logging.error("Halting process due to macroeconomic data load error")
        sys.exit(1)
    
    # Combine datasets
    combined_df = combine_datasets(bitcoin_df, macro_df)
    if combined_df is None:
        logging.error("Halting process due to data combination error")
        sys.exit(1)
    
    # Save combined data
    if save_combined_data(combined_df):
        logging.info("--- Data Combination Completed Successfully ---")
    else:
        logging.error("--- Data Combination Failed (Save Error) ---")
        sys.exit(1)

if __name__ == "__main__":
    main() 