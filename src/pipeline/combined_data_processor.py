import pandas as pd
import logging
from pathlib import Path
import sys
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Determine project root based on script location
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CRYPTO_DIR = PROJECT_ROOT / "data" / "raw" / "crypto"
RAW_MACRO_DIR = PROJECT_ROOT / "data" / "raw" / "macro"
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate" / "combined"

# Input files - Opdateret til at bruge handelsdage filer
BITCOIN_FILENAME = "bitcoin_usd_trading_days.csv"  # Opdateret til handelsdage fil
MACRO_FILENAME = "macro_economic_trading_days.csv"  # Opdateret til handelsdage fil

# Output file
OUTPUT_FILENAME = "bitcoin_macro_combined_trading_days.csv"  # Opdateret navn

# Ensure output directory exists
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

def load_bitcoin_data():
    """Load Bitcoin data from raw directory"""
    try:
        bitcoin_path = RAW_CRYPTO_DIR / BITCOIN_FILENAME
        if not bitcoin_path.exists():
            logging.error(f"Bitcoin data file not found: {bitcoin_path}")
            return None
            
        df = pd.read_csv(bitcoin_path)
        # Ensure timestamp is in datetime format
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        logging.info(f"Bitcoin trading days data loaded successfully from {bitcoin_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading Bitcoin data: {e}")
        return None

def load_macro_data():
    """Load macroeconomic data"""
    try:
        macro_path = RAW_MACRO_DIR / MACRO_FILENAME
        if not macro_path.exists():
            logging.error(f"Macro data file not found: {macro_path}")
            return None
            
        df = pd.read_csv(macro_path)
        # Handle the index column, which likely contains the date
        if 'Unnamed: 0' in df.columns:
            df['date'] = pd.to_datetime(df['Unnamed: 0'])
            df.drop(columns=['Unnamed: 0'], inplace=True)
        logging.info(f"Macroeconomic trading days data loaded successfully from {macro_path}")
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
            
        # Make a copy to avoid warnings
        bitcoin_df = bitcoin_df.copy()
        macro_df = macro_df.copy()
        
        # Ensure there is a timestamp/date column in both datasets
        if 'date' not in macro_df.columns and 'timestamp' not in macro_df.columns:
            logging.error("Macro data missing a date or timestamp column")
            return None
            
        # Standardize column names
        if 'date' in macro_df.columns and 'timestamp' not in macro_df.columns:
            macro_df['timestamp'] = macro_df['date']
            macro_df.drop(columns=['date'], inplace=True, errors='ignore')
            
        # Konverter 'Unnamed: 0' kolonner til timestamp kolonner
        if 'Unnamed: 0' in bitcoin_df.columns and 'timestamp' not in bitcoin_df.columns:
            logging.info("Konverterer 'Unnamed: 0' til timestamp kolonne i Bitcoin data")
            bitcoin_df['timestamp'] = pd.to_datetime(bitcoin_df['Unnamed: 0'])
        
        if 'Unnamed: 0' in macro_df.columns and 'timestamp' not in macro_df.columns:
            logging.info("Konverterer 'Unnamed: 0' til timestamp kolonne i Macro data")
            macro_df['timestamp'] = pd.to_datetime(macro_df['Unnamed: 0'])
        
        # Tjek om vi har timestamp kolonner i begge datasæt
        if 'timestamp' not in bitcoin_df.columns:
            logging.error("Bitcoin data mangler timestamp kolonne efter konvertering")
            return None
            
        if 'timestamp' not in macro_df.columns:
            logging.error("Macro data mangler timestamp kolonne efter konvertering")
            return None
            
        # Debug logs for at tjekke timestamp formater
        logging.info(f"Bitcoin timestamp eksempler: {bitcoin_df['timestamp'].head(3).tolist()}")
        logging.info(f"Macro timestamp eksempler: {macro_df['timestamp'].head(3).tolist()}")
        
        # Konverter timestamp til samme format (dato uden tidspunkt)
        bitcoin_df['timestamp'] = pd.to_datetime(bitcoin_df['timestamp']).dt.normalize()
        macro_df['timestamp'] = pd.to_datetime(macro_df['timestamp']).dt.normalize()
        
        logging.info(f"Bitcoin timestamp efter normalisering: {bitcoin_df['timestamp'].head(3).tolist()}")
        logging.info(f"Macro timestamp efter normalisering: {macro_df['timestamp'].head(3).tolist()}")
        
        # Merger datasets
        logging.info(f"Bitcoin data shape before merge: {bitcoin_df.shape}")
        logging.info(f"Macro data shape before merge: {macro_df.shape}")
        logging.info(f"Bitcoin columns: {bitcoin_df.columns.tolist()}")
        logging.info(f"Macro columns: {macro_df.columns.tolist()}")
        
        # Tjek for overlap i datoer mellem de to datasæt
        bitcoin_dates = set(bitcoin_df['timestamp'].dt.date)
        macro_dates = set(macro_df['timestamp'].dt.date)
        common_dates = bitcoin_dates.intersection(macro_dates)
        
        logging.info(f"Antal datoer i Bitcoin data: {len(bitcoin_dates)}")
        logging.info(f"Antal datoer i Macro data: {len(macro_dates)}")
        logging.info(f"Antal fælles datoer: {len(common_dates)}")
        
        if len(common_dates) == 0:
            logging.error("Ingen fælles datoer mellem datasættene!")
            # Hvis der ikke er nogen fælles datoer, brug Bitcoin data og udfyld med tomme værdier
            logging.warning("Bruger Bitcoin data og udfylder med tomme værdier for makroøkonomiske features")
            
            # Opret en kopi af bitcoin_df og tilføj tomme kolonner for makrodata
            combined_df = bitcoin_df.copy()
            for col in macro_df.columns:
                if col != 'timestamp' and col not in combined_df.columns:
                    combined_df[col] = np.nan
                    
            logging.info(f"Kombineret datasæt med tomme makro-kolonner: {combined_df.shape}")
        else:
            # Merge based on timestamp
            combined_df = pd.merge(
                bitcoin_df,
                macro_df,
                on='timestamp',
                how='inner'  # Ændret til 'inner' for kun at beholde datoer der findes i begge datasæt
            )
            
        logging.info(f"Combined shape after merge: {combined_df.shape}")
        
        # Check for NaN values
        nan_cols = combined_df.columns[combined_df.isna().any()].tolist()
        if nan_cols:
            logging.warning(f"NaN values found in the following columns after merge: {nan_cols}")
            logging.info("Applying forward fill (ffill) followed by backward fill (bfill) to handle NaN values")
            combined_df = combined_df.ffill().bfill()
            
        # Check again if there are still NaN values
        nan_rows = combined_df.isna().any(axis=1).sum()
        if nan_rows > 0:
            logging.warning(f"There are still {nan_rows} rows with NaN values after ffill/bfill")
            # If there are still NaN values, replace them with 0 for numeric columns
            for col in combined_df.select_dtypes(include=['float64', 'int64']).columns:
                combined_df[col] = combined_df[col].fillna(0)
        
        logging.info(f"Datasets combined successfully. Shape: {combined_df.shape}")
        
        if not combined_df.empty:
            logging.info(f"Date range: from {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
        else:
            logging.warning("Combined dataset is empty!")
            
        logging.info(f"Dataset contains only US trading days (weekdays excluding US holidays)")
        
        return combined_df
    except Exception as e:
        logging.error(f"Error combining datasets: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None

def save_combined_data(df):
    """Save combined dataset to file"""
    try:
        output_path = INTERMEDIATE_DIR / OUTPUT_FILENAME
        df.to_csv(output_path, index=False)
        logging.info(f"Combined trading days data saved successfully to {output_path}")
        return True
    except Exception as e:
        logging.error(f"Error saving combined data: {e}")
        return False

def main():
    """Main function to run the data combination process"""
    logging.info("--- Starting Data Combination Process (Trading Days Only) ---")
    
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
        logging.info("--- Data Combination (Trading Days Only) Completed Successfully ---")
    else:
        logging.error("--- Data Combination Failed (Save Error) ---")
        sys.exit(1)

if __name__ == "__main__":
    main() 