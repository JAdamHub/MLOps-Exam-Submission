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
RAW_STOCKS_DIR = PROJECT_ROOT / "data" / "raw" / "stocks"
RAW_MACRO_DIR = PROJECT_ROOT / "data" / "raw" / "macro"
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate" / "combined"

# Input files - Vestas aktiedata
# Daglige data filer
VESTAS_DAILY_FILENAME = "vestas_daily.csv"
MACRO_DAILY_FILENAME = "macro_economic_trading_days.csv"

# Output files
OUTPUT_DAILY_FILENAME = "vestas_macro_combined_trading_days.csv"

# Ensure output directory exists
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

def load_vestas_data(interval="daily"):
    """
    Load Vestas stock data from raw directory
    
    Args:
        interval: for nu kun "daily" understøttet
    """
    try:
        if interval == "daily":
            filename = VESTAS_DAILY_FILENAME
        else:
            logging.error(f"Unsupported interval: {interval}. Only 'daily' is supported for Vestas data.")
            return None
            
        vestas_path = RAW_STOCKS_DIR / filename
        if not vestas_path.exists():
            logging.error(f"Vestas {interval} data file not found: {vestas_path}")
            return None
            
        df = pd.read_csv(vestas_path)
        
        # Ensure timestamp/date is in datetime format
        if 'Unnamed: 0' in df.columns:
            df['date'] = pd.to_datetime(df['Unnamed: 0'])
            df.drop(columns=['Unnamed: 0'], inplace=True)
        elif 'date' not in df.columns:
            # Hvis filen allerede har et indeks, men intet datofelt
            df['date'] = pd.to_datetime(df.index)
            
        # Sæt dato som indeks
        if 'date' in df.columns:
            df.set_index('date', inplace=True)
            
        logging.info(f"Vestas {interval} data loaded successfully from {vestas_path}")
        logging.info(f"Vestas data columns: {df.columns.tolist()}")
        logging.info(f"Vestas data shape: {df.shape}")
        
        return df
    except Exception as e:
        logging.error(f"Error loading Vestas {interval} data: {e}")
        return None

def load_macro_data(interval="daily"):
    """
    Load macroeconomic data
    
    Args:
        interval: for nu kun "daily" understøttet
    """
    try:
        if interval == "daily":
            filename = MACRO_DAILY_FILENAME
        else:
            logging.error(f"Unsupported interval: {interval}. Only 'daily' is supported.")
            return None
            
        macro_path = RAW_MACRO_DIR / filename
        if not macro_path.exists():
            logging.error(f"Macro {interval} data file not found: {macro_path}")
            return None
            
        df = pd.read_csv(macro_path)
        
        # Handle the index column, which likely contains the date
        if 'Unnamed: 0' in df.columns:
            df['date'] = pd.to_datetime(df['Unnamed: 0'])
            df.drop(columns=['Unnamed: 0'], inplace=True)
            df.set_index('date', inplace=True)
        
        logging.info(f"Macroeconomic {interval} data loaded successfully from {macro_path}")
        logging.info(f"Macro data columns: {df.columns.tolist()}")
        logging.info(f"Macro data shape: {df.shape}")
        
        return df
    except Exception as e:
        logging.error(f"Error loading macroeconomic {interval} data: {e}")
        return None

def combine_datasets(vestas_df, macro_df, interval="daily"):
    """
    Combine Vestas stock and macroeconomic datasets
    
    Args:
        vestas_df: Vestas dataframe
        macro_df: Macroeconomic dataframe
        interval: for nu kun "daily" understøttet
    """
    try:
        if vestas_df is None or macro_df is None:
            logging.error("Cannot combine datasets: one or both datasets are missing")
            return None
            
        # Make a copy to avoid warnings
        vestas_df = vestas_df.copy()
        macro_df = macro_df.copy()
        
        # Kontroller at begge datasæt har dato som indeks
        if not isinstance(vestas_df.index, pd.DatetimeIndex):
            logging.warning("Vestas data does not have DatetimeIndex, attempting to convert")
            if 'date' in vestas_df.columns:
                vestas_df.set_index('date', inplace=True)
            else:
                logging.error("Cannot find date column in Vestas data")
                return None
                
        if not isinstance(macro_df.index, pd.DatetimeIndex):
            logging.warning("Macro data does not have DatetimeIndex, attempting to convert")
            if 'date' in macro_df.columns:
                macro_df.set_index('date', inplace=True)
            else:
                logging.error("Cannot find date column in Macro data")
                return None
                
        # Debug logs for at tjekke indeks formater
        logging.info(f"Vestas {interval} index eksempler: {vestas_df.index[:3].tolist()}")
        logging.info(f"Macro {interval} index eksempler: {macro_df.index[:3].tolist()}")
        
        # Konverter indeks til samme format for daglige data
        if interval == "daily":
            vestas_df.index = pd.to_datetime(vestas_df.index).normalize()
            macro_df.index = pd.to_datetime(macro_df.index).normalize()
        
        # Merger datasets på indeks (dato)
        logging.info(f"Vestas {interval} data shape before merge: {vestas_df.shape}")
        logging.info(f"Macro {interval} data shape before merge: {macro_df.shape}")
        
        # Tjek for overlap i datoer mellem de to datasæt
        vestas_dates = set(vestas_df.index)
        macro_dates = set(macro_df.index)
        common_dates = vestas_dates.intersection(macro_dates)
        
        logging.info(f"Antal datoer i Vestas {interval} data: {len(vestas_dates)}")
        logging.info(f"Antal datoer i Macro {interval} data: {len(macro_dates)}")
        logging.info(f"Antal fælles datoer: {len(common_dates)}")
        
        if len(common_dates) == 0:
            logging.error(f"Ingen fælles datoer mellem {interval} datasættene!")
            # Hvis der ikke er nogen fælles datoer, brug Vestas data og udfyld med tomme værdier
            logging.warning(f"Bruger Vestas {interval} data og udfylder med tomme værdier for makroøkonomiske features")
            
            # Opret en kopi af vestas_df og tilføj tomme kolonner for makrodata
            combined_df = vestas_df.copy()
            for col in macro_df.columns:
                if col not in combined_df.columns:
                    combined_df[col] = np.nan
        else:
            # Brug indre join for at undgå for mange NaN-værdier
            combined_df = pd.merge(
                vestas_df, 
                macro_df,
                left_index=True, 
                right_index=True,
                how='left'  # Behold alle Vestas-datopunkter og match med makro hvor muligt
            )
            
        logging.info(f"Combined {interval} shape after merge: {combined_df.shape}")
        
        # Check for NaN values
        nan_cols = combined_df.columns[combined_df.isna().any()].tolist()
        nan_percentage = (combined_df.isna().sum() / len(combined_df)) * 100
        
        logging.info("Procent af manglende værdier per kolonne:")
        for col, pct in nan_percentage[nan_percentage > 0].items():
            logging.info(f"  {col}: {pct:.2f}%")
            
        if nan_cols:
            logging.warning(f"NaN values found in {len(nan_cols)} columns after merge in {interval} data")
            logging.info("Applying forward fill (ffill) followed by backward fill (bfill) to handle NaN values")
            
            # For hver kolonne med NaN-værdier, udfyld med ffill -> bfill
            for col in nan_cols:
                combined_df[col] = combined_df[col].fillna(method='ffill').fillna(method='bfill')
                
            # Tjek for resterende NaN-værdier
            remaining_nan = combined_df.isna().sum().sum()
            if remaining_nan > 0:
                logging.warning(f"There are still {remaining_nan} NaN values after filling")
                # Udfyld resterende NaN-værdier med 0 (eller en anden passende strategi)
                combined_df.fillna(0, inplace=True)
                
        return combined_df
        
    except Exception as e:
        logging.error(f"Error combining datasets: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None

def save_combined_data(df, interval="daily"):
    """
    Save the combined dataset to CSV
    
    Args:
        df: Combined dataframe
        interval: for nu kun "daily" understøttet
    """
    try:
        if df is None:
            logging.error(f"Cannot save {interval} combined data: DataFrame is None")
            return False
            
        if interval == "daily":
            output_file = INTERMEDIATE_DIR / OUTPUT_DAILY_FILENAME
        else:
            logging.error(f"Unsupported interval: {interval}. Only 'daily' is supported.")
            return False
            
        # Gem med dato som indeks
        df.to_csv(output_file)
        logging.info(f"Combined {interval} data saved successfully to {output_file}")
        return True
    except Exception as e:
        logging.error(f"Error saving combined {interval} data: {e}")
        return False

def process_interval_data(interval="daily"):
    """
    Process data for a specific interval
    
    Args:
        interval: for nu kun "daily" understøttet
    """
    try:
        logging.info(f"Processing {interval} data")
        
        # Load Vestas data
        vestas_df = load_vestas_data(interval)
        if vestas_df is None:
            logging.error(f"Failed to load Vestas {interval} data")
            return False
            
        # Load macroeconomic data
        macro_df = load_macro_data(interval)
        if macro_df is None:
            logging.warning(f"No macro {interval} data available, proceeding with Vestas data only")
            
        # Combine datasets
        combined_df = combine_datasets(vestas_df, macro_df, interval)
        if combined_df is None:
            logging.error(f"Failed to combine {interval} datasets")
            return False
            
        # Save combined data
        if not save_combined_data(combined_df, interval):
            logging.error(f"Failed to save combined {interval} data")
            return False
            
        logging.info(f"Processing {interval} data completed successfully")
        return True
        
    except Exception as e:
        logging.error(f"Error in process_interval_data for {interval}: {e}")
        return False

def main():
    """Main function to run the data combination process."""
    logging.info("--- Starting Data Combination Process ---")
    
    # Process daily data
    if process_interval_data("daily"):
        logging.info("--- Daily Data Combination Completed Successfully ---")
    else:
        logging.error("--- Daily Data Combination Failed ---")
        sys.exit(1)
    
    logging.info("--- Data Combination Process Complete ---")

if __name__ == "__main__":
    main() 