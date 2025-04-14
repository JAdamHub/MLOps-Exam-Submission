import pandas as pd
import logging
from pathlib import Path
import sys
import numpy as np

# configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# determine project root based on script location
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_STOCKS_DIR = PROJECT_ROOT / "data" / "raw" / "stocks"
RAW_MACRO_DIR = PROJECT_ROOT / "data" / "raw" / "macro"
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate" / "combined"

# input files - vestas stock data
# daily data files
VESTAS_DAILY_FILENAME = "vestas_daily.csv"
MACRO_DAILY_FILENAME = "macro_economic_trading_days.csv"

# output files
OUTPUT_DAILY_FILENAME = "vestas_macro_combined_trading_days.csv"

# ensure output directory exists
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

def load_vestas_data(interval="daily"):
    """
    load vestas stock data from raw directory
    
    args:
        interval: currently only "daily" supported
    """
    try:
        if interval == "daily":
            filename = VESTAS_DAILY_FILENAME
        else:
            logging.error(f"unsupported interval: {interval}. only 'daily' is supported for vestas data.")
            return None
            
        vestas_path = RAW_STOCKS_DIR / filename
        if not vestas_path.exists():
            logging.error(f"vestas {interval} data file not found: {vestas_path}")
            return None
            
        df = pd.read_csv(vestas_path)
        
        # ensure timestamp/date is in datetime format
        if 'Unnamed: 0' in df.columns:
            df['date'] = pd.to_datetime(df['Unnamed: 0'])
            df.drop(columns=['Unnamed: 0'], inplace=True)
        elif 'date' not in df.columns:
            # if the file already has an index but no date field
            df['date'] = pd.to_datetime(df.index)
            
        # set date as index
        if 'date' in df.columns:
            df.set_index('date', inplace=True)
            
        logging.info(f"vestas {interval} data loaded successfully from {vestas_path}")
        logging.info(f"vestas data columns: {df.columns.tolist()}")
        logging.info(f"vestas data shape: {df.shape}")
        
        return df
    except Exception as e:
        logging.error(f"error loading vestas {interval} data: {e}")
        return None

def load_macro_data(interval="daily"):
    """
    load macroeconomic data
    
    args:
        interval: currently only "daily" supported
    """
    try:
        if interval == "daily":
            filename = MACRO_DAILY_FILENAME
        else:
            logging.error(f"unsupported interval: {interval}. only 'daily' is supported.")
            return None
            
        macro_path = RAW_MACRO_DIR / filename
        if not macro_path.exists():
            logging.error(f"macro {interval} data file not found: {macro_path}")
            return None
            
        df = pd.read_csv(macro_path)
        
        # handle the index column, which likely contains the date
        if 'Unnamed: 0' in df.columns:
            df['date'] = pd.to_datetime(df['Unnamed: 0'])
            df.drop(columns=['Unnamed: 0'], inplace=True)
            df.set_index('date', inplace=True)
        
        logging.info(f"macroeconomic {interval} data loaded successfully from {macro_path}")
        logging.info(f"macro data columns: {df.columns.tolist()}")
        logging.info(f"macro data shape: {df.shape}")
        
        return df
    except Exception as e:
        logging.error(f"error loading macroeconomic {interval} data: {e}")
        return None

def combine_datasets(vestas_df, macro_df, interval="daily"):
    """
    combine vestas stock and macroeconomic datasets
    
    args:
        vestas_df: vestas dataframe
        macro_df: macroeconomic dataframe
        interval: currently only "daily" supported
    """
    try:
        if vestas_df is None or macro_df is None:
            logging.error("cannot combine datasets: one or both datasets are missing")
            return None
            
        # make a copy to avoid warnings
        vestas_df = vestas_df.copy()
        macro_df = macro_df.copy()
        
        # check that both datasets have date as index
        if not isinstance(vestas_df.index, pd.DatetimeIndex):
            logging.warning("vestas data does not have datetimeindex, attempting to convert")
            if 'date' in vestas_df.columns:
                vestas_df.set_index('date', inplace=True)
            else:
                logging.error("cannot find date column in vestas data")
                return None
                
        if not isinstance(macro_df.index, pd.DatetimeIndex):
            logging.warning("macro data does not have datetimeindex, attempting to convert")
            if 'date' in macro_df.columns:
                macro_df.set_index('date', inplace=True)
            else:
                logging.error("cannot find date column in macro data")
                return None
                
        # debug logs to check index formats
        logging.info(f"vestas {interval} index examples: {vestas_df.index[:3].tolist()}")
        logging.info(f"macro {interval} index examples: {macro_df.index[:3].tolist()}")
        
        # convert index to the same format for daily data
        if interval == "daily":
            vestas_df.index = pd.to_datetime(vestas_df.index).normalize()
            macro_df.index = pd.to_datetime(macro_df.index).normalize()
        
        # merge datasets on index (date)
        logging.info(f"vestas {interval} data shape before merge: {vestas_df.shape}")
        logging.info(f"macro {interval} data shape before merge: {macro_df.shape}")
        
        # check for overlap in dates between the two datasets
        vestas_dates = set(vestas_df.index)
        macro_dates = set(macro_df.index)
        common_dates = vestas_dates.intersection(macro_dates)
        
        logging.info(f"number of dates in vestas {interval} data: {len(vestas_dates)}")
        logging.info(f"number of dates in macro {interval} data: {len(macro_dates)}")
        logging.info(f"number of common dates: {len(common_dates)}")
        
        if len(common_dates) == 0:
            logging.error(f"no common dates between {interval} datasets!")
            # if there are no common dates, use vestas data and fill with empty values
            logging.warning(f"using vestas {interval} data and filling with empty values for macroeconomic features")
            
            # create a copy of vestas_df and add empty columns for macro data
            combined_df = vestas_df.copy()
            for col in macro_df.columns:
                if col not in combined_df.columns:
                    combined_df[col] = np.nan
        else:
            # use left join to avoid losing vestas data points
            combined_df = pd.merge(
                vestas_df, 
                macro_df,
                left_index=True, 
                right_index=True,
                how='left'  # keep all vestas date points and match with macro where possible
            )
            
        logging.info(f"combined {interval} shape after merge: {combined_df.shape}")
        
        # check for nan values
        nan_cols = combined_df.columns[combined_df.isna().any()].tolist()
        nan_percentage = (combined_df.isna().sum() / len(combined_df)) * 100
        
        logging.info("percentage of missing values per column:")
        for col, pct in nan_percentage[nan_percentage > 0].items():
            logging.info(f"  {col}: {pct:.2f}%")
            
        if nan_cols:
            logging.warning(f"nan values found in {len(nan_cols)} columns after merge in {interval} data")
            logging.info("applying forward fill (ffill) followed by backward fill (bfill) to handle nan values")
            
            # for each column with nan values, fill with ffill -> bfill
            for col in nan_cols:
                combined_df[col] = combined_df[col].fillna(method='ffill').fillna(method='bfill')
                
            # check for remaining nan values
            remaining_nan = combined_df.isna().sum().sum()
            if remaining_nan > 0:
                logging.warning(f"there are still {remaining_nan} nan values after filling")
                # fill remaining nan values with 0 (or another appropriate strategy)
                combined_df.fillna(0, inplace=True)
                
        return combined_df
        
    except Exception as e:
        logging.error(f"error combining datasets: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None

def save_combined_data(df, interval="daily"):
    """
    save the combined dataset to csv
    
    args:
        df: combined dataframe
        interval: currently only "daily" supported
    """
    try:
        if df is None:
            logging.error(f"cannot save {interval} combined data: dataframe is none")
            return False
            
        if interval == "daily":
            output_file = INTERMEDIATE_DIR / OUTPUT_DAILY_FILENAME
        else:
            logging.error(f"unsupported interval: {interval}. only 'daily' is supported.")
            return False
            
        # save with date as index
        df.to_csv(output_file)
        logging.info(f"combined {interval} data saved successfully to {output_file}")
        return True
    except Exception as e:
        logging.error(f"error saving combined {interval} data: {e}")
        return False

def process_interval_data(interval="daily"):
    """
    process data for a specific interval
    
    args:
        interval: currently only "daily" supported
    """
    try:
        logging.info(f"--- processing {interval} data ---")
        
        # load data
        vestas_df = load_vestas_data(interval=interval)
        macro_df = load_macro_data(interval=interval)
        
        # combine datasets
        combined_df = combine_datasets(vestas_df, macro_df, interval=interval)
        
        # save combined data
        if combined_df is not None:
            success = save_combined_data(combined_df, interval=interval)
            if success:
                logging.info(f"--- {interval} data processing completed successfully ---")
                return True
            else:
                logging.error(f"--- failed to save {interval} combined data ---")
                return False
        else:
            logging.error(f"--- failed to combine {interval} datasets ---")
            return False
            
    except Exception as e:
        logging.error(f"error processing {interval} data: {e}")
        return False

def main():
    """
    main function to combine datasets.
    processes daily data only for now.
    """
    logging.info("=== starting dataset combination process ===")
    
    success = process_interval_data(interval="daily")
    
    if success:
        logging.info("=== dataset combination process completed successfully ===")
        return True
    else:
        logging.error("=== dataset combination process failed ===")
        return False

if __name__ == "__main__":
    main() 