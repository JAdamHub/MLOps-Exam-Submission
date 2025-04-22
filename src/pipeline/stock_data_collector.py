import requests
import pandas as pd
import logging
import sys
from pathlib import Path
import time
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from datetime import datetime, timedelta
import os
import json
import numpy as np
from dotenv import load_dotenv
import sqlite3

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
START_DATE = "2006-04-08" # Define the desired start date (used for initial full load)
# Determine project root based on script location
# Assumes the script is in src/pipeline
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" # Point directly to data directory
DB_FILE = DATA_DIR / "market_data.db" # Database file path in data/
TABLE_NAME = 'market_data' # Define table name centrally

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
if not ALPHA_VANTAGE_API_KEY:
    raise ValueError("ALPHA_VANTAGE_API_KEY not found in environment variables")
# Alpha Vantage API configuration
API_BASE_URL = "https://www.alphavantage.co/query"

# Vestas stock configuration - updated symbol to VWS.CPH
STOCK_SYMBOL = "VWSB.DEX"  # vestas wind systems ticker on frankfurt stock exchange

# OUTPUT_SIZE is now determined dynamically based on whether it's an update or initial load
# OUTPUT_SIZE = "full"

# Macroeconomic indicators from Alpha Vantage - updated symbols
MACRO_INDICATORS = {
    # Market indices - Updated symbols
    "SPY": {"name": "spy", "function": "TIME_SERIES_DAILY", "description": "S&P 500 ETF"},
    "VGK": {"name": "europe", "function": "TIME_SERIES_DAILY", "description": "Vanguard European Stock ETF"},
    
    # Currency rates relevant for Vestas (international company)
    "EUR/USD": {"name": "eurusd", "function": "FX_DAILY", "description": "Euro to US Dollar Exchange Rate"},
    
    # Commodities relevant for wind energy
    "USO": {"name": "crude_oil_etf", "function": "TIME_SERIES_DAILY", "description": "US Oil Fund ETF"},
    
    # Interest rate indicators
    "TLT": {"name": "treasury_etf", "function": "TIME_SERIES_DAILY", "description": "20+ Year Treasury Bond ETF"}
}

# API request settings
MAX_RETRIES = 5
RETRY_DELAY = 15  # seconds, increased to avoid API rate limits
MAX_DELAY = 60  # max seconds to wait in case of rate limiting

# Ensure data directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True) # Ensure the main data directory exists

def get_latest_date_from_db(db_file: Path, table_name: str) -> pd.Timestamp | None:
    """Gets the latest date from the specified table in the database."""
    if not db_file.exists():
        logging.info(f"Database file {db_file} not found. Will fetch full history.")
        return None
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        # Check if table exists
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
        if cursor.fetchone() is None:
            logging.info(f"Table '{table_name}' not found in database {db_file}. Will fetch full history.")
            conn.close()
            return None

        # Query for the max date
        query = f"SELECT MAX(date) FROM {table_name}"
        # Use read_sql_query to easily handle the result
        result_df = pd.read_sql_query(query, conn)
        conn.close()

        latest_date_str = result_df.iloc[0, 0]

        if latest_date_str:
            latest_date = pd.to_datetime(latest_date_str)
            logging.info(f"Latest date found in database table '{table_name}': {latest_date.strftime('%Y-%m-%d')}")
            return latest_date
        else:
            logging.info(f"Table '{table_name}' exists but is empty or has no date. Will fetch full history.")
            return None
    except Exception as e:
        logging.error(f"Error reading latest date from table '{table_name}' in database {db_file}: {e}")
        logging.warning("Falling back to fetching full history due to error.")
        # Fallback to fetching full history on error
        return None

def get_trading_days(start_date, end_date):
    """
    generates a list of trading days when the danish stock market is open.
    excludes weekends and danish holidays.
    
    Args:
        start_date: start date as string 'YYYY-MM-DD' or datetime
        end_date: end date as string 'YYYY-MM-DD' or datetime
        
    Returns:
        DatetimeIndex with trading days
    """
    # convert to datetime if necessary
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
        
    # define danish holidays (using us holidays as an approximation)
    # for more precision, a danish holiday calendar should be implemented
    dk_holidays = USFederalHolidayCalendar()
    holidays = dk_holidays.holidays(start=start_date, end=end_date)
    
    # define a business day that excludes weekends and holidays
    business_days = CustomBusinessDay(calendar=dk_holidays)
    
    # generate a list of trading days
    trading_days = pd.date_range(start=start_date, end=end_date, freq=business_days)
    
    logging.info(f"generated {len(trading_days)} trading days between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}")
    
    return trading_days

def fetch_data(params, retries=MAX_RETRIES, delay=RETRY_DELAY):
    """Fetches data from Alpha Vantage API with retry logic."""
    for attempt in range(retries):
        try:
            data_type = params.get('function', 'UNKNOWN')
            symbol = params.get('symbol', params.get('from_symbol', 'UNKNOWN'))
            
            logging.info(f"Fetching {data_type} data for {symbol} (Attempt {attempt + 1}/{retries})")
            response = requests.get(API_BASE_URL, params=params, timeout=30)
            
            # Check if response is empty or too small
            if not response.text or len(response.text) < 50:
                logging.warning(f"Empty or small response received: {response.text}")
                if attempt < retries - 1:
                    wait_time = delay * (attempt + 1)
                    logging.info(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    continue
                return None
            
            response.raise_for_status()
            
            # Try to parse JSON - with some APIs, invalid responses might be JSON formatted
            try:
                data = response.json()
                # Print the full response for debugging
                logging.info(f"Response from Alpha Vantage for {symbol}: {data.keys()}")
                if 'Information' in data:
                    logging.info(f"Information message: {data['Information']}")
                if 'Error Message' in data:
                    logging.info(f"Error message: {data['Error Message']}")
                if 'Note' in data:
                    logging.info(f"Note message: {data['Note']}")
            except json.JSONDecodeError:
                logging.error(f"Invalid JSON response: {response.text[:200]}...")
                if attempt < retries - 1:
                    wait_time = delay * (attempt + 1)
                    logging.info(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    continue
                return None
            
            # Check for error messages or API limits
            if "Error Message" in data:
                logging.error(f"API Error: {data['Error Message']}")
                if "Invalid API call" in data["Error Message"] and attempt < retries - 1:
                    wait_time = delay * (attempt + 1)
                    logging.info(f"Invalid API call. Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    continue
                return None
            
            # Check for rate limit messages
            if "Note" in data and "API call frequency" in data["Note"]:
                logging.warning(f"API call frequency limit reached: {data['Note']}")
                if attempt < retries - 1:
                    wait_time = min(delay * (2 ** attempt), MAX_DELAY)  # Exponential backoff
                    logging.info(f"Rate limited. Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    continue
            
            # Check if data is empty or doesn't contain expected time series keys
            expected_keys = ["Time Series (Daily)", "Weekly Time Series", "Monthly Time Series", "Time Series FX (Daily)", "data"]
            if not data or not any(key in data for key in expected_keys):
                logging.error(f"Response doesn't contain expected data keys for {symbol}. Keys: {list(data.keys())}")
                if attempt < retries - 1:
                    wait_time = delay * (attempt + 1)
                    logging.info(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    continue
                return None
                
            logging.info(f"Data successfully fetched for {symbol}")
            return data
            
        except requests.exceptions.RequestException as e:
            logging.warning(f"Request failed: {e}")
            if attempt < retries - 1:
                wait_time = delay * (attempt + 1)
                logging.info(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                logging.error(f"Failed to fetch data after {retries} attempts")
                return None
    
    return None

def process_stock_data(data, time_series_key):
    """Process Alpha Vantage stock time series data into a DataFrame."""
    try:
        # Check if the expected time series key exists
        if time_series_key not in data:
            logging.error(f"Expected key '{time_series_key}' not found in API response")
            return None
            
        # Extract time series data
        time_series = data[time_series_key]
        
        # Check if the time series is empty
        if not time_series:
            logging.error(f"Empty time series data in API response")
            return None
            
        # Initialize lists to store data
        dates = []
        open_prices = []
        high_prices = []
        low_prices = []
        close_prices = []
        volumes = []
        
        # Process data points
        for date, values in time_series.items():
            try:
                dates.append(date)
                open_prices.append(float(values.get("1. open", 0)))
                high_prices.append(float(values.get("2. high", 0)))
                low_prices.append(float(values.get("3. low", 0)))
                close_prices.append(float(values.get("4. close", 0)))
                volumes.append(int(float(values.get("5. volume", 0))))
            except (ValueError, KeyError) as e:
                logging.warning(f"Error processing data point for date {date}: {e}")
                # Skip this date rather than failing entirely
                continue
        
        # Ensure we have data to create a dataframe
        if not dates:
            logging.error("No valid dates found in time series data")
            return None
            
        # Create DataFrame
        df = pd.DataFrame({
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volumes
        }, index=pd.DatetimeIndex(dates))
        
        # Sort by date
        df.sort_index(inplace=True)
        
        # Filter data from START_DATE onwards ONLY if doing a full load initially
        # Otherwise, filtering happens based on latest_date_in_db later
        # We remove the START_DATE filtering here, it will be implicitly handled
        # by the incremental logic or the initial full load starting point.
        # try:
        #      start_datetime = pd.to_datetime(START_DATE)
        #      df = df[df.index >= start_datetime]
        #      logging.info(f"Filtered stock data to start from {START_DATE}. New shape: {df.shape}")
        # except ValueError:
        #      logging.error(f"Invalid START_DATE format: {START_DATE}. Skipping date filtering.")

        return df
        
    except Exception as e:
        logging.error(f"Error processing stock time series data: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None

def process_fx_data(data):
    """Process Alpha Vantage FX (forex) data into a DataFrame."""
    try:
        # Check if the expected key exists
        if "Time Series FX (Daily)" not in data:
            logging.error("Expected key 'Time Series FX (Daily)' not found in API response")
            return None
            
        # Extract time series data
        time_series = data["Time Series FX (Daily)"]
        
        # Initialize lists to store data
        dates = []
        open_prices = []
        high_prices = []
        low_prices = []
        close_prices = []
        
        # Process data points
        for date, values in time_series.items():
            try:
                dates.append(date)
                open_prices.append(float(values.get("1. open", 0)))
                high_prices.append(float(values.get("2. high", 0)))
                low_prices.append(float(values.get("3. low", 0)))
                close_prices.append(float(values.get("4. close", 0)))
            except (ValueError, KeyError) as e:
                logging.warning(f"Error processing FX data point for date {date}: {e}")
                continue
        
        # Ensure we have data to create a dataframe
        if not dates:
            logging.error("No valid dates found in FX time series data")
            return None
            
        # Create DataFrame
        df = pd.DataFrame({
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices
        }, index=pd.DatetimeIndex(dates))
        
        # Sort by date
        df.sort_index(inplace=True)
        return df
        
    except Exception as e:
        logging.error(f"Error processing FX data: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None

def process_economic_data(data, indicator):
    """Process Alpha Vantage economic indicator data into a DataFrame."""
    try:
        # Extract the data key
        if "data" not in data:
            logging.error(f"Expected key 'data' not found in API response for {indicator}")
            return None
            
        # Extract the data points
        data_points = data["data"]
        
        # Initialize lists to store data
        dates = []
        values = []
        
        # Process data points
        for entry in data_points:
            try:
                if "date" in entry and "value" in entry:
                    dates.append(entry["date"])
                    values.append(float(entry["value"]))
            except (ValueError, KeyError) as e:
                logging.warning(f"Error processing economic data point: {e}")
                continue
        
        # Ensure we have data to create a dataframe
        if not dates:
            logging.error("No valid dates found in economic data")
            return None
            
        # Create DataFrame
        df = pd.DataFrame({
            indicator: values
        }, index=pd.DatetimeIndex(dates))
        
        # Sort by date
        df.sort_index(inplace=True)
        
        return df
        
    except Exception as e:
        logging.error(f"Error processing economic data for {indicator}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None

def collect_macro_data(latest_date_in_db: pd.Timestamp | None):
    """Collects macroeconomic data, potentially incrementally."""
    all_macro_dfs = []

    # Determine output size based on whether we have a latest date
    output_size = "compact" if latest_date_in_db else "full"
    logging.info(f"Using outputsize='{output_size}' for macro API calls.")

    for symbol, info in MACRO_INDICATORS.items():
        logging.info(f"Collecting {info['description']} ({symbol})...")
        params = {
            "function": info['function'],
            "outputsize": output_size, # Use determined size
            "apikey": ALPHA_VANTAGE_API_KEY
        }
        if info['function'] == "FX_DAILY":
            from_currency, to_currency = symbol.split('/')
            params.update({"from_symbol": from_currency, "to_symbol": to_currency})
        else:
            params["symbol"] = symbol

        data = fetch_data(params) # Pass params dict directly
        if data:
            df = None # Initialize df
            if info['function'] == "FX_DAILY":
                df_raw = process_fx_data(data)
                if df_raw is not None:
                    # Keep only the 'close' column for FX data
                    df = df_raw[['close']].copy()
            elif info['function'] == "TIME_SERIES_DAILY": # Handle stock/ETF data
                 df_raw = process_stock_data(data, "Time Series (Daily)")
                 if df_raw is not None:
                     # Keep only the 'close' column for ETFs used as macro indicators
                     df = df_raw[['close']].copy()
            # Add elif for other function types if needed (e.g., economic indicators)

            if df is not None:
                # Rename all columns with prefix
                new_column_name = f"{info['name']}_close"
                df.rename(columns={'close': new_column_name}, inplace=True)

                # --- Incremental Filter ---
                if latest_date_in_db:
                    original_count = len(df)
                    df = df[df.index > latest_date_in_db]
                    logging.info(f"Filtered {info['name']} data to dates after {latest_date_in_db.strftime('%Y-%m-%d')}. Kept {len(df)} out of {original_count} rows.")

                if not df.empty or not latest_date_in_db: # Only append if there's new data or if it's the first run
                    # Create DataFrame with the single column before appending
                    df_to_append = pd.DataFrame({new_column_name: df[new_column_name]}, index=df.index)
                    df_to_append.sort_index(inplace=True) # Ensure sorting before appending
                    all_macro_dfs.append(df_to_append)
                    logging.info(f"Processed {info['name']} data. New/Total shape: {df_to_append.shape}")
                elif df.empty and latest_date_in_db:
                     logging.info(f"No new {info['name']} data found since last update.")

            else:
                logging.warning(f"Could not process data for {symbol}")
        else:
            logging.warning(f"Could not fetch data for {symbol}")

        time.sleep(RETRY_DELAY) # Respect API limits

    if not all_macro_dfs:
        logging.info("No new macroeconomic data found or collected.")
        # Return an empty DataFrame to signal no new data
        return pd.DataFrame()

    # Combine all macro DataFrames
    combined_macro_df = pd.concat(all_macro_dfs, axis=1)
    
    # Sort index just in case
    combined_macro_df.sort_index(inplace=True)

    logging.info(f"Combined new/total macro data shape: {combined_macro_df.shape}")
    return combined_macro_df

def collect_vestas_data(latest_date_in_db: pd.Timestamp | None):
    """Collects Vestas stock data, potentially incrementally."""
    logging.info(f"Collecting Vestas daily data ({STOCK_SYMBOL})...")

    # Determine output size based on whether we have a latest date
    output_size = "compact" if latest_date_in_db else "full"
    logging.info(f"Using outputsize='{output_size}' for API call.")

    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": STOCK_SYMBOL,
        "outputsize": output_size, # Use determined size
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    data = fetch_data(params) # Pass params dict directly
    if data:
        df = process_stock_data(data, "Time Series (Daily)")
        if df is not None:
            # Rename columns to be stock-specific
            df.rename(columns={
                'open': 'stock_open',
                'high': 'stock_high',
                'low': 'stock_low',
                'close': 'stock_close',
                'volume': 'stock_volume'
            }, inplace=True)

            # --- Incremental Filter ---
            if latest_date_in_db:
                original_count = len(df)
                df = df[df.index > latest_date_in_db]
                logging.info(f"Filtered Vestas data to dates after {latest_date_in_db.strftime('%Y-%m-%d')}. Kept {len(df)} out of {original_count} rows.")

            if df.empty and latest_date_in_db:
                 logging.info("No new Vestas data found since the last update.")
                 # Return an empty DataFrame to signal no new data of this type
                 # Ensure it has the correct columns if needed downstream, although concat handles this.
                 return pd.DataFrame(columns=['stock_open', 'stock_high', 'stock_low', 'stock_close', 'stock_volume'])

            logging.info(f"Vestas data collected successfully. New/Total shape: {df.shape}")
            return df
        else:
            logging.error("Failed to process Vestas data.")
            return None
    else:
        logging.error("Failed to fetch Vestas data.")
        return None

def save_to_db(new_df: pd.DataFrame | None, db_file: Path, table_name='market_data'):
    """Loads existing data, merges with new data, and saves back to SQLite using replace."""
    if new_df is None or new_df.empty:
        logging.info("No new data provided to add to the database. Checking existing data.")
        # Check if the database and table exist even if there's no new data.
        if not db_file.exists():
             logging.warning(f"Database file {db_file} does not exist and no new data was provided.")
             return True # Nothing to do, but not an error state necessarily.
        # If DB exists, assume it's already up-to-date.
        logging.info("Database already exists and no new data to add.")
        return True # Indicate success (no changes needed)

    try:
        existing_df = pd.DataFrame() # Default to empty df
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
        table_exists = cursor.fetchone() is not None

        if table_exists:
            logging.info(f"Loading existing data from table '{table_name}' to merge...")
            # Load existing data, making sure 'date' is the index and parsed correctly
            existing_df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn, index_col='date', parse_dates=['date'])
            logging.info(f"Loaded {len(existing_df)} existing rows.")
        else:
             logging.info(f"Table '{table_name}' does not exist in {db_file}. Creating new table.")

        conn.close() # Close connection after reading

        # Ensure new_df's index is datetime (it should be already, but double-check)
        new_df.index = pd.to_datetime(new_df.index)

        # Combine old and new data
        # Use concat, as new_df should already be filtered to have only newer dates
        # However, duplicates might still occur if the API returns the latest date already in DB.
        logging.info(f"Combining {len(existing_df)} existing rows with {len(new_df)} new rows.")
        combined_df = pd.concat([existing_df, new_df])

        # Drop duplicates based on index (date), keeping the newest entry ('last')
        # This handles potential overlaps safely.
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]

        # Sort by date index
        combined_df.sort_index(inplace=True)

        # Save the fully combined data using 'replace'
        conn = sqlite3.connect(db_file) # Reopen connection to write
        logging.info(f"Saving combined data ({len(combined_df)} rows) to table '{table_name}' using 'replace'...")
        # Use index=True to save the date index as a column named 'date'
        combined_df.to_sql(table_name, conn, if_exists='replace', index=True, index_label='date')
        conn.close() # Close connection after writing

        logging.info(f"Data successfully saved/updated in table '{table_name}' in {db_file}")

        # Verification step (optional but good)
        conn = sqlite3.connect(db_file)
        df_check = pd.read_sql(f"SELECT COUNT(*) AS count FROM {table_name}", conn)
        rows_in_db = df_check.iloc[0, 0]
        logging.info(f"Verification: Table '{table_name}' now contains {rows_in_db} rows.")
        # Log date range for verification
        if rows_in_db > 0:
            min_max_dates = pd.read_sql(f"SELECT MIN(date) as min_date, MAX(date) as max_date FROM {table_name}", conn)
            logging.info(f"Data range in DB: {min_max_dates['min_date'].iloc[0]} to {min_max_dates['max_date'].iloc[0]}")
        conn.close()

        return True
    except Exception as e:
        logging.error(f"Error saving data to database {db_file}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False

def main():
    """Main function to run the data collection (potentially incremental) and save to database."""
    logging.info("=== Starting Data Collection ===")

    # --- Get latest date from DB ---
    latest_date_in_db = get_latest_date_from_db(DB_FILE, TABLE_NAME)

    # --- Collect Vestas data (pass latest date) ---
    vestas_df_new = collect_vestas_data(latest_date_in_db)
    # If collection failed entirely (returned None), return False
    if vestas_df_new is None:
        logging.error("Critical error during Vestas stock data collection. Pipeline cannot continue.")
        return False
    # If no new data was found, vestas_df_new might be empty. That's okay.

    # --- Collect Macroeconomic data (pass latest date) ---
    macro_df_new = collect_macro_data(latest_date_in_db)
    # macro_df_new can be empty if no new data or failure for all indicators. It should not be None based on current return logic.

    # --- Combine NEW Vestas and NEW Macro data ---
    # Handle cases where one or both are empty
    combined_new_df = None # Initialize

    if not vestas_df_new.empty and macro_df_new.empty:
        combined_new_df = vestas_df_new
        logging.info("Only new Vestas data was found/collected.")
    elif vestas_df_new.empty and not macro_df_new.empty:
        combined_new_df = macro_df_new
        logging.info("Only new Macroeconomic data was found/collected.")
    elif not vestas_df_new.empty and not macro_df_new.empty:
        logging.info("Combining new Vestas and new Macroeconomic data...")
        # Use outer join on new dataframes before merging with old
        combined_new_df = pd.merge(vestas_df_new, macro_df_new, left_index=True, right_index=True, how='outer')
        combined_new_df.sort_index(inplace=True) # Sort after merge
        logging.info(f"New combined data shape before potential ffill: {combined_new_df.shape}")
        # Forward fill macro columns based on the assumption that stock data index is leading
        # This should ideally happen *after* combining with existing data for full context,
        # but let's apply preliminary ffill on the *new* chunk.
        # A better approach might be to do ffill after merging old and new in save_to_db.
        # For now, let's keep the ffill logic in the main combination step.
        if not vestas_df_new.empty: # Only ffill if vestas data exists
            macro_cols = macro_df_new.columns # Get macro columns from the macro df
            present_macro_cols = [col for col in macro_cols if col in combined_new_df.columns]
            if present_macro_cols:
                 logging.info(f"Forward filling potentially missing values in new macro columns: {present_macro_cols}")
                 combined_new_df[present_macro_cols] = combined_new_df[present_macro_cols].fillna(method='ffill')

        # Filter to only keep rows where stock data is present (if vestas data was collected)
        # This ensures the index aligns with trading days primarily.
        if 'stock_close' in combined_new_df.columns:
             logging.info("Filtering combined new data to ensure stock data is present.")
             combined_new_df = combined_new_df[combined_new_df['stock_close'].notna()]
        logging.info(f"Combined new data shape after potential ffill/filtering: {combined_new_df.shape}")
    else: # Both are empty
        logging.info("No new Vestas or Macroeconomic data found to update.")
        # combined_new_df remains None or could be set to empty df explicitly
        combined_new_df = pd.DataFrame() # Ensure it's an empty DataFrame

    # --- Save combined data (old + new) to SQLite database ---
    # The save_to_db function now handles merging existing data
    if combined_new_df.empty: # Check if effectively empty
         logging.info("No new data to add to the database.")
         logging.info("=== Data Collection Completed (No Update Needed) ===")
         return True # Success, but no changes made

    if save_to_db(combined_new_df, DB_FILE, TABLE_NAME):
        logging.info("=== Data Collection and DB Save/Update Completed Successfully ===")
        return True
    else:
        logging.error("=== Data Collection Failed During DB Save ===")
        sys.exit(1) # Exit with error code if main fails

if __name__ == "__main__":
    if not main():
        sys.exit(1) # Exit with error code if main fails 