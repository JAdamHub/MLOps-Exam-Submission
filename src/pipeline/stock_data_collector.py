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
START_DATE = "2006-04-08" # Define the desired start date
# Determine project root based on script location
# Assumes the script is in src/pipeline
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" # Point directly to data directory
DB_FILE = DATA_DIR / "market_data.db" # Database file path in data/

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
if not ALPHA_VANTAGE_API_KEY:
    raise ValueError("ALPHA_VANTAGE_API_KEY not found in environment variables")
# Alpha Vantage API configuration
API_BASE_URL = "https://www.alphavantage.co/query"

# Vestas stock configuration - updated symbol to VWS.CPH
STOCK_SYMBOL = "VWSB.DEX"  # vestas wind systems ticker on frankfurt stock exchange

OUTPUT_SIZE = "full"  # to get as much historical data as possible (up to 20 years)

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
        
        # Filter data from START_DATE onwards
        try:
             start_datetime = pd.to_datetime(START_DATE)
             df = df[df.index >= start_datetime]
             logging.info(f"Filtered stock data to start from {START_DATE}. New shape: {df.shape}")
        except ValueError:
             logging.error(f"Invalid START_DATE format: {START_DATE}. Skipping date filtering.")

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
        
        # Filter data from START_DATE onwards
        try:
             start_datetime = pd.to_datetime(START_DATE)
             df = df[df.index >= start_datetime]
             logging.info(f"Filtered FX data to start from {START_DATE}. New shape: {df.shape}")
        except ValueError:
             logging.error(f"Invalid START_DATE format: {START_DATE}. Skipping date filtering.")

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

def collect_macro_data():
    """Collects macroeconomic data and returns a combined DataFrame."""
    all_macro_dfs = []
    for symbol, info in MACRO_INDICATORS.items():
        logging.info(f"Collecting {info['description']} ({symbol})...")
        params = {
            "function": info['function'],
            "outputsize": OUTPUT_SIZE,
            "apikey": ALPHA_VANTAGE_API_KEY
        }
        if info['function'] == "FX_DAILY":
            from_currency, to_currency = symbol.split('/')
            params.update({"from_symbol": from_currency, "to_symbol": to_currency})
        else:
            params["symbol"] = symbol

        data = fetch_data(params)
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
                # Since we only kept 'close', the new name is predictable
                new_column_name = f"{info['name']}_close"
                df.rename(columns={'close': new_column_name}, inplace=True)
                new_columns = [new_column_name] # List containing the single new name

                # --- Validation Step ---
                # Ensure we have data to create a dataframe
                if not new_columns:
                    logging.error("No valid columns found in processed data")
                    continue

                # Create DataFrame
                df = pd.DataFrame({
                    new_column_name: df[new_column_name]
                }, index=df.index)
                
                # Sort by date
                df.sort_index(inplace=True)

                all_macro_dfs.append(df)
                logging.info(f"Processed {info['name']} data.")
            else:
                logging.warning(f"Could not process data for {symbol}")
        else:
            logging.warning(f"Could not fetch data for {symbol}")

        time.sleep(RETRY_DELAY) # Respect API limits

    if not all_macro_dfs:
        logging.error("Failed to collect any macroeconomic data.")
        return None

    # Combine all macro DataFrames
    combined_macro_df = pd.concat(all_macro_dfs, axis=1)
    logging.info(f"Combined macro data shape: {combined_macro_df.shape}")
    return combined_macro_df

def collect_vestas_data():
    """Collects Vestas stock data and returns a DataFrame."""
    logging.info(f"Collecting Vestas daily data ({STOCK_SYMBOL})...")
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": STOCK_SYMBOL,
        "outputsize": OUTPUT_SIZE,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    data = fetch_data(params)
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
            logging.info(f"Vestas data collected successfully. Shape: {df.shape}")
            return df
        else:
            logging.error("Failed to process Vestas data.")
            return None
    else:
        logging.error("Failed to fetch Vestas data.")
        return None

def save_to_db(df, db_file, table_name='market_data'):
    """Saves the combined DataFrame to an SQLite database."""
    if df is None or df.empty:
        logging.error("No data provided to save to database.")
        return False
    try:
        conn = sqlite3.connect(db_file)
        # Use index=True to save the date index as a column named 'date'
        df.to_sql(table_name, conn, if_exists='replace', index=True, index_label='date')
        conn.close()
        logging.info(f"Data successfully saved to table '{table_name}' in {db_file}")
        # Verify table creation
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
        if cursor.fetchone():
            logging.info(f"Table '{table_name}' verified in the database.")
            # Log first 5 rows date and stock_close
            df_check = pd.read_sql(f"SELECT date, stock_close FROM {table_name} ORDER BY date LIMIT 5", conn)
            logging.info(f"First 5 rows check:\n{df_check}")
        else:
            logging.error(f"Verification failed: Table '{table_name}' not found after saving.")
        conn.close()

        return True
    except Exception as e:
        logging.error(f"Error saving data to database {db_file}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False

def main():
    """Main function to run the data collection and save to database."""
    logging.info("=== Starting Streamlined Data Collection ===")

    # Collect Vestas data
    vestas_df = collect_vestas_data()
    if vestas_df is None:
        logging.error("Vestas stock data collection failed. Pipeline cannot continue.")
        return False

    # Collect Macroeconomic data
    macro_df = collect_macro_data()
    if macro_df is None:
        logging.warning("Macroeconomic data collection failed or returned no data. Proceeding with stock data only.")
        combined_df = vestas_df
    else:
        # Combine Vestas and Macro data using an outer join to keep all dates
        logging.info("Combining Vestas and Macroeconomic data...")
        combined_df = pd.merge(vestas_df, macro_df, left_index=True, right_index=True, how='outer')
        combined_df.sort_index(inplace=True)
        logging.info(f"Combined data shape after merge: {combined_df.shape}")

        # Optional: Filter to only include dates present in Vestas data (approximates trading days)
        # Also forward fill macro data to avoid NaNs on non-trading days for macro indicators
        combined_df = combined_df[combined_df['stock_close'].notna()].fillna(method='ffill')
        logging.info(f"Combined data shape after filtering and ffill: {combined_df.shape}")

    # Save combined data to SQLite database
    if save_to_db(combined_df, DB_FILE):
        logging.info("=== Data Collection and DB Save Completed Successfully ===")
        return True
    else:
        logging.error("=== Data Collection Failed During DB Save ===")
        return False

if __name__ == "__main__":
    if not main():
        sys.exit(1) # Exit with error code if main fails 