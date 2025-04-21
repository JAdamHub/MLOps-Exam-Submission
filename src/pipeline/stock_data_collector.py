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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Determine project root based on script location
# Assumes the script is in src/pipeline
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_STOCKS_DIR = PROJECT_ROOT / "data" / "raw" / "stocks"
RAW_MACRO_DIR = PROJECT_ROOT / "data" / "raw" / "macro"

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

# Data files
TARGET_DAILY_FILENAME = "vestas_daily.csv"  # generic filename not tied to specific symbol
TARGET_WEEKLY_FILENAME = "vestas_weekly.csv"
TARGET_MONTHLY_FILENAME = "vestas_monthly.csv"
TARGET_TRADING_DAYS_FILENAME = "vestas_trading_days.csv"

# Macroeconomic data files
MACRO_DAILY_FILENAME = "macro_economic_trading_days.csv"

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
RAW_STOCKS_DIR.mkdir(parents=True, exist_ok=True)
RAW_MACRO_DIR.mkdir(parents=True, exist_ok=True)

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

def save_data(df, filename, directory):
    """Saves the processed data to a CSV file."""
    if df is None or df.empty:
        logging.error("No data to save.")
        return False
        
    try:
        # Save to CSV
        output_path = directory / filename
        df.to_csv(output_path)
        logging.info(f"Data saved successfully to {output_path}")
        return True
    except Exception as e:
        logging.error(f"Error saving data: {e}")
        return False

def save_trading_days_only(df, symbol=STOCK_SYMBOL):
    """
    filter data to only contain trading days (days when markets are open).
    for stock data this is already the case, but we add extra features.
    """
    try:
        # copy dataframe to avoid changes to the original
        trading_days_df = df.copy()
        
        # add information about whether it's a trading day
        trading_days_df['is_trading_day'] = 1
        
        # calculate number of days since last trading day
        trading_days_df['days_since_last_trading'] = (trading_days_df.index.to_series().diff().dt.days)
        trading_days_df['days_since_last_trading'] = trading_days_df['days_since_last_trading'].fillna(0)
        
        # calculate percentage change compared to previous day
        if 'close' in trading_days_df.columns:
            trading_days_df['pct_change'] = trading_days_df['close'].pct_change() * 100
            trading_days_df['pct_change'] = trading_days_df['pct_change'].fillna(0)
        
        # save to csv
        output_path = RAW_STOCKS_DIR / TARGET_TRADING_DAYS_FILENAME
        trading_days_df.to_csv(output_path)
        logging.info(f"Trading days data saved successfully to {output_path} ({len(trading_days_df)} trading days)")
        
        return True
    except Exception as e:
        logging.error(f"Error saving trading days data: {e}")
        return False

def collect_macro_data():
    """collect macroeconomic data from alpha vantage."""
    all_macro_data = {}
    all_success = True
    
    # go through each macroeconomic indicator
    for symbol, info in MACRO_INDICATORS.items():
        logging.info(f"Collecting {info['description']} data...")
        
        # create parameters based on function type
        if info['function'] == "FX_DAILY":
            # currency rates have different parameters
            from_currency, to_currency = symbol.split('/')
            params = {
                "function": info['function'],
                "from_symbol": from_currency,
                "to_symbol": to_currency,
                "outputsize": OUTPUT_SIZE,
                "apikey": ALPHA_VANTAGE_API_KEY
            }
        else:
            # stocks and indices
            params = {
                "function": info['function'],
                "symbol": symbol,
                "outputsize": OUTPUT_SIZE,
                "apikey": ALPHA_VANTAGE_API_KEY
            }
        
        # get data
        data = fetch_data(params)
        
        if data:
            # process data based on function type
            if info['function'] == "FX_DAILY":
                df = process_fx_data(data)
            else:
                # determine time series key based on function
                if info['function'] == "TIME_SERIES_DAILY":
                    time_series_key = "Time Series (Daily)"
                elif info['function'] == "TIME_SERIES_WEEKLY":
                    time_series_key = "Weekly Time Series"
                elif info['function'] == "TIME_SERIES_MONTHLY":
                    time_series_key = "Monthly Time Series"
                else:
                    logging.error(f"Unsupported function: {info['function']}")
                    all_success = False
                    continue
                    
                df = process_stock_data(data, time_series_key)
            
            if df is not None:
                # save individual indicator
                filename = f"{info['name']}_daily.csv"
                save_data(df, filename, RAW_MACRO_DIR)
                
                # add to combined dataframe
                # we only keep 'close' column for simplicity
                all_macro_data[info['name']] = df['close']
                
                logging.info(f"Added {info['name']} to macro economic dataset")
                
                # give the api a break for rate limits
                time.sleep(RETRY_DELAY)
            else:
                logging.error(f"Failed to process {info['name']} data")
                all_success = False
        else:
            logging.error(f"Failed to fetch {info['name']} data")
            all_success = False
    
    # combine all indicators into one dataframe
    if all_macro_data:
        macro_df = pd.DataFrame(all_macro_data)
        
        # handle missing values
        macro_df = macro_df.ffill().bfill()
        
        # add is_trading_day column
        macro_df['is_trading_day'] = 1
        
        # save combined macroeconomic dataset
        save_data(macro_df, MACRO_DAILY_FILENAME, RAW_MACRO_DIR)
        logging.info(f"Combined macro economic data saved with {len(macro_df.columns)} indicators")
    else:
        # if all api calls failed, log error
        logging.error("Failed to collect any macro economic data. Pipeline cannot continue without valid macro data.")
        all_success = False  # no data was collected, mark as failed
    
    return all_success

def collect_vestas_data():
    """collect vestas stock data from alpha vantage."""
    success = False
    
    logging.info(f"Attempting to fetch Vestas data using symbol {STOCK_SYMBOL}")
        
    # get daily data
    daily_params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": STOCK_SYMBOL,
        "outputsize": OUTPUT_SIZE,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    
    daily_data = fetch_data(daily_params)
    
    if daily_data:
        daily_df = process_stock_data(daily_data, "Time Series (Daily)")
        if daily_df is not None and save_data(daily_df, TARGET_DAILY_FILENAME, RAW_STOCKS_DIR):
            logging.info(f"Daily Stock Data Collection Completed Successfully with symbol {STOCK_SYMBOL}")
            # also save version with only trading days (and extra features)
            save_trading_days_only(daily_df)
            success = True
            
            # give the api a break before next call
            time.sleep(RETRY_DELAY)
            
            # get weekly data
            weekly_params = {
                "function": "TIME_SERIES_WEEKLY",
                "symbol": STOCK_SYMBOL,
                "apikey": ALPHA_VANTAGE_API_KEY
            }
            
            weekly_data = fetch_data(weekly_params)
            
            if weekly_data:
                weekly_df = process_stock_data(weekly_data, "Weekly Time Series")
                if weekly_df is not None and save_data(weekly_df, TARGET_WEEKLY_FILENAME, RAW_STOCKS_DIR):
                    logging.info(f"Weekly Stock Data Collection Completed Successfully with symbol {STOCK_SYMBOL}")
            
            # give the api a break before next call
            time.sleep(RETRY_DELAY)
            
            # get monthly data
            monthly_params = {
                "function": "TIME_SERIES_MONTHLY",
                "symbol": STOCK_SYMBOL,
                "apikey": ALPHA_VANTAGE_API_KEY
            }
            
            monthly_data = fetch_data(monthly_params)
            
            if monthly_data:
                monthly_df = process_stock_data(monthly_data, "Monthly Time Series")
                if monthly_df is not None and save_data(monthly_df, TARGET_MONTHLY_FILENAME, RAW_STOCKS_DIR):
                    logging.info(f"Monthly Stock Data Collection Completed Successfully with symbol {STOCK_SYMBOL}")
        else:
            logging.error(f"Failed to process or save data for symbol {STOCK_SYMBOL}")
    else:
        logging.error(f"Failed to fetch data for symbol {STOCK_SYMBOL}")
    
    # if the symbol didn't work, log an error
    if not success:
        logging.error("Failed to collect Vestas stock data. Pipeline cannot continue without valid stock data.")
    
    return success

def main():
    """main function to run the data collection process."""
    logging.info("=== Starting Stock and Macro Data Collection ===")
    
    # collect vestas stock data
    logging.info("=== Starting Vestas Stock Data Collection ===")
    vestas_success = collect_vestas_data()
    
    if not vestas_success:
        logging.error("Vestas stock data collection failed. Pipeline cannot continue.")
        return False
    
    # collect macroeconomic data
    logging.info("=== Starting Macro Economic Data Collection ===")
    macro_success = collect_macro_data()
    
    if not macro_success:
        logging.error("Macro economic data collection failed. Pipeline cannot continue.")
        return False
    
    # If we reach here, both steps succeeded
    logging.info("=== All Data Collection Completed Successfully ===")
    return True

if __name__ == "__main__":
    main() 