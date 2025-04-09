import requests
import pandas as pd
import logging
import sys
from pathlib import Path
import time
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Determine project root based on script location
# Assumes the script is in src/pipeline
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "crypto"  # Updated path for cryptocurrency data
API_BASE_URL = "https://api.coingecko.com/api/v3"
# Example: Fetch Bitcoin data in USD for the last year
COIN_ID = "bitcoin"
VS_CURRENCY = "usd"
DAYS = 365 # Number of days of historical data
TARGET_FILENAME = f"{COIN_ID}_{VS_CURRENCY}_365d.csv"  # Simplified filename
TARGET_TRADING_DAYS_FILENAME = f"{COIN_ID}_{VS_CURRENCY}_trading_days.csv"  # Ny fil med kun handelsdage
API_ENDPOINT = f"/coins/{COIN_ID}/market_chart"
MAX_RETRIES = 3
RETRY_DELAY = 5 # seconds

# Ensure data directory exists
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

def get_trading_days(start_date, end_date):
    """
    Genererer en liste over handelsdage (trading days), hvor den amerikanske børs er åben.
    Ekskluderer weekender og amerikanske helligdage.
    
    Args:
        start_date: Startdato som string 'YYYY-MM-DD' eller datetime
        end_date: Slutdato som string 'YYYY-MM-DD' eller datetime
        
    Returns:
        DatetimeIndex med handelsdage
    """
    # Konverter til datetime hvis nødvendigt
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
        
    # Definer amerikanske helligdage
    us_holidays = USFederalHolidayCalendar()
    holidays = us_holidays.holidays(start=start_date, end=end_date)
    
    # Definer en business day, der ekskluderer weekender og helligdage
    business_days = CustomBusinessDay(calendar=us_holidays)
    
    # Generer en liste over handelsdage
    trading_days = pd.date_range(start=start_date, end=end_date, freq=business_days)
    
    logging.info(f"Genererede {len(trading_days)} handelsdage mellem {start_date.strftime('%Y-%m-%d')} og {end_date.strftime('%Y-%m-%d')}")
    
    return trading_days

def fetch_data(endpoint: str, params: dict) -> dict | None:
    """Fetches data from the CoinGecko API with retry logic."""
    url = f"{API_BASE_URL}{endpoint}"
    for attempt in range(MAX_RETRIES):
        try:
            logging.info(f"Attempting to fetch data from {url} (Attempt {attempt + 1}/{MAX_RETRIES})")
            response = requests.get(url, params=params, timeout=30) # Increased timeout
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            logging.info("Data fetched successfully.")
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.warning(f"Request failed: {e}. Retrying in {RETRY_DELAY} seconds...")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                logging.error("Max retries reached. Failed to fetch data.")
                return None
    return None # Should not be reached if loop completes, but added for clarity

def save_data(data: dict, filename: Path) -> bool:
    """Processes and saves the fetched data to a CSV file."""
    if not data or 'prices' not in data or 'market_caps' not in data or 'total_volumes' not in data:
        logging.error("Incomplete data received from API. Cannot save.")
        return False

    try:
        prices_df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        market_caps_df = pd.DataFrame(data['market_caps'], columns=['timestamp', 'market_cap'])
        total_volumes_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'total_volume'])

        # Merge dataframes on timestamp
        df = pd.merge(prices_df, market_caps_df, on='timestamp', how='outer')
        df = pd.merge(df, total_volumes_df, on='timestamp', how='outer')

        # Convert timestamp (milliseconds) to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Set timestamp as index
        df.set_index('timestamp', inplace=True)

        # Remove potential duplicate timestamps if any (though unlikely from API)
        df = df[~df.index.duplicated(keep='first')]

        # Sort by time
        df.sort_index(inplace=True)

        # Save to CSV
        output_path = RAW_DATA_DIR / filename
        df.to_csv(output_path)
        logging.info(f"Data saved successfully to {output_path}")
        
        # Gem også en version, der kun indeholder handelsdage
        save_trading_days_only(df)
        
        return True
    except KeyError as e:
        logging.error(f"Missing expected key in API response: {e}")
        return False
    except Exception as e:
        logging.error(f"Error processing or saving data: {e}")
        return False

def save_trading_days_only(df):
    """
    Filtrer data til kun at indeholde handelsdage (børsåbningsdage).
    Gemmer en separat fil med kun handelsdage.
    """
    try:
        # Find start- og slutdato fra data
        start_date = df.index.min()
        end_date = df.index.max()
        
        # Få liste over handelsdage i perioden
        trading_days = get_trading_days(start_date, end_date)
        
        # Filtrer data til kun at indeholde handelsdage
        trading_days_df = df.reindex(trading_days).dropna(how='all')
        
        # Hvis der er manglende data, brug forward/backward fill
        if trading_days_df.isnull().any().any():
            logging.info("Udfylder manglende værdier for handelsdage ved hjælp af forward/backward fill")
            trading_days_df = trading_days_df.ffill().bfill()
        
        # Tilføj information om det er en handelsdag
        trading_days_df['is_trading_day'] = 1
        
        # Beregn antal dage siden sidste handelsdag (kan være nyttigt for modellen)
        trading_days_df['days_since_last_trading'] = (trading_days_df.index.to_series().diff().dt.days)
        trading_days_df['days_since_last_trading'] = trading_days_df['days_since_last_trading'].fillna(0)
        
        # Gem til CSV
        output_path = RAW_DATA_DIR / TARGET_TRADING_DAYS_FILENAME
        trading_days_df.to_csv(output_path)
        logging.info(f"Trading days data saved successfully to {output_path} ({len(trading_days_df)} trading days)")
        
        return True
    except Exception as e:
        logging.error(f"Error saving trading days data: {e}")
        return False

def main():
    """Main function to run the data ingestion process."""
    logging.info("--- Starting Cryptocurrency Data Collection ---")
    params = {
        'vs_currency': VS_CURRENCY,
        'days': str(DAYS),
        'interval': 'daily' # or 'hourly' if needed, check API limits
    }
    raw_data = fetch_data(API_ENDPOINT, params)

    if raw_data:
        if save_data(raw_data, TARGET_FILENAME):
            logging.info("--- Cryptocurrency Data Collection Completed Successfully ---")
        else:
            logging.error("--- Cryptocurrency Data Collection Failed (Save Error) ---")
            sys.exit(1) # Exit with error code
    else:
        logging.error("--- Cryptocurrency Data Collection Failed (Fetch Error) ---")
        sys.exit(1) # Exit with error code

if __name__ == "__main__":
    main() 