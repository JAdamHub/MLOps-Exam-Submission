import requests
import pandas as pd
import logging
import sys
from pathlib import Path
import time

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
API_ENDPOINT = f"/coins/{COIN_ID}/market_chart"
MAX_RETRIES = 3
RETRY_DELAY = 5 # seconds

# Ensure data directory exists
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

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
        return True
    except KeyError as e:
        logging.error(f"Missing expected key in API response: {e}")
        return False
    except Exception as e:
        logging.error(f"Error processing or saving data: {e}")
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