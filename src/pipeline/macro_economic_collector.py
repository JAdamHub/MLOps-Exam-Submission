import pandas as pd
import requests
from datetime import datetime, timedelta
import yfinance as yf
import logging
from pathlib import Path
import os
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MacroDataCollector:
    def __init__(self):
        self.base_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "macro"
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.fred_api_key = os.getenv('FRED_API_KEY')
        if not self.fred_api_key or self.fred_api_key == "your_fred_api_key_here":
            logger.error("FRED_API_KEY not found or set to default value. Please set a valid API key in the .env file.")
            raise ValueError("FRED_API_KEY not found or set to default value. Please set a valid API key in the .env file.")

    def get_fred_data(self, series_id, start_date=None):
        """Fetches data from FRED (Federal Reserve Economic Data)"""
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        url = f"https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": self.fred_api_key,
            "file_type": "json",
            "observation_start": start_date
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raise an exception for HTTP errors
            data = response.json()
            df = pd.DataFrame(data['observations'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            # Ensure dates are timezone-naive
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            return df.set_index('date')['value']
        except Exception as e:
            logger.error(f"Error fetching FRED data for {series_id}: {e}")
            raise

    def get_yahoo_finance_data(self, symbol, start_date=None):
        """Fetches data from Yahoo Finance"""
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date)
            # Ensure index is timezone-naive
            df.index = df.index.tz_localize(None)
            return df['Close']
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data for {symbol}: {e}")
            raise

    def collect_all_macro_data(self):
        """Collects all macroeconomic data"""
        data = {}
        
        # CPI (Consumer Price Index)
        data['cpi'] = self.get_fred_data('CPIAUCSL')
        
        # Federal Funds Rate
        data['fed_rate'] = self.get_fred_data('DFF')
        
        # DXY (Dollar Index)
        data['dxy'] = self.get_yahoo_finance_data('DX-Y.NYB')
        
        # S&P 500
        data['sp500'] = self.get_yahoo_finance_data('^GSPC')
        
        # Ensure all time series have timezone-naive indices
        for key, series in data.items():
            if series is not None and hasattr(series.index, 'tz'):
                data[key].index = series.index.tz_localize(None)
        
        # Combine all data into a DataFrame
        df = pd.DataFrame(data)
        
        # Calculate percentage changes for each series (except interest rate)
        for column in df.columns:
            if column != 'fed_rate':  # Fed rate is already in percentage
                df[f'{column}_pct_change'] = df[column].pct_change(periods=1, fill_method=None)
        
        # Save to CSV
        output_file = self.base_path / "macro_economic.csv"
        df.to_csv(output_file)
        logger.info(f"Macroeconomic data saved to {output_file}")
        
        return df

    def update_daily(self):
        """Performs daily update of macroeconomic data"""
        try:
            df = self.collect_all_macro_data()
            logger.info("Daily macroeconomic data update completed")
            return True
        except Exception as e:
            logger.error(f"Error during daily update: {e}")
            raise

if __name__ == "__main__":
    collector = MacroDataCollector()
    collector.collect_all_macro_data() 