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
        self.base_path = Path(__file__).resolve().parents[2] / "data" / "macro"
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.fred_api_key = os.getenv('FRED_API_KEY')
        if not self.fred_api_key or self.fred_api_key == "your_fred_api_key_here":
            logger.warning("FRED_API_KEY ikke fundet eller er standard værdi. Genererer mock data.")
            self.use_mock_data = True
        else:
            self.use_mock_data = False

    def get_fred_data(self, series_id, start_date=None):
        """Henter data fra FRED (Federal Reserve Economic Data)"""
        if self.use_mock_data:
            return self._generate_mock_fred_data(series_id)
            
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
            data = response.json()
            df = pd.DataFrame(data['observations'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df['date'] = pd.to_datetime(df['date'])
            return df.set_index('date')['value']
        except Exception as e:
            logger.error(f"Fejl ved hentning af FRED data for {series_id}: {e}")
            return None

    def get_yahoo_finance_data(self, symbol, start_date=None):
        """Henter data fra Yahoo Finance"""
        if self.use_mock_data:
            return self._generate_mock_yahoo_data(symbol)
            
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date)
            return df['Close']
        except Exception as e:
            logger.error(f"Fejl ved hentning af Yahoo Finance data for {symbol}: {e}")
            return None
            
    def _generate_mock_fred_data(self, series_id):
        """Genererer mock data for FRED serier"""
        today = datetime.now()
        dates = [today - timedelta(days=i) for i in range(365)]
        dates.sort()  # Sorter i kronologisk rækkefølge
        
        if series_id == 'CPIAUCSL':  # CPI
            values = np.linspace(280, 300, len(dates))  # Stiger lidt over tid
            # Tilføj lidt støj
            values += np.random.normal(0, 1, len(dates))
        elif series_id == 'DFF':  # Federal Funds Rate
            # Starter ved omkring 5% og falder lidt
            values = np.linspace(5.0, 4.75, len(dates))
            # Tilføj små ændringer
            values += np.random.normal(0, 0.05, len(dates))
        else:
            # Generisk mock data
            values = np.random.normal(100, 10, len(dates))
            # Tilføj trend
            values = values + np.linspace(0, 20, len(dates))
            
        return pd.Series(values, index=dates)
        
    def _generate_mock_yahoo_data(self, symbol):
        """Genererer mock data for Yahoo Finance symboler"""
        today = datetime.now()
        dates = [today - timedelta(days=i) for i in range(365)]
        dates.sort()  # Sorter i kronologisk rækkefølge
        
        if symbol == 'DX-Y.NYB':  # Dollar Index
            # Start omkring 100 og lav en svag stigning
            values = np.linspace(100, 105, len(dates))
            # Tilføj lidt volatilitet
            values += np.random.normal(0, 1, len(dates))
        elif symbol == '^GSPC':  # S&P 500
            # Start omkring 4000 og lav en svag stigning
            values = np.linspace(4000, 4800, len(dates))
            # Tilføj lidt volatilitet
            values += np.random.normal(0, 30, len(dates))
        else:
            # Generisk mock data
            values = np.random.normal(100, 5, len(dates))
            # Tilføj trendy bevægelse
            values = values + np.linspace(0, 30, len(dates))
            
        return pd.Series(values, index=dates)

    def collect_all_macro_data(self):
        """Indsamler alle makroøkonomiske data"""
        data = {}
        
        # CPI (Consumer Price Index)
        data['cpi'] = self.get_fred_data('CPIAUCSL')
        
        # Federal Funds Rate
        data['fed_rate'] = self.get_fred_data('DFF')
        
        # DXY (Dollar Index)
        data['dxy'] = self.get_yahoo_finance_data('DX-Y.NYB')
        
        # S&P 500
        data['sp500'] = self.get_yahoo_finance_data('^GSPC')
        
        # Kombiner alle data
        df = pd.DataFrame(data)
        
        # Beregn ændringer og normaliser
        for column in df.columns:
            if column != 'fed_rate':  # Fed rate er allerede i procent
                df[f'{column}_pct_change'] = df[column].pct_change()
        
        # Gem data
        output_file = self.base_path / "macro_economic_data.csv"
        df.to_csv(output_file)
        logger.info(f"Makroøkonomiske data gemt til {output_file}")
        
        return df

    def update_daily(self):
        """Opdaterer data dagligt"""
        try:
            df = self.collect_all_macro_data()
            logger.info("Daglig opdatering af makroøkonomiske data gennemført")
            return True
        except Exception as e:
            logger.error(f"Fejl ved daglig opdatering: {e}")
            return False

if __name__ == "__main__":
    collector = MacroDataCollector()
    collector.collect_all_macro_data()