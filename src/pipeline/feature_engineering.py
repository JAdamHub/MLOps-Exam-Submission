import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Konfigurer logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Klasse til at generere features fra rå data."""
    
    def __init__(self):
        """Initialiserer FeatureEngineer med standard parametre."""
        self.price_column = 'price'
        self.volume_column = 'total_volume'
        self.market_cap_column = 'market_cap'
        
        # Tekniske indikatorer parametre
        self.lag_periods = [1, 3, 7]
        self.sma_windows = [7, 30]
        self.volatility_window = 14
        self.rsi_periods = [7, 14, 21]
        self.macd_params = {'fast': 12, 'slow': 26, 'signal': 9}
        self.bb_window = 20
        
        # Target variabel
        self.target_shift = -1  # Prædiker næste dags prisbevægelse
    
    def calculate_rsi(self, data: pd.Series, period: int) -> pd.Series:
        """Beregner Relative Strength Index (RSI)."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, data: pd.Series) -> tuple:
        """Beregner Moving Average Convergence Divergence (MACD)."""
        exp1 = data.ewm(span=self.macd_params['fast'], adjust=False).mean()
        exp2 = data.ewm(span=self.macd_params['slow'], adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=self.macd_params['signal'], adjust=False).mean()
        histogram = macd - signal
        return macd, signal, histogram
    
    def calculate_bollinger_bands(self, data: pd.Series) -> tuple:
        """Beregner Bollinger Bands."""
        sma = data.rolling(window=self.bb_window).mean()
        std = data.rolling(window=self.bb_window).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        width = (upper - lower) / sma
        position = (data - lower) / (upper - lower)
        return upper, sma, lower, width, position
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Genererer features fra input data.
        
        Args:
            df: DataFrame med rå data
            
        Returns:
            DataFrame med genererede features
        """
        try:
            logger.info("Starter feature engineering...")
            features_df = df.copy()
            
            # 1. Lagged Features
            for lag in self.lag_periods:
                features_df[f'price_lag_{lag}'] = df[self.price_column].shift(lag)
            
            # 2. Moving Averages
            for window in self.sma_windows:
                features_df[f'price_sma_{window}'] = df[self.price_column].rolling(window=window).mean()
                features_df[f'volume_sma_{window}'] = df[self.volume_column].rolling(window=window).mean()
            
            # 3. Volatility
            features_df['price_volatility_14'] = df[self.price_column].rolling(window=self.volatility_window).std()
            
            # 4. RSI
            for period in self.rsi_periods:
                features_df[f'rsi_{period}'] = self.calculate_rsi(df[self.price_column], period)
            
            # 5. MACD
            macd, signal, histogram = self.calculate_macd(df[self.price_column])
            features_df['macd'] = macd
            features_df['macd_signal'] = signal
            features_df['macd_histogram'] = histogram
            
            # 6. Bollinger Bands
            upper, middle, lower, width, position = self.calculate_bollinger_bands(df[self.price_column])
            features_df['bb_upper'] = upper
            features_df['bb_middle'] = middle
            features_df['bb_lower'] = lower
            features_df['bb_width'] = width
            features_df['bb_position'] = position
            
            # 7. Volume Features
            features_df['volume_ratio'] = df[self.volume_column] / features_df['volume_sma_7']
            features_df['volume_ratio_30'] = df[self.volume_column] / features_df['volume_sma_30']
            features_df['volume_momentum'] = df[self.volume_column].pct_change(periods=1)
            
            # 8. Price Momentum
            for period in [1, 7, 30, 90]:
                features_df[f'price_momentum_{period}'] = df[self.price_column].pct_change(periods=period)
            
            # 9. Market Features
            features_df['market_cap_to_volume'] = df[self.market_cap_column] / df[self.volume_column]
            features_df['market_cap_momentum_1'] = df[self.market_cap_column].pct_change(periods=1)
            features_df['market_cap_momentum_7'] = df[self.market_cap_column].pct_change(periods=7)
            features_df['market_cap_momentum_30'] = df[self.market_cap_column].pct_change(periods=30)
            
            # 10. Time Features
            features_df['day_of_week'] = df.index.dayofweek
            features_df['month'] = df.index.month
            features_df['year'] = df.index.year
            features_df['day_of_month'] = df.index.day
            features_df['is_weekend'] = features_df['day_of_week'].isin([5, 6]).astype(int)
            
            # 11. Target Variable
            features_df['target'] = (df[self.price_column].shift(self.target_shift) > df[self.price_column]).astype(int)
            
            # Fjern rækker med NaN værdier
            initial_rows = len(features_df)
            features_df.dropna(inplace=True)
            final_rows = len(features_df)
            logger.info(f"Fjernet {initial_rows - final_rows} rækker med NaN værdier")
            
            return features_df
            
        except Exception as e:
            logger.error(f"Fejl under feature engineering: {str(e)}")
            raise

def main():
    """Hovedfunktion til at køre feature engineering."""
    try:
        # Initialiser FeatureEngineer
        engineer = FeatureEngineer()
        
        # Indlæs rå data
        data_dir = Path(__file__).resolve().parents[2] / "data"
        raw_data_path = data_dir / "raw" / "bitcoin_usd_365d_raw.csv"
        features_data_path = data_dir / "features" / "bitcoin_usd_365d_features.csv"
        
        # Opret output mappe hvis den ikke findes
        features_data_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Indlæs data
        df = pd.read_csv(raw_data_path, index_col='timestamp', parse_dates=True)
        logger.info(f"Indlæste data fra {raw_data_path}")
        
        # Generer features
        features_df = engineer.create_features(df)
        
        # Gem features
        features_df.to_csv(features_data_path)
        logger.info(f"Gemte features til {features_data_path}")
        
    except Exception as e:
        logger.error(f"Fejl i main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
