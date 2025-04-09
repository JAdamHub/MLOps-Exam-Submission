import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Determine project root based on script location
# Assumes the script is in src/pipeline
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERMEDIATE_PREPROCESSED_DIR = PROJECT_ROOT / "data" / "intermediate" / "preprocessed"  # Updated path
PROCESSED_FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "features"  # Updated path

# Input file from preprocessing step
INPUT_FILENAME = "bitcoin_macro_preprocessed.csv"  # Updated filename
# Output file
OUTPUT_FILENAME = "bitcoin_features.csv"  # Updated filename

# Feature Engineering Parameters
PRICE_COLUMN = 'price' # Assuming 'price' is the column name after preprocessing
LAG_PERIODS = [1, 3, 7] # Lag periods in days
SMA_WINDOWS = [7, 30] # Simple Moving Average windows in days
VOLATILITY_WINDOW = 14 # Window for rolling standard deviation (volatility)
TARGET_SHIFT = -1 # Predict next day's price movement

# Ensure output directory exists
PROCESSED_FEATURES_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE_PATH = INTERMEDIATE_PREPROCESSED_DIR / INPUT_FILENAME
OUTPUT_FILE_PATH = PROCESSED_FEATURES_DIR / OUTPUT_FILENAME

def load_data(filepath: Path) -> pd.DataFrame | None:
    """Loads data from a CSV file."""
    if not filepath.exists():
        logging.error(f"Input file not found: {filepath}")
        return None
    try:
        df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
        logging.info(f"Processed data loaded successfully from {filepath}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {filepath}: {e}")
        return None

def create_features(df: pd.DataFrame) -> pd.DataFrame | None:
    """Generates features for the model."""
    if PRICE_COLUMN not in df.columns:
        logging.error(f"Required column '{PRICE_COLUMN}' not found in the DataFrame.")
        return None

    try:
        logging.info("Starting feature engineering...")
        features_df = df.copy()

        # 1. Lagged Features
        for lag in LAG_PERIODS:
            features_df[f'{PRICE_COLUMN}_lag_{lag}'] = df[PRICE_COLUMN].shift(lag)
            logging.debug(f"Created lag feature: {PRICE_COLUMN}_lag_{lag}")

        # 2. Moving Averages
        for window in SMA_WINDOWS:
            features_df[f'{PRICE_COLUMN}_sma_{window}'] = df[PRICE_COLUMN].rolling(window=window, min_periods=1).mean()
            logging.debug(f"Created SMA feature: {PRICE_COLUMN}_sma_{window}")

        # 3. Volatility
        features_df[f'{PRICE_COLUMN}_volatility_{VOLATILITY_WINDOW}'] = df[PRICE_COLUMN].rolling(window=VOLATILITY_WINDOW, min_periods=1).std()
        logging.debug(f"Created volatility feature: {PRICE_COLUMN}_volatility_{VOLATILITY_WINDOW}")

        # 4. Time-based Features
        features_df['day_of_week'] = df.index.dayofweek
        features_df['month'] = df.index.month
        features_df['year'] = df.index.year # Keep year if useful for longer trends
        logging.debug("Created time-based features: day_of_week, month, year")

        # 5. Target Variable: Price up (1) or down/same (0) tomorrow?
        features_df['target_price_next_day'] = df[PRICE_COLUMN].shift(TARGET_SHIFT)
        features_df['target_price_up'] = (features_df['target_price_next_day'] > df[PRICE_COLUMN]).astype(int)
        features_df.drop(columns=['target_price_next_day'], inplace=True) # Drop the intermediate column
        logging.debug("Created target variable: target_price_up")

        # Drop rows with NaNs introduced by lags/rolling windows/target shift
        initial_rows = len(features_df)
        features_df.dropna(inplace=True)
        final_rows = len(features_df)
        logging.info(f"Dropped {initial_rows - final_rows} rows due to NaN values from feature creation.")

        if features_df.empty:
            logging.error("DataFrame is empty after feature engineering and NaN removal.")
            return None

        logging.info("Feature engineering completed.")
        return features_df

    except Exception as e:
        logging.error(f"Error during feature engineering: {e}", exc_info=True)
        return None

def save_features(df: pd.DataFrame, filepath: Path):
    """Saves the features DataFrame to a CSV file."""
    try:
        df.to_csv(filepath)
        logging.info(f"Features data saved successfully to {filepath}")
    except Exception as e:
        logging.error(f"Error saving features data to {filepath}: {e}")
        sys.exit(1)

def calculate_rsi(data, periods=14):
    """Calculate RSI (Relative Strength Index)"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger_bands(data, window=20):
    """Calculate Bollinger Bands"""
    sma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return upper_band, sma, lower_band

def calculate_adx(data, period=14):
    """Calculate Average Directional Index (ADX)"""
    high = data['high']
    low = data['low']
    close = data['price']
    
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    # Calculate Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Calculate Directional Indicators
    plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr
    
    # Calculate ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return adx, plus_di, minus_di

def calculate_stochastic(data, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator"""
    low_min = data['low'].rolling(window=k_period).min()
    high_max = data['high'].rolling(window=k_period).max()
    
    k = 100 * (data['price'] - low_min) / (high_max - low_min)
    d = k.rolling(window=d_period).mean()
    
    return k, d

def calculate_market_features(data):
    """Calculate market-based features"""
    # Market cap to volume ratio (with safety check)
    data['market_cap_to_volume'] = np.where(
        data['total_volume'] != 0,
        data['market_cap'] / data['total_volume'],
        0
    )
    
    # Market cap momentum
    data['market_cap_momentum_1'] = data['market_cap'].pct_change(periods=1)
    data['market_cap_momentum_7'] = data['market_cap'].pct_change(periods=7)
    data['market_cap_momentum_30'] = data['market_cap'].pct_change(periods=30)
    
    # Volume to market cap ratio (with safety check)
    data['volume_to_market_cap'] = np.where(
        data['market_cap'] != 0,
        data['total_volume'] / data['market_cap'],
        0
    )
    
    return data

def calculate_macro_features(data):
    """Calculate macroeconomic-based features"""
    # Check if macroeconomic columns exist
    macro_columns = ['cpi', 'fed_rate', 'dxy', 'sp500']
    available_columns = [col for col in macro_columns if col in data.columns]
    
    if not available_columns:
        logging.warning("No macroeconomic indicators found in dataset")
        return data
    
    logging.info(f"Found the following macroeconomic indicators: {available_columns}")
    
    # Fed rate lag features
    if 'fed_rate' in data.columns:
        data['fed_rate_lag_1'] = data['fed_rate'].shift(1)
        data['fed_rate_change'] = data['fed_rate'] - data['fed_rate_lag_1']
        data['fed_rate_diff_7d'] = data['fed_rate'] - data['fed_rate'].shift(7)
    
    # CPI features
    if 'cpi' in data.columns:
        data['cpi_mom'] = data['cpi'].pct_change(periods=1)
        data['cpi_3m'] = data['cpi'].pct_change(periods=90)
        data['cpi_6m'] = data['cpi'].pct_change(periods=180)
    
    # DXY (Dollar Index) features
    if 'dxy' in data.columns:
        data['dxy_change'] = data['dxy'].pct_change(periods=1)
        data['dxy_ma7'] = data['dxy'].rolling(window=7).mean()
        data['dxy_ma30'] = data['dxy'].rolling(window=30).mean()
        data['dxy_volatility'] = data['dxy'].rolling(window=14).std()
        
        # DXY RSI
        data['dxy_rsi_14'] = calculate_rsi(data['dxy'], periods=14)
    
    # S&P 500 features
    if 'sp500' in data.columns:
        data['sp500_change'] = data['sp500'].pct_change(periods=1)
        data['sp500_ma7'] = data['sp500'].rolling(window=7).mean()
        data['sp500_ma30'] = data['sp500'].rolling(window=30).mean()
        data['sp500_volatility'] = data['sp500'].rolling(window=14).std()
        
        # S&P 500 RSI
        data['sp500_rsi_14'] = calculate_rsi(data['sp500'], periods=14)
    
    # Correlation between Bitcoin and S&P 500 (rolling 30-day correlation)
    if 'sp500' in data.columns:
        data['btc_sp500_corr_30d'] = data['price'].rolling(window=30).corr(data['sp500'])
    
    # Correlation between Bitcoin and DXY (rolling 30-day correlation)
    if 'dxy' in data.columns:
        data['btc_dxy_corr_30d'] = data['price'].rolling(window=30).corr(data['dxy'])
    
    return data

def main():
    """Main function to run the feature engineering process."""
    logging.info("--- Starting Feature Engineering ---")
    
    try:
        # Load processed data
        input_file = INTERMEDIATE_PREPROCESSED_DIR / INPUT_FILENAME
        df = pd.read_csv(input_file)
        logging.info(f"Processed data loaded successfully from {input_file}")
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Basic price features
        df['price_lag_1'] = df['price'].shift(1)
        df['price_lag_3'] = df['price'].shift(3)
        df['price_lag_7'] = df['price'].shift(7)
        df['price_sma_7'] = df['price'].rolling(window=7).mean()
        df['price_sma_30'] = df['price'].rolling(window=30).mean()
        df['price_volatility_14'] = df['price'].rolling(window=14).std()
        
        # RSI with multiple periods
        df['rsi_14'] = calculate_rsi(df['price'], periods=14)
        df['rsi_7'] = calculate_rsi(df['price'], periods=7)
        df['rsi_21'] = calculate_rsi(df['price'], periods=21)
        
        # MACD with different parameters
        df['macd'], df['macd_signal'] = calculate_macd(df['price'])
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        df['macd_fast'], df['macd_signal_fast'] = calculate_macd(df['price'], fast=8, slow=17, signal=9)
        
        # Bollinger Bands with different windows
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = calculate_bollinger_bands(df['price'])
        df['bb_width'] = np.where(
            df['bb_middle'] != 0,
            (df['bb_upper'] - df['bb_lower']) / df['bb_middle'],
            0
        )
        df['bb_position'] = np.where(
            (df['bb_upper'] - df['bb_lower']) != 0,
            (df['price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']),
            0.5
        )
        
        # Volume indicators with more windows
        df['volume_sma_7'] = df['total_volume'].rolling(window=7).mean()
        df['volume_sma_30'] = df['total_volume'].rolling(window=30).mean()
        df['volume_ratio'] = np.where(
            df['volume_sma_7'] != 0,
            df['total_volume'] / df['volume_sma_7'],
            1
        )
        df['volume_ratio_30'] = np.where(
            df['volume_sma_30'] != 0,
            df['total_volume'] / df['volume_sma_30'],
            1
        )
        df['volume_momentum'] = df['total_volume'].pct_change(periods=1)
        
        # Price momentum with more periods
        df['price_momentum_1'] = df['price'].pct_change(periods=1)
        df['price_momentum_7'] = df['price'].pct_change(periods=7)
        df['price_momentum_30'] = df['price'].pct_change(periods=30)
        df['price_momentum_90'] = df['price'].pct_change(periods=90)
        
        # Price volatility with different windows
        df['volatility_7'] = df['price'].rolling(window=7).std()
        df['volatility_14'] = df['price'].rolling(window=14).std()
        df['volatility_30'] = df['price'].rolling(window=30).std()
        
        # Market-based features
        df = calculate_market_features(df)
        
        # Macroeconomic features
        df = calculate_macro_features(df)
        
        # Interaction features
        df['price_volatility_ratio'] = np.where(
            df['volatility_30'] != 0,
            df['volatility_7'] / df['volatility_30'],
            1
        )
        df['momentum_volatility_ratio'] = np.where(
            df['volatility_14'] != 0,
            df['price_momentum_7'] / df['volatility_14'],
            0
        )
        df['volume_price_ratio'] = np.where(
            df['price'] != 0,
            df['total_volume'] / df['price'],
            0
        )
        
        # Time-based features with more granularity
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['year'] = df['timestamp'].dt.year
        df['day_of_month'] = df['timestamp'].dt.day
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Target variable (next day's price movement)
        df['target_price_up'] = (df['price'].shift(-1) > df['price']).astype(int)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Replace infinite values with 0
        df = df.replace([np.inf, -np.inf], 0)
        
        logging.info(f"Dropped {len(df)} rows due to NaN values from feature creation.")
        
        # Save features
        output_file = PROCESSED_FEATURES_DIR / OUTPUT_FILENAME
        df.to_csv(output_file, index=False)
        logging.info(f"Features data saved successfully to {output_file}")
        
        logging.info("--- Feature Engineering Completed Successfully ---")
        
    except Exception as e:
        logging.error(f"Error during feature engineering: {e}")
        raise

if __name__ == "__main__":
    main()
