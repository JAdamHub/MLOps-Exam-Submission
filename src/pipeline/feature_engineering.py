import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import joblib

# Konfigurer logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Determine project root based on script location
# Assumes the script is in src/pipeline
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERMEDIATE_PREPROCESSED_DIR = PROJECT_ROOT / "data" / "intermediate" / "preprocessed"  # Updated path
PROCESSED_FEATURES_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"

# Input file from preprocessing step
INPUT_FILENAME = "bitcoin_macro_preprocessed_trading_days.csv"  # Opdateret til trading days version
# Output file
OUTPUT_FILENAME = "bitcoin_features_trading_days.csv"  # Opdateret til trading days version

# Feature Engineering Parameters
PRICE_COLUMN = 'price' # Assuming 'price' is the column name after preprocessing
LAG_PERIODS = [1, 3, 7] # Lag periods in days
SMA_WINDOWS = [7, 30] # Simple Moving Average windows in days
VOLATILITY_WINDOW = 14 # Window for rolling standard deviation (volatility)
FORECAST_HORIZONS = [1, 3, 7] # Forudsig prisen 1, 3 og 7 dage frem
# TARGET_SHIFT = -1 # Predict next day's price movement - ikke brugt længere

# Ensure output directory exists
PROCESSED_FEATURES_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE_PATH = INTERMEDIATE_PREPROCESSED_DIR / INPUT_FILENAME
OUTPUT_FILE_PATH = PROCESSED_FEATURES_DIR / OUTPUT_FILENAME

def load_data(filepath: Path) -> pd.DataFrame | None:
    """Load preprocessed data."""
    try:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        logging.info(f"Processed data loaded successfully from {filepath}")
        return df
    except Exception as e:
        logging.error(f"Error loading processed data: {e}")
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

        # 5. Target Variables: Faktiske priser i fremtiden for forskellige tidshorisonter
        for horizon in FORECAST_HORIZONS:
            target_col = f'price_target_{horizon}d'
            features_df[target_col] = df[PRICE_COLUMN].shift(-horizon)
            logging.debug(f"Created target variable: {target_col}")

        # Fjern den gamle binære target
        if 'target_price_up' in features_df.columns:
            features_df.drop(columns=['target_price_up'], inplace=True)
            logging.debug("Removed old binary target variable")

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
    # Check for macroeconomic columns
    macro_columns = [
        'treasury_10y', 'treasury_30y', 'treasury_5y', 'vix', 
        'fed_rate', 'dxy', 'sp500', 'nasdaq', 'dow', 
        'gold', 'oil', 'eurusd'
    ]
    available_columns = [col for col in macro_columns if col in data.columns]
    
    if not available_columns:
        logging.warning("No macroeconomic indicators found in dataset")
        return data
    
    logging.info(f"Found {len(available_columns)} macroeconomic indicators: {available_columns}")
    
    # Treasury yield features
    for treasury in ['treasury_10y', 'treasury_30y', 'treasury_5y']:
        if treasury in data.columns:
            data[f'{treasury}_lag_1'] = data[treasury].shift(1)
            data[f'{treasury}_change'] = data[treasury] - data[f'{treasury}_lag_1']
            data[f'{treasury}_diff_7d'] = data[treasury] - data[treasury].shift(7)
            data[f'{treasury}_ma7'] = data[treasury].rolling(window=7).mean()
            data[f'{treasury}_volatility'] = data[treasury].rolling(window=14).std()
    
    # Fed rate features
    if 'fed_rate' in data.columns:
        data['fed_rate_lag_1'] = data['fed_rate'].shift(1)
        data['fed_rate_change'] = data['fed_rate'] - data['fed_rate_lag_1']
        data['fed_rate_diff_7d'] = data['fed_rate'] - data['fed_rate'].shift(7)
        data['fed_rate_ma7'] = data['fed_rate'].rolling(window=7).mean()
    
    # Volatility index features
    if 'vix' in data.columns:
        data['vix_lag_1'] = data['vix'].shift(1)
        data['vix_change'] = data['vix'] - data['vix_lag_1']
        data['vix_ma7'] = data['vix'].rolling(window=7).mean()
        data['vix_ma30'] = data['vix'].rolling(window=30).mean()
        data['vix_rsi_14'] = calculate_rsi(data['vix'], periods=14)
    
    # DXY (Dollar Index) features
    if 'dxy' in data.columns:
        data['dxy_lag_1'] = data['dxy'].shift(1)
        data['dxy_change'] = data['dxy'] - data['dxy_lag_1']
        data['dxy_ma7'] = data['dxy'].rolling(window=7).mean()
        data['dxy_ma30'] = data['dxy'].rolling(window=30).mean()
        data['dxy_volatility'] = data['dxy'].rolling(window=14).std()
        data['dxy_rsi_14'] = calculate_rsi(data['dxy'], periods=14)
    
    # Stock market indices features (S&P 500, NASDAQ, Dow Jones)
    for index in ['sp500', 'nasdaq', 'dow']:
        if index in data.columns:
            data[f'{index}_lag_1'] = data[index].shift(1)
            data[f'{index}_change'] = data[index] - data[f'{index}_lag_1']
            data[f'{index}_ma7'] = data[index].rolling(window=7).mean()
            data[f'{index}_ma30'] = data[index].rolling(window=30).mean()
            data[f'{index}_volatility'] = data[index].rolling(window=14).std()
            data[f'{index}_rsi_14'] = calculate_rsi(data[index], periods=14)
            
            # Correlation between Bitcoin and stock indices
            data[f'btc_{index}_corr_30d'] = data['price'].rolling(window=30).corr(data[index])
    
    # Commodities features (Gold, Oil)
    for commodity in ['gold', 'oil']:
        if commodity in data.columns:
            data[f'{commodity}_lag_1'] = data[commodity].shift(1)
            data[f'{commodity}_change'] = data[commodity] - data[f'{commodity}_lag_1']
            data[f'{commodity}_ma7'] = data[commodity].rolling(window=7).mean()
            data[f'{commodity}_volatility'] = data[commodity].rolling(window=14).std()
            data[f'{commodity}_rsi_14'] = calculate_rsi(data[commodity], periods=14)
            
            # Correlation between Bitcoin and commodities
            data[f'btc_{commodity}_corr_30d'] = data['price'].rolling(window=30).corr(data[commodity])
    
    # Forex features (EUR/USD)
    if 'eurusd' in data.columns:
        data['eurusd_lag_1'] = data['eurusd'].shift(1)
        data['eurusd_change'] = data['eurusd'] - data['eurusd_lag_1']
        data['eurusd_ma7'] = data['eurusd'].rolling(window=7).mean()
        data['eurusd_volatility'] = data['eurusd'].rolling(window=14).std()
        data['eurusd_rsi_14'] = calculate_rsi(data['eurusd'], periods=14)
        
        # Correlation between Bitcoin and EUR/USD
        data['btc_eurusd_corr_30d'] = data['price'].rolling(window=30).corr(data['eurusd'])
    
    # Cross-asset correlations
    if 'dxy' in data.columns and 'eurusd' in data.columns:
        data['dxy_eurusd_corr_30d'] = data['dxy'].rolling(window=30).corr(data['eurusd'])
    
    if 'gold' in data.columns and 'dxy' in data.columns:
        data['gold_dxy_corr_30d'] = data['gold'].rolling(window=30).corr(data['dxy'])
    
    if 'oil' in data.columns and 'dxy' in data.columns:
        data['oil_dxy_corr_30d'] = data['oil'].rolling(window=30).corr(data['dxy'])
    
    # Feature ratios
    if 'vix' in data.columns and 'sp500' in data.columns:
        data['vix_to_sp500'] = data['vix'] / data['sp500']
    
    if 'gold' in data.columns and 'sp500' in data.columns:
        data['gold_to_sp500'] = data['gold'] / data['sp500']
    
    logging.info(f"Generated {len(data.columns) - len(available_columns)} macroeconomic features")
    
    return data

def main():
    """Main function to run the feature engineering process."""
    logging.info("Starting feature engineering")
    
    try:
        # Indlæs præprocesseret data med både Bitcoin og makroøkonomisk data
        input_file = INPUT_FILE_PATH
        
        # Indlæs data
        df = load_data(input_file)
        if df is None:
            return
        
        logging.info("Starting feature engineering...")
        
        # Generer basis-features
        features_df = create_features(df)
        if features_df is None:
            return
        
        # Generer market features
        features_df = calculate_market_features(features_df)
        
        # Generer makroøkonomiske features
        features_df = calculate_macro_features(features_df)
        
        # Drop rows with NaNs after all feature generation
        initial_rows = len(features_df)
        features_df.dropna(inplace=True)
        final_rows = len(features_df)
        logging.info(f"Dropped {initial_rows - final_rows} rows due to NaN values from feature creation.")
        
        # Save features
        output_file = PROCESSED_FEATURES_DIR / OUTPUT_FILENAME
        features_df.to_csv(output_file, index=True)
        logging.info(f"Features data saved successfully to {output_file}")
        
        logging.info("--- Feature Engineering Completed Successfully ---")
        
    except Exception as e:
        logging.error(f"Fejl i main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
