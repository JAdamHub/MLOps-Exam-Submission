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
INPUT_FILENAME = "vestas_macro_preprocessed_trading_days.csv"  # Opdateret til Vestas trading days version
# Output file
OUTPUT_FILENAME = "vestas_features_trading_days.csv"  # Opdateret til Vestas trading days version

# Feature Engineering Parameters
PRICE_COLUMN = 'close'  # Aktiekurs bruger typisk "close" som den primære priskolonne
VOLUME_COLUMN = 'volume'  # Volumen kolonne for aktier
LAG_PERIODS = [1, 3, 7]  # Lag periods in days
SMA_WINDOWS = [7, 30]  # Simple Moving Average windows in days
VOLATILITY_WINDOW = 14  # Window for rolling standard deviation (volatility)
FORECAST_HORIZONS = [1, 3, 7]  # Forudsig prisen 1, 3 og 7 dage frem

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
    """Generates features for the stock price model."""
    if PRICE_COLUMN not in df.columns:
        logging.error(f"Required column '{PRICE_COLUMN}' not found in the DataFrame.")
        return None

    try:
        logging.info("Starting feature engineering for Vestas stock data...")
        features_df = df.copy()

        # 1. Lagged Features - flere lags for at fange mere komplekse mønstre
        for lag in LAG_PERIODS + [14, 21, 30]:
            features_df[f'{PRICE_COLUMN}_lag_{lag}'] = df[PRICE_COLUMN].shift(lag)
            logging.debug(f"Created lag feature: {PRICE_COLUMN}_lag_{lag}")

        # 2. Moving Averages - flere vinduer og eksponentielle glidende gennemsnit
        for window in SMA_WINDOWS + [14, 30, 60, 90]:
            # Simple Moving Average (SMA)
            features_df[f'{PRICE_COLUMN}_sma_{window}'] = df[PRICE_COLUMN].rolling(window=window, min_periods=1).mean()
            
            # Exponential Moving Average (EMA)
            features_df[f'{PRICE_COLUMN}_ema_{window}'] = df[PRICE_COLUMN].ewm(span=window, adjust=False).mean()
            
            logging.debug(f"Created SMA and EMA features with window {window}")

        # 3. Volatility - forskellige vinduesstørrelser for at fange kortere og længere volatilitet
        for window in [7, 14, 21, 30]:
            features_df[f'{PRICE_COLUMN}_volatility_{window}'] = df[PRICE_COLUMN].rolling(window=window, min_periods=1).std()
            logging.debug(f"Created volatility feature with window {window}")

        # 4. Avancerede tekniske indikatorer for aktier
        # 4.1 RSI (Relative Strength Index)
        for period in [7, 14, 21]:
            features_df[f'{PRICE_COLUMN}_rsi_{period}'] = calculate_rsi(df[PRICE_COLUMN], periods=period)
            logging.debug(f"Created RSI feature with period {period}")

        # 4.2 MACD (Moving Average Convergence Divergence)
        features_df['price_macd'], features_df['price_macd_signal'], features_df['price_macd_hist'] = calculate_macd(df[PRICE_COLUMN])
        logging.debug("Created MACD features")

        # 4.3 Bollinger Bands
        for window in [20, 30]:
            mid, upper, lower = calculate_bollinger_bands(df[PRICE_COLUMN], window=window)
            features_df[f'price_bb_mid_{window}'] = mid
            features_df[f'price_bb_upper_{window}'] = upper
            features_df[f'price_bb_lower_{window}'] = lower
            
            # Procentvis afstand til båndene (normaliseret)
            features_df[f'price_bb_pct_b_{window}'] = (df[PRICE_COLUMN] - lower) / (upper - lower)
            
            # Båndbredde (volatilitetsmål)
            features_df[f'price_bb_bandwidth_{window}'] = (upper - lower) / mid
            
            logging.debug(f"Created Bollinger Bands features with window {window}")

        # 4.4 Rate of Change (RoC)
        for period in [1, 3, 7, 14, 30]:
            features_df[f'price_roc_{period}'] = ((df[PRICE_COLUMN] / df[PRICE_COLUMN].shift(period)) - 1) * 100
            logging.debug(f"Created Rate of Change feature with period {period}")
            
        # 4.5 Momentum indikatorer
        for period in [3, 7, 14, 30]:
            features_df[f'price_momentum_{period}'] = df[PRICE_COLUMN] - df[PRICE_COLUMN].shift(period)
            logging.debug(f"Created Momentum feature with period {period}")
            
        # 5. Aktiespecifikke indikatorer
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # 5.1 Average True Range (ATR) - volatilitetsindikator
            features_df['atr_14'] = calculate_atr(df, 14)
            
            # 5.2 Money Flow Index (MFI) - volumenbaseret oscillator
            if VOLUME_COLUMN in df.columns:
                features_df['mfi_14'] = calculate_mfi(df['high'], df['low'], df['close'], df[VOLUME_COLUMN], 14)
            
            # 5.3 Stochastic Oscillator
            features_df['stoch_k'], features_df['stoch_d'] = calculate_stochastic(df)
            
            # 5.4 Average Directional Index (ADX) - trendstyrkemåling
            adx_df = calculate_adx(df)
            features_df['adx'] = adx_df['adx']
            features_df['di_plus'] = adx_df['di_plus']
            features_df['di_minus'] = adx_df['di_minus']
            
            logging.debug("Created stock-specific technical indicators")
            
        # 6. Pris-transformationer
        # Log transformation kan hjælpe med at linearisere eksponentielle mønstre
        features_df['price_log'] = np.log1p(df[PRICE_COLUMN])  # log1p håndterer 0-værdier
        
        # Differenser (absolutte ændringer)
        features_df['price_diff_1d'] = df[PRICE_COLUMN].diff(1)
        features_df['price_diff_3d'] = df[PRICE_COLUMN].diff(3)
        features_df['price_diff_7d'] = df[PRICE_COLUMN].diff(7)
        
        # Procentvise ændringer
        features_df['price_pct_change_1d'] = df[PRICE_COLUMN].pct_change(1)
        features_df['price_pct_change_3d'] = df[PRICE_COLUMN].pct_change(3)
        features_df['price_pct_change_7d'] = df[PRICE_COLUMN].pct_change(7)
        
        logging.debug("Created price transformations")
        
        # 7. Time-based Features - udvidet med flere datotransformationer
        features_df['day_of_week'] = df.index.dayofweek
        features_df['day_of_month'] = df.index.day
        features_df['week_of_year'] = df.index.isocalendar().week
        features_df['month'] = df.index.month
        features_df['quarter'] = df.index.quarter
        features_df['year'] = df.index.year
        features_df['is_month_start'] = df.index.is_month_start.astype(int)
        features_df['is_month_end'] = df.index.is_month_end.astype(int)
        features_df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        
        # Sæsonkomponenter med sinus/cosinus transformation (cirkulære features)
        # Disse transformationer hjælper modellen med at forstå periodiske/cykliske mønstre
        features_df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        features_df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
        features_df['day_of_week_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        features_df['day_of_week_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        
        logging.debug("Created advanced time-based features")
        
        # 8. Polynomielle features for de vigtigste indikatorer
        price_lag_1 = features_df[f'{PRICE_COLUMN}_lag_1'].values
        price_lag_7 = features_df[f'{PRICE_COLUMN}_lag_7'].values
        price_sma_7 = features_df[f'{PRICE_COLUMN}_sma_7'].values
        
        features_df['price_lag1_squared'] = price_lag_1 ** 2
        features_df['price_lag7_squared'] = price_lag_7 ** 2
        features_df['price_sma7_squared'] = price_sma_7 ** 2
        
        # Interaktionsled mellem vigtige features
        features_df['price_lag1_lag7_interact'] = price_lag_1 * price_lag_7
        features_df['price_lag1_sma7_interact'] = price_lag_1 * price_sma_7
        
        logging.debug("Created polynomial and interaction features")
        
        # 9. Volume-based Features - ekstra volumenfeatues for aktier
        if VOLUME_COLUMN in df.columns:
            # Volumenindikatorer
            for window in [3, 7, 14, 30]:
                features_df[f'volume_sma_{window}'] = df[VOLUME_COLUMN].rolling(window=window, min_periods=1).mean()
                features_df[f'volume_std_{window}'] = df[VOLUME_COLUMN].rolling(window=window, min_periods=1).std()
                # Volumen Rate of Change (normaliseret)
                features_df[f'volume_roc_{window}'] = df[VOLUME_COLUMN].pct_change(window) * 100
            
            # On-Balance Volume (OBV) - akkumulerende volumen baseret på prisretning
            features_df['obv'] = calculate_obv(df[PRICE_COLUMN], df[VOLUME_COLUMN])
            
            # Volume Price Trend (VPT)
            features_df['vpt'] = calculate_vpt(df[PRICE_COLUMN], df[VOLUME_COLUMN])
            
            # Pris-volumen forhold
            features_df['price_to_volume'] = df[PRICE_COLUMN] / (df[VOLUME_COLUMN] + 1)  # +1 for at undgå division med nul
            
            # Volumen Profil
            features_df['volume_price_ratio'] = df[VOLUME_COLUMN] / (df[PRICE_COLUMN] + 0.1)
            
            # Chaikin Money Flow - en indikator for købspres vs. salgspres
            if all(col in df.columns for col in ['high', 'low', 'close']):
                features_df['chaikin_money_flow'] = calculate_chaikin_money_flow(
                    df['high'], df['low'], df[PRICE_COLUMN], df[VOLUME_COLUMN]
                )
                
            logging.debug("Created advanced volume-based features")
            
        # 10. Prisforskel-features for OHLC (Open, High, Low, Close) data
        if all(col in df.columns for col in ['open', 'high', 'low']):
            # Daglig range (High - Low)
            features_df['daily_range'] = df['high'] - df['low']
            features_df['daily_range_pct'] = (df['high'] - df['low']) / df['low'] * 100
            
            # Åbning til lukning ændring
            features_df['open_close_change'] = df[PRICE_COLUMN] - df['open']
            features_df['open_close_pct'] = (df[PRICE_COLUMN] - df['open']) / df['open'] * 100
            
            # Afstand fra dagens laveste til lukkekurs
            features_df['close_to_low'] = df[PRICE_COLUMN] - df['low']
            features_df['close_to_low_pct'] = (df[PRICE_COLUMN] - df['low']) / df['low'] * 100
            
            # Afstand fra dagens højeste til lukkekurs
            features_df['high_to_close'] = df['high'] - df[PRICE_COLUMN]
            features_df['high_to_close_pct'] = (df['high'] - df[PRICE_COLUMN]) / df[PRICE_COLUMN] * 100
            
            logging.debug("Created OHLC price difference features")
            
        # 11. Target Features (prisforudsigelser frem i tiden)
        for horizon in FORECAST_HORIZONS:
            # Absolut pris
            features_df[f'price_target_{horizon}d'] = df[PRICE_COLUMN].shift(-horizon)
            
            # Prisændring i procent
            features_df[f'price_change_{horizon}d'] = df[PRICE_COLUMN].pct_change(periods=-horizon) * 100
            
            # Binær bevægelsesretning (op/ned)
            features_df[f'price_direction_{horizon}d'] = (features_df[f'price_change_{horizon}d'] > 0).astype(int)
            
            logging.debug(f"Created target features for {horizon} day horizon")

        # Fjern rækker med NaN værdier
        nan_count_before = features_df.isna().sum().sum()
        logging.info(f"NaN values before cleaning: {nan_count_before}")
        
        # Fyld NaN værdier
        # For lag features, brug backfill for de første rækker
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        features_df[numeric_columns] = features_df[numeric_columns].fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        # Special behandling til target columns - behold NaN for de sidste rækker, da vi ikke har fremtidige data
        for horizon in FORECAST_HORIZONS:
            # Genindstil target kolonner til NaN for de sidste rækker
            target_col = f'price_target_{horizon}d'
            change_col = f'price_change_{horizon}d'
            direction_col = f'price_direction_{horizon}d'
            
            # Tidligere NaN værdier i target-kolonnerne (sidst i datasættet) skal forblive NaN
            mask = pd.isna(df[PRICE_COLUMN].shift(-horizon))
            features_df.loc[mask, [target_col, change_col, direction_col]] = np.nan
        
        nan_count_after = features_df.isna().sum().sum()
        logging.info(f"NaN values after cleaning: {nan_count_after}")
        
        logging.info(f"Feature engineering completed. Total features created: {len(features_df.columns)}")
        
        return features_df

    except Exception as e:
        logging.error(f"Error during feature engineering: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None

def calculate_rsi(series, periods=14):
    """Calculate Relative Strength Index"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window=periods, min_periods=1).mean()
    avg_loss = loss.rolling(window=periods, min_periods=1).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence/Divergence) indicator"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(series, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return rolling_mean, upper_band, lower_band

def calculate_obv(price, volume):
    """Calculate On-Balance Volume (OBV)"""
    obv = pd.Series(index=price.index, dtype=float)
    obv.iloc[0] = volume.iloc[0]
    
    for i in range(1, len(price)):
        if price.iloc[i] > price.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif price.iloc[i] < price.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv

def calculate_chaikin_money_flow(high, low, close, volume, period=20):
    """Calculate Chaikin Money Flow indicator"""
    mf_multiplier = ((close - low) - (high - close)) / (high - low + np.finfo(float).eps)
    mf_volume = mf_multiplier * volume
    cmf = mf_volume.rolling(window=period, min_periods=1).sum() / volume.rolling(window=period, min_periods=1).sum()
    return cmf

def save_features(df: pd.DataFrame, filepath: Path):
    """Saves the features DataFrame to a CSV file."""
    try:
        df.to_csv(filepath)
        logging.info(f"Features data saved successfully to {filepath}")
    except Exception as e:
        logging.error(f"Error saving features data to {filepath}: {e}")
        sys.exit(1)

def calculate_adx(data, period=14):
    """Calculate Average Directional Index (ADX)"""
    if not all(col in data.columns for col in ['high', 'low']):
        logging.warning("Cannot calculate ADX: missing required columns 'high' and 'low'")
        return pd.DataFrame({'adx': pd.Series(dtype=float), 'di_plus': pd.Series(dtype=float), 'di_minus': pd.Series(dtype=float)})
    
    high = data['high']
    low = data['low']
    close = data[PRICE_COLUMN]
    
    # Create result dataframe
    result = pd.DataFrame(index=data.index)
    
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
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + np.finfo(float).eps)  # Undgå division med nul
    adx = dx.rolling(window=period).mean()
    
    # Resultater til dataframe
    result['adx'] = adx
    result['di_plus'] = plus_di
    result['di_minus'] = minus_di
    
    return result

def calculate_stochastic(data, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator"""
    if not all(col in data.columns for col in ['high', 'low']):
        logging.warning("Cannot calculate Stochastic: missing required columns 'high' and 'low'")
        return pd.Series(dtype=float), pd.Series(dtype=float)
    
    low_min = data['low'].rolling(window=k_period).min()
    high_max = data['high'].rolling(window=k_period).max()
    
    # Undgå division med nul
    denom = high_max - low_min
    denom = denom.replace(0, np.finfo(float).eps)
    
    k = 100 * (data[PRICE_COLUMN] - low_min) / denom
    d = k.rolling(window=d_period).mean()
    
    return k, d

def calculate_vpt(price, volume):
    """
    Calculate Volume Price Trend (VPT)
    VPT er en kumulativ volumen-baseret indikator, der viser forholdet mellem prisændringer og volumen
    """
    vpt = volume * (price.pct_change())
    vpt.iloc[0] = 0  # Første værdi kan ikke beregnes, så vi sætter den til 0
    return vpt.cumsum()

def calculate_atr(data, period=14):
    """
    Calculate Average True Range (ATR)
    ATR er et volatilitetsmål, der typisk bruges til at vurdere hvor meget en aktie kan svinge i værdi
    """
    if not all(col in data.columns for col in ['high', 'low']):
        logging.warning("Cannot calculate ATR: missing required columns 'high' and 'low'")
        return pd.Series(dtype=float)
    
    high = data['high']
    low = data['low']
    close = data[PRICE_COLUMN]
    
    # Beregn True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Beregn Average True Range
    atr = tr.rolling(window=period).mean()
    
    return atr

def calculate_mfi(high, low, close, volume, period=14):
    """
    Calculate Money Flow Index (MFI)
    MFI er en oscillator, der bruger både pris og volumen til at identificere overkøbte eller oversolgte forhold
    """
    # Beregn typisk pris
    typical_price = (high + low + close) / 3
    
    # Beregn raw money flow
    raw_money_flow = typical_price * volume
    
    # Beregn positive og negative money flow
    money_flow_positive = np.where(typical_price > typical_price.shift(1), raw_money_flow, 0)
    money_flow_negative = np.where(typical_price < typical_price.shift(1), raw_money_flow, 0)
    
    # Beregn money flow ratio for perioden
    positive_flow = pd.Series(money_flow_positive).rolling(window=period).sum()
    negative_flow = pd.Series(money_flow_negative).rolling(window=period).sum()
    
    # Undgå division med nul
    negative_flow = negative_flow.replace(0, np.finfo(float).eps)
    
    money_flow_ratio = positive_flow / negative_flow
    
    # Beregn MFI
    mfi = 100 - (100 / (1 + money_flow_ratio))
    
    return mfi

def calculate_market_features(data):
    """Calculate market-based features for stock data"""
    result_df = data.copy()
    
    # Aktiespecifikke beregninger
    # Pris/volumen forhold
    if VOLUME_COLUMN in data.columns:
        result_df['volume_to_price_ratio'] = np.where(
            data[PRICE_COLUMN] != 0,
            data[VOLUME_COLUMN] / data[PRICE_COLUMN],
            0
        )
        
        # Akkumuleret volumen over 10 dage
        result_df['volume_10d_sum'] = data[VOLUME_COLUMN].rolling(window=10).sum()
        
        # Gennemsnitlig daglig volumen (forskellige perioder)
        for period in [5, 10, 20, 50]:
            result_df[f'adv_{period}d'] = data[VOLUME_COLUMN].rolling(window=period).mean()
        
        # Relativ volumen (aktuel volumen ift. gennemsnit)
        result_df['relative_volume_10d'] = data[VOLUME_COLUMN] / result_df['adv_10d']
    
    # Momentum indikatorer
    for period in [5, 10, 20, 50, 200]:
        # Simple Moving Average
        result_df[f'sma_{period}d'] = data[PRICE_COLUMN].rolling(window=period).mean()
        
        # Relative Strength (pris ift. SMA)
        result_df[f'rs_{period}d'] = data[PRICE_COLUMN] / result_df[f'sma_{period}d']
        
    # Moving Average Crossovers
    result_df['sma_cross_5_20'] = np.where(
        result_df['sma_5d'] > result_df['sma_20d'], 1, 
        np.where(result_df['sma_5d'] < result_df['sma_20d'], -1, 0)
    )
    
    result_df['sma_cross_20_50'] = np.where(
        result_df['sma_20d'] > result_df['sma_50d'], 1, 
        np.where(result_df['sma_20d'] < result_df['sma_50d'], -1, 0)
    )
    
    result_df['sma_cross_50_200'] = np.where(
        result_df['sma_50d'] > result_df['sma_200d'], 1, 
        np.where(result_df['sma_50d'] < result_df['sma_200d'], -1, 0)
    )
    
    logging.info(f"Generated stock-specific market features")
    
    return result_df

def calculate_macro_features(data):
    """Calculate macroeconomic features relative to stock data"""
    # Check for macroeconomic columns
    macro_columns = [
        'treasury_10y', 'vix', 'fed_rate', 'dxy', 'sp500', 
        'omxc25', 'eurusd', 'eurdkk', 'oil', 'gold'
    ]
    available_columns = [col for col in macro_columns if col in data.columns]
    
    if not available_columns:
        logging.warning("No macroeconomic indicators found in dataset")
        return data
    
    result_df = data.copy()
    logging.info(f"Found {len(available_columns)} macroeconomic indicators: {available_columns}")
    
    # Treasury yield features
    if 'treasury_10y' in data.columns:
        result_df['treasury_10y_lag_1'] = data['treasury_10y'].shift(1)
        result_df['treasury_10y_change'] = data['treasury_10y'] - result_df['treasury_10y_lag_1']
        result_df['treasury_10y_diff_7d'] = data['treasury_10y'] - data['treasury_10y'].shift(7)
        result_df['treasury_10y_ma7'] = data['treasury_10y'].rolling(window=7).mean()
        result_df['treasury_10y_volatility'] = data['treasury_10y'].rolling(window=14).std()
    
    # Volatility index features
    if 'vix' in data.columns:
        result_df['vix_lag_1'] = data['vix'].shift(1)
        result_df['vix_change'] = data['vix'] - result_df['vix_lag_1']
        result_df['vix_ma7'] = data['vix'].rolling(window=7).mean()
        result_df['vix_ma30'] = data['vix'].rolling(window=30).mean()
        result_df['vix_rsi_14'] = calculate_rsi(data['vix'], periods=14)
    
    # DXY (Dollar Index) features
    if 'dxy' in data.columns:
        result_df['dxy_lag_1'] = data['dxy'].shift(1)
        result_df['dxy_change'] = data['dxy'] - result_df['dxy_lag_1']
        result_df['dxy_ma7'] = data['dxy'].rolling(window=7).mean()
        result_df['dxy_volatility'] = data['dxy'].rolling(window=14).std()
    
    # Stock market indices features
    for index in ['sp500', 'omxc25']:
        if index in data.columns:
            result_df[f'{index}_lag_1'] = data[index].shift(1)
            result_df[f'{index}_change'] = data[index] - result_df[f'{index}_lag_1']
            result_df[f'{index}_ma7'] = data[index].rolling(window=7).mean()
            result_df[f'{index}_ma30'] = data[index].rolling(window=30).mean()
            result_df[f'{index}_volatility'] = data[index].rolling(window=14).std()
            
            # Correlation between Vestas and market indices
            result_df[f'vestas_{index}_corr_30d'] = data[PRICE_COLUMN].rolling(window=30).corr(data[index])
            result_df[f'vestas_{index}_beta_30d'] = (
                data[PRICE_COLUMN].pct_change().rolling(window=30).cov(data[index].pct_change()) /
                data[index].pct_change().rolling(window=30).var()
            )
    
    # Commodities features (Oil is particularly relevant for wind energy companies)
    for commodity in ['oil', 'gold']:
        if commodity in data.columns:
            result_df[f'{commodity}_lag_1'] = data[commodity].shift(1)
            result_df[f'{commodity}_change'] = data[commodity] - result_df[f'{commodity}_lag_1']
            result_df[f'{commodity}_ma7'] = data[commodity].rolling(window=7).mean()
            result_df[f'{commodity}_volatility'] = data[commodity].rolling(window=14).std()
            
            # Correlation between Vestas and commodities
            result_df[f'vestas_{commodity}_corr_30d'] = data[PRICE_COLUMN].rolling(window=30).corr(data[commodity])
    
    # Forex features (EUR/USD, EUR/DKK) - relevant for Vestas as an international company
    for forex in ['eurusd', 'eurdkk']:
        if forex in data.columns:
            result_df[f'{forex}_lag_1'] = data[forex].shift(1)
            result_df[f'{forex}_change'] = data[forex] - result_df[f'{forex}_lag_1']
            result_df[f'{forex}_ma7'] = data[forex].rolling(window=7).mean()
            result_df[f'{forex}_volatility'] = data[forex].rolling(window=14).std()
            
            # Correlation between Vestas and forex
            result_df[f'vestas_{forex}_corr_30d'] = data[PRICE_COLUMN].rolling(window=30).corr(data[forex])
    
    # Rentespænd - relevant for aktier i forhold til obligationsrenter
    if 'treasury_10y' in data.columns and 'fed_rate' in data.columns:
        result_df['yield_spread'] = data['treasury_10y'] - data['fed_rate'] 
    
    logging.info(f"Generated macroeconomic features related to Vestas stock")
    
    return result_df

def main():
    """Main function to run the feature engineering process."""
    logging.info("Starting feature engineering for Vestas stock data")
    
    try:
        # Indlæs præprocesseret data med Vestas og makroøkonomisk data
        input_file = INPUT_FILE_PATH
        
        # Indlæs data
        df = load_data(input_file)
        if df is None:
            return
        
        logging.info(f"Loaded preprocessed data with {len(df)} rows and {len(df.columns)} columns")
        
        # Generer basis-features
        features_df = create_features(df)
    if features_df is None:
            return
        
        # Generer market features
        logging.info("Generating market-specific features...")
        features_df = calculate_market_features(features_df)
        
        # Generer makroøkonomiske features
        logging.info("Generating macroeconomic features...")
        features_df = calculate_macro_features(features_df)
        
        # Gem det endelige datasæt med alle features
    save_features(features_df, OUTPUT_FILE_PATH)

        # Gem også en simplere version med kun de mest relevante features (for reference)
        latest_data = features_df.tail(100).copy()
        # Sikre at vi kun beholder numeriske kolonner
        latest_data = latest_data.select_dtypes(include=[np.number])
        
        # Gem de sidste 100 rækker som seneste data for simpel reference
        latest_filepath = PROCESSED_FEATURES_DIR / "latest_data.csv"
        latest_data.to_csv(latest_filepath)
        logging.info(f"Latest sample data saved to {latest_filepath}")
        
        logging.info("Feature engineering completed successfully!")
        
    except Exception as e:
        logging.error(f"Error during feature engineering process: {e}")
        import traceback
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()
