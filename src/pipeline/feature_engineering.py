import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import joblib

# configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# determine project root based on script location
# assumes the script is in src/pipeline
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# INTERMEDIATE_PREPROCESSED_DIR = PROJECT_ROOT / "data" / "intermediate" / "preprocessed"  # updated path
PROCESSED_FEATURES_DIR = PROJECT_ROOT / "data" / "features" # Keep for potential future use or debugging output
MODELS_DIR = PROJECT_ROOT / "models"

# feature engineering parameters
PRICE_COLUMN = 'close'  # stock price typically uses "close" as the primary price column
VOLUME_COLUMN = 'volume'  # volume column for stocks
LAG_PERIODS = [1, 3, 7]  # lag periods in days
SMA_WINDOWS = [7, 30]  # simple moving average windows in days
VOLATILITY_WINDOW = 14  # window for rolling standard deviation (volatility)
FORECAST_HORIZONS = [1, 3, 7]  # predict the price 1, 3, and 7 days ahead

def create_features(df: pd.DataFrame) -> pd.DataFrame | None:
    """generates features for the stock price model."""
    if df is None or df.empty:
        logging.error("Input DataFrame for feature engineering is None or empty.")
        return None

    try:
        logging.info("starting feature engineering for vestas stock data...")
        features_df = df.copy()

        # Percentage change for the main price column
        features_df[f'{PRICE_COLUMN}_pct_change'] = df[PRICE_COLUMN].pct_change() * 100
        logging.debug(f"created {PRICE_COLUMN}_pct_change feature")
        
        # Volatility (14-day rolling standard deviation of daily returns)
        features_df['volatility_14d'] = df[PRICE_COLUMN].pct_change().rolling(window=14).std() * 100
        logging.debug("created volatility_14d feature")

        # 1. lagged features - more lags to capture more complex patterns
        for lag in LAG_PERIODS + [14, 21, 30]:
            features_df[f'{PRICE_COLUMN}_lag_{lag}'] = df[PRICE_COLUMN].shift(lag)
            logging.debug(f"created lag feature: {PRICE_COLUMN}_lag_{lag}")

        # 2. moving averages - more windows and exponential moving averages
        for window in SMA_WINDOWS + [14, 30, 60, 90]:
            # simple moving average (sma)
            features_df[f'{PRICE_COLUMN}_sma_{window}'] = df[PRICE_COLUMN].rolling(window=window, min_periods=1).mean()
            
            # exponential moving average (ema)
            features_df[f'{PRICE_COLUMN}_ema_{window}'] = df[PRICE_COLUMN].ewm(span=window, adjust=False).mean()
            
            logging.debug(f"created sma and ema features with window {window}")

        # 3. volatility - different window sizes to capture shorter and longer volatility
        for window in [7, 14, 21, 30]:
            features_df[f'{PRICE_COLUMN}_volatility_{window}'] = df[PRICE_COLUMN].rolling(window=window, min_periods=1).std()
            logging.debug(f"created volatility feature with window {window}")

        # 4. advanced technical indicators for stocks
        # 4.1 rsi (relative strength index)
        for period in [7, 14, 21]:
            features_df[f'{PRICE_COLUMN}_rsi_{period}'] = calculate_rsi(df[PRICE_COLUMN], periods=period)
            logging.debug(f"created rsi feature with period {period}")

        # 4.2 macd (moving average convergence divergence)
        features_df['price_macd'], features_df['price_macd_signal'], features_df['price_macd_hist'] = calculate_macd(df[PRICE_COLUMN])
        logging.debug("created macd features")

        # 4.3 bollinger bands
        for window in [20, 30]:
            mid, upper, lower = calculate_bollinger_bands(df[PRICE_COLUMN], window=window)
            features_df[f'price_bb_mid_{window}'] = mid
            features_df[f'price_bb_upper_{window}'] = upper
            features_df[f'price_bb_lower_{window}'] = lower
            
            # percentage distance to bands (normalized)
            features_df[f'price_bb_pct_b_{window}'] = (df[PRICE_COLUMN] - lower) / (upper - lower)
            
            # bandwidth (volatility measure)
            features_df[f'price_bb_bandwidth_{window}'] = (upper - lower) / mid
            
            logging.debug(f"created bollinger bands features with window {window}")

        # 4.4 rate of change (roc)
        for period in [1, 3, 7, 14, 30]:
            features_df[f'price_roc_{period}'] = ((df[PRICE_COLUMN] / df[PRICE_COLUMN].shift(period)) - 1) * 100
            logging.debug(f"created rate of change feature with period {period}")
            
        # 4.5 momentum indicators
        for period in [3, 7, 14, 30]:
            features_df[f'price_momentum_{period}'] = df[PRICE_COLUMN] - df[PRICE_COLUMN].shift(period)
            logging.debug(f"created momentum feature with period {period}")
            
        # 5. stock-specific indicators
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # 5.1 average true range (atr) - volatility indicator
            features_df['atr_14'] = calculate_atr(df, 14)
            
            # 5.2 money flow index (mfi) - volume-based oscillator
            if VOLUME_COLUMN in df.columns:
                features_df['mfi_14'] = calculate_mfi(df['high'], df['low'], df['close'], df[VOLUME_COLUMN], 14)
            
            # 5.3 stochastic oscillator
            features_df['stoch_k'], features_df['stoch_d'] = calculate_stochastic(df)
            
            # 5.4 average directional index (adx) - trend strength measurement
            adx_df = calculate_adx(df)
            features_df['adx'] = adx_df['adx']
            features_df['di_plus'] = adx_df['di_plus']
            features_df['di_minus'] = adx_df['di_minus']
            
            logging.debug("created stock-specific technical indicators")
            
        # 6. price transformations
        # log transformation can help linearize exponential patterns
        features_df['price_log'] = np.log1p(df[PRICE_COLUMN])  # log1p handles 0 values
        
        # differences (absolute changes)
        features_df['price_diff_1d'] = df[PRICE_COLUMN].diff(1)
        features_df['price_diff_3d'] = df[PRICE_COLUMN].diff(3)
        features_df['price_diff_7d'] = df[PRICE_COLUMN].diff(7)
        
        # percentage changes
        features_df['price_pct_change_1d'] = df[PRICE_COLUMN].pct_change(1)
        features_df['price_pct_change_3d'] = df[PRICE_COLUMN].pct_change(3)
        features_df['price_pct_change_7d'] = df[PRICE_COLUMN].pct_change(7)
        
        logging.debug("created price transformations")
        
        # 7. time-based features - expanded with more date transformations
        features_df['day_of_week'] = df.index.dayofweek
        features_df['day_of_month'] = df.index.day
        features_df['week_of_year'] = df.index.isocalendar().week
        features_df['month'] = df.index.month
        features_df['quarter'] = df.index.quarter
        features_df['year'] = df.index.year
        features_df['is_month_start'] = df.index.is_month_start.astype(int)
        features_df['is_month_end'] = df.index.is_month_end.astype(int)
        features_df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        
        # seasonal components with sine/cosine transformation (circular features)
        # these transformations help the model understand periodic/cyclical patterns
        features_df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        features_df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
        features_df['day_of_week_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        features_df['day_of_week_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        
        logging.debug("created advanced time-based features")
        
        # 8. polynomial features for the most important indicators
        price_lag_1 = features_df[f'{PRICE_COLUMN}_lag_1'].values
        price_lag_7 = features_df[f'{PRICE_COLUMN}_lag_7'].values
        price_sma_7 = features_df[f'{PRICE_COLUMN}_sma_7'].values
        
        features_df['price_lag1_squared'] = price_lag_1 ** 2
        features_df['price_lag7_squared'] = price_lag_7 ** 2
        features_df['price_sma7_squared'] = price_sma_7 ** 2
        
        # interaction terms between important features
        features_df['price_lag1_lag7_interact'] = price_lag_1 * price_lag_7
        features_df['price_lag1_sma7_interact'] = price_lag_1 * price_sma_7
        
        logging.debug("created polynomial and interaction features")
        
        # 9. volume-based features - extra volume features for stocks
        if VOLUME_COLUMN in df.columns:
            # volume indicators
            for window in [3, 7, 14, 30]:
                features_df[f'volume_sma_{window}'] = df[VOLUME_COLUMN].rolling(window=window, min_periods=1).mean()
                features_df[f'volume_std_{window}'] = df[VOLUME_COLUMN].rolling(window=window, min_periods=1).std()
                # volume rate of change (normalized)
                features_df[f'volume_roc_{window}'] = df[VOLUME_COLUMN].pct_change(window) * 100
            
            # on-balance volume (obv) - accumulating volume based on price direction
            features_df['obv'] = calculate_obv(df[PRICE_COLUMN], df[VOLUME_COLUMN])
            
            # volume price trend (vpt)
            features_df['vpt'] = calculate_vpt(df[PRICE_COLUMN], df[VOLUME_COLUMN])
            
            # price-volume ratio
            features_df['price_to_volume'] = df[PRICE_COLUMN] / (df[VOLUME_COLUMN] + 1)  # +1 to avoid division by zero
            
            # volume profile
            features_df['volume_price_ratio'] = df[VOLUME_COLUMN] / (df[PRICE_COLUMN] + 0.1)
            
            # chaikin money flow - an indicator of buying pressure vs. selling pressure
            if all(col in df.columns for col in ['high', 'low', 'close']):
                features_df['chaikin_money_flow'] = calculate_chaikin_money_flow(
                    df['high'], df['low'], df[PRICE_COLUMN], df[VOLUME_COLUMN]
                )
                
            logging.debug("created advanced volume-based features")
            
        # 10. price difference features for ohlc (open, high, low, close) data
        if all(col in df.columns for col in ['open', 'high', 'low']):
            # daily range (high - low)
            features_df['daily_range'] = df['high'] - df['low']
            features_df['daily_range_pct'] = (df['high'] - df['low']) / df['low'] * 100
            
            # open to close change
            features_df['open_close_change'] = df[PRICE_COLUMN] - df['open']
            features_df['open_close_pct'] = (df[PRICE_COLUMN] - df['open']) / df['open'] * 100
            
            # distance from day's low to close price
            features_df['close_to_low'] = df[PRICE_COLUMN] - df['low']
            features_df['close_to_low_pct'] = (df[PRICE_COLUMN] - df['low']) / df['low'] * 100
            
            # distance from day's high to close price
            features_df['high_to_close'] = df['high'] - df[PRICE_COLUMN]
            features_df['high_to_close_pct'] = (df['high'] - df[PRICE_COLUMN]) / df[PRICE_COLUMN] * 100
            
            logging.debug("created ohlc price difference features")
            
        # 11. target features (price predictions ahead in time) - CREATE ABSOLUTE TARGETS FIRST
        absolute_target_columns = []
        for horizon in FORECAST_HORIZONS:
            target_col_name = f'price_target_{horizon}d'
            features_df[target_col_name] = df[PRICE_COLUMN].shift(-horizon)
            absolute_target_columns.append(target_col_name)
            logging.debug(f"created absolute target feature: {target_col_name}")

        # Convert absolute targets to percentage change
        # This transformation is done here instead of in the training script
        percent_target_columns = []
        for target_col in absolute_target_columns:
            horizon = int(target_col.split('_')[-1].replace('d', ''))
            pct_target_col_name = f'pct_change_{target_col}'
            # Calculate percentage change from current price to target price
            # Need to handle potential division by zero if PRICE_COLUMN is zero
            features_df[pct_target_col_name] = (
                (features_df[target_col] / (df[PRICE_COLUMN].replace(0, np.nan))) - 1
            ) * 100
            percent_target_columns.append(pct_target_col_name)
            logging.info(f"Converted {target_col} to percent change target: {pct_target_col_name}")

        # IMPORTANT: Do NOT handle NaNs here anymore. 
        # NaN handling (filling) will happen in the training script after splitting data
        # to prevent data leakage from validation/test sets into training set during imputation.
        nan_count_before_save = features_df.isna().sum().sum()
        if nan_count_before_save > 0:
             logging.warning(f"Feature engineering introduced {nan_count_before_save} NaN values. They will be handled in the training script.")

        logging.info("feature engineering completed.")
        logging.info(f"final feature set shape (before NaN handling in training): {features_df.shape}")
        logging.info(f"sample features: {features_df.columns.tolist()[:10]}...")
        logging.info(f"final target columns (percentage change): {percent_target_columns}")

        return features_df

    except Exception as e:
        logging.error(f"error during feature engineering: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None

def calculate_rsi(series, periods=14):
    """calculate the relative strength index (rsi)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    """calculate the moving average convergence divergence (macd)."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def calculate_bollinger_bands(series, window=20, num_std=2):
    """calculate bollinger bands."""
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return rolling_mean, upper_band, lower_band

def calculate_obv(price, volume):
    """calculate on-balance volume (obv)."""
    obv = pd.Series(index=price.index)
    obv.iloc[0] = 0
    for i in range(1, len(price)):
        if price.iloc[i] > price.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif price.iloc[i] < price.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    return obv

def calculate_chaikin_money_flow(high, low, close, volume, period=20):
    """calculate chaikin money flow (cmf)."""
    clv = ((close - low) - (high - close)) / (high - low)
    clv = clv.fillna(0)  # handle division by zero if high == low
    mfv = clv * volume
    cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
    return cmf

def calculate_adx(data, period=14):
    """calculate the average directional index (adx)."""
    df = data.copy()
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1)))
    )
    df['dm_plus'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']), df['high'] - df['high'].shift(1), 0)
    df['dm_plus'] = np.where(df['dm_plus'] < 0, 0, df['dm_plus'])
    df['dm_minus'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)), df['low'].shift(1) - df['low'], 0)
    df['dm_minus'] = np.where(df['dm_minus'] < 0, 0, df['dm_minus'])

    tr_smooth = df['tr'].ewm(alpha=1/period, adjust=False).mean()
    dm_plus_smooth = df['dm_plus'].ewm(alpha=1/period, adjust=False).mean()
    dm_minus_smooth = df['dm_minus'].ewm(alpha=1/period, adjust=False).mean()

    di_plus = 100 * (dm_plus_smooth / tr_smooth)
    di_minus = 100 * (dm_minus_smooth / tr_smooth)
    dx = 100 * (abs(di_plus - di_minus) / (di_plus + di_minus))
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    return pd.DataFrame({'adx': adx, 'di_plus': di_plus, 'di_minus': di_minus})

def calculate_stochastic(data, k_period=14, d_period=3):
    """calculate the stochastic oscillator (%k and %d)."""
    df = data.copy()
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    
    df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
    df['stoch_k'].fillna(df['stoch_k'].mean(), inplace=True) # fill potential nans
    
    df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
    df['stoch_d'].fillna(df['stoch_d'].mean(), inplace=True) # fill potential nans
    
    return df['stoch_k'], df['stoch_d']

def calculate_vpt(price, volume):
    """calculate volume price trend (vpt)."""
    vpt = (volume * price.pct_change()).cumsum()
    return vpt

def calculate_atr(data, period=14):
    """calculate average true range (atr)."""
    df = data.copy()
    df['h-l'] = df['high'] - df['low']
    df['h-pc'] = abs(df['high'] - df['close'].shift(1))
    df['l-pc'] = abs(df['low'] - df['close'].shift(1))
    
    df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    atr = df['tr'].rolling(window=period).mean()
    # alternativt: atr = df['tr'].ewm(alpha=1/period, adjust=false).mean()
    return atr

def calculate_mfi(high, low, close, volume, period=14):
    """calculate money flow index (mfi)."""
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    
    positive_flow = []
    negative_flow = []

    for i in range(1, len(typical_price)):
        if typical_price.iloc[i] > typical_price.iloc[i-1]:
            positive_flow.append(raw_money_flow.iloc[i])
            negative_flow.append(0)
        elif typical_price.iloc[i] < typical_price.iloc[i-1]:
            negative_flow.append(raw_money_flow.iloc[i])
            positive_flow.append(0)
        else:
            positive_flow.append(0)
            negative_flow.append(0)
    
    positive_mf = pd.Series(positive_flow, index=high.index[1:]).rolling(window=period).sum()
    negative_mf = pd.Series(negative_flow, index=high.index[1:]).rolling(window=period).sum()
    
    # avoid division by zero
    money_flow_ratio = positive_mf / (negative_mf + 1e-9)
    
    mfi = 100 - (100 / (1 + money_flow_ratio))
    
    # reindex to match original dataframe
    mfi = mfi.reindex(high.index)
    
    return mfi

def calculate_market_features(data):
    """calculate features based on other market indicators (e.g., s&p 500)."""
    features_df = pd.DataFrame(index=data.index)
    # example: calculate correlation with s&p 500 over a rolling window
    if 'spy_close' in data.columns and PRICE_COLUMN in data.columns:
        features_df['corr_spy_30d'] = data[PRICE_COLUMN].rolling(window=30).corr(data['spy_close'])
    
    if 'vgk_close' in data.columns and PRICE_COLUMN in data.columns:
        features_df['corr_vgk_30d'] = data[PRICE_COLUMN].rolling(window=30).corr(data['vgk_close'])
        
    # example: calculate relative performance to s&p 500
    if 'spy_close' in data.columns and PRICE_COLUMN in data.columns:
        features_df['relative_strength_spy'] = data[PRICE_COLUMN] / data['spy_close']
    
    # add more market-related features as needed
    return features_df

def calculate_macro_features(data):
    """calculate features based on macroeconomic indicators."""
    features_df = pd.DataFrame(index=data.index)
    
    # example: calculate rate of change for gdp (assuming quarterly data is forward filled)
    if 'real_gdp' in data.columns:
        # assuming quarterly data, so lag by approx 63 trading days
        features_df['gdp_qoq_change'] = data['real_gdp'].pct_change(periods=63) * 100
        
    # example: calculate moving average of inflation
    if 'inflation_us' in data.columns:
        features_df['inflation_sma_6m'] = data['inflation_us'].rolling(window=126).mean() # approx 6 months
        
    # example: difference from treasury yield
    if 'treasury_etf_close' in data.columns:
        # using the etf as a proxy for interest rates
        features_df['treasury_yield_proxy_change'] = data['treasury_etf_close'].pct_change(30) * 100
        
    # eur/usd rate change
    if 'eurusd_close' in data.columns:
        features_df['eurusd_change_30d'] = data['eurusd_close'].pct_change(30) * 100
        
    # crude oil price change
    if 'crude_oil_etf_close' in data.columns:
        features_df['oil_change_30d'] = data['crude_oil_etf_close'].pct_change(30) * 100
        
    # add more macro features as needed
    return features_df

def main(df_preprocessed: pd.DataFrame):
    """main function to run the feature engineering process."""
    logging.info("--- starting feature engineering process ---")

    # load data
    # df = load_data(INPUT_FILE_PATH)
    if df_preprocessed is None:
        logging.error("halting pipeline: No preprocessed data received.")
        return None # Return None on error

    # create features
    features_df = create_features(df_preprocessed)
    if features_df is None:
        logging.error("halting pipeline due to feature creation error.")
        return None # Return None on error

    # calculate market and macro features separately and merge
    market_features = calculate_market_features(features_df) # use features_df as it contains needed columns
    macro_features = calculate_macro_features(features_df)
    
    # merge market features
    if not market_features.empty:
        features_df = features_df.merge(market_features, left_index=True, right_index=True, how='left')
        logging.info(f"merged {market_features.shape[1]} market features.")
        
    # merge macro features
    if not macro_features.empty:
        features_df = features_df.merge(macro_features, left_index=True, right_index=True, how='left')
        logging.info(f"merged {macro_features.shape[1]} macro features.")
        
    # --- Final NaN drop before returning --- 
    # This removes rows with NaNs from initial feature calculations (lags, rolling) 
    # or from merges, and rows where targets are NaN (end of series)
    initial_rows = len(features_df)
    features_df.dropna(inplace=True)
    rows_dropped = initial_rows - len(features_df)
    if rows_dropped > 0:
        logging.info(f"Dropped {rows_dropped} rows containing NaN values (e.g., initial periods, merge gaps, or end-of-series targets). Final shape: {features_df.shape}")
    else:
        logging.info("No rows dropped due to NaNs.")

    # save features
    # save_features(features_df, OUTPUT_FILE_PATH)

    logging.info(f"final dataset with features shape: {features_df.shape}")
    logging.info("--- feature engineering process completed successfully ---")
    return features_df # Return the features DataFrame

if __name__ == "__main__":
    # Add placeholder logic for running standalone if needed, e.g., load from DB
    logging.warning("This script is intended to be run as part of the pipeline.")
    sys.exit(0) # Indicate success if run standalone without error (though not typical use)
