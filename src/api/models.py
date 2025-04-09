from pydantic import BaseModel

class InputFeatures(BaseModel):
    """
    Defines the input features required for the prediction endpoint.
    Assumes that the client provides features consistent with the training data,
    meaning base features (price, market_cap, total_volume) and derived features
    (lags, SMAs, volatility) are appropriately scaled/calculated.
    """
    price: float
    market_cap: float
    total_volume: float
    price_lag_1: float
    price_lag_3: float
    price_lag_7: float
    price_sma_7: float
    price_sma_30: float
    price_volatility_14: float
    day_of_week: int
    month: int
    year: int
    rsi_14: float
    rsi_7: float
    rsi_21: float
    macd: float
    macd_signal: float
    macd_histogram: float
    macd_fast: float
    macd_signal_fast: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_width: float
    bb_position: float
    volume_sma_7: float
    volume_sma_30: float
    volume_ratio: float
    volume_ratio_30: float
    volume_momentum: float
    price_momentum_1: float
    price_momentum_7: float
    price_momentum_30: float
    price_momentum_90: float
    volatility_7: float
    volatility_14: float
    volatility_30: float
    market_cap_to_volume: float
    market_cap_momentum_1: float
    market_cap_momentum_7: float
    market_cap_momentum_30: float
    volume_to_market_cap: float
    price_volatility_ratio: float
    momentum_volatility_ratio: float
    volume_price_ratio: float
    day_of_month: int
    is_weekend: int

    # Example to guide users in the API docs
    class Config:
        schema_extra = {
            "example": {
                "price": 0.5, # Example scaled value
                "market_cap": 0.6,
                "total_volume": 0.4,
                "price_lag_1": 0.49,
                "price_lag_3": 0.48,
                "price_lag_7": 0.45,
                "price_sma_7": 0.48,
                "price_sma_30": 0.46,
                "price_volatility_14": 0.05,
                "day_of_week": 3, # Wednesday
                "month": 10,
                "year": 2023,
                "rsi_14": 0.45,
                "rsi_7": 0.42,
                "rsi_21": 0.47,
                "macd": 0.02,
                "macd_signal": 0.01,
                "macd_histogram": 0.01,
                "macd_fast": 0.03,
                "macd_signal_fast": 0.02,
                "bb_upper": 0.55,
                "bb_middle": 0.5,
                "bb_lower": 0.45,
                "bb_width": 0.1,
                "bb_position": 0.5,
                "volume_sma_7": 0.4,
                "volume_sma_30": 0.42,
                "volume_ratio": 0.95,
                "volume_ratio_30": 0.97,
                "volume_momentum": 0.01,
                "price_momentum_1": 0.01,
                "price_momentum_7": 0.03,
                "price_momentum_30": 0.05,
                "price_momentum_90": 0.08,
                "volatility_7": 0.03,
                "volatility_14": 0.04,
                "volatility_30": 0.05,
                "market_cap_to_volume": 1.5,
                "market_cap_momentum_1": 0.01,
                "market_cap_momentum_7": 0.02,
                "market_cap_momentum_30": 0.04,
                "volume_to_market_cap": 0.67,
                "price_volatility_ratio": 0.75,
                "momentum_volatility_ratio": 0.8,
                "volume_price_ratio": 0.9,
                "day_of_month": 15,
                "is_weekend": 0
            }
        }

class PredictionResponse(BaseModel):
    """
    Defines the response structure for the prediction endpoint.
    """
    prediction: int # 0 for price not up, 1 for price up
    probability: float # Probability of the price being up (class 1)
    timestamp: str # ISO format timestamp of when the prediction was made
    features_used: list # List of features used for the prediction
