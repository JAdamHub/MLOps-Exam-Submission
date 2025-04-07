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
                "year": 2023
            }
        }

class PredictionResponse(BaseModel):
    """
    Defines the response structure for the prediction endpoint.
    """
    prediction: int # 0 for price not up, 1 for price up
    probability: float # Probability of the price being up (class 1)
