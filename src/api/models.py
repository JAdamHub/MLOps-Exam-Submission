from pydantic import BaseModel, Field
from typing import Dict, Any, List

class InputFeatures(BaseModel):
    """
    Defines the input features required for the prediction endpoint.
    Pydantic model that dynamically accepts all features used in the model training.
    """
    # Dette er nu en dynamisk model, der kan acceptere alle features
    # Felter defineres ved runtime baseret på model.feature_names_in_
    
    def __init__(self, **data: Any):
        super().__init__(**data)
    
    # Tillad ekstra felter som ikke er eksplicit defineret (alle features)
    model_config = {
        "extra": "allow",
        "json_schema_extra": {
            "example": {
                "price": 0.5,
                "total_volume": 0.4,
                "price_lag_1": 0.49,
                "price_lag_3": 0.48,
                "price_lag_7": 0.45,
                "price_sma_7": 0.48,
                "price_sma_30": 0.46,
                "price_volatility_14": 0.05,
                "market_cap_to_volume": 1.5,
                "treasury_10y": 0.5,
                "sp500": 0.7,
                # etc.
            }
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
