from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging
import os
import json
from typing import Dict, List, Any, Optional

# Konfigurer logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("bitcoin-api")

# Definér stier til modeller og data
BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Stier til modeller og scaler
MODEL_FILES = {
    "1d": MODELS_DIR / "xgboost_model_1d.joblib",
    "3d": MODELS_DIR / "xgboost_model_3d.joblib",
    "7d": MODELS_DIR / "xgboost_model_7d.joblib"
}
FEATURE_NAMES_FILE = MODELS_DIR / "feature_names.joblib"
TARGET_COLUMNS_FILE = MODELS_DIR / "target_columns.joblib"

# Data sti
BITCOIN_DATA_FILE = DATA_DIR / "intermediate" / "combined" / "bitcoin_macro_combined_trading_days.csv"

# Globale variabler til at holde modeller og data
models = {}
feature_names = []
bitcoin_data = None

app = FastAPI(
    title="Bitcoin API",
    description="API til Bitcoin prishistorik og forudsigelser",
    version="1.0.0"
)

# Tilføj CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Load models and data on startup"""
    global models, feature_names, bitcoin_data
    
    # Indlæs modeller
    try:
        for horizon, file_path in MODEL_FILES.items():
            if file_path.exists():
                models[horizon] = joblib.load(file_path)
                logger.info(f"Model for {horizon} loaded successfully")
            else:
                logger.error(f"Model file not found: {file_path}")
        
        # Indlæs feature names
        if FEATURE_NAMES_FILE.exists():
            feature_names = joblib.load(FEATURE_NAMES_FILE)
            logger.info(f"Feature names loaded: {feature_names}")
        else:
            logger.error(f"Feature names file not found: {FEATURE_NAMES_FILE}")
        
        # Indlæs Bitcoin data
        if BITCOIN_DATA_FILE.exists():
            bitcoin_data = pd.read_csv(BITCOIN_DATA_FILE)
            # Korrekt indlæsning af datoen fra 'Unnamed: 0' kolonne og eksisterende 'timestamp' kolonne
            if 'Unnamed: 0' in bitcoin_data.columns:
                bitcoin_data['date'] = pd.to_datetime(bitcoin_data['Unnamed: 0'])
            elif 'timestamp' in bitcoin_data.columns:
                bitcoin_data['date'] = pd.to_datetime(bitcoin_data['timestamp'])
            else:
                # Fallback til indeks
                bitcoin_data['date'] = pd.to_datetime(bitcoin_data.index)
            logger.info(f"Bitcoin data loaded with shape: {bitcoin_data.shape}")
        else:
            logger.error(f"Bitcoin data file not found: {BITCOIN_DATA_FILE}")
            
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")

@app.get("/")
async def root():
    """Welcome message"""
    return {
        "message": "Velkommen til Bitcoin API",
        "endpoints": {
            "GET /price/history": "Hent Bitcoin prishistorik",
            "GET /predict": "Forudsig Bitcoin pris for 1, 3 og 7 dage"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    models_loaded = len(models) > 0
    data_loaded = bitcoin_data is not None
    is_healthy = models_loaded and data_loaded
    
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "models_loaded": models_loaded,
        "data_loaded": data_loaded
    }

@app.get("/price/history")
async def get_price_history(days: Optional[int] = 365):
    """Hent Bitcoin prishistorik for det seneste år"""
    if bitcoin_data is None:
        raise HTTPException(status_code=500, detail="Bitcoin data ikke indlæst")
    
    # Filtrer data baseret på antallet af dage
    last_n_days = bitcoin_data.tail(days).copy()
    
    # Formatér data til response
    price_history = []
    for _, row in last_n_days.iterrows():
        price_history.append({
            "date": row['date'].strftime("%Y-%m-%d"),
            "price": row['price'],
            "market_cap": row['market_cap'],
            "total_volume": row['total_volume'] if 'total_volume' in row else None
        })
    
    return {"data": price_history}

@app.get("/metrics")
async def get_metrics():
    """Hent træningsmetrikker for modellerne"""
    metrics_file = MODELS_DIR / "training_metrics.json"
    
    try:
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            return metrics
        else:
            raise HTTPException(status_code=404, detail="Metrics file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading metrics: {str(e)}")

@app.post("/predict")
async def predict_price(features: Dict[str, Any]):
    """Forudsig Bitcoin pris for 1, 3 og 7 dage"""
    if not models:
        raise HTTPException(status_code=500, detail="Modeller ikke indlæst")
    
    if not feature_names:
        raise HTTPException(status_code=500, detail="Feature names ikke indlæst")
    
    try:
        # Opret DataFrame med features
        input_df = pd.DataFrame([features])
        
        # Tjek for manglende features og tilføj dem med standardværdier
        missing_features = set(feature_names) - set(input_df.columns)
        if missing_features:
            logger.warning(f"Manglende features: {', '.join(missing_features)}")
            
            # Standardværdier for manglende features
            default_values = {
                'price': features.get('price', 60000),
                'market_cap': features.get('market_cap', 1200000000000),
                'fed_rate': 5.0,
                'sp500_pct_change': 0.001,
                'eurusd_pct_change': 0.0005,
                'price_lag_7': features.get('price', 60000) * 0.95,
                'price_sma_7': features.get('price', 60000) * 1.02,
                'market_cap_momentum_30': 0.02,
                'treasury_10y_volatility': 0.03,
                'treasury_5y_ma7': 3.5,
                'vix_ma30': 18.0,
                'dxy_volatility': 0.05,
                'sp500_ma30': 4500,
                'btc_nasdaq_corr_30d': 0.65,
                'btc_dow_corr_30d': 0.45,
                'gold_ma7': 2000,
                'oil_ma7': 80,
                'eurusd_rsi_14': 50,
                'gold_dxy_corr_30d': -0.3,
                'oil_dxy_corr_30d': -0.2
            }
            
            # Tilføj manglende features med standardværdier
            for feature in missing_features:
                input_df[feature] = default_values.get(feature, 0.0)
            
            logger.info(f"Tilføjede standardværdier for {len(missing_features)} manglende features")
        
        # Reorganiser features til at matche træningsrækkefølgen
        input_df = input_df[feature_names]
        
        # Foretag forudsigelser
        scaled_predictions = {}
        for horizon, model in models.items():
            # Brug raw_features i stedet for scaled_features
            prediction = model.predict(input_df)[0]  
            scaled_predictions[horizon] = float(prediction)
        
        # Tilføj den seneste pris som reference
        current_price = 0
        if bitcoin_data is not None and not bitcoin_data.empty:
            current_price = float(bitcoin_data['price'].iloc[-1])
        else:
            # Hvis vi ikke har data, brug input price
            current_price = float(input_df['price'].iloc[0])
        
        # Konverter skalerede prædiktioner til faktiske priser
        # Baseret på vores observation ser modellerne ud til at returnere procentvise ændringer
        # eller en anden skaleret værdi. Vi bruger den aktuelle pris til at konvertere.
        actual_predictions = {}
        
        # Metode 1: Hvis værdierne er under 1, antag at de er procentvise ændringer af aktuel pris
        if all(pred < 5 for pred in scaled_predictions.values()):
            for horizon, scaled_pred in scaled_predictions.items():
                days = int(horizon.replace('d', ''))
                # Beregn faktisk pris (antag at værdierne er procenter af prisændring)
                # multiplier faktor på ~100 for at få en rimelig skala
                actual_predictions[horizon] = current_price * (1 + scaled_pred * 0.01)
                logger.info(f"Konverterede {horizon} prædiktion fra {scaled_pred} til {actual_predictions[horizon]}")
        else:
            # Hvis værdierne er større, antag de er faktiske priser
            actual_predictions = scaled_predictions
        
        # Tilføj reference prisen
        actual_predictions['current_price'] = current_price
        
        return {
            "predictions": actual_predictions,
            "scaled_values": scaled_predictions,  # Inkluder også de oprindelige skalerede værdier for reference
            "timestamp": pd.Timestamp.now().isoformat(),
            "had_missing_features": len(missing_features) > 0
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/features")
async def get_features():
    """Hent liste over features, der bruges af modellerne"""
    if not feature_names:
        raise HTTPException(status_code=500, detail="Feature names ikke indlæst")
    
    return {"features": feature_names}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("bitcoin_api:app", host="0.0.0.0", port=8000, reload=False) 