from src.monitoring.prediction_store import PredictionStore
import random
from datetime import datetime, timedelta
import logging

# Konfigurer logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Test PredictionStore funktionalitet"""
    try:
        # Opret PredictionStore
        store = PredictionStore()
        
        # Opret nogle dummy forudsigelser for test
        logger.info("Opretter test forudsigelser...")
        
        # Generate a mix of accurate and inaccurate predictions
        num_predictions = 20
        
        # Simuler forudsigelser over de seneste 10 dage
        for i in range(num_predictions):
            # Simuler at nogle forudsigelser er korrekte og andre ikke
            prediction = random.choice([0, 1])
            
            # 70% chance for at forudsigelsen er korrekt
            if random.random() < 0.7:
                actual_value = prediction
            else:
                actual_value = 1 - prediction
            
            # Opret dummy features
            features_used = {
                "price": random.uniform(40000, 60000),
                "market_cap": random.uniform(800e9, 1.2e12),
                "rsi_14": random.uniform(30, 70),
                "sma_cross": random.choice([0, 1])
            }
            
            # Simuler at forudsigelsen blev lavet for i dage siden
            prediction_date = datetime.now() - timedelta(days=i//2)
            
            # Gem forudsigelse
            store.store_prediction(
                prediction=prediction,
                actual_value=actual_value,
                features_used=features_used,
                model_version="v1.0.0"
            )
            
        logger.info(f"Gemt {num_predictions} test forudsigelser")
        
        # Hent de seneste forudsigelser
        recent_predictions = store.get_recent_predictions(days=7)
        
        # Vis nogle statistikker
        print("\n=== Seneste Forudsigelser ===")
        print(f"Antal seneste forudsigelser: {len(recent_predictions)}")
        
        # Beregn metrics
        metrics = store.get_prediction_metrics()
        
        if metrics:
            print("\n=== Prediction Metrics ===")
            print(f"Total antal forudsigelser: {metrics['total_predictions']}")
            print(f"Accuracy: {metrics['accuracy']:.2f}")
            print(f"Forudsigelse distribution: {metrics['prediction_distribution']}")
        
    except Exception as e:
        logger.error(f"Fejl under test af PredictionStore: {e}")

if __name__ == "__main__":
    main() 