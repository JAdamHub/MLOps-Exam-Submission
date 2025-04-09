from src.monitoring.evaluation import ModelEvaluator
from src.monitoring.prediction_store import PredictionStore
import pandas as pd
import numpy as np
import random
import logging
import matplotlib.pyplot as plt
from pathlib import Path

# Konfigurer logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Test ModelEvaluator med simulerede data og rigtige forudsigelser"""
    try:
        # Opret evaluator
        evaluator = ModelEvaluator()
        
        # Hent gemte forudsigelser fra PredictionStore
        store = PredictionStore()
        predictions_df = store.get_recent_predictions()
        
        if predictions_df.empty:
            logger.warning("Ingen forudsigelser fundet i PredictionStore")
            # Opret dummy predictions
            predictions = [random.choice([0, 1]) for _ in range(20)]
            actuals = [p if random.random() < 0.7 else 1-p for p in predictions]
        else:
            logger.info(f"Indlæste {len(predictions_df)} forudsigelser fra PredictionStore")
            predictions = predictions_df['prediction'].tolist()
            actuals = predictions_df['actual_value'].tolist()
        
        # Simuler pris-bevægelser for backtesting
        price_movements = [random.uniform(-500, 700) for _ in range(len(predictions))]
        
        # Evaluer modellen
        logger.info("Evaluerer model...")
        metrics = evaluator.evaluate_predictions(
            predictions=predictions,
            actual_values=actuals,
            price_movement=price_movements
        )
        
        # Vis resultater
        print("\n=== Model Evaluering ===")
        print(f"Accuracy: {metrics['daily_accuracy']:.4f}")
        print(f"Precision: {metrics['daily_precision']:.4f}")
        print(f"Recall: {metrics['daily_recall']:.4f}")
        print(f"F1 Score: {metrics['daily_f1']:.4f}")
        
        # Vis opsamlet metrics
        summary = evaluator.get_metrics_summary()
        
        print("\n=== Samlet Performance ===")
        print(f"Gennemsnitlig accuracy: {summary['average_accuracy']:.4f}")
        print(f"Accuracy trend: {summary['accuracy_trend']}")
        
        if 'profit_loss' in summary and summary['profit_loss'] is not None:
            print(f"Profit/Loss: {summary['profit_loss']:.2f}")
            
        # Vis figur-placering
        figures_dir = Path("models/figures")
        print(f"\nFigurer er gemt i: {figures_dir.absolute()}")
        if figures_dir.exists():
            print("Figurer inkluderer:")
            for fig_file in figures_dir.glob("*.png"):
                print(f"  - {fig_file.name}")
    
    except Exception as e:
        logger.error(f"Fejl under model evaluering: {e}")

if __name__ == "__main__":
    main() 