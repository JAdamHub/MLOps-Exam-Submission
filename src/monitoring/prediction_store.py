import pandas as pd
import json
from pathlib import Path
import logging
from datetime import datetime

class PredictionStore:
    def __init__(self, store_path="data/predictions"):
        self.project_root = Path(__file__).resolve().parents[2]
        self.store_path = self.project_root / store_path
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.predictions_file = self.store_path / "predictions_history.csv"
        
        # Opret predictions fil hvis den ikke eksisterer
        if not self.predictions_file.exists():
            self._create_empty_predictions_file()
            
    def _create_empty_predictions_file(self):
        """Opret tom predictions fil med korrekt struktur"""
        df = pd.DataFrame(columns=[
            'timestamp',
            'prediction',
            'actual_value',
            'features_used',
            'model_version'
        ])
        df.to_csv(self.predictions_file, index=False)
        
    def store_prediction(self, prediction, actual_value, features_used, model_version):
        """Gem en ny prædiktion"""
        try:
            # Læs eksisterende predictions
            df = pd.read_csv(self.predictions_file)
            
            # Opret ny prediction record
            new_prediction = pd.DataFrame([{
                'timestamp': datetime.now().isoformat(),
                'prediction': prediction,
                'actual_value': actual_value,
                'features_used': json.dumps(features_used),
                'model_version': model_version
            }])
            
            # Tilføj til eksisterende predictions
            df = pd.concat([df, new_prediction], ignore_index=True)
            
            # Gem opdateret predictions
            df.to_csv(self.predictions_file, index=False)
            logging.info(f"Stored new prediction: {prediction}")
            
        except Exception as e:
            logging.error(f"Error storing prediction: {e}")
            
    def get_recent_predictions(self, days=7):
        """Hent seneste prædiktioner"""
        try:
            df = pd.read_csv(self.predictions_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filtrer på seneste X dage
            recent_date = datetime.now() - pd.Timedelta(days=days)
            recent_predictions = df[df['timestamp'] >= recent_date]
            
            return recent_predictions
            
        except Exception as e:
            logging.error(f"Error getting recent predictions: {e}")
            return pd.DataFrame()
            
    def get_prediction_metrics(self, days=7):
        """Beregn metrics for seneste prædiktioner"""
        try:
            recent_predictions = self.get_recent_predictions(days)
            
            if recent_predictions.empty:
                return None
                
            metrics = {
                'total_predictions': len(recent_predictions),
                'accuracy': (recent_predictions['prediction'] == recent_predictions['actual_value']).mean(),
                'prediction_distribution': recent_predictions['prediction'].value_counts().to_dict(),
                'actual_distribution': recent_predictions['actual_value'].value_counts().to_dict()
            }
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error calculating prediction metrics: {e}")
            return None 