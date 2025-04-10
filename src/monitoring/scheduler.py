import schedule
import time
import logging
from pathlib import Path
import json
from datetime import datetime
import sys
import numpy as np
sys.path.append(str(Path(__file__).resolve().parents[1]))
from pipeline.training import train_model, load_data, prepare_data, select_features
from pipeline.training_lstm import load_data as load_lstm_data
from pipeline.training_lstm import prepare_data as prepare_lstm_data
from pipeline.training_lstm import train_model as train_lstm_model
from monitoring.evaluation import ModelEvaluator

# Konfigurer logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelUpdateScheduler:
    def __init__(self):
        self.is_running = False
        self.metrics_file = Path("src/monitoring/model_metrics.json")
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialiser metrics fil hvis den ikke eksisterer
        if not self.metrics_file.exists():
            self._initialize_metrics_file()
    
    def _initialize_metrics_file(self):
        """Initialiserer metrics filen med tom struktur"""
        initial_metrics = {
            "lstm_accuracy_history": [],
            "xgboost_accuracy_history": [],
            "last_update": None,
            "lstm_model_versions": [],
            "xgboost_model_versions": []
        }
        with open(self.metrics_file, 'w') as f:
            json.dump(initial_metrics, f, indent=4)
    
    def update_model_metrics(self, model_type, accuracy):
        """Opdaterer metrics filen med nye metrics for en specifik model type"""
        try:
            with open(self.metrics_file, 'r') as f:
                metrics = json.load(f)
            
            # Vælg den korrekte historik og version liste baseret på model type
            history_key = f"{model_type}_accuracy_history"
            versions_key = f"{model_type}_model_versions"
            
            # Tilføj nye metrics
            metrics[history_key].append({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "accuracy": accuracy
            })
            metrics["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            metrics[versions_key].append({
                "version": len(metrics[versions_key]) + 1,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "accuracy": accuracy
            })
            
            # Gem opdaterede metrics
            with open(self.metrics_file, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            logger.info(f"{model_type} metrics opdateret med ny accuracy: {accuracy:.4f}")
            
        except Exception as e:
            logger.error(f"Fejl ved opdatering af {model_type} metrics: {e}")
    
    def update_models(self):
        """Opdaterer både LSTM og XGBoost modeller med ny data og evaluerer performance"""
        try:
            logger.info("Starter daglig model opdatering...")
            
            # Opdater XGBoost model
            df = load_data()
            if df is None:
                raise ValueError("Kunne ikke indlæse data for XGBoost")
            
            X, y_dict, feature_columns, target_columns, scaler = prepare_data(df)
            X_selected, selected_features = select_features(X, y_dict, feature_columns)
            
            # Træn ny XGBoost model
            models, metrics, feature_importances = train_model(
                X_selected, y_dict, selected_features, target_columns
            )
            
            # Opdater XGBoost metrics
            xgboost_accuracy = metrics['accuracy']
            self.update_model_metrics('xgboost', xgboost_accuracy)
            
            # Opdater LSTM model
            df_lstm = load_lstm_data()
            if df_lstm is None:
                raise ValueError("Kunne ikke indlæse data for LSTM")
            
            # Forbered data for LSTM
            data_splits, feature_columns, target_columns, target_scalers = prepare_lstm_data(df_lstm)
            
            # Træn LSTM model
            models, metrics, histories, test_metrics = train_lstm_model(
                data_splits, feature_columns, target_columns, target_scalers
            )
            
            # Opdater LSTM metrics (brug gennemsnitlig accuracy over alle horizons)
            lstm_accuracy = np.mean([m['accuracy'] for m in test_metrics.values()])
            self.update_model_metrics('lstm', lstm_accuracy)
            
            logger.info(f"Modeller opdateret succesfuldt. XGBoost accuracy: {xgboost_accuracy:.4f}, LSTM accuracy: {lstm_accuracy:.4f}")
            
        except Exception as e:
            logger.error(f"Fejl under model opdatering: {e}")
    
    def start(self):
        """Starter scheduler"""
        if self.is_running:
            logger.warning("Scheduler kører allerede")
            return
        
        self.is_running = True
        logger.info("Starter model update scheduler...")
        
        # Planlæg daglig opdatering kl. 01:00
        schedule.every().day.at("01:00").do(self.update_models)
        
        # Kør første opdatering med det samme
        self.update_models()
        
        # Hold scheduler kørende
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)
    
    def stop(self):
        """Stopper scheduler"""
        self.is_running = False
        logger.info("Stopper model update scheduler...") 