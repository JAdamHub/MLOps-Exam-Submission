import schedule
import time
import logging
from pathlib import Path
import json
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from pipeline.training import train_model, load_data, prepare_data, select_features
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
            "accuracy_history": [],
            "last_update": None,
            "model_versions": []
        }
        with open(self.metrics_file, 'w') as f:
            json.dump(initial_metrics, f, indent=4)
    
    def update_model_metrics(self, accuracy):
        """Opdaterer metrics filen med nye metrics"""
        try:
            with open(self.metrics_file, 'r') as f:
                metrics = json.load(f)
            
            # Tilføj nye metrics
            metrics["accuracy_history"].append({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "accuracy": accuracy
            })
            metrics["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            metrics["model_versions"].append({
                "version": len(metrics["model_versions"]) + 1,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "accuracy": accuracy
            })
            
            # Gem opdaterede metrics
            with open(self.metrics_file, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            logger.info(f"Metrics opdateret med ny accuracy: {accuracy:.4f}")
            
        except Exception as e:
            logger.error(f"Fejl ved opdatering af metrics: {e}")
    
    def update_model(self):
        """Opdaterer modellen med ny data og evaluerer performance"""
        try:
            logger.info("Starter daglig model opdatering...")
            
            # Load og forbered data
            df = load_data()
            if df is None:
                raise ValueError("Kunne ikke indlæse data")
            
            X, y_dict, feature_columns, target_columns, scaler = prepare_data(df)
            X_selected, selected_features = select_features(X, y_dict, feature_columns)
            
            # Træn ny model
            models, metrics, feature_importances = train_model(
                X_selected, y_dict, selected_features, target_columns
            )
            
            # Opdater metrics
            accuracy = metrics['accuracy']
            self.update_model_metrics(accuracy)
            
            logger.info(f"Model opdateret succesfuldt. Ny accuracy: {accuracy:.4f}")
            
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
        schedule.every().day.at("01:00").do(self.update_model)
        
        # Kør første opdatering med det samme
        self.update_model()
        
        # Hold scheduler kørende
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)
    
    def stop(self):
        """Stopper scheduler"""
        self.is_running = False
        logger.info("Stopper model update scheduler...") 