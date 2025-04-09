import logging
from pathlib import Path
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from .drift_detector import DriftDetector
from .evaluation import ModelEvaluator
from .prediction_store import PredictionStore
import sys
import time

# Konfigurer logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelUpdateScheduler:
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.drift_detector = DriftDetector()
        self.model_evaluator = ModelEvaluator()
        self.prediction_store = PredictionStore()
        self.project_root = Path(__file__).resolve().parents[2]
        self.is_running = False
        
    def start(self):
        """Start scheduler med daglige jobs"""
        try:
            if self.is_running:
                logger.warning("Scheduler er allerede kørende")
                return
                
            # Planlæg drift detection hver dag kl. 01:00
            self.scheduler.add_job(
                self.check_and_update_model,
                'cron',
                hour=1,
                minute=0,
                id='daily_model_check'
            )
            
            # Start scheduler
            self.scheduler.start()
            self.is_running = True
            logger.info("Model update scheduler started successfully")
            
        except Exception as e:
            logger.error(f"Error starting scheduler: {e}")
            raise
            
    def check_and_update_model(self):
        """Tjek for drift og opdater model hvis nødvendigt"""
        try:
            logger.info("Starting daily model check...")
            
            # Tjek for drift
            drift_results = self.drift_detector.detect_drift()
            
            if self.drift_detector.should_retrain(drift_results):
                logger.info("Drift detected - initiating model retraining")
                self.retrain_model()
            else:
                logger.info("No significant drift detected - model remains unchanged")
                
            # Evaluer model performance
            self.evaluate_model_performance()
            
        except Exception as e:
            logger.error(f"Error in model check: {e}")
            
    def retrain_model(self):
        """Retrain model med ny data"""
        try:
            # Import træningsmoduler her for at undgå cirkulære imports
            sys.path.append(str(self.project_root))
            from src.pipeline.training import main as train_model
            
            # Kør model træning
            logger.info("Starting model retraining...")
            train_model()
            logger.info("Model retraining completed successfully")
            
            # Vent kort tid for at sikre at alle filer er gemt
            time.sleep(2)
            
            # Evaluer den nye model med det samme
            self.evaluate_model_performance()
            
        except Exception as e:
            logger.error(f"Error during model retraining: {e}")
            
    def evaluate_model_performance(self):
        """Evaluer model performance og gem metrics"""
        try:
            # Hent seneste prædiktioner
            recent_predictions = self.prediction_store.get_recent_predictions(days=7)
            
            if not recent_predictions.empty:
                # Evaluer prædiktioner
                metrics = self.model_evaluator.evaluate_predictions(
                    predictions=recent_predictions['prediction'].tolist(),
                    actual_values=recent_predictions['actual_value'].tolist(),
                    timestamp=datetime.now().isoformat()
                )
                
                # Hent prediction metrics
                prediction_metrics = self.prediction_store.get_prediction_metrics()
                
                # Kombinér metrics
                combined_metrics = {
                    **metrics,
                    'prediction_metrics': prediction_metrics
                }
                
                logger.info(f"Model evaluation completed. Metrics: {combined_metrics}")
            else:
                logger.warning("No recent predictions available for evaluation")
                
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            
    def stop(self):
        """Stop scheduler og cleanup"""
        try:
            if self.is_running:
                self.scheduler.shutdown()
                self.is_running = False
                logger.info("Model update scheduler stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}") 