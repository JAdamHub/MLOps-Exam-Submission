import logging
import uvicorn
import threading
from pathlib import Path
import sys
import time

# Tilføj project root til Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from monitoring.scheduler import ModelUpdateScheduler
from pipeline.training import train_model
from monitoring.drift_detector import DriftDetector
from monitoring.evaluation import ModelEvaluator

# Konfigurer logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / "pipeline.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def run_api():
    """Kør FastAPI serveren."""
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)

def run_scheduler():
    """Kør model update scheduler."""
    scheduler = ModelUpdateScheduler()
    scheduler.start()

def initial_training():
    """Udfør initial model træning og generer visualiseringer."""
    try:
        logger.info("Starter initial model træning...")
        train_model()
        
        logger.info("Genererer drift detection visualiseringer...")
        drift_detector = DriftDetector()
        drift_detector.detect_drift()
        
        logger.info("Genererer model evaluering visualiseringer...")
        model_evaluator = ModelEvaluator()
        model_evaluator.evaluate_model()
        
        logger.info("Initial træning og visualiseringer færdig!")
    except Exception as e:
        logger.error(f"Fejl under initial træning: {e}")

def main():
    """Hovedfunktion der starter hele pipeline."""
    try:
        # Start initial træning i en separat tråd
        training_thread = threading.Thread(target=initial_training)
        training_thread.start()
        
        # Start scheduler i en separat tråd
        scheduler_thread = threading.Thread(target=run_scheduler)
        scheduler_thread.start()
        
        # Start API server
        logger.info("Starter API server...")
        run_api()
        
    except KeyboardInterrupt:
        logger.info("Afslutter pipeline...")
    except Exception as e:
        logger.error(f"Fejl i main: {e}")

if __name__ == "__main__":
    main() 