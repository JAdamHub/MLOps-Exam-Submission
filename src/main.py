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

def initial_setup():
    """Udfør initial setup af systemet."""
    try:
        logger.info("Starter initial system setup...")
        
        # Initialiser komponenter
        drift_detector = DriftDetector()
        model_evaluator = ModelEvaluator()
        
        # Tjek om model eksisterer
        model_path = project_root / "models" / "xgboost_model.joblib"
        if not model_path.exists():
            logger.info("Ingen eksisterende model fundet - starter initial træning")
            train_model()
        else:
            logger.info("Eksisterende model fundet - springer initial træning over")
        
        # Initial evaluering
        logger.info("Udfører initial model evaluering...")
        model_evaluator.evaluate_model()
        
        logger.info("Initial setup færdig!")
        
    except Exception as e:
        logger.error(f"Fejl under initial setup: {e}")
        raise

def main():
    """Hovedfunktion der starter hele pipeline."""
    try:
        # Start initial setup i en separat tråd
        setup_thread = threading.Thread(target=initial_setup)
        setup_thread.start()
        
        # Vent på at setup er færdigt
        setup_thread.join()
        
        # Initialiser og start scheduler
        scheduler = ModelUpdateScheduler()
        scheduler_thread = threading.Thread(target=scheduler.start)
        scheduler_thread.daemon = True  # Så den stopper når hovedprogrammet stopper
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