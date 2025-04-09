import logging
import uvicorn
import threading
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from monitoring.scheduler import ModelUpdateScheduler
from visualization.model_metrics_viz import main as generate_visualizations

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

def run_api():
    """Kører FastAPI serveren"""
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)

def run_scheduler():
    """Kører model update scheduler"""
    scheduler = ModelUpdateScheduler()
    scheduler.start()

def main():
    """Hovedfunktion der starter alle komponenter"""
    try:
        # Start API i en separat thread
        api_thread = threading.Thread(target=run_api)
        api_thread.daemon = True
        api_thread.start()
        logger.info("API server startet")

        # Start scheduler i en separat thread
        scheduler_thread = threading.Thread(target=run_scheduler)
        scheduler_thread.daemon = True
        scheduler_thread.start()
        logger.info("Model update scheduler startet")

        # Generer initial visualiseringer
        generate_visualizations()
        logger.info("Initial visualiseringer genereret")

        # Hold hovedtråden i live
        while True:
            try:
                # Generer visualiseringer hver time
                generate_visualizations()
                threading.Event().wait(3600)  # Vent 1 time
            except KeyboardInterrupt:
                logger.info("Afslutter program...")
                break
            except Exception as e:
                logger.error(f"Fejl under kørsel: {e}")
                continue

    except Exception as e:
        logger.error(f"Kritisk fejl: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 