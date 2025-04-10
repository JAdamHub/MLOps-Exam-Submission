from src.monitoring.scheduler import ModelUpdateScheduler
import logging
import time

# Konfigurer logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Test ModelUpdateScheduler funktionalitet"""
    try:
        logger.info("Starter ModelUpdateScheduler i testmode...")
        
        # Opret scheduler
        scheduler = ModelUpdateScheduler()
        
        # Nu vil vi manuelt udføre checks i stedet for at bruge scheduling
        logger.info("Udfører manuel tjek af model og drift...")
        
        # Kør drift detection og model opdatering tjek
        scheduler.check_and_update_model()
        
        # Kør model evaluering
        logger.info("Udfører manuel model evaluering...")
        scheduler.evaluate_model_performance()
        
        # Vent lidt for at vise logs
        logger.info("Scheduler testning er afsluttet.")
        
    except Exception as e:
        logger.error(f"Fejl under test af scheduler: {e}")

if __name__ == "__main__":
    main() 