import logging
import uvicorn
import threading
from pathlib import Path
import sys
import importlib

# add src to path
sys.path.append(str(Path(__file__).resolve().parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# try to import scheduler
try:
    from monitoring.scheduler import ModelUpdateScheduler
    scheduler_available = True
except ImportError as e:
    logger.warning(f"scheduler module not available: {e}")
    scheduler_available = False

def run_api():
    """Runs the FastAPI server"""
    try:
        uvicorn.run("api.stock_api:app", host="0.0.0.0", port=8000, reload=False)
    except Exception as e:
        logger.error(f"error starting api: {e}")
        raise

def run_scheduler():
    """Runs the model update scheduler"""
    if scheduler_available:
        try:
            scheduler = ModelUpdateScheduler()
            scheduler.start()
        except Exception as e:
            logger.error(f"error starting scheduler: {e}")
    else:
        logger.warning("scheduler not available, skipping")

def main():
    """Main function that starts all components"""
    try:
        # Run pipeline first
        logger.info("Starting initial pipeline run...")
        from pipeline.pipeline_start import main as run_pipeline
        run_pipeline()
        
        # Start API in a separate thread
        api_thread = threading.Thread(target=run_api)
        api_thread.daemon = True
        api_thread.start()
        logger.info("API server started")
        
        # Start streamlit app in separate thread
        def run_streamlit():
            import subprocess
            subprocess.run(["streamlit", "run", "src/streamlit/app.py"])
        
        streamlit_thread = threading.Thread(target=run_streamlit)
        streamlit_thread.daemon = True
        streamlit_thread.start()
        logger.info("Streamlit app started")
        
        # Start scheduler for timed pipeline runs
        if scheduler_available:
            scheduler_thread = threading.Thread(target=run_scheduler)
            scheduler_thread.daemon = True
            scheduler_thread.start()
            logger.info("Model update scheduler started - next run scheduled at 20:30")
        
        # Keep main thread alive
        while True:
            threading.Event().wait(60)  # Check every minute
    except KeyboardInterrupt:
        logger.info("Exiting program...")
    except Exception as e:
        logger.error(f"Critical error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 