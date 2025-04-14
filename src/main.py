import logging
import uvicorn
import threading
from pathlib import Path
import sys
import importlib
import time
import schedule

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
        
        # Start API in a separate thread - store the process/thread reference
        api_thread = start_api_server()
        logger.info("API server started")
        
        # Start streamlit app in separate thread
        streamlit_thread = start_streamlit_app()
        logger.info("Streamlit app started")
        
        # Schedule pipeline run at 20:30
        schedule.every().day.at("20:30").do(run_scheduled_pipeline, api_thread)
        logger.info("Pipeline update scheduled for 20:30 daily")
        
        # Keep main thread alive and check for scheduled tasks
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
            
    except KeyboardInterrupt:
        logger.info("Exiting program...")
    except Exception as e:
        logger.error(f"Critical error: {e}")
        sys.exit(1)

def run_scheduled_pipeline(api_thread):
    """Run pipeline and restart API"""
    try:
        logger.info("TIK TOK!!! 20:30 - Starting scheduled pipeline run at 20:30...")
        from pipeline.pipeline_start import main as run_pipeline
        success = run_pipeline()
        
        if success:
            logger.info("Pipeline update completed successfully, restarting API...")
            # Stop the current API thread
            api_thread.terminate()  # (needs proper implementation)
            # Start new API thread with updated model
            new_api_thread = start_api_server()
            return new_api_thread
        else:
            logger.error("Scheduled pipeline run failed")
            return api_thread
    except Exception as e:
        logger.error(f"Error during scheduled pipeline: {e}")
        return api_thread

if __name__ == "__main__":
    main() 