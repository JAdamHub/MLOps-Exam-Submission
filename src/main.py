import logging
import uvicorn
import threading
import multiprocessing
import signal
from pathlib import Path
import sys
import time
import schedule
import subprocess
import os

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

# Global process variables
api_process = None
streamlit_process = None

def start_api_server():
    """Starts the FastAPI server as a separate process so it can be terminated later"""
    global api_process
    
    # If there's an existing process, terminate it
    if api_process is not None and api_process.is_alive():
        logger.info("Stopping existing API server...")
        api_process.terminate()
        api_process.join(timeout=5)
    
    # Create a new process
    api_process = multiprocessing.Process(
        target=run_api,
        daemon=True
    )
    api_process.start()
    logger.info(f"API server started (PID: {api_process.pid})")
    return api_process

def run_api():
    """Runs the FastAPI server"""
    try:
        uvicorn.run("api.stock_api:app", host="0.0.0.0", port=8000, reload=False)
    except Exception as e:
        logger.error(f"Error starting API: {e}")
        raise

def start_streamlit_app():
    """Starts the Streamlit app as a separate process"""
    global streamlit_process
    
    # If there's an existing process, terminate it
    if streamlit_process is not None and streamlit_process.is_alive():
        logger.info("Stopping existing Streamlit app...")
        streamlit_process.terminate()
        streamlit_process.join(timeout=5)
    
    # Create a new process
    streamlit_process = multiprocessing.Process(
        target=run_streamlit,
        daemon=True
    )
    streamlit_process.start()
    logger.info(f"Streamlit app started (PID: {streamlit_process.pid})")
    return streamlit_process

def run_streamlit():
    """Runs the Streamlit app"""
    try:
        subprocess.run(["streamlit", "run", "src/streamlit/app.py"], 
                      check=True)
    except Exception as e:
        logger.error(f"Error starting Streamlit: {e}")

def run_scheduled_pipeline():
    """Run pipeline and restart API"""
    try:
        logger.info("TIK TOK!!! 08:30 - Starting scheduled pipeline run...")
        from pipeline.pipeline_start import main as run_pipeline
        success = run_pipeline()
        
        if success:
            logger.info("Pipeline update completed successfully, restarting API...")
            # Restart the API server to use the new model
            start_api_server()
        else:
            logger.error("Scheduled pipeline run failed")
    except Exception as e:
        logger.error(f"Error during scheduled pipeline: {e}")

def main():
    """Main function that starts all components"""
    try:
        # Enable clean process management
        multiprocessing.set_start_method('spawn', force=True)
        
        # Run pipeline first
        logger.info("Starting initial pipeline run...")
        # from pipeline.pipeline_start import main as run_pipeline
        # run_pipeline() # Commented out to skip initial pipeline run
        logger.info("Skipping initial pipeline run as requested.")
        
        # Start API in a separate process
        start_api_server()
        
        # Start streamlit app in separate process
        start_streamlit_app()
        
        # Schedule pipeline run at 08:30
        schedule.every().day.at("08:30").do(run_scheduled_pipeline)
        logger.info("Pipeline update scheduled for 08:30 daily")
        
        # Keep main thread alive and check for scheduled tasks
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
            
    except KeyboardInterrupt:
        logger.info("Exiting program...")
        # Clean shutdown of processes
        if api_process and api_process.is_alive():
            api_process.terminate()
        if streamlit_process and streamlit_process.is_alive():
            streamlit_process.terminate()
    except Exception as e:
        logger.error(f"Critical error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 