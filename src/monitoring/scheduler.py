import schedule
import time
import logging
import json
import os
from datetime import datetime
import numpy as np
from pathlib import Path
import sys
import threading
import uvicorn

# add project root to path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from src.pipeline.main import main as run_pipeline
from src.api.stock_api import app

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_api():
    """Starts the api server in a separate thread"""
    try:
        logger.info("Starting api server...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logger.error(f"Error starting api server: {str(e)}")
        raise

def run_daily_pipeline():
    """run the entire pipeline from scratch every day"""
    try:
        logger.info("starting daily pipeline run...")
        success = run_pipeline()
        
        if success:
            logger.info("pipeline run completed successfully")
        else:
            logger.error("pipeline run failed")
            
    except Exception as e:
        logger.error(f"error during pipeline run: {str(e)}")
        raise

def main():
    """main function to start the scheduler"""
    # Start api server in a separate thread
    api_thread = threading.Thread(target=run_api)
    api_thread.daemon = True
    api_thread.start()
    
    # run pipeline immediately on startup
    run_daily_pipeline()
    
    # schedule daily run at 8:00 am
    schedule.every().day.at("08:00").do(run_daily_pipeline)
    
    logger.info("scheduler started - running pipeline daily at 8:00 am")
    
    # run scheduler in an infinite loop
    while True:
        schedule.run_pending()
        time.sleep(60)  # check every minute

if __name__ == "__main__":
    main() 