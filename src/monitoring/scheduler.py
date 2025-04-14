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

from pipeline.pipeline_start import main as run_pipeline
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

class ModelUpdateScheduler:
    """Class to manage scheduling of model updates and API server"""
    
    def __init__(self, run_at_startup=False, schedule_time="20:30"):
        """
        Initialize scheduler
        
        Args:
            run_at_startup: Whether to run the pipeline immediately on startup
            schedule_time: Time to run the daily pipeline (24-hour format)
        """
        self.run_at_startup = run_at_startup
        self.schedule_time = schedule_time
        logger.info(f"Scheduler initialized (daily run at {schedule_time})")
    
    def run_daily_pipeline(self):
        """Run the entire pipeline and restart API server"""
        try:
            logger.info("TIK TOK!!! 20:30 - STARTING DAILY PIPELINE RUN...")
            success = run_pipeline()
            
            if success:
                logger.info("Pipeline run completed successfully")
                # Signal to restart API (could be implemented via shared file or message queue)
                logger.info("Restarting API server to use new model...")
                # Implementation for API restart
            else:
                logger.error("Pipeline run failed")
                
        except Exception as e:
            logger.error(f"Error during pipeline run: {str(e)}")
            raise
    
    def start(self):
        """Start the scheduler without API management"""
        # Run pipeline immediately on startup if configured
        if self.run_at_startup:
            logger.info("Running initial pipeline as requested by run_at_startup=True")
            self.run_daily_pipeline()
        else:
            logger.info(f"Skipping initial pipeline run, will run at scheduled time: {self.schedule_time}")
        
        # Schedule daily run at configured time
        schedule.every().day.at(self.schedule_time).do(self.run_daily_pipeline)
        
        logger.info(f"Scheduler started - running pipeline daily at {self.schedule_time}")
        
        # Run scheduler in an infinite loop
        while True:
            schedule.run_pending()
            time.sleep(60)  # check every minute

def main():
    """Main function to start the scheduler"""
    scheduler = ModelUpdateScheduler()
    scheduler.start()

if __name__ == "__main__":
    main() 