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

# Add project root to path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from src.pipeline.main import main as run_pipeline
from src.api.stock_api import app

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_api():
    """Starter API serveren i en separat tråd"""
    try:
        logger.info("Starter API server...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logger.error(f"Fejl ved start af API server: {str(e)}")
        raise

def run_daily_pipeline():
    """Kør hele pipelinen fra bunden hver dag"""
    try:
        logger.info("Starter daglig pipeline kørsel...")
        success = run_pipeline()
        
        if success:
            logger.info("Pipeline kørsel gennemført succesfuldt")
        else:
            logger.error("Pipeline kørsel fejlede")
            
    except Exception as e:
        logger.error(f"Fejl under pipeline kørsel: {str(e)}")
        raise

def main():
    """Hovedfunktion til at starte scheduler"""
    # Start API server i en separat tråd
    api_thread = threading.Thread(target=run_api)
    api_thread.daemon = True
    api_thread.start()
    
    # Kør pipeline med det samme ved opstart
    run_daily_pipeline()
    
    # Planlæg daglig kørsel kl. 8:00
    schedule.every().day.at("08:00").do(run_daily_pipeline)
    
    logger.info("Scheduler startet - kører pipeline dagligt kl. 8:00")
    
    # Kør scheduler i en uendelig løkke
    while True:
        schedule.run_pending()
        time.sleep(60)  # Tjek hvert minut

if __name__ == "__main__":
    main() 