import logging
import sys
import importlib
import os
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_module(module_name, skip_errors=False):
    """Run a Python module by name"""
    try:
        logging.info(f"Starting module: {module_name}")
        module = importlib.import_module(module_name)
        if hasattr(module, 'main'):
            module.main()
        else:
            logging.warning(f"Module {module_name} has no main() function")
        logging.info(f"Completed module: {module_name}")
        return True
    except Exception as e:
        logging.error(f"Error running {module_name}: {e}")
        if not skip_errors:
            raise
        return False

def main():
    """
    Main function to run the entire data pipeline.
    Processing steps:
    1. Collect cryptocurrency data (Bitcoin)
    2. Collect macroeconomic data
    3. Combine datasets (kun handelsdage - US stock market åbningsdage)
    4. Preprocess combined data
    5. Feature engineering
    6. Train machine learning model
    """
    # Load environment variables from .env file
    dotenv_path = Path(__file__).resolve().parents[2] / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        logging.info(f"Loaded environment variables from {dotenv_path}")
    else:
        logging.warning(f".env file not found at {dotenv_path}")
        
    logging.info("=== Starting ML Pipeline ===")
    logging.info("BEMÆRK: Denne pipeline arbejder kun med data fra handelsdage (US stock market åbningsdage)")
    
    # Data Collection steps
    if not run_module("src.pipeline.crypto_data_collector", skip_errors=False):
        logging.error("Halting pipeline due to error in crypto data collection")
        sys.exit(1)
    
    # Kører makroøkonomisk datahentning (nu med Yahoo Finance)
    if not run_module("src.pipeline.macro_economic_collector", skip_errors=False):
        logging.error("Halting pipeline due to error in macro economic data collection")
        sys.exit(1)
    
    # Data Integration
    if not run_module("src.pipeline.combined_data_processor", skip_errors=False):
        logging.error("Halting pipeline due to error in dataset combination")
        sys.exit(1)
    
    # Data Preprocessing
    if not run_module("src.pipeline.preprocessing", skip_errors=False):
        logging.error("Halting pipeline due to error in data preprocessing")
        sys.exit(1)
    
    # Feature Engineering
    if not run_module("src.pipeline.feature_engineering", skip_errors=False):
        logging.error("Halting pipeline due to error in feature engineering")
        sys.exit(1)
    
    # Model Training 
    if not run_module("src.pipeline.training", skip_errors=False):
        logging.error("Halting pipeline due to error in model training")
        sys.exit(1)
    
    logging.info("=== ML Pipeline Completed Successfully ===")

if __name__ == "__main__":
    main() 