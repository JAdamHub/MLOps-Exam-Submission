import logging
import sys
import importlib
import os
from pathlib import Path
from dotenv import load_dotenv
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_module(module_name, skip_errors=False):
    """Run a Python module by name"""
    try:
        logging.info(f"Starting module: {module_name}")
        module = importlib.import_module(module_name)
        if hasattr(module, 'main'):
            result = module.main()
            # Check if the module returns a boolean success status
            if isinstance(result, bool):
                if not result and not skip_errors:
                    logging.warning(f"Module {module_name} returned False, indicating errors")
                    return False
            logging.info(f"Completed module: {module_name}")
            return True
        else:
            logging.warning(f"Module {module_name} has no main() function")
            return skip_errors
    except Exception as e:
        logging.error(f"Error running {module_name}: {e}")
        if not skip_errors:
            raise
        return False

def main():
    """
    Main function to run the entire data pipeline.
    Processing steps:
    1. Collect stock data (Vestas) and macroeconomic data
    2. Combine datasets (kun handelsdage - danske børs åbningsdage)
    3. Preprocess combined data
    4. Feature engineering
    5. Train machine learning model (LSTM eller XGBoost)
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the ML pipeline')
    parser.add_argument('--model', choices=['xgboost', 'lstm'], default='lstm',
                      help='Choose which model to train (default: lstm)')
    args = parser.parse_args()
    
    # Load environment variables from .env file
    dotenv_path = Path(__file__).resolve().parents[2] / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        logging.info(f"Loaded environment variables from {dotenv_path}")
    else:
        logging.warning(f".env file not found at {dotenv_path}")
        
    logging.info("=== Starting ML Pipeline ===")
    logging.info("BEMÆRK: Denne pipeline arbejder kun med data fra handelsdage (danske børs åbningsdage)")
    
    # Data Collection steps - indsamler både Vestas aktiedata og makroøkonomiske data
    # Brug skip_errors=True, da vi har implementeret fallback-mekanismer i stock_data_collector.py
    if not run_module("src.pipeline.stock_data_collector", skip_errors=True):
        logging.warning("Stock data collection encountered errors but continuing with fallback data")
    else:
        logging.info("Stock data collection completed successfully")
    
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
    
    # Model Training - vælg mellem LSTM eller XGBoost
    if args.model == 'lstm':
        logging.info("Training LSTM model...")
        training_module = "src.pipeline.training-lstm"
    else:
        logging.info("Training XGBoost model...")
        training_module = "src.pipeline.training"
    
    if not run_module(training_module, skip_errors=False):
        logging.error(f"Halting pipeline due to error in {args.model} model training")
        sys.exit(1)
    
    logging.info(f"=== ML Pipeline Completed Successfully with {args.model.upper()} model ===")

if __name__ == "__main__":
    main() 