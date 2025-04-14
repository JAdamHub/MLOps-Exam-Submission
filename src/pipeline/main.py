import logging
import sys
import importlib
import os
from pathlib import Path
from dotenv import load_dotenv

# configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_module(module_name, skip_errors=False):
    """run a python module by name"""
    try:
        logging.info(f"starting module: {module_name}")
        # add sys.path to include project root
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        module = importlib.import_module(module_name)
        if hasattr(module, 'main'):
            result = module.main()
            # check if the module returns a boolean success status
            if isinstance(result, bool):
                if not result and not skip_errors:
                    logging.warning(f"module {module_name} returned false, indicating errors")
                    return False
            logging.info(f"completed module: {module_name}")
            return True
        else:
            logging.warning(f"module {module_name} has no main() function")
            return skip_errors
    except Exception as e:
        logging.error(f"error running {module_name}: {e}")
        if not skip_errors:
            raise
        return False

def main():
    """
    main function to run the entire data pipeline.
    processing steps:
    1. collect stock data (vestas) and macroeconomic data
    2. combine datasets (only trading days - danish stock exchange opening days)
    3. preprocess combined data
    4. feature engineering
    5. train lstm model
    """    
    # add sys.path to include project root
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    
    # load environment variables from .env file
    dotenv_path = project_root / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        logging.info(f"loaded environment variables from {dotenv_path}")
    else:
        logging.warning(f".env file not found at {dotenv_path}")
        
    logging.info("=== starting ML Pipeline ===")
    logging.info("note: this pipeline only works with data from trading days (danish stock exchange opening days)")
    
    # data collection steps - collects both vestas stock data and macroeconomic data
    # use skip_errors=true, as we have implemented fallback mechanisms in stock_data_collector.py
    if not run_module("src.pipeline.stock_data_collector", skip_errors=True):
        logging.warning("stock data collection encountered errors but continuing with fallback data")
    else:
        logging.info("stock data collection completed successfully")
    
    # data integration
    if not run_module("src.pipeline.combined_data_processor", skip_errors=False):
        logging.error("halting pipeline due to error in dataset combination")
        sys.exit(1)
    
    # data preprocessing
    if not run_module("src.pipeline.preprocessing", skip_errors=False):
        logging.error("halting pipeline due to error in data preprocessing")
        sys.exit(1)
    
    # feature engineering
    if not run_module("src.pipeline.feature_engineering", skip_errors=False):
        logging.error("halting pipeline due to error in feature engineering")
        sys.exit(1)
    
    # model training - only lstm
    logging.info("training lstm model...")
    training_module = "src.pipeline.training-lstm"
    
    if not run_module(training_module, skip_errors=False):
        logging.error("halting pipeline due to error in lstm model training")
        sys.exit(1)
    
    logging.info("lstm model training completed successfully")
    
    # save lstm metrics
    logging.info(" saving lstm model metric results ")
    training_module = "src.pipeline.model_results_visualizer"
    
    if not run_module(training_module, skip_errors=False):
        logging.error("halting pipeline due to error in saving lstm model metrics...")
        sys.exit(1)

    logging.info("=== ML Pipeline completed successfully ===")

if __name__ == "__main__":
    main() 