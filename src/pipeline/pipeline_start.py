import logging
import sys
import importlib
import os
from pathlib import Path
from dotenv import load_dotenv
import sqlite3

# Import main functions directly
from src.pipeline import stock_data_collector, preprocessing, feature_engineering, training_lstm, model_results_visualizer

# configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def run_module(module_name, skip_errors=False):
#     """run a python module by name"""
#     try:
#         logging.info(f"starting module: {module_name}")
#         # add sys.path to include project root
#         sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
#         module = importlib.import_module(module_name)
#         if hasattr(module, 'main'):
#             result = module.main()
#             # check if the module returns a boolean success status
#             if isinstance(result, bool):
#                 if not result and not skip_errors:
#                     logging.warning(f"module {module_name} returned false, indicating errors")
#                     return False
#             logging.info(f"completed module: {module_name}")
#             return True
#         else:
#             logging.warning(f"module {module_name} has no main() function")
#             return skip_errors
#     except Exception as e:
#         logging.error(f"error running {module_name}: {e}")
#         if not skip_errors:
#             raise
#         return False

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
    # if not run_module("src.pipeline.stock_data_collector", skip_errors=False):
    #     logging.error("stock data collection failed - stopping pipeline")
    #     sys.exit(1)
    # else:
    #     logging.info("stock data collection completed successfully")
    logging.info("--- 1. Running Stock Data Collection ---")
    if not stock_data_collector.main():
        logging.error("Stock data collection failed. Stopping pipeline.")
        return False
    logging.info("Stock data collection completed successfully.")
    
    # data preprocessing
    # if not run_module("src.pipeline.preprocessing", skip_errors=False):
    #     logging.error("halting pipeline due to error in data preprocessing")
    #     sys.exit(1)
    logging.info("--- 2. Running Data Preprocessing ---")
    df_preprocessed = preprocessing.main()
    if df_preprocessed is None:
        logging.error("Data preprocessing failed. Stopping pipeline.")
        return False
    logging.info("Data preprocessing completed successfully.")
    
    # feature engineering
    # if not run_module("src.pipeline.feature_engineering", skip_errors=False):
    #     logging.error("halting pipeline due to error in feature engineering")
    #     sys.exit(1)
    logging.info("--- 3. Running Feature Engineering ---")
    df_features = feature_engineering.main(df_preprocessed)
    if df_features is None:
        logging.error("Feature engineering failed. Stopping pipeline.")
        return False
    logging.info("Feature engineering completed successfully.")
    
    # model training - only lstm
    # training_module = "src.pipeline.training-lstm"
    # if not run_module(training_module, skip_errors=False):
    #     logging.error("halting pipeline due to error in lstm model training")
    #     sys.exit(1)
    logging.info("--- 4. Training LSTM Model ---")
    if not training_lstm.main(df_features):
        logging.error("LSTM model training failed. Stopping pipeline.")
        return False
    logging.info("LSTM model training completed successfully.")
    
    # save lstm metrics
    # training_module = "src.pipeline.model_results_visualizer"
    # if not run_module(training_module, skip_errors=False):
    #     logging.error("halting pipeline due to error in saving lstm model metrics...")
    #     sys.exit(1)
    logging.info("--- 5. Visualizing LSTM Model Metrics ---")
    if not model_results_visualizer.main():
         logging.error("Saving LSTM model metrics failed.")
         # Continue pipeline even if visualization fails, but log error
    else:
        logging.info("LSTM model metrics visualization completed successfully.")

    logging.info("=== ML Pipeline completed successfully ===")
    return True

if __name__ == "__main__":
    if not main():
        sys.exit(1) 