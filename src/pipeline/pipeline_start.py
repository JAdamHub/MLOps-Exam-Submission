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
    
    logging.info("--- 1. Running Stock Data Collection ---")
    if not stock_data_collector.main():
        logging.error("Stock data collection failed. Stopping pipeline.")
        return False
    logging.info("Stock data collection completed successfully.")
    
    logging.info("--- 2. Running Data Preprocessing ---")
    df_preprocessed = preprocessing.main()
    if df_preprocessed is None:
        logging.error("Data preprocessing failed. Stopping pipeline.")
        return False
    logging.info("Data preprocessing completed successfully.")
    
    logging.info("--- 3. Running Feature Engineering ---")
    df_features = feature_engineering.main(df_preprocessed)
    if df_features is None:
        logging.error("Feature engineering failed. Stopping pipeline.")
        return False
    logging.info("Feature engineering completed successfully.")

    logging.info("--- 4. Training LSTM Model ---")
    if not training_lstm.main(df_features):
        logging.error("LSTM model training failed. Stopping pipeline.")
        return False
    logging.info("LSTM model training completed successfully.")
    
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