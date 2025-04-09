import logging
import sys
from pathlib import Path

# Add src directory to Python path to allow importing pipeline modules
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Import the main functions from each pipeline step
from pipeline import crypto_data_collector, preprocessing, feature_engineering, training
from pipeline import macro_economic_collector, combined_data_processor  # Updated module names

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_pipeline():
    """Runs the complete data processing and model training pipeline."""
    logging.info("========== Starting Full Pipeline Execution ==========")

    try:
        # Step 1: Cryptocurrency Data Collection
        logging.info("--- Running Cryptocurrency Data Collection ---")
        crypto_data_collector.main()
        logging.info("--- Cryptocurrency Data Collection Finished ---")

        # Step 2: Macroeconomic Data Collection
        logging.info("--- Running Macroeconomic Data Collection ---")
        collector = macro_economic_collector.MacroDataCollector()
        collector.collect_all_macro_data()
        logging.info("--- Macroeconomic Data Collection Finished ---")
        
        # Step 3: Combine Bitcoin and Macroeconomic Data
        logging.info("--- Running Data Combination ---")
        combined_data_processor.main()
        logging.info("--- Data Combination Finished ---")

        # Step 4: Data Preprocessing
        logging.info("--- Running Data Preprocessing ---")
        preprocessing.main()
        logging.info("--- Data Preprocessing Finished ---")

        # Step 5: Feature Engineering
        logging.info("--- Running Feature Engineering ---")
        feature_engineering.main()
        logging.info("--- Feature Engineering Finished ---")

        # Step 6: Model Training
        logging.info("--- Running Model Training ---")
        training.main()
        logging.info("--- Model Training Finished ---")

        logging.info("========== Full Pipeline Execution Completed Successfully ==========")

    except SystemExit as e:
        # Catch SystemExit raised by pipeline steps on failure
        logging.error(f"Pipeline execution halted with exit code {e.code}. Check previous logs for errors.")
        sys.exit(e.code)
    except Exception as e:
        logging.error(f"An unexpected error occurred during pipeline execution: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline() 