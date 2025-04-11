import schedule
import time
import logging
import json
import os
from datetime import datetime
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.lstm_model import LSTMModel
from src.data.data_processor import DataProcessor
from src.pipeline.stock_data_collector import collect_stock_data
from src.pipeline.preprocessing import preprocess_data, save_processed_data
from src.pipeline.combined_data_processor import process_interval_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelScheduler:
    def __init__(self):
        self.running = False
        self.metrics_file = 'model_metrics.json'
        self.load_metrics()
        
    def load_metrics(self):
        """Load existing metrics or create new metrics file"""
        if os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'r') as f:
                self.metrics = json.load(f)
        else:
            self.metrics = {
                'lstm': {
                    'accuracy_history': [],
                    'versions': [],
                    'last_update': None
                }
            }
            self.save_metrics()
    
    def save_metrics(self):
        """Save metrics to file"""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)
    
    def collect_new_data(self):
        """Collect new data from APIs"""
        try:
            logger.info("Starting daily data collection")
            
            # Collect new stock data
            stock_data = collect_stock_data()
            if stock_data is None:
                raise ValueError("Failed to collect new stock data")
            
            # Process the new data
            if not process_interval_data(interval="daily"):
                raise ValueError("Failed to process new data")
            
            logger.info("New data collected and processed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error collecting new data: {str(e)}", exc_info=True)
            return False
    
    def update_lstm_model(self):
        """Update LSTM model with new data"""
        try:
            logger.info("Starting LSTM model update")
            
            # First collect new data
            if not self.collect_new_data():
                raise ValueError("Failed to collect new data")
            
            # Initialize data processor and load data
            data_processor = DataProcessor()
            data_processor.load_data()
            
            # Prepare sequential data
            sequence_length = 10
            X_train, y_train, X_test, y_test = data_processor.prepare_sequential_data(
                sequence_length=sequence_length
            )
            
            # Initialize and train LSTM model
            lstm_model = LSTMModel()
            lstm_model.train(X_train, y_train, X_test, y_test)
            
            # Evaluate model
            test_predictions = lstm_model.predict(X_test)
            test_accuracy = np.mean(np.argmax(test_predictions, axis=1) == np.argmax(y_test, axis=1))
            
            # Update metrics
            self.metrics['lstm']['accuracy_history'].append(float(test_accuracy))
            self.metrics['lstm']['versions'].append({
                'version': len(self.metrics['lstm']['versions']) + 1,
                'accuracy': float(test_accuracy),
                'timestamp': datetime.now().isoformat()
            })
            self.metrics['lstm']['last_update'] = datetime.now().isoformat()
            
            # Save updated metrics
            self.save_metrics()
            
            logger.info(f"LSTM model updated successfully. New accuracy: {test_accuracy:.4f}")
            
        except Exception as e:
            logger.error(f"Error updating LSTM model: {str(e)}", exc_info=True)
    
    def run_scheduler(self):
        """Run the scheduler"""
        self.running = True
        logger.info("Starting scheduler")
        
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def start(self):
        """Start the scheduler"""
        # Schedule daily model updates
        schedule.every().day.at("01:00").do(self.update_lstm_model)
        
        # Run the scheduler in a separate thread
        import threading
        scheduler_thread = threading.Thread(target=self.run_scheduler)
        scheduler_thread.start()
        
        logger.info("Scheduler started")
    
    def stop(self):
        """Stop the scheduler"""
        self.running = False
        logger.info("Scheduler stopped")

if __name__ == "__main__":
    scheduler = ModelScheduler()
    scheduler.start() 