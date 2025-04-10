import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd
import logging

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

class ModelMetricsVisualizer:
    def __init__(self, metrics_file='model_metrics.json'):
        self.metrics_file = metrics_file
        self.metrics = self.load_metrics()
        
    def load_metrics(self):
        """Load metrics from file"""
        try:
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Metrics file {self.metrics_file} not found")
            return None
        except json.JSONDecodeError:
            logger.error(f"Error decoding metrics file {self.metrics_file}")
            return None
    
    def plot_accuracy_history(self, save_path=None):
        """Plot accuracy history for LSTM model"""
        if not self.metrics or 'lstm' not in self.metrics:
            logger.error("No LSTM metrics available")
            return
        
        plt.figure(figsize=(12, 6))
        accuracy_history = self.metrics['lstm']['accuracy_history']
        
        plt.plot(accuracy_history, marker='o', linestyle='-', color='blue')
        plt.title('LSTM Model Accuracy History')
        plt.xlabel('Update Number')
        plt.ylabel('Accuracy')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Accuracy history plot saved to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_version_comparison(self, save_path=None):
        """Plot version comparison for LSTM model"""
        if not self.metrics or 'lstm' not in self.metrics:
            logger.error("No LSTM metrics available")
            return
        
        versions = self.metrics['lstm']['versions']
        if not versions:
            logger.error("No version data available")
            return
        
        df = pd.DataFrame(versions)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='version', y='accuracy', color='blue')
        plt.title('LSTM Model Version Comparison')
        plt.xlabel('Model Version')
        plt.ylabel('Accuracy')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Version comparison plot saved to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def generate_report(self, save_dir='reports'):
        """Generate comprehensive visualization report"""
        try:
            import os
            os.makedirs(save_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Generate plots
            self.plot_accuracy_history(f"{save_dir}/lstm_accuracy_history_{timestamp}.png")
            self.plot_version_comparison(f"{save_dir}/lstm_version_comparison_{timestamp}.png")
            
            logger.info(f"Report generated successfully in {save_dir}")
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}", exc_info=True)

if __name__ == "__main__":
    visualizer = ModelMetricsVisualizer()
    visualizer.generate_report() 