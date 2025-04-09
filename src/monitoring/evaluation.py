import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
from pathlib import Path

class ModelEvaluator:
    def __init__(self, metrics_file="models/evaluation_metrics.json"):
        self.metrics_file = metrics_file
        self.metrics_history = self._load_metrics_history()
        
    def _load_metrics_history(self):
        """Load existing metrics history or create new if not exists"""
        if os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        return {
            "daily_accuracy": [],
            "cumulative_accuracy": [],
            "timestamps": []
        }
    
    def evaluate_predictions(self, predictions, actual_values, timestamp=None):
        """
        Evaluate model predictions and update metrics
        
        Args:
            predictions (list): List of model predictions (0 or 1)
            actual_values (list): List of actual values (0 or 1)
            timestamp (str): Timestamp for the evaluation (defaults to current time)
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
            
        # Calculate daily accuracy
        daily_accuracy = np.mean(np.array(predictions) == np.array(actual_values))
        
        # Update metrics history
        self.metrics_history["daily_accuracy"].append(float(daily_accuracy))
        self.metrics_history["timestamps"].append(timestamp)
        
        # Calculate cumulative accuracy
        cumulative_accuracy = np.mean(self.metrics_history["daily_accuracy"])
        self.metrics_history["cumulative_accuracy"].append(float(cumulative_accuracy))
        
        # Save updated metrics
        self._save_metrics()
        
        return {
            "daily_accuracy": daily_accuracy,
            "cumulative_accuracy": cumulative_accuracy,
            "timestamp": timestamp
        }
    
    def _save_metrics(self):
        """Save metrics history to file"""
        os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=4)
    
    def get_metrics_summary(self):
        """Get summary of model performance metrics"""
        if not self.metrics_history["daily_accuracy"]:
            return "No evaluation metrics available yet"
            
        return {
            "latest_accuracy": self.metrics_history["daily_accuracy"][-1],
            "average_accuracy": np.mean(self.metrics_history["daily_accuracy"]),
            "accuracy_trend": self._calculate_trend(),
            "total_evaluations": len(self.metrics_history["daily_accuracy"])
        }
    
    def _calculate_trend(self, window=7):
        """Calculate accuracy trend over the last n days"""
        if len(self.metrics_history["daily_accuracy"]) < window:
            return "Insufficient data for trend calculation"
            
        recent_accuracy = self.metrics_history["daily_accuracy"][-window:]
        trend = np.mean(np.diff(recent_accuracy))
        
        if trend > 0.01:
            return "Improving"
        elif trend < -0.01:
            return "Declining"
        else:
            return "Stable" 