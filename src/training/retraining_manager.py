import os
import json
import logging
from datetime import datetime
import pandas as pd
from ..monitoring.drift_detector import DriftDetector
from ..monitoring.evaluation import ModelEvaluator
from .train import train_model

class RetrainingManager:
    def __init__(self, 
                 model_path="models/model.joblib",
                 drift_detector=None,
                 evaluator=None):
        self.model_path = model_path
        self.drift_detector = drift_detector or DriftDetector()
        self.evaluator = evaluator or ModelEvaluator()
        self.logger = logging.getLogger(__name__)
        
    def check_and_retrain(self, new_data, actual_values=None):
        """
        Check if retraining is needed and perform retraining if necessary
        
        Args:
            new_data (pd.DataFrame): New data to check for drift
            actual_values (pd.Series, optional): Actual values for evaluation
            
        Returns:
            dict: Results of retraining check and process
        """
        # Check for data drift
        drift_results = self.drift_detector.detect_drift(new_data)
        
        # Check model performance
        metrics_summary = self.evaluator.get_metrics_summary()
        
        should_retrain = (
            self.drift_detector.should_retrain(drift_results) or
            (metrics_summary and metrics_summary.get("trend") == "declining")
        )
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "should_retrain": should_retrain,
            "drift_detected": drift_results.get("has_significant_drift", False),
            "performance_trend": metrics_summary.get("trend") if metrics_summary else None
        }
        
        if should_retrain:
            self.logger.info("Initiating model retraining...")
            try:
                # Backup existing model
                if os.path.exists(self.model_path):
                    backup_path = f"{self.model_path}.backup"
                    os.rename(self.model_path, backup_path)
                
                # Retrain model
                new_model = train_model(new_data)
                
                # Save new model
                new_model.save(self.model_path)
                
                # Update reference distribution
                self.drift_detector.compute_reference_stats(new_data)
                
                results["retraining_success"] = True
                self.logger.info("Model retraining completed successfully")
                
            except Exception as e:
                self.logger.error(f"Error during retraining: {str(e)}")
                results["retraining_success"] = False
                results["error"] = str(e)
                
                # Restore backup if retraining failed
                if os.path.exists(f"{self.model_path}.backup"):
                    os.rename(f"{self.model_path}.backup", self.model_path)
        
        return results 