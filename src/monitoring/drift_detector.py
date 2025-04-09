import numpy as np
import pandas as pd
from scipy import stats
import joblib
import os
from datetime import datetime
import json

class DriftDetector:
    def __init__(self, reference_data_path="data/features/reference_distribution.json"):
        self.reference_data_path = reference_data_path
        self.reference_stats = self._load_reference_stats()
        
    def _load_reference_stats(self):
        """Load reference distribution statistics or create new if not exists"""
        if os.path.exists(self.reference_data_path):
            with open(self.reference_data_path, 'r') as f:
                return json.load(f)
        return None
    
    def compute_reference_stats(self, df):
        """
        Compute and save reference statistics from a DataFrame
        
        Args:
            df (pd.DataFrame): DataFrame containing feature data
        """
        stats_dict = {}
        for column in df.select_dtypes(include=[np.number]).columns:
            stats_dict[column] = {
                "mean": float(df[column].mean()),
                "std": float(df[column].std()),
                "min": float(df[column].min()),
                "max": float(df[column].max())
            }
            
        os.makedirs(os.path.dirname(self.reference_data_path), exist_ok=True)
        with open(self.reference_data_path, 'w') as f:
            json.dump(stats_dict, f, indent=4)
            
        self.reference_stats = stats_dict
        return stats_dict
    
    def detect_drift(self, new_data, threshold=0.05):
        """
        Detect drift in new data compared to reference distribution
        
        Args:
            new_data (pd.DataFrame): New data to check for drift
            threshold (float): Significance level for drift detection
            
        Returns:
            dict: Dictionary containing drift detection results
        """
        if self.reference_stats is None:
            return {"error": "No reference statistics available. Please compute reference stats first."}
            
        drift_results = {}
        has_significant_drift = False
        
        for column in new_data.select_dtypes(include=[np.number]).columns:
            if column not in self.reference_stats:
                continue
                
            ref_mean = self.reference_stats[column]["mean"]
            ref_std = self.reference_stats[column]["std"]
            
            # Perform Kolmogorov-Smirnov test
            ks_statistic, p_value = stats.ks_2samp(
                new_data[column],
                np.random.normal(ref_mean, ref_std, size=len(new_data))
            )
            
            drift_results[column] = {
                "ks_statistic": float(ks_statistic),
                "p_value": float(p_value),
                "has_drift": p_value < threshold
            }
            
            if p_value < threshold:
                has_significant_drift = True
        
        return {
            "timestamp": datetime.now().isoformat(),
            "has_significant_drift": has_significant_drift,
            "feature_drift": drift_results
        }
    
    def should_retrain(self, drift_results):
        """
        Determine if model should be retrained based on drift results
        
        Args:
            drift_results (dict): Results from detect_drift method
            
        Returns:
            bool: True if model should be retrained
        """
        if isinstance(drift_results, dict) and "has_significant_drift" in drift_results:
            return drift_results["has_significant_drift"]
        return False 