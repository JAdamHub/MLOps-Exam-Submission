import numpy as np
import pandas as pd
from scipy import stats
import joblib
import os
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class DriftDetector:
    def __init__(self, reference_data_path="data/features/reference_distribution.json"):
        self.reference_data_path = reference_data_path
        self.reference_stats = self._load_reference_stats()
        self.figures_dir = Path(os.path.dirname(reference_data_path)).parent / "models" / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
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
                "max": float(df[column].max()),
                "median": float(df[column].median()),
                "q1": float(df[column].quantile(0.25)),
                "q3": float(df[column].quantile(0.75)),
                "skewness": float(stats.skew(df[column].dropna())),
                "kurtosis": float(stats.kurtosis(df[column].dropna())),
                "distribution_histogram": self._get_histogram_data(df[column])
            }
            
        os.makedirs(os.path.dirname(self.reference_data_path), exist_ok=True)
        with open(self.reference_data_path, 'w') as f:
            json.dump(stats_dict, f, indent=4)
            
        self.reference_stats = stats_dict
        return stats_dict
    
    def _get_histogram_data(self, series, bins=10):
        """Get histogram data for a series"""
        hist, bin_edges = np.histogram(series.dropna(), bins=bins)
        return {
            "counts": [int(count) for count in hist],
            "bin_edges": [float(edge) for edge in bin_edges]
        }
    
    def detect_drift(self, new_data, threshold=0.05, method='ks'):
        """
        Detect drift in new data compared to reference distribution
        
        Args:
            new_data (pd.DataFrame): New data to check for drift
            threshold (float): Significance level for drift detection
            method (str): Method to use for drift detection ('ks', 'psi', or 'wasserstein')
            
        Returns:
            dict: Dictionary containing drift detection results
        """
        if self.reference_stats is None:
            return {"error": "No reference statistics available. Please compute reference stats first."}
            
        drift_results = {}
        has_significant_drift = False
        top_drifted_features = []
        
        for column in new_data.select_dtypes(include=[np.number]).columns:
            if column not in self.reference_stats:
                continue
                
            ref_mean = self.reference_stats[column]["mean"]
            ref_std = self.reference_stats[column]["std"]
            
            if method == 'ks':
                # Perform Kolmogorov-Smirnov test
                ks_statistic, p_value = stats.ks_2samp(
                    new_data[column].dropna(),
                    np.random.normal(ref_mean, ref_std, size=len(new_data))
                )
                
                drift_results[column] = {
                    "ks_statistic": float(ks_statistic),
                    "p_value": float(p_value),
                    "has_drift": p_value < threshold,
                    "drift_magnitude": float(ks_statistic)  # Higher KS statistic = more drift
                }
            
            elif method == 'psi':
                # Population Stability Index
                psi_value = self._calculate_psi(new_data[column], column)
                
                drift_results[column] = {
                    "psi_value": float(psi_value),
                    "has_drift": psi_value > 0.2,  # PSI > 0.2 indicates significant drift
                    "drift_magnitude": float(psi_value)
                }
                
            elif method == 'wasserstein':
                # Wasserstein distance (Earth Mover's Distance)
                ref_data = np.random.normal(ref_mean, ref_std, size=len(new_data))
                w_distance = stats.wasserstein_distance(
                    new_data[column].dropna(),
                    ref_data
                )
                
                drift_results[column] = {
                    "wasserstein_distance": float(w_distance),
                    "has_drift": w_distance > 0.1,  # Threshold value can be adjusted
                    "drift_magnitude": float(w_distance)
                }
            
            # Check if this feature has drift
            if drift_results[column].get("has_drift", False):
                has_significant_drift = True
                top_drifted_features.append({
                    "feature": column,
                    "drift_magnitude": drift_results[column]["drift_magnitude"]
                })
        
        # Sort drifted features by magnitude
        top_drifted_features = sorted(top_drifted_features, key=lambda x: x["drift_magnitude"], reverse=True)
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "has_significant_drift": has_significant_drift,
            "feature_drift": drift_results,
            "top_drifted_features": top_drifted_features[:5] if top_drifted_features else []
        }
        
        # Visualize the drift results
        self._visualize_drift_results(result, new_data)
        
        return result
    
    def _calculate_psi(self, new_series, column_name, bins=10):
        """Calculate Population Stability Index"""
        if column_name not in self.reference_stats:
            return 0.0
            
        # Get reference histogram data
        ref_hist = self.reference_stats[column_name]["distribution_histogram"]
        ref_counts = np.array(ref_hist["counts"])
        bin_edges = ref_hist["bin_edges"]
        
        # Calculate new histogram using same bins
        new_hist, _ = np.histogram(new_series.dropna(), bins=bin_edges)
        
        # Convert to proportions
        ref_props = ref_counts / np.sum(ref_counts)
        new_props = new_hist / np.sum(new_hist)
        
        # Replace zeros with small value to avoid division by zero
        ref_props = np.where(ref_props == 0, 0.0001, ref_props)
        new_props = np.where(new_props == 0, 0.0001, new_props)
        
        # Calculate PSI
        psi = np.sum((new_props - ref_props) * np.log(new_props / ref_props))
        
        return psi
    
    def _visualize_drift_results(self, drift_results, new_data):
        """Visualize the drift detection results"""
        # 1. Create bar chart of drift magnitude for top drifted features
        top_features = drift_results.get("top_drifted_features", [])
        if top_features:
            plt.figure(figsize=(10, 6))
            
            features = [item["feature"] for item in top_features]
            magnitudes = [item["drift_magnitude"] for item in top_features]
            
            plt.barh(features, magnitudes)
            plt.xlabel('Drift Magnitude')
            plt.ylabel('Feature')
            plt.title('Top Features with Drift')
            plt.tight_layout()
            plt.savefig(self.figures_dir / 'feature_drift_magnitude.png')
            plt.close()
            
            # 2. Create distribution comparison plots for top 3 features
            for i, feature_info in enumerate(top_features[:3]):
                feature = feature_info["feature"]
                if feature in self.reference_stats and feature in new_data.columns:
                    plt.figure(figsize=(10, 6))
                    
                    # Plot new data distribution
                    sns.kdeplot(new_data[feature].dropna(), label='New Data')
                    
                    # Generate reference distribution data
                    ref_mean = self.reference_stats[feature]["mean"]
                    ref_std = self.reference_stats[feature]["std"]
                    ref_data = np.random.normal(ref_mean, ref_std, size=1000)
                    
                    # Plot reference distribution
                    sns.kdeplot(ref_data, label='Reference Data')
                    
                    plt.title(f'Distribution Comparison: {feature}')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(self.figures_dir / f'distribution_comparison_{feature}.png')
                    plt.close()
    
    def should_retrain(self, drift_results, drift_threshold=0.3, feature_threshold=0.2):
        """
        Determine if model should be retrained based on drift results
        
        Args:
            drift_results (dict): Results from detect_drift method
            drift_threshold (float): Threshold for overall drift to trigger retraining
            feature_threshold (float): Threshold for proportion of features with drift
            
        Returns:
            bool: True if model should be retrained
        """
        if "error" in drift_results:
            return False
            
        if drift_results.get("has_significant_drift", False):
            # Count how many features have drift
            feature_drift = drift_results.get("feature_drift", {})
            total_features = len(feature_drift)
            
            if total_features == 0:
                return False
                
            drifted_count = sum(1 for feature, result in feature_drift.items() 
                              if result.get("has_drift", False))
            
            # Calculate proportion of features with drift
            drift_proportion = drifted_count / total_features
            
            # Check top drifted features
            top_drifted = drift_results.get("top_drifted_features", [])
            max_drift = max([item["drift_magnitude"] for item in top_drifted]) if top_drifted else 0
            
            # Decide if retraining is needed
            needs_retraining = (drift_proportion > feature_threshold) or (max_drift > drift_threshold)
            
            if needs_retraining:
                return True
        
        return False
    
    def monitor_feature_stability(self, data_sequence, feature, window_size=10):
        """
        Monitor the stability of a specific feature over time
        
        Args:
            data_sequence (list): List of DataFrames containing feature data over time
            feature (str): Feature name to monitor
            window_size (int): Size of the sliding window for stability calculation
            
        Returns:
            dict: Stability monitoring results
        """
        if feature not in self.reference_stats:
            return {"error": f"Feature {feature} not in reference statistics"}
            
        if len(data_sequence) < 2:
            return {"error": "Need at least 2 data points for stability monitoring"}
            
        # Extract the feature values from each DataFrame
        feature_values = [df[feature].dropna() for df in data_sequence if feature in df.columns]
        
        if not all(feature_values):
            return {"error": f"Feature {feature} missing in some data points"}
            
        # Calculate mean and std for each time point
        means = [values.mean() for values in feature_values]
        stds = [values.std() for values in feature_values]
        
        # Calculate stability metrics
        stability_results = {
            "feature": feature,
            "reference_mean": self.reference_stats[feature]["mean"],
            "reference_std": self.reference_stats[feature]["std"],
            "current_mean": means[-1],
            "current_std": stds[-1],
            "mean_change": means[-1] - self.reference_stats[feature]["mean"],
            "std_change": stds[-1] - self.reference_stats[feature]["std"],
            "mean_trend": np.mean(np.diff(means[-window_size:])) if len(means) >= window_size else None,
            "std_trend": np.mean(np.diff(stds[-window_size:])) if len(stds) >= window_size else None,
        }
        
        # Plot the stability over time
        plt.figure(figsize=(12, 6))
        
        # Plot mean over time
        plt.subplot(1, 2, 1)
        plt.plot(means, 'b-')
        plt.axhline(y=self.reference_stats[feature]["mean"], color='r', linestyle='--')
        plt.title(f'{feature} Mean Over Time')
        plt.xlabel('Time Point')
        plt.ylabel('Mean Value')
        
        # Plot std over time
        plt.subplot(1, 2, 2)
        plt.plot(stds, 'g-')
        plt.axhline(y=self.reference_stats[feature]["std"], color='r', linestyle='--')
        plt.title(f'{feature} Std Dev Over Time')
        plt.xlabel('Time Point')
        plt.ylabel('Std Dev')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / f'stability_{feature}.png')
        plt.close()
        
        return stability_results 