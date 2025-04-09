import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef
import joblib

class ModelEvaluator:
    def __init__(self, metrics_file="models/evaluation_metrics.json"):
        self.metrics_file = metrics_file
        self.metrics_history = self._load_metrics_history()
        self.figures_dir = Path(os.path.dirname(metrics_file)) / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_metrics_history(self):
        """Load existing metrics history or create new if not exists"""
        if os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        return {
            "daily_accuracy": [],
            "daily_precision": [],
            "daily_recall": [],
            "daily_f1": [],
            "daily_roc_auc": [],
            "daily_mcc": [],
            "cumulative_accuracy": [],
            "timestamps": [],
            "profit_loss": [],
            "drawdown": []
        }
    
    def evaluate_predictions(self, predictions, actual_values, price_movement=None, timestamp=None):
        """
        Evaluate model predictions and update metrics
        
        Args:
            predictions (list): List of model predictions (0 or 1)
            actual_values (list): List of actual values (0 or 1)
            price_movement (list, optional): List of price movements for backtesting
            timestamp (str): Timestamp for the evaluation (defaults to current time)
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
            
        # Convert to numpy arrays for calculation
        y_pred = np.array(predictions)
        y_true = np.array(actual_values)
        
        # Calculate daily metrics
        daily_accuracy = accuracy_score(y_true, y_pred)
        daily_precision = precision_score(y_true, y_pred, zero_division=0)
        daily_recall = recall_score(y_true, y_pred)
        daily_f1 = f1_score(y_true, y_pred)
        daily_mcc = matthews_corrcoef(y_true, y_pred)
        
        # Calculate ROC AUC if available (requires probabilities)
        daily_roc_auc = 0.5  # Default value if probabilities not available
        
        # Update metrics history
        self.metrics_history["daily_accuracy"].append(float(daily_accuracy))
        self.metrics_history["daily_precision"].append(float(daily_precision))
        self.metrics_history["daily_recall"].append(float(daily_recall))
        self.metrics_history["daily_f1"].append(float(daily_f1))
        self.metrics_history["daily_mcc"].append(float(daily_mcc))
        self.metrics_history["daily_roc_auc"].append(float(daily_roc_auc))
        self.metrics_history["timestamps"].append(timestamp)
        
        # Calculate cumulative accuracy
        cumulative_accuracy = np.mean(self.metrics_history["daily_accuracy"])
        self.metrics_history["cumulative_accuracy"].append(float(cumulative_accuracy))
        
        # Calculate backtest metrics if price movement data is provided
        if price_movement is not None and len(price_movement) == len(predictions):
            pnl, dd = self._calculate_backtest_metrics(predictions, price_movement)
            self.metrics_history["profit_loss"].append(float(pnl))
            self.metrics_history["drawdown"].append(float(dd))
        
        # Save updated metrics
        self._save_metrics()
        
        # Generate evaluation plots (altid)
        self._generate_evaluation_plots()
        
        return {
            "daily_accuracy": daily_accuracy,
            "daily_precision": daily_precision,
            "daily_recall": daily_recall,
            "daily_f1": daily_f1,
            "daily_mcc": daily_mcc,
            "daily_roc_auc": daily_roc_auc,
            "cumulative_accuracy": cumulative_accuracy,
            "timestamp": timestamp
        }
    
    def _calculate_backtest_metrics(self, predictions, price_movements):
        """
        Calculate profit/loss and drawdown from a simple trading strategy
        based on predictions and actual price movements
        """
        # Convert to numpy arrays
        preds = np.array(predictions)
        moves = np.array(price_movements)
        
        # Calculate daily P&L
        # 1 if prediction is correct and we made money, -1 if we're wrong and lost money
        daily_pnl = np.where(preds == 1, moves, -moves)
        
        # Calculate cumulative P&L
        cumulative_pnl = np.cumsum(daily_pnl)
        
        # Calculate drawdown
        peak = np.maximum.accumulate(cumulative_pnl)
        drawdown = peak - cumulative_pnl
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Calculate final P&L
        final_pnl = cumulative_pnl[-1] if len(cumulative_pnl) > 0 else 0
        
        return final_pnl, max_drawdown
    
    def _save_metrics(self):
        """Save metrics history to file"""
        os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=4)
    
    def _generate_evaluation_plots(self):
        """Generate plots for evaluation metrics"""
        # Create timestamps for x-axis
        timestamps = pd.to_datetime(self.metrics_history["timestamps"])
        
        # Plot accuracy over time
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, self.metrics_history["daily_accuracy"], 'b-', label='Daily Accuracy')
        plt.plot(timestamps, self.metrics_history["cumulative_accuracy"], 'r-', label='Cumulative Accuracy')
        plt.title('Model Accuracy Over Time')
        plt.xlabel('Date')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.figures_dir / 'accuracy_over_time.png')
        plt.close()
        
        # Plot precision, recall, F1 over time
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, self.metrics_history["daily_precision"], 'g-', label='Precision')
        plt.plot(timestamps, self.metrics_history["daily_recall"], 'b-', label='Recall')
        plt.plot(timestamps, self.metrics_history["daily_f1"], 'r-', label='F1 Score')
        plt.title('Precision, Recall, and F1 Score Over Time')
        plt.xlabel('Date')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.figures_dir / 'precision_recall_f1.png')
        plt.close()
        
        # Plot backtest metrics if available
        if len(self.metrics_history["profit_loss"]) > 0:
            plt.figure(figsize=(12, 6))
            plt.plot(timestamps[:len(self.metrics_history["profit_loss"])], 
                     np.cumsum(self.metrics_history["profit_loss"]), 'g-')
            plt.title('Cumulative Profit/Loss from Model Predictions')
            plt.xlabel('Date')
            plt.ylabel('Cumulative P&L')
            plt.grid(True)
            plt.savefig(self.figures_dir / 'cumulative_pnl.png')
            plt.close()
    
    def get_metrics_summary(self):
        """Get summary of model performance metrics"""
        if not self.metrics_history["daily_accuracy"]:
            return "No evaluation metrics available yet"
            
        return {
            "latest_accuracy": self.metrics_history["daily_accuracy"][-1],
            "latest_precision": self.metrics_history["daily_precision"][-1],
            "latest_recall": self.metrics_history["daily_recall"][-1],
            "latest_f1": self.metrics_history["daily_f1"][-1],
            "latest_mcc": self.metrics_history["daily_mcc"][-1],
            "average_accuracy": np.mean(self.metrics_history["daily_accuracy"]),
            "accuracy_trend": self._calculate_trend("daily_accuracy"),
            "total_evaluations": len(self.metrics_history["daily_accuracy"]),
            "profit_loss": sum(self.metrics_history["profit_loss"]) if self.metrics_history["profit_loss"] else None,
            "max_drawdown": max(self.metrics_history["drawdown"]) if self.metrics_history["drawdown"] else None
        }
    
    def _calculate_trend(self, metric="daily_accuracy", window=7):
        """Calculate metric trend over the last n days"""
        if len(self.metrics_history[metric]) < window:
            return "Insufficient data for trend calculation"
            
        recent_values = self.metrics_history[metric][-window:]
        trend = np.mean(np.diff(recent_values))
        
        if trend > 0.01:
            return "Improving"
        elif trend < -0.01:
            return "Declining"
        else:
            return "Stable"
            
    def analyze_prediction_errors(self, predictions, actual_values, feature_values=None, feature_names=None):
        """
        Analyze where and why the model makes errors
        
        Args:
            predictions (list): Model predictions
            actual_values (list): Actual values
            feature_values (array): Feature values used for prediction
            feature_names (list): Names of features
        """
        # Convert to arrays
        y_pred = np.array(predictions)
        y_true = np.array(actual_values)
        
        # Identify errors
        errors = (y_pred != y_true)
        error_rate = np.mean(errors)
        
        # Calculate confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Calculate error types
        false_positives = conf_matrix[0, 1]  # Actually down, predicted up
        false_negatives = conf_matrix[1, 0]  # Actually up, predicted down
        
        error_analysis = {
            "error_rate": float(error_rate),
            "false_positive_rate": float(false_positives) / max(1, np.sum(conf_matrix[0, :])),
            "false_negative_rate": float(false_negatives) / max(1, np.sum(conf_matrix[1, :])),
            "total_errors": int(np.sum(errors)),
            "false_positives": int(false_positives),
            "false_negatives": int(false_negatives)
        }
        
        # Analyze feature patterns in errors if feature data is provided
        if feature_values is not None and feature_names is not None:
            X = np.array(feature_values)
            error_indices = np.where(errors)[0]
            
            if len(error_indices) > 0:
                error_features = X[error_indices]
                # Calculate mean feature values for errors
                error_feature_means = np.mean(error_features, axis=0)
                
                # Add feature analysis to error analysis
                error_analysis["feature_analysis"] = {
                    name: float(mean) for name, mean in zip(feature_names, error_feature_means)
                }
        
        return error_analysis

def evaluate_model():
    """Evaluate the model and generate initial metrics"""
    try:
        # Load model and data
        model_path = "models/xgboost_model.joblib"
        feature_names_path = "models/feature_names.joblib"
        
        if not os.path.exists(model_path):
            print(f"Model file not found at {model_path}")
            return
            
        if not os.path.exists(feature_names_path):
            print(f"Feature names file not found at {feature_names_path}")
            return
            
        model = joblib.load(model_path)
        feature_names = joblib.load(feature_names_path)
        n_features = len(feature_names)
        
        # Generate dummy data with correct number of features
        X_test = np.random.rand(100, n_features)  # 100 samples, correct number of features
        y_test = np.random.randint(0, 2, 100)  # Binary labels
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Initialize evaluator
        evaluator = ModelEvaluator()
        
        # Evaluate predictions
        metrics = evaluator.evaluate_predictions(
            predictions=y_pred.tolist(),
            actual_values=y_test.tolist(),
            timestamp=datetime.now().isoformat()
        )
        
        print("Initial evaluation completed!")
        print(f"Accuracy: {metrics['daily_accuracy']:.3f}")
        print(f"Precision: {metrics['daily_precision']:.3f}")
        print(f"Recall: {metrics['daily_recall']:.3f}")
        print(f"F1 Score: {metrics['daily_f1']:.3f}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    evaluate_model() 