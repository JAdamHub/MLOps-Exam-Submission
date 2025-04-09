import numpy as np
import pandas as pd
import json
from datetime import datetime
import logging
from pathlib import Path

# Konfigurer logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_for_drift(current_features, recent_predictions, threshold=0.05):
    """
    Simpel drift detection funktion, der sammenligner nuværende features med tidligere.
    
    Args:
        current_features (dict): Nuværende features brugt til forudsigelse
        recent_predictions (list): Liste af tidligere forudsigelser med features
        threshold (float): Tærskelværdi for drift detection (default: 0.05)
        
    Returns:
        bool: True hvis drift er detekteret, ellers False
    """
    try:
        logger.info(f"Checking for drift with {len(recent_predictions)} previous predictions")
        
        if len(recent_predictions) < 10:
            logger.info("Not enough historical data for drift detection")
            return False
            
        # Konverter tidligere features til DataFrame
        hist_features = []
        for pred in recent_predictions:
            if "features" in pred:
                hist_features.append(pred["features"])
                
        if not hist_features:
            logger.warning("No valid feature data in recent predictions")
            return False
            
        # Beregn gennemsnit og standardafvigelse for hver feature
        hist_df = pd.DataFrame(hist_features)
        mean_values = hist_df.mean()
        std_values = hist_df.std()
        
        # Beregn z-scores for nuværende features
        drift_detected = False
        drift_features = []
        
        for feature in current_features:
            if feature in mean_values and feature in std_values:
                # Beregn z-score
                if std_values[feature] > 0:  # Undgå division med nul
                    z_score = abs((current_features[feature] - mean_values[feature]) / std_values[feature])
                    
                    # Hvis z-score er over en tærskel, er der måske drift
                    if z_score > 3.0:  # 3 standardafvigelser
                        drift_features.append({
                            "feature": feature,
                            "current_value": current_features[feature],
                            "mean_value": float(mean_values[feature]),
                            "std_value": float(std_values[feature]),
                            "z_score": float(z_score)
                        })
        
        # Hvis mere end threshold % af features viser tegn på drift
        if len(drift_features) / len(current_features) > threshold:
            drift_detected = True
            
        # Gem resultater til fil
        result = {
            "drift_detected": drift_detected,
            "timestamp": datetime.now().isoformat(),
            "details": {
                "drift_features": drift_features,
                "drift_percentage": len(drift_features) / len(current_features),
                "threshold": threshold
            }
        }
        
        # Gem til fil
        with open("drift_detection_results.json", "w") as f:
            json.dump(result, f, indent=2)
            
        if drift_detected:
            logger.warning(f"DRIFT DETECTED in {len(drift_features)} features!")
        else:
            logger.info("No significant drift detected")
            
        return drift_detected
            
    except Exception as e:
        logger.error(f"Error in drift detection: {str(e)}")
        return False

# Resten af den eksisterende DriftDetector klasse beholdes herunder... 