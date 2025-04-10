from src.monitoring.drift_detector import DriftDetector
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Konfigurer logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Kør drift detection på seneste data"""
    try:
        # Opret drift detector
        drift_detector = DriftDetector()
        
        # Forsøg at indlæse seneste data
        data_path = Path("data/features/latest_data.csv")
        
        if not data_path.exists():
            # Hvis filen ikke findes, lav dummy data
            logger.info("Ingen data fundet. Opretter dummy data for demonstration...")
            n_samples = 100
            n_features = 10
            feature_names = [f"feature_{i}" for i in range(n_features)]
            
            # Opret dummy data med lidt drift
            dummy_data = np.random.randn(n_samples, n_features) + 0.5  # Add shift to introduce drift
            df = pd.DataFrame(dummy_data, columns=feature_names)
            
            # Gem data for fremtidig reference
            data_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(data_path, index=False)
            
            # Opret reference statistics hvis de ikke findes
            if drift_detector.reference_stats is None:
                # Brug en lidt anderledes distribution til reference
                reference_data = np.random.randn(n_samples, n_features)
                ref_df = pd.DataFrame(reference_data, columns=feature_names)
                drift_detector.compute_reference_stats(ref_df)
                logger.info("Oprettet reference statistik for drift detection")
        else:
            # Indlæs eksisterende data
            df = pd.read_csv(data_path)
            logger.info(f"Indlæst data med shape {df.shape}")
            
        # Kør drift detection
        drift_results = drift_detector.detect_drift(df)
        
        # Vis resultater
        print("\n=== Drift Detection Resultater ===")
        print(f"Signifikant drift opdaget: {drift_results['has_significant_drift']}")
        
        if drift_results['top_drifted_features']:
            print("\nTop features med drift:")
            for feature_info in drift_results['top_drifted_features']:
                print(f"  - {feature_info['feature']}: {feature_info['drift_magnitude']:.4f}")
        
        # Tjek om vi bør gentræne modellen
        should_retrain = drift_detector.should_retrain(drift_results)
        print(f"\nAnbefalet gentræning: {should_retrain}")
        
    except Exception as e:
        logger.error(f"Fejl under drift detection: {e}")

if __name__ == "__main__":
    main() 