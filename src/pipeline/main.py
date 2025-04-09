import logging
import sys
from pathlib import Path
from typing import Optional

# Tilføj src directory til Python path for at tillade import af pipeline moduler
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Import pipeline moduler
from pipeline import ingestion, preprocessing, feature_engineering, training
from pipeline import makro_data_collector, combined_data_processor

# Konfigurer logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "logs" / "pipeline.log")
    ]
)
logger = logging.getLogger(__name__)

class PipelineRunner:
    """Klasse til at køre den komplette data processing og model trænings pipeline."""
    
    def __init__(self):
        """Initialiserer PipelineRunner."""
        self.project_root = PROJECT_ROOT
        self.logs_dir = self.project_root / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
    
    def run_step(self, step_name: str, step_func: callable) -> Optional[bool]:
        """
        Kører en enkelt pipeline step med fejlhåndtering.
        
        Args:
            step_name: Navn på pipeline step
            step_func: Funktion der skal køres
            
        Returns:
            True hvis step lykkedes, False hvis fejl, None hvis ikke kørt
        """
        try:
            logger.info(f"--- Starter {step_name} ---")
            step_func()
            logger.info(f"--- {step_name} gennemført ---")
            return True
        except Exception as e:
            logger.error(f"Fejl under {step_name}: {str(e)}", exc_info=True)
            return False
    
    def run_pipeline(self) -> bool:
        """
        Kører den komplette pipeline.
        
        Returns:
            True hvis pipeline gennemført succesfuldt, False ellers
        """
        logger.info("========== Starter Pipeline Kørsel ==========")
        
        # Definer pipeline steps
        steps = [
            ("Data Indlæsning", ingestion.main),
            ("Makroøkonomisk Data Indsamling", lambda: makro_data_collector.MacroDataCollector().collect_all_macro_data()),
            ("Data Kombination", combined_data_processor.main),
            ("Data Preprocessing", preprocessing.main),
            ("Feature Engineering", feature_engineering.main),
            ("Model Træning", training.main)
        ]
        
        # Kør hver step
        for step_name, step_func in steps:
            if not self.run_step(step_name, step_func):
                logger.error(f"Pipeline stopped ved {step_name}")
                return False
        
        logger.info("========== Pipeline Gennemført Succesfuldt ==========")
        return True

def main():
    """Hovedfunktion til at køre pipeline."""
    runner = PipelineRunner()
    success = runner.run_pipeline()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 