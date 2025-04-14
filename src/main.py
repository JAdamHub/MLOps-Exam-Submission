import logging
import uvicorn
import threading
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from monitoring.scheduler import ModelUpdateScheduler
from pipelinevizoptions.src.pipelinevizoptions.model_metrics_viz import main as generate_visualizations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_api():
    """Runs the FastAPI server"""
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)

def run_scheduler():
    """Runs the model update scheduler"""
    scheduler = ModelUpdateScheduler()
    scheduler.start()

def main():
    """Main function that starts all components"""
    try:
        # Start API in a separate thread
        api_thread = threading.Thread(target=run_api)
        api_thread.daemon = True
        api_thread.start()
        logger.info("API server started")

        # Start scheduler in a separate thread
        scheduler_thread = threading.Thread(target=run_scheduler)
        scheduler_thread.daemon = True
        scheduler_thread.start()
        logger.info("Model update scheduler started")

        # Generate initial visualizations
        generate_visualizations()
        logger.info("Initial visualizations generated")

        # Keep the main thread alive
        while True:
            try:
                # Generate visualizations every hour
                generate_visualizations()
                threading.Event().wait(3600)  # Wait 1 hour
            except KeyboardInterrupt:
                logger.info("Exiting program...")
                break
            except Exception as e:
                logger.error(f"Error during execution: {e}")
                continue

    except Exception as e:
        logger.error(f"Critical error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 