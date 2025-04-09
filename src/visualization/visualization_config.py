from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"

# Model visualizations
MODEL_FIGURES_DIR = MODELS_DIR / "figures"
MODEL_METRICS_FILE = MODELS_DIR / "evaluation_metrics.json"
MODEL_TRAINING_METRICS_FILE = MODELS_DIR / "training_metrics.json"

# Pipeline visualizations
PIPELINE_VIZ_DIR = VISUALIZATIONS_DIR / "pipeline"
PIPELINE_VIZ_FILE = PIPELINE_VIZ_DIR / "pipeline_architecture.png"

# Ensure directories exist
for directory in [MODEL_FIGURES_DIR, PIPELINE_VIZ_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Visualization settings
PLOT_STYLE = {
    'figure.figsize': (12, 6),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'font.size': 12
}

# Color schemes
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'warning': '#d62728',
    'neutral': '#7f7f7f'
} 