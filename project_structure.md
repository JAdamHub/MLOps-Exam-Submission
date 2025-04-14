# Project Structure 🌳

This document outlines the file structure of the MLOps Exam Submission project.

```text
.
├── .env                    # Environment variables (API key) 🔑 NEEDS TO BE CREATED WITH YOUR ALPHA VANTAGE API KEY
├── data/                   # Contains datasets used in the project 📁
│   ├── features/           # Processed features for model training
│   ├── intermediate/       # Intermediate data files generated during processing
│   └── raw/                # Raw, original data files
├── models/                 # Trained models and related artifacts 🤖
│   ├── *.joblib            # Saved preprocessing objects (scalers, feature names, etc.)
│   ├── *.keras             # Saved Keras model files
│   └── figures/            # Figures related to model analysis (if any)
├── plots/                  # General plots and visualizations generated during analysis (outside reports) 📉
├── pipeline.log            # Log file for the main pipeline execution 📝
├── README.md               # Project overview, setup instructions, etc. 📖
├── reports/                # Generated reports, including evaluation metrics and plots 📊
│   └── lstm_*.*            # Timestamped CSVs and PNGs with metrics and visualizations
├── requirements.txt        # Python package dependencies 📄
├── results/                # Raw evaluation results from model runs 📈
│   └── *.csv               # CSV files containing detailed model evaluation metrics with timestamp for each csv file 
├── src/                    # Source code for the project 💻
│   ├── data_processing/    # Scripts for data loading and preprocessing
│   ├── modeling/           # Scripts for model training and evaluation
│   ├── deployment/         # Scripts/code related to model deployment (e.g., API)
│   ├── pipelines/          # Code defining pipelines (e.g., training, prediction)
│   └── utils/              # Utility functions and helper scripts
└── visualizations/         # Static visualization files 🖼️
    └── pipeline_architecture.png # Diagram showing the pipeline architecture
```

## Folder Descriptions

*   **`data/`**: Holds all data used in the project, separated into raw, intermediate, and feature stages.
*   **`models/`**: Stores trained machine learning models, scalers, and other necessary artifacts for prediction.
*   **`plots/`**: Contains general plots generated during data exploration or analysis that are not part of the formal reports.
*   **`reports/`**: Contains automatically generated reports, typically including model performance metrics (CSVs) and visualizations (PNGs) from specific runs.
*   **`results/`**: Stores the raw, detailed output files from model evaluation runs, usually in CSV format.
*   **`src/`**: The core source code of the project, organized into subdirectories based on functionality (data processing, modeling, deployment, pipelines, utilities).
*   **`visualizations/`**: Stores key static visualizations, such as architecture diagrams.
*   **`venv/`**: Standard directory for Python virtual environments. Not tracked by Git.

## Key Files

*   **`.env`**: Stores sensitive information like API keys. *Do not commit this file to Git.*
*   **`pipeline.log`**: Records logs from pipeline runs, useful for debugging.
*   **`README.md`**: Provides essential information about the project.
*   **`requirements.txt`**: Lists all Python dependencies required to run the project. Use `pip install -r requirements.txt` to install them.
``` 