# Project Structure 🌳

This document outlines the file structure of the MLOps Exam Submission project.

```text
.
├── .env                    # Environment variables (API key) 🔑 NEEDS TO BE CREATED WITH YOUR ALPHA VANTAGE API KEY
├── .gitignore              # Specifies intentionally untracked files that Git should ignore
├── .dockerignore           # Specifies files and directories to ignore when building Docker images
├── Dockerfile              # Instructions to build the Docker image for the application
├── README.md               # Project overview, setup instructions, etc. 📖
├── docker-compose.yml      # Defines and runs multi-container Docker applications (API, Streamlit)
├── project_structure.md    # This file, describing the project layout
├── requirements.txt        # Python package dependencies 📄
├── data/                   # Contains datasets used in the project 📁
│   ├── features/           #  Processed features for model training
│   ├── intermediate/       #  Intermediate data files generated during processing
│   └── raw/                #  Raw, original data files
├── docker/                 # Contains Docker-related configuration files 🐳
│   ├── healthcheck.sh      #  Script to check API health within the Docker container
│   └── run.sh              #  Helper script to build and run Docker containers
├── models/                 # Trained models and related artifacts 🤖
│   ├── *.joblib            #  Saved preprocessing objects (scalers, feature names, etc.)
│   └── *.keras             #  Saved Keras model files
├── plots/                  # General plots and visualizations generated during analysis (outside reports) 📉
├── reports/                # Generated reports, including evaluation metrics and plots 📊
│   └── *.png               #  PNG files with visualizations of model performance
├── results/                # Raw evaluation results from model runs 📈
│   └── *.csv               #  CSV files containing detailed model evaluation metrics with timestamp
├── src/                    # Source code for the project 💻
│   ├── api/                #  FastAPI application code
│   │   └── stock_api.py    #  API endpoints for stock data and predictions
│   ├── main.py             #  Main entry point to run API and Streamlit, scheduling pipeline runs
│   ├── pipeline/           #  Scripts for the data processing and model training pipeline
│   │   ├── combined_data_processor.py      # Combines stock and macro data
│   │   ├── feature_engineering.py          # Creates features for the model
│   │   ├── model_results_visualizer.py     # Generates plots from model results
│   │   ├── pipeline_start.py               # Orchestrates the pipeline execution
│   │   ├── preprocessing.py                # Data preprocessing of the combined data
│   │   ├── stock_data_collector.py         # Fetches stock and macro data from Alpha Vantage
│   │   └── training-lstm.py                # Trains the LSTM model
│   └── streamlit/                          # Streamlit application code
│       └── app.py          #    Defines the Streamlit dashboard UI and logic
└── venv/                   # Python virtual environment directory (usually not tracked by Git) 🐍
```

## Folder Descriptions

*   **`data/`**: Holds all data used in the project, separated into raw, intermediate, and feature stages.
*   **`docker/`**: Contains configurations and scripts specifically for running the application using Docker.
*   **`models/`**: Stores trained machine learning models, scalers, and other necessary artifacts for prediction.
*   **`plots/`**: Contains general plots generated during data exploration or analysis that are not part of the formal reports.
*   **`reports/`**: Contains automatically generated reports, typically including model performance metrics (CSVs) and visualizations (PNGs) from specific runs.
*   **`results/`**: Stores the raw, detailed output files from model evaluation runs, usually in CSV format.
*   **`src/`**: The core source code of the project, organized into subdirectories for the API, the main application logic, the data pipeline, and the Streamlit UI.
*   **`venv/`**: Standard directory for Python virtual environments. Not tracked by Git.

## Key Files

*   **`.env`**: Stores sensitive information like API keys. *Do not commit this file to Git.*
*   **`Dockerfile`**: Instructions for building the Docker image.
*   **`docker-compose.yml`**: Configuration for running the application stack (API, potentially others) with Docker Compose.
*   **`README.md`**: Provides essential information about the project, setup, and how to run it.
*   **`requirements.txt`**: Lists all Python dependencies required to run the project. Use `pip install -r requirements.txt` to install them.
*   **`src/main.py`**: Entry point for running the application, including starting the API, Streamlit app, and scheduling the pipeline.
*   **`src/pipeline/pipeline_start.py`**: Orchestrates the execution of the data collection, preprocessing, feature engineering, and model training steps.
*   **`src/api/stock_api.py`**: Defines the FastAPI endpoints for serving data and predictions.
*   **`src/streamlit/app.py`**: Defines the user interface for the Streamlit dashboard.
``` 