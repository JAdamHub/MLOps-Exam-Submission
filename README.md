# 📈 Vestas Stock Price Prediction MLOps Pipeline 🚀

A comprehensive MLOps pipeline for predicting Vestas stock prices using historical market data and macroeconomic indicators. The system leverages LSTM deep learning models to forecast stock prices at multiple time horizons (1, 3, and 7 days ahead).

## 🔍 Project Overview

This project demonstrates a complete machine learning operations (MLOps) pipeline for stock price prediction:
- Data collection from Alpha Vantage API for stock and macroeconomic data
- Data preprocessing and feature engineering
- Model training using LSTM neural networks
- API endpoints for predictions and visualizations
- Streamlit dashboard for interactive analysis
- Containerization with Docker for easy deployment
- Daily pipeline retraining for up-to-date models

## 📁 Project Structure

```text
.
├── .env                    # Environment variables (API key) 🔑 NEEDS TO BE CREATED WITH YOUR ALPHA VANTAGE API KEY
├── .gitignore              # Specifies intentionally untracked files to ignore
├── .dockerignore           # Specifies files to ignore when building Docker images
├── Dockerfile              # Instructions to build the Docker image
├── README.md               # Project overview, setup instructions, etc. 📖
├── docker-compose.yml      # Defines Docker applications (API, Streamlit)
├── requirements.txt        # Python package dependencies 📄
├── data/                   # Contains datasets used in the project 📁
│   ├── features/           #  Processed features for model training
│   ├── intermediate/       #  Intermediate data files during processing
│   │   ├── combined/       #   Combined stock and macro data
│   │   └── preprocessed/   #   Preprocessed data ready for feature engineering
│   └── raw/                #  Raw, original data files
│       ├── stocks/         #   Raw stock price data
│       └── macro/          #   Raw macroeconomic data
├── docker/                 # Docker-related configuration files 🐳
│   ├── healthcheck.sh      #  Script to check API health within containers
│   └── run.sh              #  Helper script to build and run Docker containers
├── models/                 # Trained models and related artifacts 🤖
│   ├── *.joblib            #  Saved preprocessing objects (scalers, feature names)
│   └── *.keras             #  Saved Keras model files
├── plots/                  # Visualizations generated during analysis 📉
├── reports/                # Generated reports, including model metrics 📊
├── results/                # Raw evaluation results from model runs 📈
├── src/                    # Source code for the project 💻
│   ├── api/                #  FastAPI application code
│   │   └── stock_api.py    #  API endpoints for stock data and predictions
│   ├── main.py             #  Main entry point for the application
│   ├── pipeline/           #  Data processing and model training pipeline
│   │   ├── combined_data_processor.py      # Combines stock and macro data
│   │   ├── feature_engineering.py          # Creates features for the model
│   │   ├── model_results_visualizer.py     # Generates plots from model results
│   │   ├── pipeline_start.py               # Orchestrates the pipeline execution
│   │   ├── preprocessing.py                # Data preprocessing
│   │   ├── stock_data_collector.py         # Fetches data from Alpha Vantage
│   │   └── training-lstm.py                # Trains the LSTM model
│   └── streamlit/                          # Streamlit dashboard application
│       └── app.py          #  Defines the Streamlit dashboard UI and logic
```

## ⚙️ Setup and Installation

### Prerequisites
- Python 3.10+ 🐍
- Docker and Docker Compose (for containerized deployment) 🐳
- Alpha Vantage API key (get a free key at [Alpha Vantage](https://www.alphavantage.co/support/#api-key))

### Local Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/JAdamHub/MLOps-Exam-Submission.git
   cd MLOps-Exam-Submission
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create `.env` file with your Alpha Vantage API key:**
   ```
   ALPHA_VANTAGE_API_KEY="your_api_key_here"
   ```

### Running the Pipeline + API & Streamlit App

To run the complete data pipeline (collection, preprocessing, feature engineering, and model training):

```bash
python -m src.main
```

This will:
- Run the initial data pipeline
- Train a LSTM model
- Start the FastAPI server on http://localhost:8000
- Start the Streamlit dashboard on http://localhost:8501
- Schedule daily pipeline runs at 8:30 AM

## 🐳 Docker Deployment

### Using the Run Script (Recommended)

The easiest way to start the application is using the provided helper script:

```bash
./docker/run.sh
```

This script will:
- Create all necessary directories
- Check for the `.env` file with your API key
- Start the Docker container

### Manual Docker Setup

1. **Build and start using Docker Compose:**
   ```bash
   docker-compose up -d
   ```

2. **View logs:**
   ```bash
   docker-compose logs -f
   ```

3. **Stop the container:**
   ```bash
   docker-compose down
   ```

## 🌐 API Endpoints

The API provides the following endpoints:

- **GET /** - Welcome message and API information
- **GET /health** - Health check for the API
- **GET /price/history** - Get Vestas stock price history
  - Parameters: `days` (optional, default=7300) (last 20 years)
- **POST /predict/lstm** - Predict Vestas stock prices
  - Parameters: `days_ahead` (optional, values: 1, 3, or 7)

Access the interactive API documentation at http://localhost:8000/docs

## 📊 Streamlit Dashboard

The Streamlit dashboard provides an interactive interface for:
- Visualizing historical stock prices
- Viewing price predictions
- Analyzing price volatility
- Examining technical indicators

Access the dashboard at http://localhost:8501

## 🔄 Daily Updates

The system automatically runs the pipeline daily at 8:30 AM to:
- Fetch the latest stock and macroeconomic data
- Retrain the LSTM model with the latest data
- Update the API with the new model

## 🛠️ Model Details

- **Model Type:** Sequence-to-Sequence LSTM (Long Short-Term Memory)
- **Input Features:** Price history, technical indicators, macroeconomic data (S&P 500 ETF (SPY), Vanguard FTSE Europe ETF (VGK), EUR/USD exchange rates, and United States Oil Fund (USO) to contextualize market trends
- **Forecast Horizons:** 1-day, 3-day, and 7-day price predictions
- **Evaluation Metrics:** RMSE, MAE, R²

## 📝 License

This project is provided as-is for educational and demonstration purposes.
