# 📈 Vestas Stock Price Prediction MLOps Pipeline V2.0 🚀

Welcome to Version 2.0 of the MLOps pipeline for predicting Vestas (VWSB.DEX) stock prices! This project demonstrates a complete machine learning operations workflow, leveraging historical market data, macroeconomic indicators, and an LSTM deep learning model to forecast future stock prices across multiple time horizons (1, 3, and 7 days).

## ✨ V2.0 Highlights

*   **Database-Centric:** Pipeline steps now primarily use an SQLite database (`market_data.db`) for data persistence, reducing reliance on intermediate CSV files.
*   **In-Memory Data Transfer:** Preprocessing, Feature Engineering, and Training steps pass DataFrames directly in memory for improved efficiency.
*   **Refined API Data Loading:** The prediction API now sources its data directly from the database and performs necessary feature engineering, ensuring consistency with the latest pipeline run.
*   **Code Simplification:** Removed unused functions and redundant code, particularly in the training script.

## 🌟 Key Features

*   **Data Collection:** Fetches Vestas stock data and relevant macroeconomic indicators (ETFs, FX rates) from Alpha Vantage API. ↔️
*   **Data Storage:** Persists collected data in an SQLite database. 💾
*   **Preprocessing:** Cleans and prepares the data. ✨
*   **Feature Engineering:** Generates a rich set of technical indicators and time-based features. ⚙️
*   **LSTM Model Training:** Trains a Sequence-to-Sequence LSTM model for multi-horizon price prediction. 🧠
*   **API Server:** Provides RESTful endpoints (built with FastAPI) to serve price history and predictions. 🌐
*   **Interactive Dashboard:** Visualizes historical data and model predictions using Streamlit. 📊
*   **Containerization:** Uses Docker and Docker Compose for easy setup and deployment. 🐳
*   **Scheduled Retraining:** Automatically retrains the model daily with fresh data. ⏰

## 🔄 Pipeline Overview V2.0

The core pipeline now operates as follows:

```text
                                                       (Scheduled Daily @ 08:30)
                                                                  |
                                                                  v
[Alpha Vantage API] ---> (1. Data Collection) ---> [data/raw/stocks/market_data.db]
                                                                  |
                               +----------------------------------+
                               |
                               v
[market_data.db] ---> (2. Preprocessing) ---------> [DataFrame (Memory)]
                                                                  |
                                                                  v
[DataFrame (Memory)] --> (3. Feature Engineering) --> [DataFrame (Memory)]
                                                                  |
                                                                  v
[DataFrame (Memory)] --> (4. LSTM Training) --------> [models/* artifacts]
                                                                  |
                                                                  v
[models/* artifacts] & [market_data.db] ---> (5. API & Streamlit Startup) ---> [User via Browser]

```

**Explanation:**

1.  **Data Collection:** Fetches data from Alpha Vantage and stores the combined raw/preprocessed data in `market_data.db`.
2.  **Preprocessing:** Loads data from the database, performs initial cleaning, and returns a DataFrame.
3.  **Feature Engineering:** Takes the preprocessed DataFrame, calculates numerous features, cleans NaNs, and returns the feature-rich DataFrame.
4.  **LSTM Training:** Takes the features DataFrame, trains the Seq2Seq LSTM model, and saves the model (`.keras`) and supporting artifacts (`.joblib` scalers, feature names, etc.) to the `models/` directory.
5.  **API & Streamlit:** The FastAPI server and Streamlit dashboard are started. The API loads the necessary model artifacts from `models/`. For predictions or historical data display, it queries the `market_data.db` and performs necessary processing (like feature engineering for predictions).
6.  **Scheduling:** The entire pipeline (Steps 1-4) is scheduled to run daily to retrain the model. The API server is restarted after a successful pipeline run to load the new model.

## 📁 Project Structure

```text
.
├── .env                    # Environment variables (API key) 🔑 NEEDS TO BE CREATED
├── .gitignore              # Files ignored by Git
├── .dockerignore           # Files ignored by Docker build
├── Dockerfile              # Docker image build instructions
├── README.md               # This file! 📖 (V2.0)
├── docker-compose.yml      # Docker service definitions
├── requirements.txt        # Python dependencies 📄
├── data/                   # Datasets 📁
│   └── market_data.db      # Primary data store 💾
├── docker/                 # Docker helper scripts 🐳
│   ├── healthcheck.sh
│   └── run.sh
├── models/                 # Trained models and artifacts 🤖
├── plots/                  # Saved plot images 📉 (e.g., from training)
├── reports/                # Generated reports 📊 (placeholder)
├── results/                # Model evaluation metrics 📈
└── src/                    # Source code 💻
    ├── api/                # FastAPI application
    ├── main.py             # Main entry point (starts pipeline, API, Streamlit)
    ├── pipeline/           # Data processing and model training scripts
    └── streamlit/          # Streamlit dashboard application
```

## ⚙️ Setup and Installation

### Prerequisites
*   Python 3.10+ 🐍
*   Docker and Docker Compose (Recommended for ease of use) 🐳
*   Alpha Vantage API key (Get a free key: [alpha_vantage.co](https://www.alphavantage.co/support/#api-key))

### Local Development Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/JAdamHub/MLOps-Exam-Submission.git
    cd MLOps-Exam-Submission
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create `.env` file:** Create a file named `.env` in the project root directory and add your Alpha Vantage API key:
    ```
    ALPHA_VANTAGE_API_KEY="YOUR_API_KEY_HERE"
    ```

## ▶️ Running the Application

### Option 1: Full Pipeline Run & Serve (Recommended for First Time)

This runs the entire data collection and training pipeline first, then starts the API and Streamlit app.

```bash
python -m src.main
```

*   The initial pipeline run might take some time (data fetching, training).
*   API will be available at `http://localhost:8000`.
*   Streamlit Dashboard will be available at `http://localhost:8501`.
*   Daily retraining is scheduled for 08:30 AM.

### Option 2: Start API & Streamlit Only (Using Existing Data/Model)

If you have already run the pipeline and have the `market_data.db` and `models/` artifacts, you can skip the initial pipeline run:

1.  **Temporarily edit `src/main.py`:** Comment out the line that calls `run_pipeline()`.
    ```python
    # from pipeline.pipeline_start import main as run_pipeline
    # run_pipeline() # Commented out
    logger.info("Skipping initial pipeline run.")
    ```
2.  **Run the main script:**
    ```bash
    python -m src.main
    ```
    *Remember to uncomment the lines in `src/main.py` later if you want the full startup behavior.*

## 🐳 Docker Deployment

Using Docker is the recommended way for a consistent environment.

### Using the Run Script (Easiest)

The helper script handles directory creation and checks:

```bash
./docker/run.sh
```

### Manual Docker Compose

Ensure you have created the `.env` file first.

1.  **Build and start services:**
    ```bash
    docker-compose up --build -d
    ```
    *   `--build`: Forces rebuilding the image if changes were made.
    *   `-d`: Runs in detached mode (background).

2.  **View logs:**
    ```bash
    docker-compose logs -f
    ```

3.  **Stop services:**
    ```bash
    docker-compose down
    ```

### Running the Pre-built Image (from GHCR)

```bash
# Make sure to replace YOUR_API_KEY_HERE
docker run -d --name mlops-app-v2 \\
  -p 8000:8000 \\
  -p 8501:8501 \\
  -e ALPHA_VANTAGE_API_KEY="YOUR_API_KEY_HERE" \\
  ghcr.io/jadamhub/mlops-exam-submission:latest
```

Access API at `http://localhost:8000` and Streamlit at `http://localhost:8501`.

## 🌐 API Endpoints

*   `GET /`: Welcome message.
*   `GET /health`: Check API health and loaded components status.
*   `GET /price/history?days={num_days}`: Get historical stock data (default: all available).
*   `POST /predict/lstm?days_ahead={1|3|7}`: Get LSTM model predictions (default: all horizons).

Access interactive API documentation via Swagger UI at `http://localhost:8000/docs`.

## 📊 Streamlit Dashboard

An interactive web application for:
*   Visualizing historical stock prices and volume.
*   Displaying LSTM model predictions for 1, 3, and 7 days ahead.
*   Viewing basic volatility metrics.

Access it at `http://localhost:8501`.

## 🧠 Model Details

*   **Type:** Sequence-to-Sequence (Seq2Seq) LSTM with Attention.
*   **Input Features:** Historical OHLCV, technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.), time-based features, macroeconomic indicators (closing prices of SPY, VGK, EUR/USD, USO, TLT).
*   **Targets:** Predicts the *price* for 1, 3, and 7 **trading days** ahead.
*   **Evaluation:** Metrics like RMSE, MAE, R² are calculated during training and saved.

## 🤔 Future Improvements & Considerations

*   **Feature Selection:** The feature engineering step generates many features. A systematic analysis could identify and remove less impactful features to simplify the model.
*   **Hyperparameter Tuning:** Implement a more robust hyperparameter optimization strategy (e.g., KerasTuner, Optuna) instead of manual testing.
*   **Error Handling:** Enhance error handling and reporting throughout the pipeline and API.
*   **Monitoring:** Add more sophisticated monitoring for data drift, model performance degradation, and API health.
*   **Testing:** Implement unit and integration tests for pipeline components and the API.

## 📝 License

This project is provided as-is for educational and demonstration purposes.