# 🪙 Cryptocurrency Price Prediction MLOps Pipeline 🚀

A project demonstrating a basic MLOps pipeline for predicting short-term cryptocurrency price movements (specifically, whether the next day's closing price will be higher than the current day's) using historical data from the public CoinGecko API.

## 📁 Project Structure
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

## ⚙️ Setup

1.  **Clone the repository:**
    ```bash
    # Replace with your actual repository link after creating it
    git clone https://github.com/JAdamhub/MLOps-Exam-Submission.git
    cd MLOps-Exam-Submission
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you encounter NumPy version issues, ensure you have `numpy<2.0` in `requirements.txt` as specified.*

## ▶️ Usage

### 1. Run the Full Data Pipeline:

This command fetches data from CoinGecko, preprocesses it, engineers features, and trains the model, saving artifacts in `data/` and `models/`.

```bash
# Run from the project root directory
python -m src.pipeline.main 
```

### 2. Run the API Locally (using Uvicorn):

Starts the FastAPI server on `http://localhost:8000`.

```bash
# Make sure you are in the project root directory
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Build and Run the API with Docker:

Requires Docker Desktop to be installed and running.

```bash
# Build the Docker image
docker build -t crypto-predictor:latest .

# Run the Docker container
# Use -d to run in detached mode (background)
docker run -d -p 8000:8000 --name crypto-api crypto-predictor:latest
```

To stop the container: `docker stop crypto-api` 🛑
To remove the container: `docker rm crypto-api` 🗑️

## 🧪 Testing the API

Once the API is running (either locally with Uvicorn or via Docker), you can test the prediction endpoint:

1.  **Swagger UI:** Open `http://localhost:8000/docs` 📄 in your browser for an interactive interface.
2.  **cURL:** Use a command-line tool like `curl` 💻.

**Example `curl` command:**

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{ 
    "price": 0.5, 
    "market_cap": 0.6, 
    "total_volume": 0.4, 
    "price_lag_1": 0.49, 
    "price_lag_3": 0.48, 
    "price_lag_7": 0.45, 
    "price_sma_7": 0.48, 
    "price_sma_30": 0.46, 
    "price_volatility_14": 0.05, 
    "day_of_week": 3, 
    "month": 10, 
    "year": 2023 
  }'
```

**Getting Sample Feature Data:**

To get realistic input values for the JSON payload above:

1.  Run the pipeline (`python src/pipeline/main.py`).
2.  Open the generated file: `data/features/bitcoin_usd_365d_features.csv`.
3.  Pick any row.
4.  Copy the values for all columns **except** `timestamp` and `target_price_up`.
5.  Use these values in the JSON payload for your API request.

**API Response:**
The API will return a JSON like:
```json
{
  "prediction": 1,  // 1 = Price predicted to go up 📈, 0 = Price predicted to go down/stay same 📉
  "probability": 0.65 // Model's probability estimate for class 1 (price up)
}
```

## 🔩 Pipeline Details

*   **Data Source:** CoinGecko API (`/coins/{id}/market_chart` endpoint) 🦎.
*   **Preprocessing:** Forward/backward fill for missing values, `MinMaxScaler` for numerical features 🧼.
*   **Features:** Lagged prices, Simple Moving Averages (SMA), Rolling Standard Deviation (Volatility), Day of Week, Month, Year 📊.
*   **Model:** `LogisticRegression` (from scikit-learn) 🧠.
*   **Target:** Binary classification - Predict if `price` tomorrow > `price` today (Up=1 / Down=0) 🎯.

## 📈 Monitoring

Currently, only basic Python logging is implemented. A full monitoring solution (e.g., tracking metrics over time, data drift detection, alerts) is **not implemented** but would be a crucial addition for a production system. 🚧

## 🖥️ Frontend Link

No frontend component (e.g., Streamlit, GitHub Pages) was developed for this project. ❌

# Bitcoin Forudsigelsesværktøj

Dette projekt indeholder en API og en Streamlit-app til at analysere og forudsige Bitcoin-priser.

## Struktur

Projektet er organiseret som følger:

- `src/ny_api/`: FastAPI-applikation til at hente Bitcoin-data og lave forudsigelser
- `src/ny_streamlit/`: Streamlit-applikation til at visualisere data og forudsigelser
- `models/`: Trænede maskinlæringsmodeller og skalering
- `data/`: Datakatalog med Bitcoin og makroøkonomiske data

## Kom i gang

### Opsætning af miljø

1. Sørg for at du har Python 3.8+ installeret
2. Installer afhængigheder:

```bash
pip install -r requirements.txt
```

### Start API-serveren

```bash
# Fra projektets rod
uvicorn src.ny_api.main:app --reload --port 8000
```

API-dokumentation vil være tilgængelig på: http://localhost:8000/docs

### Start Streamlit-appen

```bash
# Fra projektets rod
streamlit run src/ny_streamlit/app.py
```

Streamlit-appen vil køre på: http://localhost:8501

## API Endpoints

- `GET /`: Root endpoint med API-information
- `GET /health`: Health check
- `GET /prices`: Hent Bitcoin-prishistorik
  - Parametre: `period` (1month, 3months, 6months, 1year, all), `limit` (antal datapunkter)
- `POST /predict`: Forudsig Bitcoin-priser
  - Body: JSON med features til forudsigelsen

## Miljøvariabler

- `API_URL`: URL til API-serveren (standard: http://localhost:8000)
- `ENVIRONMENT`: Miljøindstilling (development, production) påvirker logging

## Afhængigheder

- FastAPI
- Streamlit
- Pandas
- NumPy
- Plotly
- scikit-learn
- XGBoost
- joblib
- requests

## Licens

Dette projekt er udviklet til uddannelsesformål.

## 🐳 Docker Instructions

This project can be run in a Docker container for easy deployment and isolation. We provide both Docker Compose and standalone Docker configurations.

### Prerequisites
- Docker and Docker Compose installed on your system
- Alpha Vantage API key (you can get a free key at [Alpha Vantage](https://www.alphavantage.co/support/#api-key))

### Using the Run Script (Recommended)

1. Run the setup script:
   ```bash
   ./docker/run.sh
   ```
   This script will:
   - Create necessary directories
   - Check for the .env file with your API key
   - Start the container with Docker Compose

2. Access the API at: http://localhost:8000
   - API documentation: http://localhost:8000/docs

### Manual Docker Compose Setup

1. Create directories for data persistence:
   ```bash
   mkdir -p data/raw/stocks data/raw/macro data/intermediate/combined data/intermediate/preprocessed data/features models reports results plots
   ```

2. Ensure your `.env` file contains your Alpha Vantage API key:
   ```
   ALPHA_VANTAGE_API_KEY="your_api_key_here"
   ```

3. Start the container:
   ```bash
   docker-compose up -d
   ```

4. To view logs:
   ```bash
   docker-compose logs -f
   ```

5. To stop the container:
   ```bash
   docker-compose down
   ```

### Using Docker Directly

1. Build the image:
   ```bash
   docker build -t vestas-stock-api .
   ```

2. Run the container:
   ```bash
   docker run -d -p 8000:8000 \
     -v ./data:/app/data \
     -v ./models:/app/models \
     -v ./reports:/app/reports \
     -v ./results:/app/results \
     --env-file .env \
     --name vestas-stock-api \
     vestas-stock-api
   ```

For more detailed instructions, see [docker/README.md](docker/README.md).
