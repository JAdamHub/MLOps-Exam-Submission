# Docker Setup for Vestas Stock Price Prediction

This directory contains Docker-related files for containerizing the Vestas stock price prediction application.

## Requirements

- Docker and Docker Compose installed on your system
- Alpha Vantage API key (set in `.env` file)

## Docker Setup

The application is containerized using Docker with the following components:

- FastAPI application for serving predictions and historical data
- Scheduled pipeline for data collection and model training
- Volume mounts for persistent data storage

## Running with Docker

### Using Docker Compose (Recommended)

1. Ensure your `.env` file is set up with your Alpha Vantage API key:
   ```
   ALPHA_VANTAGE_API_KEY="your_api_key_here"
   ```

2. From the project root directory, run:
   ```bash
   docker-compose up -d
   ```

3. To view logs:
   ```bash
   docker-compose logs -f
   ```

4. To stop the container:
   ```bash
   docker-compose down
   ```

### Using Docker Directly

1. Build the Docker image:
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

3. To stop the container:
   ```bash
   docker stop vestas-stock-api
   docker rm vestas-stock-api
   ```

## API Access

Once running, the API can be accessed at:

- API endpoint: http://localhost:8000
- API documentation: http://localhost:8000/docs

## Troubleshooting

- If the container fails to start, check the logs with `docker logs vestas-stock-api`
- Ensure all required data directories exist and have appropriate permissions
- Verify that your Alpha Vantage API key is correct in the `.env` file 