#!/bin/bash

# run.sh - a utility script for running the Vestas stock prediction app with Docker

# ensure we're in the project root directory
cd "$(dirname "$0")/.."

# make directories for data persistence
echo "Creating necessary directories if they don't exist..."
mkdir -p data/raw/stocks \
    data/raw/macro \
    data/intermediate/combined \
    data/intermediate/preprocessed \
    data/features \
    models \
    reports \
    results \
    plots

# check if .env file exists
if [ ! -f .env ]; then
    echo "Warning: .env file not found. Creating a template..."
    echo "ALPHA_VANTAGE_API_KEY=\"your_api_key_here\"" > .env
    echo "Please edit .env and add your Alpha Vantage API key before continuing."
    exit 1
fi

# check if the API key is set to the default value
if grep -q "your_api_key_here" .env; then
    echo "Warning: Alpha Vantage API key not set in .env file."
    echo "Please edit .env and add your real API key before continuing."
    exit 1
fi

# start the container with docker-compose
echo "Starting Vestas stock prediction application..."
docker-compose up -d

# check if the container started
if [ $? -eq 0 ]; then
    echo "Application started successfully. The API will be available at http://localhost:8000"
    echo "API documentation can be accessed at http://localhost:8000/docs"
    echo ""
    echo "To view logs: docker-compose logs -f"
    echo "To stop: docker-compose down"
else
    echo "Failed to start the application. Please check the logs for errors."
fi 