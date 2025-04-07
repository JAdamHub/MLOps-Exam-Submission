# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies that might be needed by some Python packages
# (Example: build-essential for C extensions). Adjust if needed.
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Copy only requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
# This includes src, models, and potentially data needed at runtime
# For production, consider only copying necessary artifacts (src, models)
COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define the command to run the application
# Run uvicorn pointing to the FastAPI app object inside src/api/main.py
# Use 0.0.0.0 to listen on all available network interfaces
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
