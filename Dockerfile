FROM python:3.11-slim

# set working directory
WORKDIR /app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Europe/Copenhagen

# install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# copy project files
COPY . /app/

# ensure necessary directories exist
RUN mkdir -p /app/data/raw/stocks \
    /app/data/raw/macro \
    /app/data/intermediate/combined \
    /app/data/intermediate/preprocessed \
    /app/data/features \
    /app/models \
    /app/reports \
    /app/results \
    /app/plots

# make healthcheck script executable
RUN chmod +x /app/docker/healthcheck.sh

# install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# expose api port
EXPOSE 8000

# health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD /app/docker/healthcheck.sh

# create entrypoint script
RUN echo '#!/bin/bash\n\
python -m src.main\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# run entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"] 