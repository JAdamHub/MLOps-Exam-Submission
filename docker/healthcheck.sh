#!/bin/bash

# healthcheck.sh - script to check if the API is working correctly

# check if the api port is reachable
if ! wget --quiet --spider http://localhost:8000/health; then
  echo "API is not responding"
  exit 1
fi

# check the health endpoint
HEALTH_RESPONSE=$(curl -s http://localhost:8000/health)

if echo "$HEALTH_RESPONSE" | grep -q "\"status\":\"healthy\""; then
  echo "API is healthy"
  exit 0
else
  echo "API is unhealthy"
  exit 1
fi 