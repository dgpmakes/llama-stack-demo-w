#!/bin/bash

# Simple start script for Streamlit Chat App

set -e

# Load environment variables from .env if it exists
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Set defaults
: ${LLAMA_STACK_HOST:=localhost}
: ${LLAMA_STACK_PORT:=8080}
: ${LLAMA_STACK_SECURE:=false}

export LLAMA_STACK_HOST
export LLAMA_STACK_PORT
export LLAMA_STACK_SECURE

echo "Starting Streamlit Chat App..."
echo "LlamaStack: ${LLAMA_STACK_HOST}:${LLAMA_STACK_PORT}"
echo "Chat App: http://localhost:8501"
echo ""

# Start health server in background (Flask + Streamlit conflict in same process on reload)
echo "Starting health server on port ${HEALTH_PORT:-8081}..."
python health-server.py &
HEALTH_PID=$!
echo "Health server started with PID: $HEALTH_PID"

# Give health server a moment to start
sleep 1

# Start Streamlit (will block here)
exec streamlit run app.py

