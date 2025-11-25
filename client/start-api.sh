#!/bin/bash

# Simple start script for REST API Server

set -e

# Load environment variables from .env if it exists
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Set defaults
: ${LLAMA_STACK_HOST:=localhost}
: ${LLAMA_STACK_PORT:=8080}
: ${LLAMA_STACK_SECURE:=false}
: ${SERVER_HOST:=0.0.0.0}
: ${SERVER_PORT:=8000}

export LLAMA_STACK_HOST
export LLAMA_STACK_PORT
export LLAMA_STACK_SECURE
export SERVER_HOST
export SERVER_PORT

echo "Starting REST API Server..."
echo "LlamaStack: ${LLAMA_STACK_HOST}:${LLAMA_STACK_PORT}"
echo "REST API: ${SERVER_HOST}:${SERVER_PORT}"
echo "API docs: http://localhost:${SERVER_PORT}/docs"
echo ""

exec python server.py

