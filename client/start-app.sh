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

exec streamlit run app.py

