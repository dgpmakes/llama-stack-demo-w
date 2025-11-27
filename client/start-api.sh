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
: ${SERVER_PORT:=8700}

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

# Curl example
# curl -X 'POST' \
#   'http://0.0.0.0:8700/agents/execute' \
#   -H 'accept: application/json' \
#   -H 'Content-Type: application/json' \
#   -d '{
#   "agent_type": "default",
#   "input_text": "Tell me how much taxes do I have to pay if my yearly income is 43245€",
#   "model_name": "llama-3-1-8b-w4a16",
#   "system_instructions": "You are a helpful AI assistant.",
#   "tools": [
#       { 
#         "type": "mcp",
#         "server_label": "dmcp",
#         "server_url": "https://compatibility-engine-llama-stack-demo.apps.ocp.sandbox3322.opentlc.com/sse",
#         "transport": "sse",
#         "require_approval": "never"
#       }
#   ]
# }'

exec python server.py

