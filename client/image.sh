#!/bin/bash

set -euo pipefail

# --- Load .env (optional, for build-time defaults) ---
if [ -f .env ]; then
  source .env
fi

# --- Default values ---
COMPONENT_NAME="${COMPONENT_NAME:-$DEFAULT_COMPONENT_NAME}"
TAG="${TAG:-$DEFAULT_TAG}"
BASE_IMAGE="${BASE_IMAGE:-$DEFAULT_BASE_IMAGE}"
BASE_TAG="${BASE_TAG:-$DEFAULT_BASE_TAG}"
CONTAINER_FILE="${CONTAINER_FILE:-$DEFAULT_CONTAINER_FILE}"
CACHE_FLAG="${CACHE_FLAG:-$DEFAULT_CACHE_FLAG}"
REGISTRY="${REGISTRY:-$DEFAULT_REGISTRY}"
SERVER_PORT="${SERVER_PORT:-8000}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
HEALTH_PORT="${HEALTH_PORT:-8081}"

LOCAL_IMAGE="${COMPONENT_NAME}:${TAG}"
REMOTE_IMAGE="${REGISTRY}/${COMPONENT_NAME}:${TAG}"

# --- Build the image ---
function build() {
  echo "🔨 Building image ${LOCAL_IMAGE} with base image ${BASE_IMAGE}:${BASE_TAG}"
  podman build ${CACHE_FLAG} \
    -t "${LOCAL_IMAGE}" \
    -f "${CONTAINER_FILE}" . \
    --build-arg BASE_IMAGE="${BASE_IMAGE}:${BASE_TAG}" \
    --build-arg COMPONENT_NAME="${COMPONENT_NAME}"
  echo "✅ Build complete: ${LOCAL_IMAGE}"
}

# --- Push the image to registry ---
function push() {
  echo "📤 Pushing image to ${REMOTE_IMAGE}..."
  podman tag "${LOCAL_IMAGE}" "${REMOTE_IMAGE}"
  podman push "${REMOTE_IMAGE}"
  echo "✅ Image pushed to: ${REMOTE_IMAGE}"
}

# --- Run commands (run.py) ---
function run_command() {
  MODE="$1"
  shift
  
  USE_REMOTE=false
  if [[ "${1:-}" == "--remote" ]]; then
    USE_REMOTE=true
    shift
  fi

  IMAGE_TO_RUN=$(get_image "$USE_REMOTE")

  # Pick env file
  ENV_FILE=$(get_env_file)
  source "${ENV_FILE}"
  echo "📄 Using environment file: ${ENV_FILE}"

  # Adjust LLAMA_STACK_HOST for macOS
  adjust_host_for_macos

  echo "🚀 Running command: $*"
  
  podman run --rm -it --name llama-stack-cli \
    -e EMBEDDING_MODEL="${EMBEDDING_MODEL}" \
    -e EMBEDDING_DIMENSION="${EMBEDDING_DIMENSION}" \
    -e EMBEDDING_MODEL_PROVIDER="${EMBEDDING_MODEL_PROVIDER}" \
    -e VECTOR_STORE_PROVIDER_ID="${VECTOR_STORE_PROVIDER_ID}" \
    -e VECTOR_STORE_NAME="${VECTOR_STORE_NAME:-rag-store}" \
    -e RANKER="${RANKER:-default}" \
    -e SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.8}" \
    -e MAX_NUM_RESULTS="${MAX_NUM_RESULTS:-10}" \
    -e TEST_QUERY="${TEST_QUERY:-Tell me about taxes in Lysmark.}" \
    -e LLAMA_STACK_HOST="${LLAMA_STACK_HOST}" \
    -e LLAMA_STACK_PORT="${LLAMA_STACK_PORT}" \
    -e LLAMA_STACK_SECURE="${LLAMA_STACK_SECURE}" \
    -e DOCS_FOLDER="/app/docs" \
    -e CHUNK_SIZE_IN_TOKENS="${CHUNK_SIZE_IN_TOKENS}" \
    -e CHUNK_OVERLAP_IN_TOKENS="${CHUNK_OVERLAP_IN_TOKENS}" \
    -e NO_PROXY="localhost,127.0.0.1,host.containers.internal" \
    -v "${DOCS_FOLDER}:/app/docs:ro,Z" \
    "${IMAGE_TO_RUN}" \
    ./run.sh "$@"
}

# --- Run REST API Server ---
function run_api() {
  USE_REMOTE=false
  if [[ "${1:-}" == "--remote" ]]; then
    USE_REMOTE=true
    shift
  fi

  IMAGE_TO_RUN=$(get_image "$USE_REMOTE")

  # Pick env file
  ENV_FILE=$(get_env_file)
  source "${ENV_FILE}"
  echo "📄 Using environment file: ${ENV_FILE}"

  # Adjust LLAMA_STACK_HOST for macOS
  adjust_host_for_macos

  echo "🚀 Starting REST API Server on port ${SERVER_PORT}..."
  echo "📖 API docs will be available at: http://localhost:${SERVER_PORT}/docs"
  
  podman run --rm -it --name llama-stack-api \
    -p "${SERVER_PORT}:8000" \
    -e LLAMA_STACK_HOST="${LLAMA_STACK_HOST}" \
    -e LLAMA_STACK_PORT="${LLAMA_STACK_PORT}" \
    -e LLAMA_STACK_SECURE="${LLAMA_STACK_SECURE}" \
    -e SERVER_HOST="0.0.0.0" \
    -e SERVER_PORT="8000" \
    -e NO_PROXY="localhost,127.0.0.1,host.containers.internal" \
    "${IMAGE_TO_RUN}" \
    ./start-api.sh
}

# --- Run Streamlit App ---
function run_app() {
  USE_REMOTE=false
  if [[ "${1:-}" == "--remote" ]]; then
    USE_REMOTE=true
    shift
  fi

  IMAGE_TO_RUN=$(get_image "$USE_REMOTE")

  # Pick env file
  ENV_FILE=$(get_env_file)
  source "${ENV_FILE}"
  echo "📄 Using environment file: ${ENV_FILE}"

  # Adjust LLAMA_STACK_HOST for macOS
  adjust_host_for_macos

  echo "🚀 Starting Streamlit Chat App on port ${STREAMLIT_PORT}..."
  echo "💬 App will be available at: http://localhost:${STREAMLIT_PORT}"
  echo "🏥 Health server will be available at: http://localhost:${HEALTH_PORT}"
  echo "Image: ${IMAGE_TO_RUN}"
  
  podman run --rm -it --name llama-stack-app \
    -p "${STREAMLIT_PORT}:8501" \
    -p "${HEALTH_PORT}:8081" \
    -e VECTOR_STORE_NAME="${VECTOR_STORE_NAME:-rag-store}" \
    -e PYTHONUNBUFFERED=1 \
    -e STREAMLIT_LOGGER_LEVEL=info \
    -e LLAMA_STACK_HOST="${LLAMA_STACK_HOST}" \
    -e LLAMA_STACK_PORT="${LLAMA_STACK_PORT}" \
    -e LLAMA_STACK_SECURE="${LLAMA_STACK_SECURE}" \
    -e NO_PROXY="localhost,127.0.0.1,host.containers.internal" \
    "${IMAGE_TO_RUN}" \
    ./start-app.sh
}

# --- Run Bash ---
function run_bash() {
  USE_REMOTE=false
  if [[ "${1:-}" == "--remote" ]]; then
    USE_REMOTE=true
    shift
  fi

  IMAGE_TO_RUN=$(get_image "$USE_REMOTE")

  echo "🐚 Running bash in container: ${IMAGE_TO_RUN}"
  podman run --rm -it --name llama-stack-bash \
    -p "${HEALTH_PORT}:8081" \
    -e PYTHONUNBUFFERED=1 \
    -e STREAMLIT_LOGGER_LEVEL=info \
    -e LLAMA_STACK_HOST="${LLAMA_STACK_HOST}" \
    -e LLAMA_STACK_PORT="${LLAMA_STACK_PORT}" \
    -e LLAMA_STACK_SECURE="${LLAMA_STACK_SECURE}" \
    -e NO_PROXY="localhost,127.0.0.1,host.containers.internal" \
  "${IMAGE_TO_RUN}" /bin/bash
}

# --- Helper: Get image (local or remote) ---
function get_image() {
  local USE_REMOTE=$1
  local IMAGE_TO_RUN="${LOCAL_IMAGE}"
  
  if [ "$USE_REMOTE" = true ]; then
    echo "🌐 Pulling remote image ${REMOTE_IMAGE}..." >&2
    podman pull "${REMOTE_IMAGE}"
    IMAGE_TO_RUN="${REMOTE_IMAGE}"
    echo "✅ Image pulled from: ${REMOTE_IMAGE}" >&2
  fi
  
  echo "${IMAGE_TO_RUN}"
}

# --- Helper: Get environment file ---
function get_env_file() {
  if [ -f .test.env ]; then
    echo ".test.env"
  elif [ -f .env ]; then
    echo ".env"
  else
    echo "❌ No .test.env or .env file found." >&2
    exit 1
  fi
}

# --- Helper: Adjust host for macOS ---
function adjust_host_for_macos() {
  if [ "${LLAMA_STACK_HOST}" = "localhost" ] && [ "$(uname -s)" = "Darwin" ]; then
    LLAMA_STACK_HOST="host.containers.internal"
    echo "🍎 macOS detected: Using host.containers.internal for LLAMA_STACK_HOST"
  fi
}

# --- Show usage ---
function help() {
  cat << EOF
Usage: ./image.sh <command> [options] [args]

Commands:
  build                      Build the container image
  push                       Push the image to the registry
  
  cmd [--remote] <args>      Run CLI commands (run.py)
                            Examples:
                              ./image.sh cmd load
                              ./image.sh cmd search --query "test"
                              ./image.sh cmd model list
                              ./image.sh cmd --remote load
  
  api [--remote]             Run REST API server (port ${SERVER_PORT})
                            Examples:
                              ./image.sh api
                              ./image.sh api --remote
  
  app [--remote]             Run Streamlit chat app (port ${STREAMLIT_PORT})
                            Examples:
                              ./image.sh app
                              ./image.sh app --remote
  
  all                        Build and push the image

Options:
  --remote                   Use image from registry instead of local

Environment:
  Uses .test.env or .env for configuration
  Set SERVER_PORT (default: 8000) and STREAMLIT_PORT (default: 8501)

Examples:
  # Build and test locally
  ./image.sh build
  ./image.sh cmd load
  ./image.sh api
  
  # Push and run from registry
  ./image.sh push
  ./image.sh app --remote
EOF
}

# --- Entrypoint ---
case "${1:-}" in
  build) build ;;
  push) push ;;
  cmd|command) shift; run_command "cmd" "$@" ;;
  api|server) shift; run_api "$@" ;;
  app|streamlit) shift; run_app "$@" ;;
  bash) shift; run_bash "$@" ;;
  all)
    build
    push
    ;;
  *) help ;;
esac
