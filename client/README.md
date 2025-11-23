# RAG Loader CLI

A command-line interface for managing Retrieval-Augmented Generation (RAG) operations with LlamaStack.

## Features

- **Document Loading**: Load documents into vector stores for RAG
- **Vector Search**: Search documents with semantic similarity
- **Agent Commands**: Run AI agents with different implementations
  - Default Agent: Native Llama Stack with MCP servers
  - LangChain Agent: LangChain 1.0 with MCP adapters (see [LANGCHAIN_AGENT_README.md](./LANGCHAIN_AGENT_README.md))
- **Model Management**: List and inspect available models
- **Tool Management**: Manage MCP tool groups

## Installation

```bash
# Install dependencies
uv sync

# Or install directly with pip
pip install -e .
```

## Environment Variables

Set the following environment variables before running commands:

```bash
# LlamaStack connection
export LLAMA_STACK_HOST="localhost"
export LLAMA_STACK_PORT="8080"
export LLAMA_STACK_SECURE="false"

# Embedding model configuration
export EMBEDDING_MODEL="granite-embedding-125m"
export EMBEDDING_DIMENSION="768"
export EMBEDDING_MODEL_PROVIDER="sentence-transformers"
export CHUNK_SIZE_IN_TOKENS="512"

# Vector store configuration
export VECTOR_STORE_NAME="rag-store"
export DOCS_FOLDER="./docs"

# Search configuration
export RANKER="default"
export SCORE_THRESHOLD="0.8"
export MAX_NUM_RESULTS="10"
export TEST_QUERY="Tell me about taxes in Lysmark."
```

## Usage

### Load Command

Load documents from a folder into the vector store:

```bash
# Basic usage
python run.py load

# With delay on failure
python run.py load --delay 5
```

This command will:
1. Connect to LlamaStack
2. List and delete existing vector stores
3. Upload documents from the configured folder
4. Create a new vector store with the documents
5. Run a test search query

### Search Command

Search the vector store for relevant documents:

```bash
# Basic search
python run.py search --query "Tell me about taxes"

# Search with custom parameters
python run.py search --query "What is the capital?" --max-results 5

# Search specific vector store
python run.py search --query "Investment info" --vector-store-id my-store-id

# Advanced search with custom scoring
python run.py search \
  --query "Tell me about regulations" \
  --max-results 20 \
  --score-threshold 0.7 \
  --ranker "default"
```

### Search Command Options

- `--query`: (Required) The search query text
- `--vector-store-id`: ID of the vector store to search (default: uses the latest)
- `--max-results`: Maximum number of results to return (default: 10)
- `--score-threshold`: Minimum score threshold for results (default: 0.8)
- `--ranker`: Ranker to use for scoring (default: "default")

### Agent Command

Run AI agents with different implementations:

```bash
# Default agent (native Llama Stack)
python run.py agent --agent-type default --input "What is the weather?"

# LangChain agent (LangChain 1.0 with MCP adapters)
python run.py agent --agent-type lang_chain --input "Analyze financial data"

# With custom model
python run.py agent \
  --agent-type lang_chain \
  --input "What are the tax implications?" \
  --model "meta-llama/Llama-3.2-3B-Instruct"

# With custom system instructions
python run.py agent \
  --agent-type default \
  --input "Hello" \
  --instructions "You are a friendly assistant"
```

**See [LANGCHAIN_AGENT_README.md](./LANGCHAIN_AGENT_README.md) for detailed LangChain agent documentation.**

### Model Command

List and inspect available models:

```bash
# List all models
python run.py model list

# List with details
python run.py model list --verbose

# Filter by provider and type
python run.py model list --provider meta --type llm

# Get model info
python run.py model info --model "meta-llama/Llama-3.2-3B-Instruct"
```

### Tool Command

Manage MCP tool groups:

```bash
# List all tool groups
python run.py tool groups

# List with details
python run.py tool groups --verbose

# List all tools from all groups
python run.py tool list --all

# List tools from specific group
python run.py tool list --group "my-tool-group"

# List tools with details
python run.py tool list --all --verbose
```

## Examples

### Complete Workflow

```bash
# 1. Load documents
python run.py load

# 2. Search the loaded documents
python run.py search --query "What are the tax rates?"

# 3. Search with more results
python run.py search --query "Investment regulations" --max-results 20
```

### Using with Docker

```bash
# Run load command in container
docker run --rm \
  -v $(pwd)/docs:/app/docs \
  -e LLAMA_STACK_HOST="llama-stack" \
  -e LLAMA_STACK_PORT="8080" \
  -e EMBEDDING_MODEL="granite-embedding-125m" \
  -e EMBEDDING_DIMENSION="768" \
  -e EMBEDDING_MODEL_PROVIDER="sentence-transformers" \
  -e DOCS_FOLDER="/app/docs" \
  rag-loader:latest python run.py load
```

## Module Usage

You can also import and use the functions directly in your Python code:

```python
from rag_loader import (
    create_client,
    load_documents_to_vector_store,
    search_vector_store,
    list_vector_stores
)

# Load documents
vector_store = load_documents_to_vector_store(delay_seconds=5)

# Or use individual functions
client = create_client(host="localhost", port=8080, secure=False)
vector_stores = list_vector_stores(client)
results = search_vector_store(
    client=client,
    vector_store_id=vector_stores[0].id,
    query="Tell me about taxes",
    max_num_results=10
)
```

## Troubleshooting

### Connection Errors

If you get connection errors, verify:
- LlamaStack is running and accessible
- `LLAMA_STACK_HOST` and `LLAMA_STACK_PORT` are set correctly
- Network connectivity between client and LlamaStack

### No Vector Stores Found

If search fails with "No vector stores found":
1. Run the `load` command first to create a vector store
2. Verify the load command completed successfully
3. Check that documents were uploaded to the vector store

### Module Import Errors

If you get import errors when running `run.py`:
- Ensure you're in the correct directory
- Verify dependencies are installed: `uv sync`
- Check Python version: requires Python >= 3.12

