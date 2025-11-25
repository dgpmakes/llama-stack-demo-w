"""
RAG Loader module for loading documents into LlamaStack vector stores.

This module can be used as a standalone script or imported as a library.
"""

import os
from pathlib import Path
from typing import List

from llama_stack_client import LlamaStackClient
from llama_stack_client.types import VectorStoreSearchResponse
from llama_stack_client.types.file import File
from llama_stack_client.types.model import Model
from llama_stack_client.types.vector_store import VectorStore

# Import from our utility modules
from utils import create_client, list_models, get_embedding_model
from vector_stores import create_vector_store, list_vector_stores, delete_vector_store, search_vector_store
from files import list_files_in_folder, upload_files


# EMBEDDING_MODEL = "granite-embedding-125m"
# EMBEDDING_DIMENSION = "768"
# EMBEDDING_MODEL_PROVIDER = "sentence-transformers"
# CHUNK_SIZE_IN_TOKENS = 512
# LLAMA_STACK_HOST = "localhost"
# LLAMA_STACK_PORT = "8080"
# LLAMA_STACK_SECURE = "False"
# DOCS_FOLDER = "./docs"

def load_command() -> None:
    """
    Load documents from a folder into a LlamaStack vector store.
    """
    
    try:
        # Get embedding model id, dimension and provider
        embedding_model_id = os.environ.get("EMBEDDING_MODEL")
        if embedding_model_id is None:
            raise ValueError("EMBEDDING_MODEL environment variable must be set")
        embedding_model_dimension = os.environ.get("EMBEDDING_DIMENSION")
        if embedding_model_dimension is None:
            raise ValueError("EMBEDDING_DIMENSION environment variable must be set")
        embedding_model_provider = os.environ.get("EMBEDDING_MODEL_PROVIDER")
        if embedding_model_provider is None:
            raise ValueError("EMBEDDING_MODEL_PROVIDER environment variable must be set")

        # Get chunk size in tokens
        chunk_size_in_tokens = os.environ.get("CHUNK_SIZE_IN_TOKENS", "512")
        chunk_size_in_tokens = int(chunk_size_in_tokens)

        # Get LlamaStack host, port and secure
        host = os.environ.get("LLAMA_STACK_HOST")
        if not host:
            raise ValueError("LLAMA_STACK_HOST environment variable must be set")
        port = os.environ.get("LLAMA_STACK_PORT")
        if not port:
            raise ValueError("LLAMA_STACK_PORT environment variable must be set")
        secure = os.environ.get("LLAMA_STACK_SECURE", "false").lower() in ["true", "1", "yes"]
        
        # Get vector store name
        vector_store_name = os.environ.get("VECTOR_STORE_NAME", "rag-store")

        # Get ranker
        ranker = str(os.environ.get("RANKER", "default"))

        # Get score threshold
        score_threshold = float(os.environ.get("SCORE_THRESHOLD", "0.8"))

        # Get max num results
        max_num_results = int(os.environ.get("MAX_NUM_RESULTS", "10"))

        # Get test query
        test_query = str(os.environ.get("TEST_QUERY", "Tell me about taxes in Lysmark."))

        # Add this after line ~195 where you read the environment variables
        print(f"DEBUG - Environment variables:")
        print(f"  HOST: '{host}'")
        print(f"  PORT: '{port}' (type: {type(port)})")
        print(f"  SECURE: '{secure}'")
        print(f"  VECTOR_STORE_NAME: '{vector_store_name}'")
        print(f"  RANKER: '{ranker}'")
        print(f"  SCORE_THRESHOLD: '{score_threshold}'")
        print(f"  MAX_NUM_RESULTS: '{max_num_results}'")
        print(f"  TEST_QUERY: '{test_query}'")
        
        # Get documents folder
        docs_folder: str = os.environ.get("DOCS_FOLDER")
        if not docs_folder:
            raise ValueError("DOCS_FOLDER environment variable must be set")
        print(f"  DOCS_FOLDER=>: '{docs_folder}'")

        # Initialize client
        client: LlamaStackClient = create_client(host=host, port=int(port), secure=secure)
        print(f"Connected to LlamaStack at {host}:{port}")
        
        # List vector stores
        vector_stores: List[VectorStore] = list_vector_stores(client, name=vector_store_name)
        if not vector_stores:
            print("No vector stores found")
        for vector_store in vector_stores:
            print(f"Deleting vector store: {vector_store})")
            delete_vector_store(client, vector_store.id)
            print(f"Deleted vector store: {vector_store.id}")

        # Get embedding model
        embedding_model = get_embedding_model(client, embedding_model_id, embedding_model_provider)
        if not embedding_model:
            raise ValueError(f"Embedding model {embedding_model_id} not found for provider {embedding_model_provider}")
        print(f"Using embedding model: {embedding_model.identifier} (dimension: {embedding_model.metadata['embedding_dimension']})")

        # Load documents from folder
        files_paths: List[Path] = list_files_in_folder(docs_folder)
        
        # Upload files into the vector database
        files: List[File] = upload_files(client, files_paths)
        if not files:
            raise ValueError("No files uploaded")

        # List models
        models: List[Model] = list_models(client)
        for model in models:
            print(f"Model: {model.identifier} (provider: {model.provider_id} type: {model.api_model_type})")

        # Create vector store
        vector_store: VectorStore = create_vector_store(client, vector_store_name, files)
        
        print(f"Files uploaded into the vector store {vector_store.name} (id: {vector_store.id})")

        # Search the vector store
        search_response: VectorStoreSearchResponse = search_vector_store(client, vector_store.id, test_query, max_num_results=max_num_results, ranker=ranker, score_threshold=score_threshold)
        print(f"Search response: {search_response}")
        
        return vector_store
    except Exception as e:
        print(f"Error: {e}")
        raise e


def main() -> None:
    """Main function to load documents and insert them into the vector database"""

    # Call the main loading function
    load_command()


if __name__ == "__main__":
    main()