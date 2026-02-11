"""
Search command implementation for RAG operations.
"""

import os
import sys
from typing import Optional
from llama_stack_client.types import VectorStoreSearchResponse
from typing_extensions import List

from llama_stack_client.types.vector_store_search_response import Data
from utils import create_client
from vector_stores import search_vector_store, list_vector_stores


def search_command(
    query: str,
    vector_store_name: Optional[str] = None,
    search_mode: str = "vector",
    max_results: int = 10,
    score_threshold: float = 0.8,
    ranker: str = "default"
) -> None:
    """
    Search the vector store for relevant documents.
    
    Args:
        query: The search query
        vector_store_name: Name of the vector store to search (if None, uses latest across all)
        max_results: Maximum number of results to return (default: 10)
        score_threshold: Minimum score threshold for results (default: 0.8)
        ranker: Ranker to use for scoring (default: "default")
    """
    # Get connection parameters from environment
    host = os.environ.get("LLAMA_STACK_HOST", "localhost")
    port = int(os.environ.get("LLAMA_STACK_PORT", "8080"))
    secure = os.environ.get("LLAMA_STACK_SECURE", "false").lower() in ["true", "1", "yes"]
    
    print(f"Connecting to LlamaStack at {host}:{port}")
    client = create_client(host=host, port=port, secure=secure)
    
    # Resolve vector store: by name (latest with that name) or latest overall
    vector_stores = list(list_vector_stores(client, name=vector_store_name))
    if not vector_stores:
        if vector_store_name:
            print(f"Error: No vector store found with name '{vector_store_name}'. Please run 'load' command first or check the name.")
        else:
            print("Error: No vector stores found. Please run 'load' command first.")
        sys.exit(1)
    # Sort by creation time (newest first); fall back to list order if created_at missing or sort fails
    def _created_at_key(vs):
        created = getattr(vs, "created_at", None)
        return (0, None) if created is None else (1, created)

    try:
        vector_stores.sort(key=_created_at_key, reverse=True)
    except (TypeError, ValueError):
        pass
    vector_store = vector_stores[0]
    vector_store_id = vector_store.id
    print(f"Using vector store: {vector_store.name} (id: {vector_store_id})")
    
    # Perform search
    print(f"\nSearching for: '{query}'")
    print(f"Parameters: max_results={max_results}, score_threshold={score_threshold}, ranker={ranker}, search_mode={search_mode}")
    print("-" * 80)
    
    search_response: VectorStoreSearchResponse = search_vector_store(
        client=client,
        vector_store_id=vector_store_id,
        query=query,
        search_mode=search_mode,
        max_num_results=max_results,
        ranker=ranker,
        score_threshold=score_threshold
    )
    
    # Display results
    # The response might have different attribute names depending on the API version
    # Try common attribute names: results, chunks, data
    results: List[Data] = search_response.data
    if not results:
        print("No results found.")
        return
    
    print(f"\nFound {len(results)} results:\n")
    for idx, result in enumerate(results, 1):
        print(f"Result {idx}:")
        print(f"  Score: {result.score:.4f}")
        print(f"  Content: {result.content[:200]}..." if len(result.content) > 200 else f"  Content: {result.content}")
        if hasattr(result, 'metadata') and result.metadata:
            print(f"  Metadata: {result.metadata}")
        print()

