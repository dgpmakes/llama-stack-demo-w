# Copyright 2025 IBM, Red Hat
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-License-Identifier: Apache-2.0

"""
RAG Pipeline Runner

This script finds the latest version of the 'rag-pipeline' and creates a run 
for each vector store provider ID specified.

Usage:
    python rag_pipeline_runner.py [OPTIONS]

Options can be passed as command-line arguments or environment variables.
Command-line arguments take precedence over environment variables.

Example:
    python rag_pipeline_runner.py --vector-store-provider-ids "milvus,pgvector"
    
    # Or using environment variables:
    VECTOR_STORE_PROVIDER_IDS="milvus,pgvector" python rag_pipeline_runner.py
"""

import argparse
import os
import sys
import time

import kfp

from shared.kubeflow import (
    get_token,
    get_route_host,
    get_pipeline_id_by_name,
    get_or_create_experiment,
    create_pipeline_run,
)


def parse_args():
    """Parse command-line arguments with environment variable defaults."""
    parser = argparse.ArgumentParser(
        description="Run RAG pipeline for multiple vector store providers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Pipeline identification
    parser.add_argument(
        "--pipeline-name",
        default=os.environ.get("PIPELINE_NAME", "rag-pipeline"),
        help="Name of the pipeline to run"
    )
    parser.add_argument(
        "--experiment-name",
        default=os.environ.get("EXPERIMENT_NAME", "rag-pipeline-experiment"),
        help="Name of the experiment"
    )
    parser.add_argument(
        "--vector-store-provider-ids",
        default=os.environ.get("VECTOR_STORE_PROVIDER_IDS", "milvus, milvus-remote, pgvector"),
        help="Comma-separated list of vector store provider IDs"
    )
    
    # Connection parameters
    parser.add_argument(
        "--token",
        default=os.environ.get("KFP_TOKEN"),
        help="KFP authentication token (default: read from service account)"
    )
    parser.add_argument(
        "--kfp-endpoint",
        default=os.environ.get("KFP_ENDPOINT"),
        help="KFP API endpoint (default: auto-discover from route)"
    )
    
    # Pipeline parameters
    parser.add_argument(
        "--git-repo",
        default=os.environ.get("PIPELINE_GIT_REPO", "alpha-hack-program/llama-stack-demo"),
        help="Git repository for pipeline data"
    )
    parser.add_argument(
        "--git-context",
        default=os.environ.get("PIPELINE_GIT_CONTEXT", "client/docs_pdf"),
        help="Git context path"
    )
    parser.add_argument(
        "--git-ref",
        default=os.environ.get("PIPELINE_GIT_REF", "next"),
        help="Git reference (branch/tag)"
    )
    parser.add_argument(
        "--filenames",
        default=os.environ.get("PIPELINE_FILENAMES", "2025_61-FR_INT.html.pdf"),
        help="Comma-separated list of filenames to process"
    )
    parser.add_argument(
        "--vector-store-name",
        default=os.environ.get("PIPELINE_VECTOR_STORE_NAME", "rag-store"),
        help="Base name for vector stores (provider ID will be appended)"
    )
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("PIPELINE_EMBEDDING_MODEL", "sentence-transformers/nomic-ai/nomic-embed-text-v1.5"),
        help="Embedding model identifier"
    )
    parser.add_argument(
        "--embedding-dimension",
        type=int,
        default=int(os.environ.get("PIPELINE_EMBEDDING_DIMENSION", "768")),
        help="Embedding dimension"
    )
    parser.add_argument(
        "--ranker",
        default=os.environ.get("PIPELINE_RANKER", "default"),
        help="Ranker to use"
    )
    parser.add_argument(
        "--chunk-size-in-tokens",
        type=int,
        default=int(os.environ.get("PIPELINE_CHUNK_SIZE_IN_TOKENS", "800")),
        help="Chunk size in tokens"
    )
    parser.add_argument(
        "--chunk-overlap-in-tokens",
        type=int,
        default=int(os.environ.get("PIPELINE_CHUNK_OVERLAP_IN_TOKENS", "400")),
        help="Chunk overlap in tokens"
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=os.environ.get("PIPELINE_USE_GPU", "false").lower() in ("true", "1", "yes"),
        help="Enable GPU usage"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the RAG pipeline runner."""
    args = parse_args()
    
    print("=== RAG Pipeline Runner ===")
    print(f"Pipeline name: {args.pipeline_name}")
    print(f"Experiment name: {args.experiment_name}")
    print(f"Vector store provider IDs: {args.vector_store_provider_ids}")
    
    # Get authentication token
    token = args.token
    if not token:
        print("Token not provided, attempting to read from service account...")
        token = get_token()
    
    if not token:
        print("ERROR: Could not obtain authentication token")
        sys.exit(1)
    
    # Get KFP endpoint
    kfp_endpoint = args.kfp_endpoint
    if not kfp_endpoint:
        print("KFP endpoint not provided, attempting to discover from route...")
        dspa_host = get_route_host("ds-pipeline-dspa")
        if dspa_host:
            kfp_endpoint = f"https://{dspa_host}"
        else:
            print("ERROR: Could not discover KFP endpoint")
            sys.exit(1)
    
    # Ensure endpoint has protocol
    if not kfp_endpoint.startswith("http"):
        kfp_endpoint = f"https://{kfp_endpoint}"
    
    print(f"KFP endpoint: {kfp_endpoint}")
    
    # Create KFP client
    try:
        client = kfp.Client(host=kfp_endpoint, existing_token=token)
    except Exception as e:
        print(f"ERROR: Failed to create KFP client: {e}")
        sys.exit(1)
    
    # Get pipeline ID
    pipeline_id = get_pipeline_id_by_name(client, args.pipeline_name)
    if not pipeline_id:
        print(f"ERROR: Pipeline '{args.pipeline_name}' not found")
        sys.exit(1)
    print(f"Pipeline ID: {pipeline_id}")
    
    # Get or create experiment
    experiment_id = get_or_create_experiment(client, args.experiment_name)
    print(f"Experiment ID: {experiment_id}")
    
    # Parse vector store provider IDs
    providers = [p.strip() for p in args.vector_store_provider_ids.split(",") if p.strip()]
    
    if not providers:
        print("ERROR: No vector store provider IDs specified")
        sys.exit(1)
    
    print(f"\nWill create {len(providers)} pipeline runs for providers: {providers}")
    
    # Create runs for each vector store provider
    created_runs = []
    for provider in providers:
        run_name = f"{args.pipeline_name}-{provider}-{int(time.time())}"
        print(f"\n=== Creating run for provider: {provider} ===")
        print(f"Run name: {run_name}")
        
        # Build parameters for this run
        params = {
            "git_repo": args.git_repo,
            "git_context": args.git_context,
            "git_ref": args.git_ref,
            "filenames": args.filenames,
            "vector_store_name": f"{args.vector_store_name}-{provider}",
            "vector_store_provider_id": provider,
            "embedding_model": args.embedding_model,
            "embedding_dimension": args.embedding_dimension,
            "ranker": args.ranker,
            "chunk_size_in_tokens": args.chunk_size_in_tokens,
            "chunk_overlap_in_tokens": args.chunk_overlap_in_tokens,
            "use_gpu": args.use_gpu,
        }
        
        print(f"Parameters: {params}")
        
        try:
            run_id = create_pipeline_run(
                client,
                pipeline_id,
                experiment_id,
                run_name,
                params
            )
            print(f"✓ Successfully created run: {run_id}")
            created_runs.append((provider, run_id))
        except Exception as e:
            print(f"❌ ERROR: Failed to create run for provider '{provider}': {e}")
            sys.exit(1)
    
    print("\n=== Summary ===")
    print(f"Created {len(created_runs)} pipeline runs:")
    for provider, run_id in created_runs:
        print(f"  - {provider}: {run_id}")
    
    print("\n=== All pipeline runs created successfully ===")


if __name__ == "__main__":
    main()
