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

# ruff: noqa: PLC0415,UP007,UP035,UP006,E712
# SPDX-License-Identifier: Apache-2.0

from kfp import kubernetes, dsl
from kfp.kubernetes import add_node_selector_json, add_toleration_json


# Workbench Runtime Image: Pytorch with CUDA and Python 3.12 (UBI 9)
# The images for each release can be found in
# https://github.com/red-hat-data-services/rhoai-disconnected-install-helper/blob/main/rhoai-2.23.md
PYTORCH_CUDA_IMAGE = "quay.io/modh/odh-pipeline-runtime-pytorch-cuda-py312-ubi9@sha256:72ff2381e5cb24d6f549534cb74309ed30e92c1ca80214669adb78ad30c5ae12"


@dsl.component(
    base_image=PYTORCH_CUDA_IMAGE,
    packages_to_install=["llama-stack-client==0.3.5", "fire", "requests"],
)
def register_vector_store_and_files(
    git_repo: str,
    git_context: str,
    git_ref: str,
    vector_store_name: str,
    vector_store_provider_id: str,
    chunk_size_in_tokens: int,
    chunk_overlap_in_tokens: int,
    embedding_model: str,
    embedding_dimension: int,
    ranker: str,
    filenames: str,
):
    import io
    import os
    import requests
    from typing import List, Optional
    from llama_stack_client import LlamaStackClient
    from llama_stack_client.types.model import Model
    from llama_stack_client.types.vector_store import VectorStore

    # Validate parameters
    if not vector_store_name:
        raise ValueError("vector_store_name parameter must be set")
    if not vector_store_provider_id:
        raise ValueError("vector_store_provider_id parameter must be set")
    if not git_repo:
        raise ValueError("git_repo parameter must be set")
    if not git_context:
        raise ValueError("git_context parameter must be set")
    if not git_ref:
        raise ValueError("git_ref parameter must be set")
    if not filenames:
        raise ValueError("filenames parameter must be set")
    if not chunk_size_in_tokens:
        raise ValueError("chunk_size_in_tokens parameter must be set")
    if not chunk_overlap_in_tokens:
        raise ValueError("chunk_overlap_in_tokens parameter must be set")
    if not embedding_model:
        raise ValueError("embedding_model parameter must be set")
    if not embedding_dimension:
        raise ValueError("embedding_dimension parameter must be set")
    if not ranker:
        raise ValueError("ranker parameter must be set")

    # Read environment variables
    llama_stack_host: str = os.environ.get("LLAMA_STACK_HOST")
    llama_stack_port: int = os.environ.get("LLAMA_STACK_PORT")
    llama_stack_secure: bool = os.environ.get("LLAMA_STACK_SECURE", "").lower() in ("true", "1", "yes")

    print(f"LLAMA_STACK_HOST: {llama_stack_host}")
    print(f"LLAMA_STACK_PORT: {llama_stack_port}")
    print(f"LLAMA_STACK_SECURE: {llama_stack_secure}")
    print(f"LLAMA_STACK_BASE_URL: {'https' if llama_stack_secure else 'http'}://{llama_stack_host}:{llama_stack_port}")

    # Validate environment variables
    if not llama_stack_host:
        raise ValueError("LLAMA_STACK_HOST environment variable must be set")
    if not llama_stack_port:
        raise ValueError("LLAMA_STACK_PORT environment variable must be set")
    if os.environ.get("LLAMA_STACK_SECURE") is None:
        raise ValueError("LLAMA_STACK_SECURE environment variable must be set")

    # Initialize client
    client: LlamaStackClient = LlamaStackClient(base_url=f"{'https' if llama_stack_secure else 'http'}://{llama_stack_host}:{llama_stack_port}")

    # Base raw URL for the git repository
    base_raw_url: str = f"https://raw.githubusercontent.com/{git_repo}/{git_ref}/{git_context}"

    # Upload all files first and collect file_ids
    file_ids: List[str] = []
    for filename in filenames.split(","):
        source = f"{base_raw_url}/{filename.strip()}"
        print("Downloading and uploading document:", source)

        try:
            # Download the docs from URL
            response = requests.get(source)
            response.raise_for_status()  # Raise an exception for bad status codes

            file_content = io.BytesIO(response.content)
            file_basename = source.split("/")[-1]

            print(f"Uploading file: {file_basename} to {client.base_url}")

            # Upload file to storage
            file = client.files.create(
                file=(file_basename, file_content, "application/pdf"),
                purpose="assistants",
            )
            file_ids.append(file.id)
            print(f"Successfully uploaded {file_basename} (file_id: {file.id})")

        except Exception as e:
            print(f"ERROR: Failed to upload {filename.strip()}: {str(e)}")
            raise

    print(f"Successfully uploaded {len(file_ids)} files: {file_ids}")

    models: List[Model] = client.models.list()
    # TODO: In 0.4.2, the model.id is the identifier, but in 0.3.5, the model.id is the identifier
    matching_model: Optional[Model] = next((m for m in models if m.identifier == embedding_model), None)

    if not matching_model:
        available = [m.identifier for m in models]
        raise ValueError(
            f"Model '{embedding_model}' not found. Available: {available}"
        )

    # TODO: In 0.4.2, the model.api_model_type is the model type, but in 0.3.5, the model.api_model_type is the model type
    # model_type = (
    #     matching_model.custom_metadata.get("model_type")
    #     if matching_model.custom_metadata
    #     else None
    # )
    model_type = matching_model.api_model_type
    print(f"Matching model: {matching_model}")
    print(f"Model type: {model_type}")

    if model_type != "embedding":
        raise ValueError(
            f"Model '{embedding_model}' is not an embedding model (type={model_type})"
        )

    embedding_dimension = int(
        # TODO: In 0.4.2, the model.custom_metadata.get("embedding_dimension") is the embedding dimension, but in 0.3.5, the model.metadata.get("embedding_dimension") is the embedding dimension
        #float(matching_model.custom_metadata.get("embedding_dimension"))
        float(matching_model.metadata.get("embedding_dimension"))
    )

    # Warm up the embedding model
    client.embeddings.create(
        model=embedding_model,
        input="warmup",
    )

    # Create empty vector store first, before inserting files.
    # Purpose: Depending on the size and number of files, attempting to create the vector store
    # and add files in a single step may lead to timeouts.
    try:
        vector_store: VectorStore = client.vector_stores.create(
            name=vector_store_name,
            file_ids=[],
            chunking_strategy={
                "type": "static",
                "static": {
                    "max_chunk_size_tokens": chunk_size_in_tokens,
                    "chunk_overlap_tokens": chunk_overlap_in_tokens,
                },
            },
            extra_body={
                "embedding_model": embedding_model,
                "embedding_dimension": embedding_dimension,
                "provider_id": vector_store_provider_id,
            }
        )
        print(f"Successfully created vector store '{vector_store_name}' with ID: {vector_store.id}")
    except Exception as e:
        print(f"ERROR: Failed to create vector store '{vector_store_name}': {str(e)}")
        raise Exception(f"ERROR: Failed to create vector store '{vector_store_name}': {str(e)}")

    # Add files to vector store
    try:
        for file_id in file_ids:
            print(f"Adding file_id '{file_id}' to vector store '{vector_store_name}'")
            client.vector_stores.files.create(
                vector_store_id=vector_store.id,
                file_id=file_id,
            )
        vector_store: VectorStore = client.vector_stores.retrieve(vector_store.id)
        print(f"Vector store details: {vector_store}")
    except Exception as e:
        print(f"WARNING: Some files failed to be added to vector store: {str(e)}")


@dsl.pipeline()
def pipeline(
    git_repo: str = "alpha-hack-program/llama-stack-demo",
    git_context: str = "client/docs_pdf",
    git_ref: str = "next",
    filenames: str = "2025_61-FR_INT.html.pdf",
    vector_store_name: str = "rag-store",
    vector_store_provider_id: str = "milvus", # milvus, milvus-remote, pgvector
    embedding_model: str = "sentence-transformers/nomic-ai/nomic-embed-text-v1.5",
    embedding_dimension: int = 768,
    ranker: str = "default",
    chunk_size_in_tokens: int = 800,
    chunk_overlap_in_tokens: int = 400,
    use_gpu: bool = False,
):
    """
    Creates a vector store with embeddings from PDF files from a GitHub source.
    Args:
        git_repo: GitHub repository URL
        git_context: Context of the git repository
        git_ref: Reference of the git repository
        filenames: Comma-separated list of filenames to download and convert
        vector_store_name: Name of the vector store to store embeddings
        vector_store_provider_id: Provider ID of the vector store
        embedding_model: Model ID for embedding generation
        embedding_dimension: Dimension of the embedding model
        ranker: Ranker to use for scoring
        chunk_size_in_tokens: Maximum number of tokens per chunk
        chunk_overlap_in_tokens: Number of overlapping tokens between chunks
        use_gpu: Enable GPU usage for embedding generation
    :return:
        The vector store
    """

    with dsl.If(use_gpu == True):
        register_task = register_vector_store_and_files(
            vector_store_name=vector_store_name,
            vector_store_provider_id=vector_store_provider_id,
            chunk_size_in_tokens=chunk_size_in_tokens,
            chunk_overlap_in_tokens=chunk_overlap_in_tokens,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            ranker=ranker,
            git_repo=git_repo,
            git_context=git_context,
            git_ref=git_ref,
            filenames=filenames,
        )
        # Set the kubernetes config map to be used in the register_task task
        kubernetes.use_config_map_as_env(
            task=register_task,
            config_map_name='rag-pipeline-config',
            config_map_key_to_env={
                'LLAMA_STACK_HOST': 'LLAMA_STACK_HOST',
                'LLAMA_STACK_PORT': 'LLAMA_STACK_PORT',
                'LLAMA_STACK_SECURE': 'LLAMA_STACK_SECURE'
            })
        # Set the caching options, resource requests and limits for the register_task task
        register_task.set_caching_options(False)
        register_task.set_cpu_request("500m")
        register_task.set_cpu_limit("4")
        register_task.set_memory_request("2Gi")
        register_task.set_memory_limit("6Gi")
        register_task.set_accelerator_type("nvidia.com/gpu")
        register_task.set_accelerator_limit(1)
        add_toleration_json(
            register_task,
            [
                {
                    "effect": "NoSchedule",
                    "key": "nvidia.com/gpu",
                    "operator": "Exists",
                }
            ],
        )
        add_node_selector_json(register_task, {})

    with dsl.Else():
        register_task = register_vector_store_and_files(
            vector_store_name=vector_store_name,
            vector_store_provider_id=vector_store_provider_id,
            chunk_size_in_tokens=chunk_size_in_tokens,
            chunk_overlap_in_tokens=chunk_overlap_in_tokens,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            ranker=ranker,
            git_repo=git_repo,
            git_context=git_context,
            git_ref=git_ref,
            filenames=filenames,
        )
        # Set the kubernetes config map to be used in the register_task task
        kubernetes.use_config_map_as_env(
            task=register_task,
            config_map_name='rag-pipeline-config',
            config_map_key_to_env={
                'LLAMA_STACK_HOST': 'LLAMA_STACK_HOST',
                'LLAMA_STACK_PORT': 'LLAMA_STACK_PORT',
                'LLAMA_STACK_SECURE': 'LLAMA_STACK_SECURE'
            })
        # Set the caching options, resource requests and limits for the register_task task
        register_task.set_caching_options(False)
        register_task.set_cpu_request("500m")
        register_task.set_cpu_limit("4")
        register_task.set_memory_request("2Gi")
        register_task.set_memory_limit("6Gi")


# if __name__ == "__main__":
#     compiler.Compiler().compile(
#         vector_store_files_pipeline,
#         package_path=__file__.replace(".py", "_compiled.yaml"),
#     )

if __name__ == '__main__':
    from shared.kubeflow import compile_and_upsert_pipeline
    
    import os

    pipeline_package_path = __file__.replace('.py', '.yaml')

    # Pipeline name
    pipeline_name=os.path.basename(__file__).replace('.py', '').replace('_', '-')

    compile_and_upsert_pipeline(
        pipeline_func=pipeline, # type: ignore
        pipeline_package_path=pipeline_package_path,
        pipeline_name=pipeline_name
    )