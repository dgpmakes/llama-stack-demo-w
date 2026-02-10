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
RAGAS Pipeline Scheduler

Creates KFP recurring runs (schedules) for the ragas-pipeline, one per
(vector store name, retrieval mode). For vector store provider id milvus or
milvus-remote, only "vector" retrieval mode is used; for other providers,
all configured retrieval_modes are used.

Reference configuration: helm/values.yaml section pipelines.scheduler.

Usage:
    python ragas_pipeline_scheduler.py [OPTIONS]

Options can be passed as command-line arguments or environment variables.
Command-line arguments take precedence over environment variables.

Example:
    python ragas_pipeline_scheduler.py --interval-minutes 60

    # Or using environment variables (e.g. from Helm):
    RAGAS_SCHEDULER_INTERVAL_MINUTES=60 python ragas_pipeline_scheduler.py
"""

import argparse
import os
import sys

import kfp

from shared.kubeflow import (
    get_token,
    get_route_host,
    get_pipeline_id_by_name,
    get_or_create_experiment,
    get_latest_pipeline_version_id,
    list_recurring_runs_for_experiment,
    delete_recurring_run,
    create_recurring_run,
)

# Vector store provider IDs that support only "vector" retrieval mode
VECTOR_ONLY_PROVIDERS = ("milvus", "milvus-remote")


def parse_args():
    """Parse command-line arguments with environment variable defaults."""
    parser = argparse.ArgumentParser(
        description="Schedule recurring RAGAS pipeline runs per vector store and retrieval mode",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Pipeline identification
    parser.add_argument(
        "--pipeline-name",
        default=os.environ.get("RAGAS_SCHEDULER_PIPELINE_NAME", "ragas-pipeline"),
        help="Name of the pipeline to schedule",
    )
    parser.add_argument(
        "--experiment-name",
        default=os.environ.get("RAGAS_SCHEDULER_EXPERIMENT_NAME", "ragas-scoring-experiment"),
        help="Name of the experiment",
    )
    parser.add_argument(
        "--vector-store-provider-ids",
        default=os.environ.get(
            "RAGAS_SCHEDULER_VECTOR_STORE_PROVIDER_IDS",
            "milvus, milvus-remote, pgvector",
        ),
        help="Comma-separated list of vector store provider IDs",
    )
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=int(os.environ.get("RAGAS_SCHEDULER_INTERVAL_MINUTES", "60")),
        help="Interval between runs in minutes (used as interval_second = interval_minutes * 60)",
    )

    # Connection parameters
    parser.add_argument(
        "--token",
        default=os.environ.get("KFP_TOKEN"),
        help="KFP authentication token (default: read from service account)",
    )
    parser.add_argument(
        "--kfp-endpoint",
        default=os.environ.get("KFP_ENDPOINT"),
        help="KFP API endpoint (default: auto-discover from route)",
    )

    # Pipeline parameters (aligned with helm/values.yaml pipelines.scheduler.params)
    parser.add_argument(
        "--git-repo",
        default=os.environ.get("RAGAS_SCHEDULER_GIT_REPO", "alpha-hack-program/llama-stack-demo"),
        help="Git repository for pipeline data",
    )
    parser.add_argument(
        "--git-context",
        default=os.environ.get("RAGAS_SCHEDULER_GIT_CONTEXT", "client/docs_pdf"),
        help="Git context path",
    )
    parser.add_argument(
        "--git-ref",
        default=os.environ.get("RAGAS_SCHEDULER_GIT_REF", "next"),
        help="Git reference (branch/tag)",
    )
    parser.add_argument(
        "--base-dataset-filename",
        default=os.environ.get("RAGAS_SCHEDULER_BASE_DATASET_FILENAME", "base_dataset_small.json"),
        help="Base dataset filename in git context",
    )
    parser.add_argument(
        "--tools",
        default=os.environ.get("RAGAS_SCHEDULER_TOOLS", "all"),
        help="MCP tools filter (e.g. 'all' or comma-separated tool names)",
    )
    parser.add_argument(
        "--instructions",
        default=os.environ.get(
            "RAGAS_SCHEDULER_INSTRUCTIONS",
            "You are a helpful assistant that can answer questions about the documents in the vector store.",
        ),
        help="Instructions for the model",
    )
    parser.add_argument(
        "--retrieval-modes",
        default=os.environ.get("RAGAS_SCHEDULER_RETRIEVAL_MODES", "vector, hybrid"),
        help="Comma-separated retrieval modes (vector, text, hybrid). Milvus/milvus-remote use only 'vector'.",
    )
    parser.add_argument(
        "--file-search-max-chunks",
        type=int,
        default=int(os.environ.get("RAGAS_SCHEDULER_FILE_SEARCH_MAX_CHUNKS", "5")),
        help="File search max chunks",
    )
    parser.add_argument(
        "--file-search-score-threshold",
        type=float,
        default=float(os.environ.get("RAGAS_SCHEDULER_FILE_SEARCH_SCORE_THRESHOLD", "0.7")),
        help="File search score threshold",
    )
    parser.add_argument(
        "--file-search-max-tokens-per-chunk",
        type=int,
        default=int(os.environ.get("RAGAS_SCHEDULER_FILE_SEARCH_MAX_TOKENS_PER_CHUNK", "0")),
        help="File search max tokens per chunk",
    )
    parser.add_argument(
        "--metrics",
        default=os.environ.get(
            "RAGAS_SCHEDULER_METRICS",
            "answer_relevancy,faithfulness,context_precision,context_recall",
        ),
        help="Comma-separated RAGAS metrics",
    )
    parser.add_argument(
        "--mode",
        default=os.environ.get("RAGAS_SCHEDULER_MODE", "inline"),
        help="RAGAS evaluation mode (e.g. inline)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("RAGAS_SCHEDULER_BATCH_SIZE", "0")),
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("RAGAS_SCHEDULER_TIMEOUT", "600")),
        help="Timeout in seconds",
    )
    parser.add_argument(
        "--max-wait-seconds",
        type=int,
        default=int(os.environ.get("RAGAS_SCHEDULER_MAX_WAIT_SECONDS", "900")),
        help="Max wait seconds",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=int(os.environ.get("RAGAS_SCHEDULER_POLL_INTERVAL", "5")),
        help="Poll interval in seconds",
    )
    parser.add_argument(
        "--vector-store-base-name",
        default=os.environ.get("RAGAS_SCHEDULER_VECTOR_STORE_BASE_NAME", "rag-store"),
        help="Base name for vector stores (provider ID will be appended)",
    )
    parser.add_argument(
        "--model-id",
        default=os.environ.get("RAGAS_SCHEDULER_MODEL_ID", "llama-3-1-8b-w4a16/llama-3-1-8b-w4a16"),
        help="LLM model ID for RAGAS (e.g. from ragDefaults.modelId)",
    )

    return parser.parse_args()


def get_retrieval_modes_for_provider(provider_id: str, configured_modes: str) -> list[str]:
    """Return retrieval modes for this provider. Milvus/milvus-remote only support 'vector'."""
    provider = (provider_id or "").strip().lower()
    if provider in VECTOR_ONLY_PROVIDERS:
        return ["vector"]
    return [m.strip() for m in configured_modes.split(",") if m.strip()]


def main():
    """Create recurring runs for ragas-pipeline per vector store and retrieval mode, then exit."""
    args = parse_args()

    print("=== RAGAS Pipeline Scheduler ===")
    print(f"Pipeline name: {args.pipeline_name}")
    print(f"Experiment name: {args.experiment_name}")
    print(f"Vector store provider IDs: {args.vector_store_provider_ids}")
    print(f"Interval: every {args.interval_minutes} minutes")
    print(f"Retrieval modes (for non-Milvus): {args.retrieval_modes}")

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

    if not kfp_endpoint.startswith("http"):
        kfp_endpoint = f"https://{kfp_endpoint}"

    print(f"KFP endpoint: {kfp_endpoint}")

    try:
        client = kfp.Client(host=kfp_endpoint, existing_token=token)
    except Exception as e:
        print(f"ERROR: Failed to create KFP client: {e}")
        sys.exit(1)

    pipeline_id = get_pipeline_id_by_name(client, args.pipeline_name)
    if not pipeline_id:
        print(f"ERROR: Pipeline '{args.pipeline_name}' not found")
        sys.exit(1)
    print(f"Pipeline ID: {pipeline_id}")

    experiment_id = get_or_create_experiment(client, args.experiment_name)
    print(f"Experiment ID: {experiment_id}")

    version_id = get_latest_pipeline_version_id(client, pipeline_id)
    if not version_id:
        print(f"ERROR: No pipeline versions found for pipeline_id: {pipeline_id}")
        sys.exit(1)
    print(f"Pipeline version ID: {version_id}")

    providers = [p.strip() for p in args.vector_store_provider_ids.split(",") if p.strip()]
    if not providers:
        print("ERROR: No vector store provider IDs specified")
        sys.exit(1)

    interval_second = args.interval_minutes * 60

    # Build (provider, retrieval_mode) pairs; milvus/milvus-remote → vector only
    schedule_specs = []
    for provider in providers:
        modes = get_retrieval_modes_for_provider(provider, args.retrieval_modes)
        for mode in modes:
            schedule_specs.append((provider, mode))

    print(f"\nWill create {len(schedule_specs)} recurring run(s): {schedule_specs}")

    # KFP has no update for recurring runs; delete any existing ones we're about to recreate
    job_names_to_create = {f"ragas-{p}-{m}" for p, m in schedule_specs}
    existing = list_recurring_runs_for_experiment(client, experiment_id)
    for run in existing:
        display_name = getattr(run, "display_name", None) or getattr(run, "job_name", None)
        if display_name and display_name in job_names_to_create:
            rid = getattr(run, "recurring_run_id", None)
            if rid:
                print(f"Deleting existing recurring run: {display_name} ({rid})")
                try:
                    delete_recurring_run(client, rid)
                except Exception as e:
                    print(f"  ⚠ Failed to delete {display_name}: {e}")

    created = []
    for provider, retrieval_mode in schedule_specs:
        vector_store_name = f"{args.vector_store_base_name}-{provider}"
        job_name = f"ragas-{provider}-{retrieval_mode}"

        params = {
            "git_repo": args.git_repo,
            "git_context": args.git_context,
            "git_ref": args.git_ref,
            "base_dataset_filename": args.base_dataset_filename,
            "model_id": args.model_id,
            "vector_store_name": vector_store_name,
            "tools": args.tools,
            "instructions": args.instructions,
            "retrieval_mode": retrieval_mode,
            "file_search_max_chunks": args.file_search_max_chunks,
            "file_search_score_threshold": args.file_search_score_threshold,
            "file_search_max_tokens_per_chunk": args.file_search_max_tokens_per_chunk,
            "metrics": args.metrics,
            "mode": args.mode,
            "batch_size": args.batch_size,
            "timeout": args.timeout,
            "max_wait_seconds": args.max_wait_seconds,
            "poll_interval": args.poll_interval,
        }

        print(f"\n=== Creating recurring run: {job_name} (every {args.interval_minutes} min) ===")
        print(f"  vector_store_name={vector_store_name}, retrieval_mode={retrieval_mode}")

        try:
            recurring_run_id = create_recurring_run(
                client,
                pipeline_id,
                experiment_id,
                job_name,
                params,
                interval_second=interval_second,
                enabled=True,
            )
            print(f"  ✓ Recurring run ID: {recurring_run_id}")
            created.append((job_name, recurring_run_id))
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            sys.exit(1)

    print("\n=== Summary ===")
    print(f"Created {len(created)} recurring run(s):")
    for job_name, rid in created:
        print(f"  - {job_name}: {rid}")
    print("\n=== Scheduler finished successfully ===")


if __name__ == "__main__":
    main()
