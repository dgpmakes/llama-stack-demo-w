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

"""
RAGAS Evaluation Pipeline (v2)

Extracted from ragas_pipeline_version.yaml (KFP PipelineVersion).

This pipeline:
1. Loads a base dataset from a git repository
2. Discovers MCP tools and resolves vector store (parallel)
3. Generates a RAGAS dataset by querying a RAG system via Llama Stack
4. Runs RAGAS evaluation metrics on the generated dataset
"""

from kfp import kubernetes, dsl
from kfp.dsl import Input, Output, Dataset, Metrics

# Workbench Runtime Image: Pytorch with Python 3.12 (UBI 9)
PYTORCH_IMAGE = "quay.io/modh/odh-pipeline-runtime-pytorch-cuda-py312-ubi9@sha256:72ff2381e5cb24d6f549534cb74309ed30e92c1ca80214669adb78ad30c5ae12"

DEFAULT_METRICS = "answer_relevancy,faithfulness,context_precision,context_recall"


@dsl.component(
    base_image=PYTORCH_IMAGE,
    packages_to_install=["llama-stack-client==0.3.5", "httpx"],
)
def discover_mcp_tools(
    tools: str = "",
) -> str:
    """
    Discover MCP tools from Llama Stack server.

    Args:
        tools: MCP tools filter. Options:
            - "" (empty): Return empty list (no MCP tools)
            - "all": Discover and return all available MCP tools
            - "tool1,tool2": Comma-separated list of specific MCP tool names
              (e.g., "cluster-insights,compatibility-engine")

    Returns:
        JSON string containing list of MCP tool configurations
    """
    import json
    import os

    import httpx
    from llama_stack_client import LlamaStackClient

    if not tools or not tools.strip():
        print("[INFO] No MCP tools requested, returning empty list")
        return json.dumps([])

    tool_filter = tools.strip()

    llama_stack_host = os.environ.get("LLAMA_STACK_HOST")
    llama_stack_port = os.environ.get("LLAMA_STACK_PORT")
    llama_stack_secure = os.environ.get("LLAMA_STACK_SECURE", "").lower() in ("true", "1", "yes")

    print(f"LLAMA_STACK_HOST: {llama_stack_host}")
    print(f"LLAMA_STACK_PORT: {llama_stack_port}")
    print(f"LLAMA_STACK_SECURE: {llama_stack_secure}")

    if not llama_stack_host:
        raise ValueError("LLAMA_STACK_HOST environment variable must be set")
    if not llama_stack_port:
        raise ValueError("LLAMA_STACK_PORT environment variable must be set")

    base_url = f"{'https' if llama_stack_secure else 'http'}://{llama_stack_host}:{llama_stack_port}"
    print(f"LLAMA_STACK_BASE_URL: {base_url}")

    http_client = httpx.Client(verify=False, timeout=60)
    client = LlamaStackClient(base_url=base_url, http_client=http_client)

    print(f"\n[SEARCH] Discovering MCP tools (filter: '{tool_filter}')...")

    mcp_tools = []

    try:
        tool_groups = list(client.toolgroups.list())
    except Exception as e:
        raise ValueError(f"Failed to list tool groups from Llama Stack: {e}") from e

    print(f"   Found {len(tool_groups)} tool groups")

    requested_tools = []
    if tool_filter.lower() != "all":
        requested_tools = [t.strip().lower() for t in tool_filter.split(",") if t.strip()]

    found_tools = []

    for group in tool_groups:
        if not hasattr(group, "identifier"):
            continue

        identifier = group.identifier
        provider_id = getattr(group, "provider_id", None)

        if not identifier.startswith("mcp::"):
            continue
        if provider_id and provider_id != "model-context-protocol":
            continue

        tool_name = identifier.split("::", 1)[1] if "::" in identifier else identifier

        if tool_filter.lower() != "all":
            if tool_name.lower() not in requested_tools:
                continue

        mcp_endpoint = getattr(group, "mcp_endpoint", None)
        server_url = None
        if mcp_endpoint:
            if hasattr(mcp_endpoint, "uri"):
                server_url = mcp_endpoint.uri
            elif isinstance(mcp_endpoint, dict):
                server_url = mcp_endpoint.get("uri")

        if server_url:
            mcp_tools.append({
                "type": "mcp",
                "server_label": tool_name,
                "server_url": server_url,
            })
            found_tools.append(tool_name.lower())
            print(f"   [OK] Found MCP tool: {tool_name} ({server_url})")
        else:
            print(f"   [WARN] Skipping {tool_name} (no server URL found)")

    if requested_tools:
        missing_tools = [t for t in requested_tools if t not in found_tools]
        if missing_tools:
            available_mcp_tools = [
                g.identifier.split("::", 1)[1]
                for g in tool_groups
                if hasattr(g, "identifier") and g.identifier.startswith("mcp::")
            ]
            raise ValueError(
                f"Requested MCP tools not found: {missing_tools}. "
                f"Available MCP tools: {available_mcp_tools}"
            )

    print(f"\n[DONE] Discovered {len(mcp_tools)} MCP tool(s)")
    return json.dumps(mcp_tools, indent=2)


@dsl.component(
    base_image=PYTORCH_IMAGE,
    packages_to_install=["requests"],
)
def load_base_dataset_from_git(
    git_repo: str,
    git_context: str,
    git_ref: str,
    base_dataset_filename: str,
    base_dataset_output: Output[Dataset],
):
    """
    Load base dataset from a git repository.

    Args:
        git_repo: GitHub repository (e.g., 'alpha-hack-program/llama-stack-demo')
        git_context: Context path in the repository (e.g., 'materials/datasets')
        git_ref: Git reference/branch (e.g., 'main', 'next')
        base_dataset_filename: Name of the dataset file (e.g., 'base_dataset.json')
        base_dataset_output: Output artifact for the base dataset
    """
    import json

    import requests

    if not git_repo:
        raise ValueError("git_repo parameter must be set")
    if not git_context:
        raise ValueError("git_context parameter must be set")
    if not git_ref:
        raise ValueError("git_ref parameter must be set")
    if not base_dataset_filename:
        raise ValueError("base_dataset_filename parameter must be set")

    raw_url = f"https://raw.githubusercontent.com/{git_repo}/{git_ref}/{git_context}/{base_dataset_filename}"
    print(f"[LOAD] Loading base dataset from: {raw_url}")

    try:
        response = requests.get(raw_url)
        response.raise_for_status()
        base_dataset_content = response.text

        base_dataset = json.loads(base_dataset_content)
        print(f"[OK] Loaded {len(base_dataset)} questions from base dataset")

        with open(base_dataset_output.path, "w", encoding="utf-8") as f:
            f.write(base_dataset_content)

        print(f"[OK] Written base dataset to artifact: {base_dataset_output.path}")

    except Exception as e:
        print(f"[ERROR] Failed to load base dataset: {str(e)}")
        raise


@dsl.component(
    base_image=PYTORCH_IMAGE,
    packages_to_install=["llama-stack-client==0.3.5", "httpx"],
)
def resolve_vector_store(
    vector_store_name: str,
) -> str:
    """
    Resolve a vector store name to its ID.

    Finds the latest vector store with the given name.

    Args:
        vector_store_name: Name of the vector store to find

    Returns:
        The vector store ID
    """
    import os

    import httpx
    from llama_stack_client import LlamaStackClient

    llama_stack_host = os.environ.get("LLAMA_STACK_HOST")
    llama_stack_port = os.environ.get("LLAMA_STACK_PORT")
    llama_stack_secure = os.environ.get("LLAMA_STACK_SECURE", "").lower() in ("true", "1", "yes")

    print(f"LLAMA_STACK_HOST: {llama_stack_host}")
    print(f"LLAMA_STACK_PORT: {llama_stack_port}")
    print(f"LLAMA_STACK_SECURE: {llama_stack_secure}")

    if not llama_stack_host:
        raise ValueError("LLAMA_STACK_HOST environment variable must be set")
    if not llama_stack_port:
        raise ValueError("LLAMA_STACK_PORT environment variable must be set")
    if not vector_store_name:
        raise ValueError("vector_store_name parameter must be set")

    base_url = f"{'https' if llama_stack_secure else 'http'}://{llama_stack_host}:{llama_stack_port}"
    print(f"LLAMA_STACK_BASE_URL: {base_url}")

    http_client = httpx.Client(verify=False, timeout=60)
    client = LlamaStackClient(base_url=base_url, http_client=http_client)

    print(f"\n[SEARCH] Looking for vector store with name: '{vector_store_name}'...")

    try:
        all_vector_stores = list(client.vector_stores.list())
        print(f"   Found {len(all_vector_stores)} total vector store(s)")

        matching_stores = [
            vs for vs in all_vector_stores
            if hasattr(vs, "name") and vs.name == vector_store_name
        ]

        if not matching_stores:
            available_names = [vs.name for vs in all_vector_stores if hasattr(vs, "name")]
            raise ValueError(
                f"No vector store found with name '{vector_store_name}'. "
                f"Available vector stores: {available_names}"
            )

        if len(matching_stores) > 1:
            print(f"   Found {len(matching_stores)} vector stores with name '{vector_store_name}'")
            try:
                matching_stores.sort(
                    key=lambda vs: getattr(vs, "created_at", 0) or 0,
                    reverse=True,
                )
                print("   Using the most recently created one")
            except Exception:
                pass

        vector_store = matching_stores[0]
        vector_store_id = vector_store.id

        print(f"[OK] Found vector store: {vector_store_id}")
        print(f"   Name: {vector_store.name}")
        if hasattr(vector_store, "created_at") and vector_store.created_at:
            print(f"   Created: {vector_store.created_at}")

        return vector_store_id

    except ValueError:
        raise
    except Exception as e:
        print(f"[ERROR] Failed to resolve vector store: {str(e)}")
        raise ValueError(f"Failed to resolve vector store '{vector_store_name}': {str(e)}") from e


@dsl.component(
    base_image=PYTORCH_IMAGE,
    packages_to_install=["llama-stack-client==0.3.5", "httpx"],
)
def generate_ragas_dataset(
    base_dataset_input: Input[Dataset],
    model_id: str,
    vector_store_id: str,
    ragas_dataset_output: Output[Dataset],
    mcp_tools_json: str = "[]",
    instructions: str = "",
    timeout: int = 300,
    retrieval_mode: str = "vector", # vector, text, hybrid
    file_search_max_chunks: int = 5,
    file_search_score_threshold: float = 0.7,
    file_search_max_tokens_per_chunk: int = 512,
):
    """
    Generate RAGAS-compatible dataset with RAG answers and contexts.

    Uses Llama Stack's Responses API to query the RAG system and generate
    answers with retrieved contexts for each question in the base dataset.

    Args:
        base_dataset_input: Input artifact containing the base dataset JSON
        model_id: Model identifier to use for inference
        vector_store_id: Vector store ID containing documents
        ragas_dataset_output: Output artifact for the RAGAS dataset
        mcp_tools_json: JSON string of MCP tool configs (from discover_mcp_tools)
        instructions: System prompt/instructions for the model (optional)
        timeout: Timeout in seconds for requests
        retrieval_mode: Retrieval mode to use (vector, text, hybrid)
        file_search_max_chunks: Max number of chunks to retrieve (OpenAI: max_num_results)
        file_search_score_threshold: Min score for returned results (0–1)
        file_search_max_tokens_per_chunk: Max tokens per chunk in results
    """
    import json
    import os
    from typing import Any, Dict, List

    import httpx
    from llama_stack_client import LlamaStackClient

    llama_stack_host = os.environ.get("LLAMA_STACK_HOST")
    llama_stack_port = os.environ.get("LLAMA_STACK_PORT")
    llama_stack_secure = os.environ.get("LLAMA_STACK_SECURE", "").lower() in ("true", "1", "yes")

    print(f"LLAMA_STACK_HOST: {llama_stack_host}")
    print(f"LLAMA_STACK_PORT: {llama_stack_port}")
    print(f"LLAMA_STACK_SECURE: {llama_stack_secure}")

    if not llama_stack_host:
        raise ValueError("LLAMA_STACK_HOST environment variable must be set")
    if not llama_stack_port:
        raise ValueError("LLAMA_STACK_PORT environment variable must be set")

    base_url = f"{'https' if llama_stack_secure else 'http'}://{llama_stack_host}:{llama_stack_port}"
    print(f"LLAMA_STACK_BASE_URL: {base_url}")

    http_client = httpx.Client(verify=False, timeout=timeout)
    client = LlamaStackClient(base_url=base_url, http_client=http_client)

    print(f"\n[LOAD] Loading base dataset from artifact: {base_dataset_input.path}")
    with open(base_dataset_input.path, "r", encoding="utf-8") as f:
        base_dataset_json = f.read()
    dataset = json.loads(base_dataset_json)
    print(f"[OK] Loaded {len(dataset)} questions")

    print("\n[CONFIG] Configuring tools...")
    tools_list: List[Dict[str, Any]] = []

    if vector_store_id:
        file_search_tool: Dict[str, Any] = {
            "type": "file_search",
            "vector_store_ids": [vector_store_id],
            "file_search": {
                "retrieval_mode": retrieval_mode,
                "max_chunks": file_search_max_chunks,
                "score_threshold": file_search_score_threshold,
                "max_tokens_per_chunk": file_search_max_tokens_per_chunk,
            },
        }
        tools_list.append(file_search_tool)
        print(
            f"   [OK] Added file_search tool (vector store: {vector_store_id}, "
            f"max_chunks={file_search_max_chunks}, score_threshold={file_search_score_threshold}, "
            f"max_tokens_per_chunk={file_search_max_tokens_per_chunk})"
        )

    mcp_tools = json.loads(mcp_tools_json) if mcp_tools_json else []
    if mcp_tools:
        tools_list.extend(mcp_tools)
        print(f"   [OK] Added {len(mcp_tools)} MCP tool(s)")

    print(f"\n[INFO] Total tools configured: {len(tools_list)}")
    for tool in tools_list:
        tool_type = tool.get("type", "unknown")
        if tool_type == "mcp":
            print(f"   - MCP: {tool.get('server_label')} ({tool.get('server_url')})")
        elif tool_type == "file_search":
            fs_opts = tool.get("file_search") or {}
            print(f"   - File Search: {tool.get('vector_store_ids')} options={fs_opts}")
        else:
            print(f"   - {tool_type}")

    if instructions and instructions.strip():
        print(f"\n[INFO] System Instructions: {instructions[:100]}{'...' if len(instructions) > 100 else ''}")

    print(f"\n[PROCESS] Processing questions using Responses API...")
    print(f"Model: {model_id}")
    print(f"Vector Store: {vector_store_id}\n")

    ragas_dataset = []
    error_count = 0
    errors: List[str] = []

    for i, item in enumerate(dataset, 1):
        question_id = item.get("id", f"q_{i}")
        question = item["question"]
        ground_truth = item.get("ground_truth", "")

        print(f"[{i}/{len(dataset)}] Processing: {question_id}")

        try:
            request_config: Dict[str, Any] = {
                "model": model_id,
                "input": question,
                "tools": tools_list,
            }
            if instructions and instructions.strip():
                request_config["instructions"] = instructions.strip()
            if any(t.get("type") == "file_search" for t in tools_list):
                request_config["include"] = ["file_search_call.results"]

            response = client.responses.create(**request_config)
            answer = getattr(response, "output_text", str(response))

            contexts = []
            if hasattr(response, "output") and isinstance(response.output, list):
                for output_item in response.output:
                    if hasattr(output_item, "results") and isinstance(output_item.results, list):
                        for result in output_item.results:
                            if hasattr(result, "text") and result.text:
                                contexts.append(result.text)

            ragas_entry = {
                "id": question_id,
                "question": question,
                "answer": answer,
                "contexts": contexts if contexts else ["No context retrieved"],
                "ground_truth": ground_truth,
            }
            if "difficulty" in item:
                ragas_entry["difficulty"] = item["difficulty"]

            ragas_dataset.append(ragas_entry)
            print(f"  [OK] Answer generated ({len(contexts)} contexts retrieved)")

        except Exception as e:
            error_count += 1
            errors.append(f"{question_id}: {str(e)}")
            print(f"  [ERROR] Error processing {question_id}: {e}")
            ragas_dataset.append({
                "id": question_id,
                "question": question,
                "answer": f"ERROR: {str(e)}",
                "contexts": [],
                "ground_truth": ground_truth,
                "difficulty": item.get("difficulty", "unknown"),
                "error": True,
            })

    total_questions = len(dataset)
    success_count = total_questions - error_count

    print(f"\n[SUMMARY] RAGAS dataset generation summary:")
    print(f"   Total questions: {total_questions}")
    print(f"   Successful: {success_count}")
    print(f"   Failed: {error_count}")

    if error_count == total_questions:
        raise ValueError(
            f"All {total_questions} questions failed to process. "
            f"First error: {errors[0] if errors else 'Unknown'}"
        )
    if error_count > total_questions / 2:
        raise ValueError(
            f"Too many questions failed: {error_count}/{total_questions} ({error_count*100//total_questions}%). "
            f"Errors: {errors[:5]}{'...' if len(errors) > 5 else ''}"
        )
    if error_count > 0:
        print(f"   [WARN] Warning: {error_count} question(s) failed but continuing with {success_count} successful entries")

    print("\n[DONE] RAGAS dataset generation complete!")

    ragas_dataset_json = json.dumps(ragas_dataset, indent=2, ensure_ascii=False)
    with open(ragas_dataset_output.path, "w", encoding="utf-8") as f:
        f.write(ragas_dataset_json)

    print(f"[OK] Written RAGAS dataset to artifact: {ragas_dataset_output.path}")


@dsl.component(
    base_image=PYTORCH_IMAGE,
    packages_to_install=["llama-stack-client==0.3.5", "httpx"],
)
def run_ragas_evaluation(
    ragas_dataset_input: Input[Dataset],
    model_id: str,
    embedding_model_id: str,
    metrics: str,
    evaluation_output_metrics: Output[Metrics],
    evaluation_results_output: Output[Dataset],
    mode: str = "inline",
    batch_size: int = 0,
    timeout: int = 600,
    max_wait_seconds: int = 900,
    poll_interval: int = 5,
):
    """
    Run RAGAS evaluation on the generated dataset.

    Reads RAGAS dataset artifact, converts to evaluation format (skipping error entries),
    runs TrustyAI RAGAS evaluation via Llama Stack, and writes metrics and results to artifacts.

    Args:
        ragas_dataset_input: Input artifact containing the RAGAS dataset JSON
        model_id: LLM model identifier for scoring
        embedding_model_id: Embedding model identifier
        metrics: Comma-separated list of RAGAS metrics to compute
        evaluation_output_metrics: Kubeflow Metrics output for logging RAGAS metrics
        evaluation_results_output: Output artifact for the evaluation results JSON
        mode: Evaluation mode - "inline" or "remote"
        batch_size: Batch size for evaluation (0 = all at once)
        timeout: Timeout in seconds for requests
        max_wait_seconds: Maximum seconds to wait for evaluation job
        poll_interval: Seconds between job status checks
    """
    import json
    import os
    import time
    from datetime import datetime

    import httpx
    from llama_stack_client import LlamaStackClient

    llama_stack_host = os.environ.get("LLAMA_STACK_HOST")
    llama_stack_port = os.environ.get("LLAMA_STACK_PORT")
    llama_stack_secure = os.environ.get("LLAMA_STACK_SECURE", "").lower() in ("true", "1", "yes")

    print(f"LLAMA_STACK_HOST: {llama_stack_host}")
    print(f"LLAMA_STACK_PORT: {llama_stack_port}")
    print(f"LLAMA_STACK_SECURE: {llama_stack_secure}")

    if not llama_stack_host:
        raise ValueError("LLAMA_STACK_HOST environment variable must be set")
    if not llama_stack_port:
        raise ValueError("LLAMA_STACK_PORT environment variable must be set")

    base_url = f"{'https' if llama_stack_secure else 'http'}://{llama_stack_host}:{llama_stack_port}"
    print(f"LLAMA_STACK_BASE_URL: {base_url}")

    http_client = httpx.Client(verify=False, timeout=timeout)
    client = LlamaStackClient(base_url=base_url, http_client=http_client)

    print(f"\n[LOAD] Loading RAGAS dataset from artifact: {ragas_dataset_input.path}")
    with open(ragas_dataset_input.path, "r", encoding="utf-8") as f:
        ragas_dataset_json = f.read()
    ragas_dataset = json.loads(ragas_dataset_json)
    print(f"[OK] Loaded {len(ragas_dataset)} entries")

    metrics_list = [m.strip() for m in metrics.split(",") if m.strip()]
    print(f"[METRICS] Metrics to evaluate: {', '.join(metrics_list)}")

    print("\n[CONVERT] Converting to RAGAS evaluation format...")
    ragas_data = []
    skipped_count = 0
    for entry in ragas_dataset:
        if entry.get("error") or entry.get("answer", "").startswith("ERROR:"):
            skipped_count += 1
            print(f"   [SKIP] Skipping error entry: {entry.get('id', 'unknown')}")
            continue
        if not entry.get("contexts") or entry["contexts"] == ["No context retrieved"]:
            skipped_count += 1
            print(f"   [SKIP] Skipping entry with no contexts: {entry.get('id', 'unknown')}")
            continue

        ragas_entry = {
            "user_input": entry["question"],
            "response": entry["answer"],
            "retrieved_contexts": entry["contexts"],
        }
        if "ground_truth" in entry and entry["ground_truth"]:
            ragas_entry["reference"] = entry["ground_truth"]
        ragas_data.append(ragas_entry)

    if skipped_count > 0:
        print(f"   [WARN] Skipped {skipped_count} invalid entries")
    print(f"[OK] Converted {len(ragas_data)} valid entries for evaluation")

    if len(ragas_data) == 0:
        raise ValueError(
            f"No valid entries for RAGAS evaluation. "
            f"All {len(ragas_dataset)} entries were invalid (errors or missing contexts)."
        )

    provider_id = "trustyai_ragas_inline" if mode == "inline" else "trustyai_ragas_remote"

    if batch_size and batch_size > 0:
        batches = [ragas_data[i : i + batch_size] for i in range(0, len(ragas_data), batch_size)]
    else:
        batches = [ragas_data]

    print(f"[START] Starting evaluation in {len(batches)} batch(es) ({mode.upper()} mode)...")

    base_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    aggregated_scores = {metric: [] for metric in metrics_list}

    all_results = {
        "metrics": {},
        "individual_scores": {},
        "generations": [],
        "failures": [],
    }

    for batch_idx, batch in enumerate(batches):
        print(f"\n[BATCH] Processing Batch {batch_idx+1}/{len(batches)} (Size: {len(batch)})")

        # Unique per pipeline run (and per batch)
        dataset_id = f"ragas_dataset_{datetime.now().isoformat()}_{batch_idx}"
        benchmark_id = f"ragas_benchmark_{datetime.now().isoformat()}_{batch_idx}"

        try:
            print(f"[REGISTER] Registering dataset: {dataset_id}")
            client.datasets.register(
                dataset_id=dataset_id,
                purpose="eval/question-answer",
                source={"type": "rows", "rows": batch},
            )
            print(f"[OK] Dataset registered with {len(batch)} entries")

            print(f"[REGISTER] Registering benchmark: {benchmark_id}")
            client.benchmarks.register(
                benchmark_id=benchmark_id,
                dataset_id=dataset_id,
                scoring_functions=metrics_list,
                provider_id=provider_id,
            )
            print("[OK] Benchmark registered")

            eval_candidate = {
                "type": "model",
                "model": model_id,
                "sampling_params": {"strategy": {"type": "greedy"}, "max_tokens": 2048},
            }
            scoring_params = {}
            for metric in metrics_list:
                scoring_params[metric] = {"type": "basic", "aggregation_functions": ["average"]}
            benchmark_config = {
                "eval_candidate": eval_candidate,
                "scoring_params": scoring_params,
            }
            extra_body = {
                "provider_id": provider_id,
                "judge_model": model_id,
                "embedding_model": embedding_model_id,
            }

            print("[RUN] Running evaluation...")
            job = client.alpha.eval.run_eval(
                benchmark_id=benchmark_id,
                benchmark_config=benchmark_config,
                extra_body=extra_body,
            )
            print(f"[OK] Evaluation job started (Job ID: {job.job_id})")

            print("[WAIT] Waiting for evaluation results...")
            batch_result = None
            waited = 0

            while waited < max_wait_seconds:
                time.sleep(poll_interval)
                waited += poll_interval

                try:
                    job_status = client.alpha.eval.jobs.status(
                        benchmark_id=benchmark_id,
                        job_id=job.job_id,
                    )
                    status_value = getattr(job_status, "status", None)
                    print(f"   Job status: {status_value} ({waited}s)")

                    if status_value in ("completed", "success", "succeeded"):
                        batch_result = client.alpha.eval.jobs.retrieve(
                            benchmark_id=benchmark_id,
                            job_id=job.job_id,
                        )
                        print(f"   [OK] Results received after {waited}s")
                        break
                    elif status_value in ("failed", "error"):
                        error_msg = getattr(job_status, "error", None)
                        error_detail = getattr(job_status, "error_message", None)
                        error_info = getattr(job_status, "message", None)
                        print(f"   [DEBUG] Full job_status: {job_status}")
                        details = error_msg or error_detail or error_info or "No error details available"
                        raise RuntimeError(
                            f"Evaluation job {job.job_id} failed with status: {status_value}. "
                            f"Details: {details}"
                        )
                except RuntimeError:
                    raise
                except Exception as e:
                    print(f"   Waiting... ({waited}s) - {e}")

            if batch_result is None:
                raise RuntimeError(f"Evaluation job {job.job_id} did not complete within {max_wait_seconds}s")

            scores = getattr(batch_result, "scores", None)
            if scores:
                scores_dict = scores if isinstance(scores, dict) else dict(scores)
                for metric, score_data in scores_dict.items():
                    score_val = 0.0
                    if hasattr(score_data, "aggregated_results"):
                        agg_results = score_data.aggregated_results
                        if isinstance(agg_results, dict):
                            score_val = agg_results.get("average", agg_results.get(metric, 0.0))
                        else:
                            score_val = float(agg_results) if agg_results else 0.0
                    elif isinstance(score_data, dict) and "aggregated_results" in score_data:
                        score_val = score_data["aggregated_results"].get(
                            "average", score_data["aggregated_results"].get(metric, 0.0)
                        )
                    elif isinstance(score_data, (int, float)):
                        score_val = float(score_data)

                    aggregated_scores[metric].append((score_val, len(batch)))

                    score_rows = None
                    if hasattr(score_data, "score_rows"):
                        score_rows = score_data.score_rows
                    elif isinstance(score_data, dict) and "score_rows" in score_data:
                        score_rows = score_data["score_rows"]
                    if score_rows:
                        if metric not in all_results["individual_scores"]:
                            all_results["individual_scores"][metric] = []
                        for row in score_rows:
                            row_score = (
                                row.get("score", 0.0) if isinstance(row, dict) else getattr(row, "score", 0.0)
                            )
                            all_results["individual_scores"][metric].append(row_score)

            generations = getattr(batch_result, "generations", None)
            if generations:
                all_results["generations"].extend(list(generations))

        except Exception as e:
            print(f"[ERROR] Batch {batch_idx+1} failed: {e}")
            all_results["failures"].append({"batch_index": batch_idx + 1, "error": str(e)})

    total_batches = len(batches)
    failed_batches = len(all_results["failures"])

    if failed_batches == total_batches:
        raise ValueError(
            f"All {total_batches} evaluation batch(es) failed. "
            f"First error: {all_results['failures'][0]['error'] if all_results['failures'] else 'Unknown'}"
        )
    if failed_batches > total_batches / 2:
        raise ValueError(
            f"Too many evaluation batches failed: {failed_batches}/{total_batches}. "
            f"Errors: {[f['error'] for f in all_results['failures'][:3]]}{'...' if failed_batches > 3 else ''}"
        )

    print("\n[SUM] Aggregating results...")
    final_metrics = {}
    for metric, values in aggregated_scores.items():
        if not values:
            continue
        total_score = sum(v[0] * v[1] for v in values)
        total_count = sum(v[1] for v in values)
        if total_count > 0:
            final_metrics[metric] = total_score / total_count
        else:
            final_metrics[metric] = 0.0

    print("\n[LOG] Logging metrics to Kubeflow Pipelines...")
    for metric_name, metric_value in final_metrics.items():
        evaluation_output_metrics.log_metric(metric_name, metric_value)
        print(f"   [OK] Logged {metric_name}: {metric_value:.4f}")

    evaluation_output_metrics.log_metric("dataset_size", float(len(ragas_dataset)))
    evaluation_output_metrics.log_metric("failure_count", float(len(all_results["failures"])))
    print(f"   [OK] Logged dataset_size: {len(ragas_dataset)}")
    print(f"   [OK] Logged failure_count: {len(all_results['failures'])}")

    formatted_results = {
        "benchmark_id": f"ragas_pipeline_run_{base_timestamp}",
        "timestamp": datetime.now().isoformat(),
        "metrics": final_metrics,
        "individual_scores": all_results["individual_scores"],
        "generations": all_results["generations"],
        "failures": all_results["failures"],
        "dataset_size": len(ragas_dataset),
        "mode": mode,
        "model_id": model_id,
        "embedding_model_id": embedding_model_id,
    }

    print("\n" + "=" * 70)
    print("[RESULTS] RAGAS EVALUATION RESULTS")
    print("=" * 70)
    print(f"Benchmark ID: {formatted_results['benchmark_id']}")
    print(f"Timestamp:    {formatted_results['timestamp']}")
    print(f"Dataset Size: {formatted_results['dataset_size']} entries")
    print(f"Mode:         {formatted_results['mode']}")
    print("=" * 70)
    if formatted_results.get("failures"):
        print("\n[FAILURES] Failures:")
        for failure in formatted_results["failures"]:
            print(f"  Batch {failure.get('batch_index')}: {failure.get('error')}")
    if formatted_results["metrics"]:
        print("\n[METRICS] Aggregated Metrics:")
        print("-" * 70)
        for metric, score in formatted_results["metrics"].items():
            status = "[PASS]" if score > 0.8 else "[WARN]" if score > 0.6 else "[FAIL]"
            print(f"  {status} {metric:25s}: {score:.4f}")
        print("-" * 70)
    print("=" * 70)

    results_json = json.dumps(formatted_results, indent=2, ensure_ascii=False)
    with open(evaluation_results_output.path, "w", encoding="utf-8") as f:
        f.write(results_json)
    print(f"\n[OK] Written evaluation results to artifact: {evaluation_results_output.path}")


@dsl.pipeline(
    name="ragas-evaluation-pipeline",
    description="Pipeline to generate and evaluate RAGAS datasets from a git repository",
)
def pipeline(
    git_repo: str = "alpha-hack-program/llama-stack-demo",
    git_context: str = "materials/datasets",
    git_ref: str = "next",
    base_dataset_filename: str = "base_dataset_small.json",
    model_id: str = "llama-3-1-8b-w4a16/llama-3-1-8b-w4a16",
    embedding_model_id: str = "sentence-transformers/nomic-ai/nomic-embed-text-v1.5",
    vector_store_name: str = "",
    tools: str = "all",
    instructions: str = "",
    retrieval_mode: str = "vector", # vector, text, hybrid
    file_search_max_chunks: int = 5,
    file_search_score_threshold: float = 0.7,
    file_search_max_tokens_per_chunk: int = 512,
    metrics: str = DEFAULT_METRICS,
    mode: str = "inline",
    batch_size: int = 0,
    timeout: int = 600,
    max_wait_seconds: int = 900,
    poll_interval: int = 5,
):
    """
    RAGAS Evaluation Pipeline (v2).

    DAG:
    - load_base_dataset_from_git, resolve_vector_store, discover_mcp_tools (parallel)
    - generate_ragas_dataset (after the three above)
    - run_ragas_evaluation (after generate_ragas_dataset)
    """
    load_task = load_base_dataset_from_git(
        git_repo=git_repo,
        git_context=git_context,
        git_ref=git_ref,
        base_dataset_filename=base_dataset_filename,
    )
    load_task.set_caching_options(True)
    load_task.set_cpu_request("100m")
    load_task.set_cpu_limit("1")
    load_task.set_memory_request("256Mi")
    load_task.set_memory_limit("512Mi")

    resolve_task = resolve_vector_store(vector_store_name=vector_store_name)
    kubernetes.use_config_map_as_env(
        task=resolve_task,
        config_map_name="rag-pipeline-config",
        config_map_key_to_env={
            "LLAMA_STACK_HOST": "LLAMA_STACK_HOST",
            "LLAMA_STACK_PORT": "LLAMA_STACK_PORT",
            "LLAMA_STACK_SECURE": "LLAMA_STACK_SECURE",
        },
    )
    resolve_task.set_caching_options(False)
    resolve_task.set_cpu_request("100m")
    resolve_task.set_cpu_limit("1")
    resolve_task.set_memory_request("256Mi")
    resolve_task.set_memory_limit("512Mi")

    discover_task = discover_mcp_tools(tools=tools)
    kubernetes.use_config_map_as_env(
        task=discover_task,
        config_map_name="rag-pipeline-config",
        config_map_key_to_env={
            "LLAMA_STACK_HOST": "LLAMA_STACK_HOST",
            "LLAMA_STACK_PORT": "LLAMA_STACK_PORT",
            "LLAMA_STACK_SECURE": "LLAMA_STACK_SECURE",
        },
    )
    discover_task.set_caching_options(False)
    discover_task.set_cpu_request("100m")
    discover_task.set_cpu_limit("1")
    discover_task.set_memory_request("256Mi")
    discover_task.set_memory_limit("512Mi")

    generate_task = generate_ragas_dataset(
        base_dataset_input=load_task.output,
        model_id=model_id,
        vector_store_id=resolve_task.output,
        mcp_tools_json=discover_task.output,
        instructions=instructions,
        timeout=timeout,
        retrieval_mode=retrieval_mode,
        file_search_max_chunks=file_search_max_chunks,
        file_search_score_threshold=file_search_score_threshold,
        file_search_max_tokens_per_chunk=file_search_max_tokens_per_chunk,
    )
    generate_task.after(load_task, resolve_task, discover_task)
    kubernetes.use_config_map_as_env(
        task=generate_task,
        config_map_name="rag-pipeline-config",
        config_map_key_to_env={
            "LLAMA_STACK_HOST": "LLAMA_STACK_HOST",
            "LLAMA_STACK_PORT": "LLAMA_STACK_PORT",
            "LLAMA_STACK_SECURE": "LLAMA_STACK_SECURE",
        },
    )
    generate_task.set_caching_options(False)
    generate_task.set_cpu_request("500m")
    generate_task.set_cpu_limit("4")
    generate_task.set_memory_request("2Gi")
    generate_task.set_memory_limit("6Gi")

    evaluate_task = run_ragas_evaluation(
        ragas_dataset_input=generate_task.outputs["ragas_dataset_output"],
        model_id=model_id,
        embedding_model_id=embedding_model_id,
        metrics=metrics,
        mode=mode,
        batch_size=batch_size,
        timeout=timeout,
        max_wait_seconds=max_wait_seconds,
        poll_interval=poll_interval,
    )
    evaluate_task.after(generate_task)
    kubernetes.use_config_map_as_env(
        task=evaluate_task,
        config_map_name="rag-pipeline-config",
        config_map_key_to_env={
            "LLAMA_STACK_HOST": "LLAMA_STACK_HOST",
            "LLAMA_STACK_PORT": "LLAMA_STACK_PORT",
            "LLAMA_STACK_SECURE": "LLAMA_STACK_SECURE",
        },
    )
    evaluate_task.set_caching_options(False)
    evaluate_task.set_cpu_request("500m")
    evaluate_task.set_cpu_limit("4")
    evaluate_task.set_memory_request("2Gi")
    evaluate_task.set_memory_limit("6Gi")


if __name__ == "__main__":
    import os

    from shared.kubeflow import compile_and_upsert_pipeline

    pipeline_package_path = __file__.replace(".py", ".yaml")
    pipeline_name = os.path.basename(__file__).replace(".py", "").replace("_", "-")

    compile_and_upsert_pipeline(
        pipeline_func=pipeline,
        pipeline_package_path=pipeline_package_path,
        pipeline_name=pipeline_name,
    )
