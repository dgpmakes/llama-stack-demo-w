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
    retrieval_mode: str = "vector",  # vector, text, hybrid
    file_search_max_chunks: int = 5,
    file_search_score_threshold: float = 0.7,
    file_search_max_tokens_per_chunk: int = 512,
    ranker: str = "default",
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
        ranker: Ranker to use for scoring retrieved chunks (e.g. "default")
    """
    import json
    import os
    from typing import Any, Dict, List

    import httpx
    from llama_stack_client import LlamaStackClient

    def _serialize_for_json(val: Any) -> Any:
        """Convert a value to something JSON-serializable."""
        if val is None or isinstance(val, (bool, int, float, str)):
            return val
        if isinstance(val, (dict, list)):
            return val
        if hasattr(val, "__dict__"):
            return {k: _serialize_for_json(v) for k, v in val.__dict__.items()}
        return str(val)

    def _extract_tool_calls(response: Any) -> List[Dict[str, Any]]:
        """Extract tool calls from Llama Stack response (output list, then turns or tool_calls). Returns list of {tool_name, arguments, response}."""
        out: List[Dict[str, Any]] = []
        if hasattr(response, "output") and isinstance(response.output, list):
            for item in response.output:
                type_ = getattr(item, "type", None)
                if type_ == "file_search_call":
                    queries = getattr(item, "queries", None) or []
                    results = getattr(item, "results", None) or []
                    result_texts = [getattr(r, "text", "") or "" for r in results]
                    out.append({
                        "tool_name": "file_search",
                        "arguments": {"queries": list(queries)},
                        "response": result_texts,
                    })
                elif type_ and "mcp" in str(type_).lower() and type_ != "mcp_list_tools":
                    name = getattr(item, "tool_name", None) or getattr(item, "name", None) or getattr(item, "server_label", None) or "mcp"
                    args = getattr(item, "arguments", None) or getattr(item, "args", None) or {}
                    # Llama Stack MCP call uses "output" for the tool result (not "response" or "result")
                    resp = getattr(item, "output", None) or getattr(item, "response", None) or getattr(item, "result", None)
                    if not isinstance(args, dict):
                        try:
                            args = json.loads(args) if isinstance(args, str) else {}
                        except Exception:
                            args = {} if args is None else {"raw": str(args)}
                    out.append({"tool_name": str(name), "arguments": args, "response": _serialize_for_json(resp)})
            if out:
                return out
        if hasattr(response, "turns") and response.turns:
            for turn in response.turns:
                if hasattr(turn, "tool_calls") and turn.tool_calls:
                    for tc in turn.tool_calls:
                        name = getattr(tc, "tool_name", None) or (tc.__dict__.get("tool_name") if hasattr(tc, "__dict__") else "unknown")
                        args = getattr(tc, "arguments", None) or (tc.__dict__.get("arguments") if hasattr(tc, "__dict__") else {})
                        resp = getattr(tc, "response", None) or (tc.__dict__.get("response") if hasattr(tc, "__dict__") else None)
                        if isinstance(args, dict):
                            pass
                        else:
                            try:
                                args = json.loads(args) if isinstance(args, str) else (args or {})
                            except Exception:
                                args = {} if args is None else {"raw": str(args)}
                        out.append({"tool_name": name, "arguments": args, "response": _serialize_for_json(resp)})
                if hasattr(turn, "steps") and turn.steps:
                    for step in turn.steps:
                        name = getattr(step, "tool_name", None) or (getattr(step, "__dict__", {}).get("tool_name") or "unknown")
                        args = getattr(step, "tool_args", None) or (getattr(step, "__dict__", {}).get("tool_args") or {})
                        resp = getattr(step, "tool_response", None) or (getattr(step, "__dict__", {}).get("tool_response"))
                        if not isinstance(args, dict):
                            try:
                                args = json.loads(args) if isinstance(args, str) else {}
                            except Exception:
                                args = {} if args is None else {"raw": str(args)}
                        out.append({"tool_name": name, "arguments": args, "response": _serialize_for_json(resp)})
        elif hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                d = tc.__dict__ if hasattr(tc, "__dict__") else {}
                name = d.get("tool_name", "unknown")
                args = d.get("arguments", {})
                if not isinstance(args, dict):
                    try:
                        args = json.loads(args) if isinstance(args, str) else {}
                    except Exception:
                        args = {} if args is None else {"raw": str(args)}
                out.append({"tool_name": name, "arguments": args, "response": _serialize_for_json(d.get("response"))})
        return out

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
                "ranker": ranker,
            },
        }
        tools_list.append(file_search_tool)
        print(
            f"   [OK] Added file_search tool (vector store: {vector_store_id}, "
            f"retrieval_mode={retrieval_mode}, ranker={ranker}, "
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

            tool_calls = _extract_tool_calls(response)
            ragas_entry = {
                "id": question_id,
                "question": question,
                "answer": answer,
                "contexts": contexts if contexts else [],
                "ground_truth": ground_truth,
            }
            if tool_calls:
                ragas_entry["tool_calls"] = tool_calls
                # Append non-file_search tool call responses as contexts for RAGAS
                for tc in tool_calls:
                    if tc.get("tool_name") != "file_search":
                        resp = tc.get("response")
                        if resp is not None:
                            ctx = resp if isinstance(resp, str) else json.dumps(resp, ensure_ascii=False)
                            ragas_entry["contexts"].append(ctx)
            print(f"  [OK] Answer generated ({len(ragas_entry['contexts'])} contexts, {len(tool_calls)} tool call(s))")
            if "difficulty" in item:
                ragas_entry["difficulty"] = item["difficulty"]
            if item.get("expected_tool"):
                ragas_entry["expected_tool"] = item["expected_tool"]
            if item.get("expected_tool_parameters"):
                ragas_entry["expected_tool_parameters"] = item["expected_tool_parameters"]

            ragas_dataset.append(ragas_entry)

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
    packages_to_install=[
        "llama-stack-client==0.3.5",
        "httpx",
        "ragas",
        "langchain-openai",
    ],
)
def run_ragas_evaluation(
    ragas_dataset_input: Input[Dataset],
    model_id: str,
    embedding_model_id: str,
    metrics: str,
    evaluation_output_metrics: Output[Metrics],
    evaluation_results_output: Output[Dataset],
    batch_size: int = 0,
    show_progress: bool = True,
    timeout: int = 600,
):
    """
    Run RAGAS evaluation on the generated dataset using the RAGAS library directly.

    Same evaluation logic as run_ragas_evaluation_direct in ragas_dataset_eval_direct.py:
    converts dataset with _context_from_json_like, uses RAGAS evaluate() with Llama Stack
    LLM (ChatOpenAI at /v1) and Llama Stack embeddings.

    Args:
        ragas_dataset_input: Input artifact containing the RAGAS dataset JSON
        model_id: LLM model identifier for scoring (judge)
        embedding_model_id: Embedding model identifier
        metrics: Comma-separated list of RAGAS metrics to compute
        evaluation_output_metrics: Kubeflow Metrics output for logging RAGAS metrics
        evaluation_results_output: Output artifact for the evaluation results JSON
        batch_size: RAGAS evaluate() batch size (0 = None, no batching)
        show_progress: Whether to show progress bar during evaluation
        timeout: Timeout in seconds for Llama Stack client
    """
    import json
    import math
    import os
    from datetime import datetime

    llama_stack_host = os.environ.get("LLAMA_STACK_HOST")
    llama_stack_port = os.environ.get("LLAMA_STACK_PORT")
    if not llama_stack_host:
        raise ValueError("LLAMA_STACK_HOST environment variable must be set")
    if not llama_stack_port:
        raise ValueError("LLAMA_STACK_PORT environment variable must be set")
    print(f"LLAMA_STACK_HOST: {llama_stack_host}")
    print(f"LLAMA_STACK_PORT: {llama_stack_port}")

    print(f"\n[LOAD] Loading RAGAS dataset from artifact: {ragas_dataset_input.path}")
    with open(ragas_dataset_input.path, "r", encoding="utf-8") as f:
        ragas_dataset_json = f.read()
    ragas_dataset = json.loads(ragas_dataset_json)
    print(f"[OK] Loaded {len(ragas_dataset)} entries")

    metrics_list = [m.strip() for m in metrics.split(",") if m.strip()]
    if not metrics_list:
        raise ValueError("At least one metric required (metrics parameter).")

    batch_size_arg = None if (batch_size is None or batch_size <= 0) else batch_size

    # --- Inlined run_ragas_evaluation_direct logic (same as ragas_dataset_eval_direct.py) ---
    def _apply_eval_log_level():
        name = os.environ.get("RAGAS_EVAL_LOG_LEVEL", "").strip().upper()
        if not name:
            return
        import logging
        level = getattr(logging, name, None)
        if level is None:
            return
        for logger_name in ("httpx", "httpcore"):
            logging.getLogger(logger_name).setLevel(level)

    def _context_from_json_like(raw):
        if raw is None:
            return ""
        s = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
        s = s.strip()
        if not s:
            return ""
        try:
            obj = json.loads(s)
        except (json.JSONDecodeError, TypeError):
            return s
        if isinstance(obj, dict):
            lines = []
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    v = json.dumps(v, ensure_ascii=False)
                lines.append(f"{k}: {v}")
            return "\n".join(lines)
        if isinstance(obj, list):
            return "\n".join(
                str(x) if not isinstance(x, (dict, list)) else json.dumps(x, ensure_ascii=False)
                for x in obj
            )
        return s

    def convert_to_evaluation_format(ragas_dataset_inner):
        ragas_data = []
        for entry in ragas_dataset_inner:
            if entry.get("error"):
                continue
            answer = entry.get("answer", "")
            if not answer or (isinstance(answer, str) and answer.startswith("ERROR:")):
                continue
            question = entry.get("question", "")
            if not question:
                continue
            raw_contexts = entry.get("contexts") or []
            if raw_contexts == ["No context retrieved"]:
                raw_contexts = []
            retrieved_contexts = [_context_from_json_like(c) for c in raw_contexts]
            ragas_entry = {
                "user_input": question,
                "response": answer,
                "retrieved_contexts": retrieved_contexts,
            }
            if "ground_truth" in entry and entry["ground_truth"]:
                ragas_entry["reference"] = entry["ground_truth"]
            ragas_data.append(ragas_entry)
        return ragas_data

    def _get_llama_stack_client(timeout_sec=600):
        import httpx
        from llama_stack_client import LlamaStackClient
        host = os.environ.get("LLAMA_STACK_HOST")
        port = os.environ.get("LLAMA_STACK_PORT")
        secure = os.environ.get("LLAMA_STACK_SECURE", "").lower() in ("true", "1", "yes")
        if not host:
            raise ValueError("LLAMA_STACK_HOST must be set when using Llama Stack for evaluation")
        if not port:
            raise ValueError("LLAMA_STACK_PORT must be set when using Llama Stack for evaluation")
        base_url = f"{'https' if secure else 'http'}://{host}:{port}"
        http_client = httpx.Client(verify=False, timeout=timeout_sec)
        return LlamaStackClient(base_url=base_url, http_client=http_client)

    class _LlamaStackEmbeddings:
        def __init__(self, client, model_id_emb):
            self._client = client
            self._model_id = model_id_emb

        def embed_documents(self, texts):
            return [self._embed_one(t) for t in texts]

        def embed_query(self, text):
            return self._embed_one(text)

        def _embed_one(self, text):
            resp = self._client.embeddings.create(model=self._model_id, input=text)
            if hasattr(resp, "data") and resp.data:
                return getattr(resp.data[0], "embedding", resp.data[0])
            if hasattr(resp, "embeddings") and resp.embeddings:
                return resp.embeddings[0] if isinstance(resp.embeddings[0], list) else getattr(resp.embeddings[0], "embedding", resp.embeddings[0])
            if isinstance(resp, list):
                return resp
            raise RuntimeError(f"Unexpected embeddings response shape: {type(resp)}")

    def _get_chat_openai_for_llama_stack(model_id_inner):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise ImportError(
                "langchain-openai is required for Llama Stack LLM. pip install langchain-openai"
            ) from e
        host = os.environ.get("LLAMA_STACK_HOST")
        port = os.environ.get("LLAMA_STACK_PORT")
        secure = os.environ.get("LLAMA_STACK_SECURE", "").lower() in ("true", "1", "yes")
        if not host or not port:
            raise ValueError("LLAMA_STACK_HOST and LLAMA_STACK_PORT must be set")
        protocol = "https" if secure else "http"
        base_url = f"{protocol}://{host}:{port}/v1"
        api_key = os.environ.get("API_KEY", "fake")
        return ChatOpenAI(
            model=model_id_inner,
            api_key=api_key,
            base_url=base_url,
            temperature=0.0,
        )

    def _get_llama_stack_llm_and_embeddings(model_id_inner, embedding_model_id_inner, timeout_sec=600):
        client = _get_llama_stack_client(timeout_sec=timeout_sec)
        llm = _get_chat_openai_for_llama_stack(model_id_inner)
        embeddings = _LlamaStackEmbeddings(client, embedding_model_id_inner)
        return llm, embeddings

    def _import_ragas():
        try:
            from ragas import EvaluationDataset, SingleTurnSample, evaluate
            from ragas.metrics._faithfulness import faithfulness
            from ragas.metrics._answer_relevance import answer_relevancy
            from ragas.metrics._context_precision import context_precision
            from ragas.metrics._context_recall import context_recall
            return {
                "EvaluationDataset": EvaluationDataset,
                "SingleTurnSample": SingleTurnSample,
                "evaluate": evaluate,
                "metrics": {
                    "faithfulness": faithfulness,
                    "answer_relevancy": answer_relevancy,
                    "context_precision": context_precision,
                    "context_recall": context_recall,
                },
            }
        except ImportError as e:
            raise ImportError("ragas is required. pip install ragas") from e

    def get_metric_objects(metrics_list_inner, ragas_metrics):
        resolved = []
        for name in metrics_list_inner:
            name = name.strip().lower()
            if name not in ragas_metrics:
                raise ValueError(f"Unknown metric '{name}'. Available: {', '.join(ragas_metrics)}")
            resolved.append(ragas_metrics[name])
        return resolved

    _apply_eval_log_level()
    ragas_data = convert_to_evaluation_format(ragas_dataset)
    skipped = len(ragas_dataset) - len(ragas_data)
    if skipped:
        print(f"[WARN] Skipped {skipped} invalid/error entries")
    if not ragas_data:
        has_questions = any(e.get("question") for e in ragas_dataset)
        if has_questions and not any(e.get("answer") for e in ragas_dataset):
            raise ValueError(
                "No valid entries for RAGAS evaluation. Input looks like a base dataset "
                "(has 'question' but no 'answer'). Generate a RAGAS dataset first."
            )
        raise ValueError(
            "No valid entries for RAGAS evaluation. "
            "All entries were invalid (error, ERROR: answer, or missing question/answer)."
        )
    print(f"[OK] {len(ragas_data)} entries to evaluate")

    ragas = _import_ragas()
    EvaluationDataset = ragas["EvaluationDataset"]
    SingleTurnSample = ragas["SingleTurnSample"]
    evaluate_fn = ragas["evaluate"]
    ragas_metrics = ragas["metrics"]

    metric_objects = get_metric_objects(metrics_list, ragas_metrics)
    print(f"[METRICS] {', '.join(metrics_list)}")

    if not (model_id and embedding_model_id):
        raise ValueError(
            "When using Llama Stack for RAGAS, both model_id and embedding_model_id are required."
        )
    print(f"[LLAMA-STACK] Using LLM: {model_id} (chat), embeddings: {embedding_model_id}")
    llm, embeddings = _get_llama_stack_llm_and_embeddings(model_id, embedding_model_id, timeout_sec=timeout)

    samples = [SingleTurnSample(**entry) for entry in ragas_data]
    eval_dataset = EvaluationDataset(samples=samples)
    print("[START] Running RAGAS evaluate()...")

    eval_kw = {
        "metrics": metric_objects,
        "show_progress": show_progress,
        "batch_size": batch_size_arg,
    }
    eval_kw["llm"] = llm
    eval_kw["embeddings"] = embeddings

    result = evaluate_fn(eval_dataset, **eval_kw)

    scores_rows = result.scores
    if not scores_rows:
        raise RuntimeError("RAGAS evaluate() returned no scores")

    final_metrics = {}
    individual_scores = {m: [] for m in metrics_list}

    for row in scores_rows:
        for metric in metrics_list:
            val = row.get(metric)
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                individual_scores[metric].append(float(val))

    for metric in metrics_list:
        vals = individual_scores[metric]
        if vals:
            final_metrics[metric] = sum(vals) / len(vals)
        else:
            final_metrics[metric] = float("nan")

    base_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "benchmark_id": f"ragas_direct_{base_timestamp}",
        "timestamp": datetime.now().isoformat(),
        "metrics": final_metrics,
        "individual_scores": individual_scores,
        "generations": [],
        "failures": [],
        "dataset_size": len(ragas_dataset),
        "valid_entries": len(ragas_data),
        "mode": "ragas_direct",
        "model_id": model_id,
        "embedding_model_id": embedding_model_id,
    }

    print("\n" + "=" * 70)
    print("[RESULTS] RAGAS EVALUATION (direct)")
    print("=" * 70)
    print(f"Run:      {results['benchmark_id']}")
    print(f"Dataset:  {results['dataset_size']} entries ({results['valid_entries']} evaluated)")
    if results["metrics"]:
        print("\n[METRICS]")
        for metric, score in results["metrics"].items():
            if isinstance(score, float) and math.isnan(score):
                status, val = "[SKIP]", "N/A"
            else:
                val = f"{score:.4f}"
                status = "[PASS]" if score > 0.8 else "[WARN]" if score > 0.6 else "[FAIL]"
            print(f"  {status} {metric:25s}: {val}")
    print("=" * 70)
    # --- End inlined run_ragas_evaluation_direct logic ---

    final_metrics = results["metrics"]

    print("\n[LOG] Logging metrics to Kubeflow Pipelines...")
    for metric_name, metric_value in final_metrics.items():
        if isinstance(metric_value, float) and math.isnan(metric_value):
            continue
        evaluation_output_metrics.log_metric(metric_name, float(metric_value))
        print(f"   [OK] Logged {metric_name}: {metric_value:.4f}")

    evaluation_output_metrics.log_metric("dataset_size", float(results["dataset_size"]))
    evaluation_output_metrics.log_metric("valid_entries", float(results["valid_entries"]))
    print(f"   [OK] Logged dataset_size: {results['dataset_size']}")
    print(f"   [OK] Logged valid_entries: {results['valid_entries']}")

    formatted_results = {
        "benchmark_id": results["benchmark_id"],
        "timestamp": results["timestamp"],
        "metrics": results["metrics"],
        "individual_scores": results["individual_scores"],
        "generations": results["generations"],
        "failures": results["failures"],
        "dataset_size": results["dataset_size"],
        "valid_entries": results["valid_entries"],
        "mode": results["mode"],
        "model_id": results["model_id"],
        "embedding_model_id": results["embedding_model_id"],
    }

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
    retrieval_mode: str = "vector",  # vector, text, hybrid
    file_search_max_chunks: int = 5,
    file_search_score_threshold: float = 0.7,
    file_search_max_tokens_per_chunk: int = 512,
    ranker: str = "default",
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
        ranker=ranker,
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
        batch_size=batch_size,
        show_progress=True,
        timeout=timeout,
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
