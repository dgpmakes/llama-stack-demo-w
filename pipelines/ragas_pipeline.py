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
RAGAS Evaluation Pipeline

This pipeline:
1. Loads a base dataset from a git repository
2. Generates a RAGAS dataset by querying a RAG system via Llama Stack
3. Runs RAGAS evaluation metrics on the generated dataset
"""

from kfp import kubernetes, dsl
from kfp.dsl import Output, Metrics

# Workbench Runtime Image: Pytorch with Python 3.12 (UBI 9)
PYTORCH_IMAGE = "quay.io/modh/odh-pipeline-runtime-pytorch-cuda-py312-ubi9@sha256:72ff2381e5cb24d6f549534cb74309ed30e92c1ca80214669adb78ad30c5ae12"

# RAGAS metrics available
AVAILABLE_METRICS = [
    "answer_relevancy",      # Measures how relevant the answer is to the question
    "faithfulness",          # Measures factual consistency with retrieved contexts
    "context_precision",     # Measures relevance of retrieved contexts to the question
    "context_recall",        # Measures if all relevant information is retrieved
    "context_utilization",   # Measures how well the answer uses retrieved contexts
    "semantic_similarity",   # Measures semantic similarity between answer and reference
]

DEFAULT_METRICS = "answer_relevancy,faithfulness,context_precision,context_recall"


@dsl.component(
    base_image=PYTORCH_IMAGE,
    packages_to_install=["requests"],
)
def load_base_dataset_from_git(
    git_repo: str,
    git_context: str,
    git_ref: str,
    dataset_filename: str,
) -> str:
    """
    Load base dataset from a git repository.
    
    Args:
        git_repo: GitHub repository (e.g., 'alpha-hack-program/llama-stack-demo')
        git_context: Context path in the repository (e.g., 'materials/datasets')
        git_ref: Git reference/branch (e.g., 'main', 'next')
        dataset_filename: Name of the dataset file (e.g., 'base_dataset.json')
        
    Returns:
        JSON string containing the base dataset
    """
    import requests
    
    # Validate parameters
    if not git_repo:
        raise ValueError("git_repo parameter must be set")
    if not git_context:
        raise ValueError("git_context parameter must be set")
    if not git_ref:
        raise ValueError("git_ref parameter must be set")
    if not dataset_filename:
        raise ValueError("dataset_filename parameter must be set")
    
    # Build raw URL for the git repository
    raw_url = f"https://raw.githubusercontent.com/{git_repo}/{git_ref}/{git_context}/{dataset_filename}"
    
    print(f"📖 Loading base dataset from: {raw_url}")
    
    try:
        response = requests.get(raw_url)
        response.raise_for_status()
        dataset_content = response.text
        
        # Validate JSON
        import json
        dataset = json.loads(dataset_content)
        print(f"✓ Loaded {len(dataset)} questions from base dataset")
        
        return dataset_content
        
    except Exception as e:
        print(f"❌ ERROR: Failed to load dataset: {str(e)}")
        raise


@dsl.component(
    base_image=PYTORCH_IMAGE,
    packages_to_install=["llama-stack-client==0.3.5", "httpx"],
)
def generate_ragas_dataset(
    base_dataset_json: str,
    model_id: str,
    vector_store_id: str,
    tools: str = "",
    instructions: str = "",
    timeout: int = 300,
) -> str:
    """
    Generate RAGAS-compatible dataset with RAG answers and contexts.
    
    Uses Llama Stack's Responses API to query the RAG system and generate
    answers with retrieved contexts for each question in the base dataset.
    
    Args:
        base_dataset_json: JSON string containing the base dataset
        model_id: Model identifier to use for inference
        vector_store_id: Vector store ID containing documents
        tools: MCP tools to use. Options:
            - "" (empty): Only use file_search with the vector store
            - "all": Discover and use all available MCP tools
            - "tool1,tool2": Comma-separated list of specific MCP tool names
              (e.g., "cluster-insights,compatibility-engine")
        instructions: System prompt/instructions for the model (optional)
        timeout: Timeout in seconds for requests
        
    Returns:
        JSON string containing the RAGAS dataset
    """
    import os
    import json
    import httpx
    from typing import List, Dict, Any
    from llama_stack_client import LlamaStackClient
    
    def discover_mcp_tools(client: LlamaStackClient, tool_filter: str) -> List[Dict[str, Any]]:
        """
        Discover MCP tools from Llama Stack server.
        
        Args:
            client: LlamaStackClient instance
            tool_filter: "all" for all MCP tools, or comma-separated tool names
            
        Returns:
            List of MCP tool configurations
        """
        mcp_tools = []
        
        try:
            # Get all tool groups from the server
            tool_groups = list(client.toolgroups.list())
            print(f"   Found {len(tool_groups)} tool groups")
            
            # Filter for MCP tools (mcp::* prefix and model-context-protocol provider)
            for group in tool_groups:
                if not hasattr(group, 'identifier'):
                    continue
                
                identifier = group.identifier
                provider_id = getattr(group, 'provider_id', None)
                
                # Check if it's an MCP tool
                if not identifier.startswith('mcp::'):
                    continue
                
                # Verify provider_id if available
                if provider_id and provider_id != 'model-context-protocol':
                    continue
                
                # Extract tool name (remove 'mcp::' prefix)
                tool_name = identifier.split('::', 1)[1] if '::' in identifier else identifier
                
                # Check if we should include this tool
                if tool_filter.lower() != "all":
                    # Parse comma-separated tool names
                    requested_tools = [t.strip().lower() for t in tool_filter.split(',') if t.strip()]
                    if tool_name.lower() not in requested_tools:
                        continue
                
                # Extract MCP endpoint URL
                mcp_endpoint = getattr(group, 'mcp_endpoint', None)
                server_url = None
                
                if mcp_endpoint:
                    if hasattr(mcp_endpoint, 'uri'):
                        server_url = mcp_endpoint.uri
                    elif isinstance(mcp_endpoint, dict):
                        server_url = mcp_endpoint.get('uri')
                
                if server_url:
                    mcp_tools.append({
                        "type": "mcp",
                        "server_label": f"{tool_name}",
                        "server_url": server_url
                    })
                    print(f"   ✓ Added MCP tool: {tool_name} ({server_url})")
                else:
                    print(f"   ⚠️  Skipping {tool_name} (no server URL found)")
            
        except Exception as e:
            print(f"   ⚠️  Error discovering MCP tools: {e}")
        
        return mcp_tools
    
    # Read environment variables
    llama_stack_host = os.environ.get("LLAMA_STACK_HOST")
    llama_stack_port = os.environ.get("LLAMA_STACK_PORT")
    llama_stack_secure = os.environ.get("LLAMA_STACK_SECURE", "").lower() in ("true", "1", "yes")
    
    print(f"LLAMA_STACK_HOST: {llama_stack_host}")
    print(f"LLAMA_STACK_PORT: {llama_stack_port}")
    print(f"LLAMA_STACK_SECURE: {llama_stack_secure}")
    
    # Validate environment variables
    if not llama_stack_host:
        raise ValueError("LLAMA_STACK_HOST environment variable must be set")
    if not llama_stack_port:
        raise ValueError("LLAMA_STACK_PORT environment variable must be set")
    
    # Build base URL
    base_url = f"{'https' if llama_stack_secure else 'http'}://{llama_stack_host}:{llama_stack_port}"
    print(f"LLAMA_STACK_BASE_URL: {base_url}")
    
    # Initialize client
    http_client = httpx.Client(verify=False, timeout=timeout)
    client = LlamaStackClient(
        base_url=base_url,
        http_client=http_client
    )
    
    # Parse base dataset
    print(f"\n📖 Parsing base dataset...")
    dataset = json.loads(base_dataset_json)
    print(f"✓ Loaded {len(dataset)} questions")
    
    # Build tools list
    print(f"\n🛠️  Configuring tools...")
    tools_list: List[Dict[str, Any]] = []
    
    # Always add file_search if vector_store_id is provided
    if vector_store_id:
        tools_list.append({
            "type": "file_search",
            "vector_store_ids": [vector_store_id],
        })
        print(f"   ✓ Added file_search tool (vector store: {vector_store_id})")
    
    # Add MCP tools if requested
    if tools and tools.strip():
        print(f"\n🔍 Discovering MCP tools (filter: '{tools}')...")
        mcp_tools = discover_mcp_tools(client, tools.strip())
        if mcp_tools:
            tools_list.extend(mcp_tools)
            print(f"   ✓ Added {len(mcp_tools)} MCP tool(s)")
        else:
            print(f"   ℹ️  No matching MCP tools found")
    
    print(f"\n📋 Total tools configured: {len(tools_list)}")
    for tool in tools_list:
        tool_type = tool.get("type", "unknown")
        if tool_type == "mcp":
            print(f"   - MCP: {tool.get('server_label')} ({tool.get('server_url')})")
        elif tool_type == "file_search":
            print(f"   - File Search: {tool.get('vector_store_ids')}")
        else:
            print(f"   - {tool_type}")
    
    # Log instructions if provided
    if instructions and instructions.strip():
        print(f"\n📝 System Instructions: {instructions[:100]}{'...' if len(instructions) > 100 else ''}")
    
    # Process each question
    print(f"\n🤖 Processing questions using Responses API...")
    print(f"Model: {model_id}")
    print(f"Vector Store: {vector_store_id}\n")
    
    ragas_dataset = []
    
    for i, item in enumerate(dataset, 1):
        question_id = item.get('id', f'q_{i}')
        question = item['question']
        ground_truth = item.get('ground_truth', '')
        
        print(f"[{i}/{len(dataset)}] Processing: {question_id}")
        
        try:
            # Build request config
            request_config: Dict[str, Any] = {
                "model": model_id,
                "input": question,
                "tools": tools_list,
            }
            
            # Add instructions if provided
            if instructions and instructions.strip():
                request_config["instructions"] = instructions.strip()
            
            # Add include for file_search results if using file_search
            if any(t.get("type") == "file_search" for t in tools_list):
                request_config["include"] = ["file_search_call.results"]
            
            # Query RAG system using Responses API
            response = client.responses.create(**request_config)
            
            # Extract answer from the response
            answer = getattr(response, "output_text", str(response))
            
            # Extract contexts from response
            contexts = []
            if hasattr(response, 'output') and isinstance(response.output, list):
                for output_item in response.output:
                    # Extract file_search results
                    if hasattr(output_item, 'results') and isinstance(output_item.results, list):
                        for result in output_item.results:
                            if hasattr(result, 'text') and result.text:
                                contexts.append(result.text)
                    # Also try to extract MCP tool results if any
                    if hasattr(output_item, 'content') and output_item.content:
                        # MCP responses may have content field
                        if isinstance(output_item.content, str) and output_item.content not in contexts:
                            # Only add if it looks like context (not the final answer)
                            pass  # Skip for now, MCP results are usually processed by the model
            
            # Build RAGAS entry
            ragas_entry = {
                "id": question_id,
                "question": question,
                "answer": answer,
                "contexts": contexts if contexts else ["No context retrieved"],
                "ground_truth": ground_truth,
            }
            
            # Include optional fields from original dataset
            if 'difficulty' in item:
                ragas_entry['difficulty'] = item['difficulty']
            
            ragas_dataset.append(ragas_entry)
            print(f"  ✓ Answer generated ({len(contexts)} contexts retrieved)")
            
        except Exception as e:
            print(f"  ⚠️  Error processing {question_id}: {e}")
            # Add entry with error information
            ragas_dataset.append({
                "id": question_id,
                "question": question,
                "answer": f"ERROR: {str(e)}",
                "contexts": [],
                "ground_truth": ground_truth,
                "difficulty": item.get('difficulty', 'unknown')
            })
    
    print(f"\n✅ RAGAS dataset generation complete!")
    print(f"   Input: {len(dataset)} questions")
    print(f"   Output: {len(ragas_dataset)} entries")
    
    return json.dumps(ragas_dataset, indent=2, ensure_ascii=False)


@dsl.component(
    base_image=PYTORCH_IMAGE,
    packages_to_install=["llama-stack-client==0.3.5", "httpx"],
)
def run_ragas_evaluation(
    ragas_dataset_json: str,
    model_id: str,
    embedding_model_id: str,
    metrics: str,
    evaluation_output_metrics: Output[Metrics],
    mode: str = "inline",
    batch_size: int = 0,
    timeout: int = 600,
    max_wait_seconds: int = 900,
    poll_interval: int = 5,
) -> str:
    """
    Run RAGAS evaluation on the generated dataset.
    
    Args:
        ragas_dataset_json: JSON string containing the RAGAS dataset
        model_id: LLM model identifier for scoring
        embedding_model_id: Embedding model identifier
        metrics: Comma-separated list of RAGAS metrics to compute
        evaluation_output_metrics: Kubeflow Metrics output for logging RAGAS metrics
        mode: Evaluation mode - "inline" or "remote"
        batch_size: Batch size for evaluation (0 = all at once)
        timeout: Timeout in seconds for requests
        max_wait_seconds: Maximum seconds to wait for evaluation job
        poll_interval: Seconds between job status checks
        
    Returns:
        JSON string containing evaluation results
        
    Metrics Logged:
        - answer_relevancy, faithfulness, context_precision, context_recall (RAGAS metrics)
        - dataset_size: Number of entries evaluated
        - failure_count: Number of failed batches
    """
    import os
    import json
    import time
    import httpx
    from datetime import datetime
    from llama_stack_client import LlamaStackClient
    
    # Read environment variables
    llama_stack_host = os.environ.get("LLAMA_STACK_HOST")
    llama_stack_port = os.environ.get("LLAMA_STACK_PORT")
    llama_stack_secure = os.environ.get("LLAMA_STACK_SECURE", "").lower() in ("true", "1", "yes")
    
    print(f"LLAMA_STACK_HOST: {llama_stack_host}")
    print(f"LLAMA_STACK_PORT: {llama_stack_port}")
    print(f"LLAMA_STACK_SECURE: {llama_stack_secure}")
    
    # Validate environment variables
    if not llama_stack_host:
        raise ValueError("LLAMA_STACK_HOST environment variable must be set")
    if not llama_stack_port:
        raise ValueError("LLAMA_STACK_PORT environment variable must be set")
    
    # Build base URL
    base_url = f"{'https' if llama_stack_secure else 'http'}://{llama_stack_host}:{llama_stack_port}"
    print(f"LLAMA_STACK_BASE_URL: {base_url}")
    
    # Initialize client
    http_client = httpx.Client(verify=False, timeout=timeout)
    client = LlamaStackClient(
        base_url=base_url,
        http_client=http_client
    )
    
    # Parse RAGAS dataset
    print(f"\n📖 Parsing RAGAS dataset...")
    ragas_dataset = json.loads(ragas_dataset_json)
    print(f"✓ Loaded {len(ragas_dataset)} entries")
    
    # Parse metrics
    metrics_list = [m.strip() for m in metrics.split(",") if m.strip()]
    print(f"📊 Metrics to evaluate: {', '.join(metrics_list)}")
    
    # Convert to RAGAS format
    print("\n🔄 Converting to RAGAS evaluation format...")
    ragas_data = []
    for entry in ragas_dataset:
        ragas_entry = {
            "user_input": entry["question"],
            "response": entry["answer"],
            "retrieved_contexts": entry["contexts"],
        }
        if "ground_truth" in entry and entry["ground_truth"]:
            ragas_entry["reference"] = entry["ground_truth"]
        ragas_data.append(ragas_entry)
    print(f"✓ Converted {len(ragas_data)} entries")
    
    # Provider ID based on mode
    provider_id = "trustyai_ragas_inline" if mode == "inline" else "trustyai_ragas_remote"
    
    # Batch processing setup
    if batch_size and batch_size > 0:
        batches = [ragas_data[i:i + batch_size] for i in range(0, len(ragas_data), batch_size)]
    else:
        batches = [ragas_data]
    
    print(f"🚀 Starting evaluation in {len(batches)} batch(es) ({mode.upper()} mode)...")
    
    # Generate base timestamp
    base_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    aggregated_scores = {metric: [] for metric in metrics_list}
    
    all_results = {
        "metrics": {},
        "individual_scores": {},
        "generations": [],
        "failures": []
    }
    
    for batch_idx, batch in enumerate(batches):
        print(f"\n📦 Processing Batch {batch_idx+1}/{len(batches)} (Size: {len(batch)})")
        
        # Unique IDs for this batch
        batch_id_suffix = f"_{base_timestamp}_{batch_idx+1}"
        dataset_id = f"ragas_dataset{batch_id_suffix}"
        benchmark_id = f"ragas_benchmark{batch_id_suffix}"
        
        try:
            # Unregister existing dataset if present
            try:
                client.datasets.unregister(dataset_id)
            except Exception:
                pass
            
            # Register dataset
            print(f"📝 Registering dataset: {dataset_id}")
            client.datasets.register(
                purpose="eval/question-answer",
                source={
                    "type": "rows",
                    "rows": batch,
                },
                dataset_id=dataset_id,
            )
            print(f"✓ Dataset registered with {len(batch)} entries")
            
            # Register benchmark
            print(f"📊 Registering benchmark: {benchmark_id}")
            url = f"{client.base_url}/v1alpha/eval/benchmarks"
            payload = {
                "benchmark_id": benchmark_id,
                "dataset_id": dataset_id,
                "scoring_functions": metrics_list,
                "provider_id": provider_id
            }
            response = client._client.post(url, json=payload)
            response.raise_for_status()
            print(f"✓ Benchmark registered")
            
            # Prepare benchmark config
            eval_candidate = {
                "type": "model",
                "model": model_id,
                "sampling_params": {
                    "strategy": {"type": "greedy"},
                    "max_tokens": 2048
                }
            }
            
            scoring_params = {}
            for metric in metrics_list:
                scoring_params[metric] = {
                    "type": "basic",
                    "aggregation_functions": ["average"]
                }
            
            benchmark_config = {
                "eval_candidate": eval_candidate,
                "scoring_params": scoring_params,
            }
            
            extra_body = {
                "provider_id": provider_id,
                "judge_model": model_id,
                "embedding_model": embedding_model_id,
            }
            
            # Run evaluation
            print(f"🚀 Running evaluation...")
            job = client.alpha.eval.run_eval(
                benchmark_id=benchmark_id,
                benchmark_config=benchmark_config,
                extra_body=extra_body,
            )
            print(f"✓ Evaluation job started (Job ID: {job.job_id})")
            
            # Wait for results
            print(f"📥 Waiting for evaluation results...")
            batch_result = None
            waited = 0
            
            while waited < max_wait_seconds:
                time.sleep(poll_interval)
                waited += poll_interval
                
                try:
                    result_url = f"{client.base_url}/v1alpha/eval/benchmarks/{benchmark_id}/jobs/{job.job_id}/result"
                    response = client._client.get(result_url)
                    response.raise_for_status()
                    batch_result = response.json()
                    
                    if batch_result.get("scores"):
                        print(f"   ✓ Results received after {waited}s")
                        break
                    else:
                        print(f"   Waiting... ({waited}s)")
                except Exception as e:
                    print(f"   Waiting... ({waited}s) - {e}")
            
            if not batch_result or not batch_result.get("scores"):
                raise RuntimeError(f"Evaluation job {job.job_id} returned no scores")
            
            # Process results
            if "scores" in batch_result:
                for metric, score_data in batch_result["scores"].items():
                    score_val = 0.0
                    if isinstance(score_data, dict) and "aggregated_results" in score_data:
                        score_val = score_data["aggregated_results"].get(metric, 0.0)
                    elif isinstance(score_data, (int, float)):
                        score_val = float(score_data)
                    
                    aggregated_scores[metric].append((score_val, len(batch)))
                    
                    if isinstance(score_data, dict) and "score_rows" in score_data:
                        if metric not in all_results["individual_scores"]:
                            all_results["individual_scores"][metric] = []
                        for row in score_data["score_rows"]:
                            all_results["individual_scores"][metric].append(row.get('score', 0.0))
            
            if "generations" in batch_result:
                all_results["generations"].extend(batch_result.get("generations", []))
                
        except Exception as e:
            print(f"❌ Batch {batch_idx+1} failed: {e}")
            all_results["failures"].append({
                "batch_index": batch_idx + 1,
                "error": str(e),
            })
            continue
    
    # Final Aggregation
    print("\n∑ Aggregating results...")
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
    
    # Log metrics to Kubeflow Pipelines for visualization
    print("\n📈 Logging metrics to Kubeflow Pipelines...")
    for metric_name, metric_value in final_metrics.items():
        evaluation_output_metrics.log_metric(metric_name, metric_value)
        print(f"   ✓ Logged {metric_name}: {metric_value:.4f}")
    
    # Also log dataset size and failure count as metadata
    evaluation_output_metrics.log_metric("dataset_size", float(len(ragas_dataset)))
    evaluation_output_metrics.log_metric("failure_count", float(len(all_results["failures"])))
    print(f"   ✓ Logged dataset_size: {len(ragas_dataset)}")
    print(f"   ✓ Logged failure_count: {len(all_results['failures'])}")
    
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
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 RAGAS EVALUATION RESULTS")
    print("=" * 70)
    print(f"Benchmark ID: {formatted_results['benchmark_id']}")
    print(f"Timestamp:    {formatted_results['timestamp']}")
    print(f"Dataset Size: {formatted_results['dataset_size']} entries")
    print(f"Mode:         {formatted_results['mode']}")
    print("=" * 70)
    
    if formatted_results.get("failures"):
        print("\n❌ Failures:")
        for failure in formatted_results["failures"]:
            print(f"  Batch {failure.get('batch_index')}: {failure.get('error')}")
    
    if formatted_results["metrics"]:
        print("\n🎯 Aggregated Metrics:")
        print("-" * 70)
        for metric, score in formatted_results["metrics"].items():
            emoji = "✅" if score > 0.8 else "⚠️" if score > 0.6 else "❌"
            print(f"  {emoji} {metric:25s}: {score:.4f}")
        print("-" * 70)
    
    print("=" * 70)
    
    return json.dumps(formatted_results, indent=2, ensure_ascii=False)


@dsl.pipeline(
    name="ragas-evaluation-pipeline",
    description="Pipeline to generate and evaluate RAGAS datasets from a git repository"
)
def pipeline(
    # Git repository parameters
    git_repo: str = "alpha-hack-program/llama-stack-demo-next",
    git_context: str = "materials/datasets",
    git_ref: str = "next",
    dataset_filename: str = "base_dataset.json",
    # Model parameters
    model_id: str = "llama-3-1-8b-w4a16/llama-3-1-8b-w4a16",
    embedding_model_id: str = "sentence-transformers/nomic-ai/nomic-embed-text-v1.5",
    # Vector store parameters
    vector_store_id: str = "",
    # MCP tools and instructions parameters
    tools: str = "",
    instructions: str = "",
    # Evaluation parameters
    metrics: str = DEFAULT_METRICS,
    mode: str = "inline",
    batch_size: int = 0,
    timeout: int = 600,
    max_wait_seconds: int = 900,
    poll_interval: int = 5,
):
    """
    RAGAS Evaluation Pipeline
    
    This pipeline:
    1. Loads a base dataset from a git repository
    2. Generates a RAGAS dataset by querying a RAG system via Llama Stack
    3. Runs RAGAS evaluation metrics on the generated dataset
    
    Args:
        git_repo: GitHub repository (e.g., 'alpha-hack-program/llama-stack-demo-next')
        git_context: Context path in the repository (e.g., 'materials/datasets')
        git_ref: Git reference/branch (e.g., 'main', 'next')
        dataset_filename: Name of the dataset file (e.g., 'base_dataset.json')
        model_id: Model identifier for RAG queries and evaluation
        embedding_model_id: Embedding model identifier
        vector_store_id: Vector store ID containing ingested documents
        tools: MCP tools to use. Options:
            - "" (empty): Only use file_search with the vector store
            - "all": Discover and use all available MCP tools
            - "tool1,tool2": Comma-separated list of specific MCP tool names
              (e.g., "cluster-insights,compatibility-engine")
        instructions: System prompt/instructions for the model when generating answers
        metrics: Comma-separated RAGAS metrics (answer_relevancy,faithfulness,context_precision,context_recall)
        mode: Evaluation mode - "inline" or "remote"
        batch_size: Batch size for evaluation (0 = all at once)
        timeout: Timeout in seconds for requests
        max_wait_seconds: Maximum seconds to wait for evaluation job
        poll_interval: Seconds between job status checks
    
    Returns:
        Evaluation results as JSON string
    """
    
    # Step 1: Load base dataset from git
    load_dataset_task = load_base_dataset_from_git(
        git_repo=git_repo,
        git_context=git_context,
        git_ref=git_ref,
        dataset_filename=dataset_filename,
    )
    load_dataset_task.set_caching_options(False)
    load_dataset_task.set_cpu_request("100m")
    load_dataset_task.set_cpu_limit("1")
    load_dataset_task.set_memory_request("256Mi")
    load_dataset_task.set_memory_limit("512Mi")
    
    # Step 2: Generate RAGAS dataset
    generate_dataset_task = generate_ragas_dataset(
        base_dataset_json=load_dataset_task.output,
        model_id=model_id,
        vector_store_id=vector_store_id,
        tools=tools,
        instructions=instructions,
        timeout=timeout,
    )
    kubernetes.use_config_map_as_env(
        task=generate_dataset_task,
        config_map_name='rag-pipeline-config',
        config_map_key_to_env={
            'LLAMA_STACK_HOST': 'LLAMA_STACK_HOST',
            'LLAMA_STACK_PORT': 'LLAMA_STACK_PORT',
            'LLAMA_STACK_SECURE': 'LLAMA_STACK_SECURE'
        }
    )
    generate_dataset_task.set_caching_options(False)
    generate_dataset_task.set_cpu_request("500m")
    generate_dataset_task.set_cpu_limit("4")
    generate_dataset_task.set_memory_request("2Gi")
    generate_dataset_task.set_memory_limit("6Gi")
    
    # Step 3: Run RAGAS evaluation
    evaluate_task = run_ragas_evaluation(
        ragas_dataset_json=generate_dataset_task.output,
        model_id=model_id,
        embedding_model_id=embedding_model_id,
        metrics=metrics,
        mode=mode,
        batch_size=batch_size,
        timeout=timeout,
        max_wait_seconds=max_wait_seconds,
        poll_interval=poll_interval,
    )
    kubernetes.use_config_map_as_env(
        task=evaluate_task,
        config_map_name='rag-pipeline-config',
        config_map_key_to_env={
            'LLAMA_STACK_HOST': 'LLAMA_STACK_HOST',
            'LLAMA_STACK_PORT': 'LLAMA_STACK_PORT',
            'LLAMA_STACK_SECURE': 'LLAMA_STACK_SECURE'
        }
    )
    evaluate_task.set_caching_options(False)
    evaluate_task.set_cpu_request("500m")
    evaluate_task.set_cpu_limit("4")
    evaluate_task.set_memory_request("2Gi")
    evaluate_task.set_memory_limit("6Gi")


if __name__ == '__main__':
    from shared.kubeflow import compile_and_upsert_pipeline
    
    import os

    pipeline_package_path = __file__.replace('.py', '.yaml')

    # Pipeline name
    pipeline_name = os.path.basename(__file__).replace('.py', '').replace('_', '-')

    compile_and_upsert_pipeline(
        pipeline_func=pipeline,  # type: ignore
        pipeline_package_path=pipeline_package_path,
        pipeline_name=pipeline_name
    )
