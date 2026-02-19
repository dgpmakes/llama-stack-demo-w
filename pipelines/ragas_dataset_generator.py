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
RAGAS Dataset Generator (CLI)

Generates a RAGAS-compatible dataset by querying a RAG system via Llama Stack,
using the same logic as the generate_ragas_dataset component in ragas_pipeline.py.
Vector store selection matches the search command: by name (optional; if omitted, uses latest).

Usage:
    # From a local base dataset file (recommended)
    python ragas_dataset_generator.py ./base_dataset_small.json --vector-store-name rag-store -o ragas_dataset.json

    # Same with explicit flags
    python ragas_dataset_generator.py --base-dataset ./base_dataset_small.json --vector-store-name rag-store -o ragas_dataset.json

    # Use latest vector store (no name)
    python ragas_dataset_generator.py ./base_dataset_small.json -o ragas_dataset.json

    # From git instead of a file
    python ragas_dataset_generator.py --from-git --git-repo alpha-hack-program/llama-stack-demo \\
        --git-context materials/datasets --git-ref next --base-dataset-filename base_dataset_small.json \\
        --vector-store-name rag-store -o ragas_dataset.json

Requires: LLAMA_STACK_HOST, LLAMA_STACK_PORT (and optionally LLAMA_STACK_SECURE).
"""

import argparse
import json
import os
import re
import sys
from pprint import pprint
from typing import Any, Dict, List

import httpx
from llama_stack_client import LlamaStackClient


def get_llama_stack_client(timeout: int = 300) -> LlamaStackClient:
    """Build Llama Stack client from environment."""
    llama_stack_host = os.environ.get("LLAMA_STACK_HOST")
    llama_stack_port = os.environ.get("LLAMA_STACK_PORT")
    llama_stack_secure = os.environ.get("LLAMA_STACK_SECURE", "").lower() in ("true", "1", "yes")

    if not llama_stack_host:
        raise ValueError("LLAMA_STACK_HOST environment variable must be set")
    if not llama_stack_port:
        raise ValueError("LLAMA_STACK_PORT environment variable must be set")

    base_url = f"{'https' if llama_stack_secure else 'http'}://{llama_stack_host}:{llama_stack_port}"
    http_client = httpx.Client(verify=False, timeout=timeout)
    return LlamaStackClient(base_url=base_url, http_client=http_client)


def load_base_dataset_from_file(path: str) -> List[Dict[str, Any]]:
    """Load base dataset from a local JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Base dataset must be a JSON array of items; got {type(data)}")
    return data


def load_base_dataset_from_git(
    git_repo: str,
    git_context: str,
    git_ref: str,
    base_dataset_filename: str,
) -> List[Dict[str, Any]]:
    """Load base dataset from a git repository (raw GitHub URL)."""
    raw_url = f"https://raw.githubusercontent.com/{git_repo}/{git_ref}/{git_context}/{base_dataset_filename}"
    print(f"[LOAD] Loading base dataset from: {raw_url}")
    with httpx.Client(timeout=30) as http:
        response = http.get(raw_url)
        response.raise_for_status()
        data = response.json()
    if not isinstance(data, list):
        raise ValueError(f"Base dataset must be a JSON array of items; got {type(data)}")
    return data


def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from model output so the stored answer is clean."""
    if not text or not isinstance(text, str):
        return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _serialize_for_json(val: Any) -> Any:
    """Convert a value to something JSON-serializable (str, dict, list, number, bool, None)."""
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

    # Llama Stack Responses API: response.output is a list (file_search_call, mcp_call, message, mcp_list_tools, etc.)
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
                    if not isinstance(args, dict):
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


def _created_at_key(vs: Any) -> tuple:
    """Sort key for vector stores: prefer newest by created_at."""
    created = getattr(vs, "created_at", None)
    return (0, None) if created is None else (1, created)


def resolve_vector_store_id(
    client: LlamaStackClient,
    vector_store_name: str | None = None,
) -> str:
    """
    Resolve vector store to ID via Llama Stack.
    Same behavior as the search command: if name is given, use latest store with that name;
    if None, use latest store overall (any name).
    """
    all_vector_stores = list(client.vector_stores.list())
    if not all_vector_stores:
        raise ValueError(
            "No vector stores found. Create one first (e.g. via the load command)."
        )
    if vector_store_name:
        matching = [
            vs for vs in all_vector_stores
            if hasattr(vs, "name") and vs.name == vector_store_name
        ]
        if not matching:
            available = [getattr(vs, "name", None) for vs in all_vector_stores if hasattr(vs, "name")]
            raise ValueError(
                f"No vector store found with name '{vector_store_name}'. "
                f"Available: {available}"
            )
        stores = matching
    else:
        stores = all_vector_stores
    try:
        stores.sort(key=_created_at_key, reverse=True)
    except (TypeError, ValueError):
        pass
    chosen = stores[0]
    return chosen.id


def discover_mcp_tools(client: LlamaStackClient, tools: str) -> List[Dict[str, Any]]:
    """Discover MCP tools from Llama Stack. tools: '' or 'none' (no MCP tools), 'all', or 'tool1,tool2'."""
    tool_filter = (tools or "").strip().lower()
    if not tool_filter or tool_filter == "none":
        return []
    tool_groups = list(client.toolgroups.list())
    requested = [] if tool_filter == "all" else [t.strip().lower() for t in tools.split(",") if t.strip()]
    mcp_tools = []
    for group in tool_groups:
        if not getattr(group, "identifier", "").startswith("mcp::"):
            continue
        if getattr(group, "provider_id", None) and getattr(group, "provider_id") != "model-context-protocol":
            continue
        tool_name = (group.identifier.split("::", 1)[1] if "::" in group.identifier else group.identifier).lower()
        if requested and tool_name not in requested:
            continue
        mcp_endpoint = getattr(group, "mcp_endpoint", None)
        server_url = None
        if mcp_endpoint:
            server_url = getattr(mcp_endpoint, "uri", None) or (mcp_endpoint.get("uri") if isinstance(mcp_endpoint, dict) else None)
        if server_url:
            mcp_tools.append({
                "type": "mcp",
                "server_label": tool_name,
                "server_url": server_url,
            })
    return mcp_tools


def generate_ragas_dataset(
    base_dataset: List[Dict[str, Any]],
    client: LlamaStackClient,
    model_id: str,
    vector_store_id: str,
    mcp_tools: List[Dict[str, Any]],
    instructions: str = "",
    retrieval_mode: str = "vector",
    file_search_max_chunks: int = 5,
    file_search_score_threshold: float = 0.7,
    file_search_max_tokens_per_chunk: int = 512,
) -> List[Dict[str, Any]]:
    """
    Generate RAGAS dataset by querying Llama Stack Responses API for each question.
    Same logic as generate_ragas_dataset in ragas_pipeline.py.
    """
    tools_list: List[Dict[str, Any]] = []

    if vector_store_id:
        tools_list.append({
            "type": "file_search",
            "vector_store_ids": [vector_store_id],
            "file_search": {
                "retrieval_mode": retrieval_mode,
                "max_chunks": file_search_max_chunks,
                "max_num_results": file_search_max_chunks,  # OpenAI-compatible name; server may use either
                "score_threshold": file_search_score_threshold,
                "max_tokens_per_chunk": file_search_max_tokens_per_chunk,
            },
        })
    tools_list.extend(mcp_tools)

    ragas_dataset = []
    errors: List[str] = []
    error_count = 0

    for i, item in enumerate(base_dataset, 1):
        question_id = item.get("id", f"q_{i}")
        question = item["question"]
        ground_truth = item.get("ground_truth", "")

        print(f"[{i}/{len(base_dataset)}] Processing: {question_id}")
        try:
            request_config: Dict[str, Any] = {
                "model": model_id,
                "input": question,
                "tools": tools_list,
            }
            # DEBUG: print the request config
            # print(f"Request config: {request_config}")
            # print(f"Tools list: {tools_list}")
            if instructions and instructions.strip():
                request_config["instructions"] = instructions.strip()
            if any(t.get("type") == "file_search" for t in tools_list):
                request_config["include"] = ["file_search_call.results"]

            # Run the request using the client
            response = client.responses.create(**request_config)
            # DEBUG: print the response (pretty-printed, works with non-JSON-serializable objects)
            print("########## Response:")
            pprint(response.__dict__ if hasattr(response, "__dict__") else response)
            print(f"########## Response type: {type(response)}")

            answer = getattr(response, "output_text", str(response))
            if isinstance(answer, str):
                answer = _strip_think_blocks(answer)

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
            print(f"  [ERROR] {question_id}: {e}")
            ragas_dataset.append({
                "id": question_id,
                "question": question,
                "answer": f"ERROR: {str(e)}",
                "contexts": [],
                "ground_truth": ground_truth,
                "difficulty": item.get("difficulty", "unknown"),
                "error": True,
            })

    total = len(base_dataset)
    success = total - error_count
    print(f"\n[SUMMARY] Total: {total}, Successful: {success}, Failed: {error_count}")

    if error_count == total:
        raise ValueError(f"All questions failed. First error: {errors[0] if errors else 'Unknown'}")
    if error_count > total / 2:
        raise ValueError(f"Too many failures: {error_count}/{total}. Errors: {errors[:5]}")

    return ragas_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a RAGAS dataset by querying a RAG system via Llama Stack",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Base dataset: file path (positional or --base-dataset) or --from-git
    parser.add_argument(
        "base_dataset_file",
        nargs="?",
        default=None,
        metavar="PATH",
        help="Path to base dataset JSON file (array of {id, question, ground_truth?}). Use --from-git to load from git instead.",
    )
    parser.add_argument(
        "--base-dataset",
        dest="base_dataset_path",
        metavar="PATH",
        default=None,
        help="Path to base dataset JSON (alternative to positional PATH)",
    )
    parser.add_argument(
        "--from-git",
        action="store_true",
        help="Load base dataset from git; use with --git-repo, --git-context, --git-ref, --base-dataset-filename",
    )
    parser.add_argument("--git-repo", default="alpha-hack-program/llama-stack-demo", help="GitHub repo (owner/repo)")
    parser.add_argument("--git-context", default="materials/datasets", help="Path inside repo")
    parser.add_argument("--git-ref", default="next", help="Branch or tag")
    parser.add_argument("--base-dataset-filename", default="base_dataset_small.json", help="Dataset filename in repo")

    # Model and vector store (same as search command: optional name, else latest)
    parser.add_argument(
        "--model-id",
        default="llama-3-1-8b-w4a16/llama-3-1-8b-w4a16",
        help="Model identifier for inference",
    )
    parser.add_argument(
        "--vector-store-name",
        default=None,
        metavar="NAME",
        help="Vector store name (resolved via Llama Stack; if omitted, uses latest vector store, same as search command)",
    )

    # Output
    parser.add_argument("-o", "--output", default="ragas_dataset.json", help="Output RAGAS dataset JSON path")

    # Optional: MCP tools, instructions, retrieval
    parser.add_argument("--tools", default="", help="MCP tools: '' or 'none' (file_search only), 'all', or 'tool1,tool2'")
    parser.add_argument("--instructions", default="", help="System prompt / instructions for the model")
    parser.add_argument("--timeout", type=int, default=300, help="Request timeout in seconds")
    parser.add_argument("--retrieval-mode", choices=["vector", "text", "hybrid"], default="vector")
    parser.add_argument("--file-search-max-chunks", type=int, default=5)
    parser.add_argument("--file-search-score-threshold", type=float, default=0.7)
    parser.add_argument("--file-search-max-tokens-per-chunk", type=int, default=512)

    args = parser.parse_args()
    # Normalize: prefer positional, then --base-dataset
    args.base_dataset_path = args.base_dataset_path or args.base_dataset_file
    return args


def main() -> int:
    args = parse_args()

    # Base dataset: file path (primary) or --from-git
    if args.from_git:
        if args.base_dataset_path:
            print("Error: Cannot use both a base dataset file path and --from-git.", file=sys.stderr)
            return 1
        base_dataset = load_base_dataset_from_git(
            args.git_repo,
            args.git_context,
            args.git_ref,
            args.base_dataset_filename,
        )
    elif args.base_dataset_path:
        print(f"[LOAD] Loading base dataset from file: {args.base_dataset_path}")
        base_dataset = load_base_dataset_from_file(args.base_dataset_path)
    else:
        print("Error: Provide a base dataset file path (e.g. ./base_dataset.json) or use --from-git.", file=sys.stderr)
        return 1

    print(f"[OK] Loaded {len(base_dataset)} questions")

    client = get_llama_stack_client(timeout=args.timeout)
    host = os.environ.get("LLAMA_STACK_HOST", "")
    port = os.environ.get("LLAMA_STACK_PORT", "")
    print(f"[CONFIG] Llama Stack: {host}:{port}")

    # Vector store: by name (optional; if omitted, use latest — same as search command)
    print(f"[RESOLVE] Resolving vector store (name={args.vector_store_name!r}, or latest if omitted)")
    vector_store_id = resolve_vector_store_id(client, args.vector_store_name)
    print(f"[OK] Vector store ID: {vector_store_id}")

    mcp_tools = discover_mcp_tools(client, args.tools)
    if mcp_tools:
        print(f"[CONFIG] MCP tools: {len(mcp_tools)}")

    ragas_dataset = generate_ragas_dataset(
        base_dataset=base_dataset,
        client=client,
        model_id=args.model_id,
        vector_store_id=vector_store_id,
        mcp_tools=mcp_tools,
        instructions=args.instructions,
        retrieval_mode=args.retrieval_mode,
        file_search_max_chunks=args.file_search_max_chunks,
        file_search_score_threshold=args.file_search_score_threshold,
        file_search_max_tokens_per_chunk=args.file_search_max_tokens_per_chunk,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(ragas_dataset, f, indent=2, ensure_ascii=False)
    print(f"[OK] Written RAGAS dataset to {args.output} ({len(ragas_dataset)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
