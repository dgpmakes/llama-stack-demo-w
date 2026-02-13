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
RAGAS Dataset Evaluator (direct RAGAS library)

Runs RAGAS evaluation on a RAGAS-compatible dataset using the RAGAS Python
library directly (no Llama Stack / benchmark API). Reads a RAGAS dataset JSON,
converts to RAGAS EvaluationDataset, calls ragas.evaluate() with the requested
metrics, and writes aggregated metrics and per-row results to a JSON file.

Requires: pip install ragas.
When --model-id and --embedding-model-id are provided with LLAMA_STACK_HOST
and LLAMA_STACK_PORT, uses Llama Stack for the LLM and embeddings.

LLM options:
  - Default: ChatOpenAI with base_url to /v1/openai/v1 (OpenAI-compatible chat
    completions). May show a deprecation notice on the server; works well with RAGAS.
  - --use-openai-responses-api: ChatOpenAI(use_responses_api=True). Not recommended:
    RAGAS often raises AttributeError('str' object has no attribute 'content') due to
    response format mismatch. Use the default (chat) instead.
  - --use-responses-api: Custom LLM using Llama Stack's native API
    (client.responses.create()). Some RAGAS metrics may hit parser errors.

Embeddings always use Llama Stack's embeddings API. Otherwise RAGAS uses its defaults.

Set RAGAS_EVAL_LOG_LEVEL=ERROR (or WARNING) to reduce httpx/httpcore request log noise.
(typically OpenAI; set OPENAI_API_KEY).

Usage:
    # Evaluate a generated RAGAS dataset (default metrics, output ragas_eval_results.json)
    python ragas_dataset_eval_new.py ./ragas_dataset.json

    # Custom metrics and output path
    python ragas_dataset_eval_new.py ./ragas_dataset.json -o results.json \\
        --metrics "faithfulness,answer_relevancy,context_precision,context_recall"

    # With progress and batching
    python ragas_dataset_eval_new.py ./ragas_dataset.json --batch-size 5 --no-progress

    # Using Llama Stack for LLM and embeddings (set LLAMA_STACK_HOST, LLAMA_STACK_PORT)
    python ragas_dataset_eval_new.py ./ragas_dataset.json --model-id <llm> --embedding-model-id <emb> -o results.json
"""

import argparse
import asyncio
import json
import logging
import math
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple

# Env: RAGAS_EVAL_LOG_LEVEL (e.g. ERROR, WARNING) sets httpx/httpcore log level to reduce request log noise
def _apply_eval_log_level() -> None:
    name = os.environ.get("RAGAS_EVAL_LOG_LEVEL", "").strip().upper()
    if not name:
        return
    level = getattr(logging, name, None)
    if level is None:
        return
    for logger_name in ("httpx", "httpcore"):
        logging.getLogger(logger_name).setLevel(level)

# RAGAS imports are done inside run_ragas_evaluation_direct() so --help works without ragas installed


def _context_from_json_like(raw: Any) -> str:
    """
    Convert JSON-like context to key: value lines so RAGAS/LLM see readable text
    instead of raw JSON (avoids parser confusion from tool responses like calc_tax).
    """
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


def load_ragas_dataset(path: str) -> List[Dict[str, Any]]:
    """Load RAGAS dataset from a local JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"RAGAS dataset must be a JSON array of entries; got {type(data)}")
    return data


def convert_to_evaluation_format(ragas_dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert RAGAS dataset entries to RAGAS SingleTurnSample-compatible format.
    Skips entries with error, ERROR: answer, or missing question/answer
    (e.g. base dataset only has question/ground_truth; run ragas_dataset_generator first).
    """
    ragas_data = []
    for entry in ragas_dataset:
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
        ragas_entry: Dict[str, Any] = {
            "user_input": question,
            "response": answer,
            "retrieved_contexts": retrieved_contexts,
        }
        if "ground_truth" in entry and entry["ground_truth"]:
            ragas_entry["reference"] = entry["ground_truth"]
        ragas_data.append(ragas_entry)
    return ragas_data


def _get_llama_stack_client(timeout: int = 600):
    """Build Llama Stack client from environment. Requires LLAMA_STACK_HOST and LLAMA_STACK_PORT."""
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
    http_client = httpx.Client(verify=False, timeout=timeout)
    return LlamaStackClient(base_url=base_url, http_client=http_client)


class _LlamaStackEmbeddings:
    """LangChain-compatible embeddings using Llama Stack's embeddings API."""

    def __init__(self, client: Any, model_id: str):
        self._client = client
        self._model_id = model_id

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        out = []
        for text in texts:
            emb = self._embed_one(text)
            out.append(emb)
        return out

    def embed_query(self, text: str) -> List[float]:
        return self._embed_one(text)

    def _embed_one(self, text: str) -> List[float]:
        resp = self._client.embeddings.create(model=self._model_id, input=text)
        # Handle common response shapes: .data[0].embedding or .embeddings[0] or single list
        if hasattr(resp, "data") and resp.data:
            return getattr(resp.data[0], "embedding", resp.data[0])
        if hasattr(resp, "embeddings") and resp.embeddings:
            return resp.embeddings[0] if isinstance(resp.embeddings[0], list) else getattr(resp.embeddings[0], "embedding", resp.embeddings[0])
        if isinstance(resp, list):
            return resp
        raise RuntimeError(f"Unexpected embeddings response shape: {type(resp)}")


def _get_chat_openai_for_llama_stack(model_id: str):
    """
    Build ChatOpenAI pointing at Llama Stack's OpenAI-compatible endpoint.
    Uses LLAMA_STACK_HOST, LLAMA_STACK_PORT (and optional LLAMA_STACK_SECURE).
    RAGAS and LangChain expect this interface; avoids custom LLM and parser issues.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise ImportError(
            "langchain-openai is required for Llama Stack LLM. "
            "Install with: pip install langchain-openai"
        ) from e
    host = os.environ.get("LLAMA_STACK_HOST")
    port = os.environ.get("LLAMA_STACK_PORT")
    secure = os.environ.get("LLAMA_STACK_SECURE", "").lower() in ("true", "1", "yes")
    if not host:
        raise ValueError("LLAMA_STACK_HOST must be set when using Llama Stack for evaluation")
    if not port:
        raise ValueError("LLAMA_STACK_PORT must be set when using Llama Stack for evaluation")
    protocol = "https" if secure else "http"
    base_url = f"{protocol}://{host}:{port}/v1/openai/v1"
    api_key = os.environ.get("API_KEY", "fake")
    return ChatOpenAI(
        model=model_id,
        api_key=api_key,
        base_url=base_url,
        temperature=0.0,
    )


def _get_chat_openai_responses_for_llama_stack(model_id: str):
    """
    Build ChatOpenAI with use_responses_api=True (OpenAI Responses API).
    Same base_url as chat; LangChain uses the Responses API endpoint when this flag is set.
    Requires a server that exposes an OpenAI-compatible Responses API.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise ImportError(
            "langchain-openai is required for Llama Stack LLM. "
            "Install with: pip install langchain-openai"
        ) from e
    host = os.environ.get("LLAMA_STACK_HOST")
    port = os.environ.get("LLAMA_STACK_PORT")
    secure = os.environ.get("LLAMA_STACK_SECURE", "").lower() in ("true", "1", "yes")
    if not host:
        raise ValueError("LLAMA_STACK_HOST must be set when using Llama Stack for evaluation")
    if not port:
        raise ValueError("LLAMA_STACK_PORT must be set when using Llama Stack for evaluation")
    protocol = "https" if secure else "http"
    base_url = f"{protocol}://{host}:{port}/v1/openai/v1"
    api_key = os.environ.get("API_KEY", "fake")
    llm = ChatOpenAI(
        model=model_id,
        api_key=api_key,
        base_url=base_url,
        temperature=0.0,
        use_responses_api=True,
    )
    # OpenAI Responses API does not accept 'n'; RAGAS passes n for multiple completions. Strip it.
    # Responses API also returns AIMessage.content as list of blocks; RAGAS expects a string. Normalize.
    return _StripNLLMWrapper(llm)


def _normalize_ai_message_content(msg: Any) -> Any:
    """
    If msg is an AIMessage with content that is a list (Responses API format),
    return a new AIMessage with content as a single string. If msg is already
    a string (e.g. raw completion), return AIMessage(content=msg). Otherwise return msg unchanged.
    RAGAS expects .content to be a string (it does not handle list-of-blocks).
    """
    from langchain_core.messages import AIMessage

    if isinstance(msg, str):
        return AIMessage(content=msg)
    if not isinstance(msg, AIMessage):
        return msg
    content = getattr(msg, "content", None)
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if text is not None:
                    parts.append(str(text))
            elif hasattr(block, "content"):
                parts.append(str(block.content))
            elif hasattr(block, "text"):
                parts.append(str(block.text))
            elif isinstance(block, str):
                parts.append(block)
        content = "\n".join(parts) if parts else ""
        kwargs = {"content": content, "response_metadata": getattr(msg, "response_metadata", {})}
        if hasattr(msg, "id") and getattr(msg, "id", None) is not None:
            kwargs["id"] = msg.id
        return AIMessage(**kwargs)
    return msg


def _normalize_llm_result(result: Any) -> Any:
    """Normalize AIMessages inside an LLMResult so content is always a string. Build new generations so .text etc. are consistent."""
    if result is None:
        return result
    from langchain_core.outputs import ChatGeneration, LLMResult

    if not hasattr(result, "generations") or not isinstance(result.generations, list):
        return result
    new_batches = []
    for batch in result.generations:
        if not isinstance(batch, list):
            new_batches.append(batch)
            continue
        new_batch = []
        for gen in batch:
            if hasattr(gen, "message"):
                raw = gen.message
                norm_msg = _normalize_ai_message_content(raw)
                new_batch.append(ChatGeneration(message=norm_msg))
            else:
                new_batch.append(gen)
        new_batches.append(new_batch)
    return LLMResult(generations=new_batches)


class _StripNLLMWrapper:
    """
    Wraps an LLM and strips the 'n' kwarg before delegating. Used with
    ChatOpenAI(use_responses_api=True) because the OpenAI Responses API
    create() does not accept 'n'. Also normalizes AIMessage.content from
    list-of-blocks to string so RAGAS does not raise 'str' has no attribute 'content'.
    """

    def __init__(self, llm: Any):
        self._llm = llm

    def invoke(self, messages: List[Any], **kwargs: Any) -> Any:
        kwargs = {k: v for k, v in kwargs.items() if k != "n"}
        out = self._llm.invoke(messages, **kwargs)
        return _normalize_ai_message_content(out)

    async def ainvoke(self, messages: List[Any], **kwargs: Any) -> Any:
        kwargs = {k: v for k, v in kwargs.items() if k != "n"}
        out = await self._llm.ainvoke(messages, **kwargs)
        return _normalize_ai_message_content(out)

    def generate(
        self,
        messages: List[Any],
        stop: List[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        kwargs = {k: v for k, v in kwargs.items() if k != "n"}
        out = self._llm.generate(messages, stop=stop, **kwargs)
        return _normalize_llm_result(out)

    async def agenerate(
        self,
        messages: List[Any],
        stop: List[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        kwargs = {k: v for k, v in kwargs.items() if k != "n"}
        out = await self._llm.agenerate(messages, stop=stop, **kwargs)
        return _normalize_llm_result(out)

    def bind(self, **kwargs: Any) -> "_StripNLLMWrapper":
        """Wrap bound LLM so normalized output is still applied (RAGAS often uses llm.bind(...))."""
        bound = self._llm.bind(**kwargs)
        return _StripNLLMWrapper(bound)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)


def _messages_to_input(messages: List[Any]) -> str:
    """Convert LangChain messages to a single string for Responses API input."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    parts = []
    for m in messages:
        if isinstance(m, HumanMessage):
            parts.append(str(m.content))
        elif isinstance(m, SystemMessage):
            parts.append(str(m.content))
        elif isinstance(m, AIMessage):
            parts.append(str(m.content))
        else:
            parts.append(str(getattr(m, "content", m)))
    return "\n\n".join(parts) if parts else ""


class _LlamaStackResponsesLLM:
    """
    LangChain-compatible LLM using Llama Stack's Responses API (client.responses.create).
    Use with --use-responses-api when you need the non-deprecated route. Some RAGAS
    metrics may raise output parser errors (StringIO vs direct-schema).
    """

    _client: Any
    _model_id: str

    def __init__(self, client: Any, model_id: str):
        self._client = client
        self._model_id = model_id

    def invoke(self, messages: List[Any], **kwargs: Any) -> Any:
        result = self._generate(messages, run_manager=None, **kwargs)
        return result.generations[0][0].message

    def generate(
        self,
        messages: List[Any],
        stop: List[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        return self.agenerate(messages, stop=stop, **kwargs)

    async def agenerate(
        self,
        messages: List[Any],
        stop: List[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._generate(messages, stop=stop, run_manager=None, **kwargs),
        )

    def _generate(
        self,
        messages: List[Any],
        stop: List[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, LLMResult

        prompt = _messages_to_input(messages)
        n = max(1, min(int(kwargs.get("n", 1)), 5))
        generations: List[ChatGeneration] = []
        for _ in range(n):
            resp = self._client.responses.create(
                model=self._model_id,
                input=prompt,
            )
            text = getattr(resp, "output_text", None) or getattr(resp, "output", str(resp))
            if hasattr(text, "content"):
                text = getattr(text, "content", str(text))
            if not isinstance(text, str):
                text = json.dumps(text) if isinstance(text, (dict, list)) else str(text)
            generations.append(ChatGeneration(message=AIMessage(content=text)))
        return LLMResult(generations=[generations])

    @property
    def _llm_type(self) -> str:
        return "llama_stack_responses"


def _get_llama_stack_llm_and_embeddings(
    model_id: str,
    embedding_model_id: str,
    timeout: int = 600,
    use_responses_api: bool = False,
    use_openai_responses_api: bool = False,
) -> Tuple[Any, Any]:
    """
    Build LangChain-compatible LLM and embeddings for Llama Stack.
    - use_openai_responses_api: ChatOpenAI(use_responses_api=True) — OpenAI Responses API.
    - use_responses_api: custom wrapper around client.responses.create() — Llama Stack native.
    - else: ChatOpenAI() at /v1/openai/v1 — chat completions.
    Embeddings always use Llama Stack's embeddings API.
    """
    client = _get_llama_stack_client(timeout=timeout)
    if use_openai_responses_api:
        llm = _get_chat_openai_responses_for_llama_stack(model_id)
    elif use_responses_api:
        llm = _LlamaStackResponsesLLM(client, model_id)
    else:
        llm = _get_chat_openai_for_llama_stack(model_id)
    embeddings = _LlamaStackEmbeddings(client, embedding_model_id)
    return llm, embeddings


def _import_ragas():
    """Import RAGAS only when needed."""
    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        # Use pre-initialised metric instances (llm/embeddings=None); evaluate() assigns our llm/embeddings
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
        raise ImportError(
            "ragas is required. Install with: pip install ragas"
        ) from e


def get_metric_objects(metrics_list: List[str], ragas_metrics: Dict[str, Any]):
    """Resolve comma-separated metric names to RAGAS metric objects."""
    resolved = []
    for name in metrics_list:
        name = name.strip().lower()
        if name not in ragas_metrics:
            raise ValueError(
                f"Unknown metric '{name}'. Available: {', '.join(ragas_metrics)}"
            )
        resolved.append(ragas_metrics[name])
    return resolved


def run_ragas_evaluation_direct(
    ragas_dataset: List[Dict[str, Any]],
    metrics_list: List[str],
    batch_size: int | None = None,
    show_progress: bool = True,
    model_id: str | None = None,
    embedding_model_id: str | None = None,
    use_responses_api: bool = False,
    use_openai_responses_api: bool = False,
) -> Dict[str, Any]:
    """
    Run RAGAS evaluation using the RAGAS library directly (no benchmark API).
    """
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
                "(has 'question' but no 'answer'). Generate a RAGAS dataset first with "
                "ragas_dataset_generator.py, then run this evaluator on the generated file."
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

    # Llama Stack: when both model_id and embedding_model_id are set
    llm, embeddings = None, None
    if model_id or embedding_model_id:
        if not (model_id and embedding_model_id):
            raise ValueError(
                "When using Llama Stack for RAGAS, both --model-id and --embedding-model-id are required. "
                "Set LLAMA_STACK_HOST and LLAMA_STACK_PORT."
            )
        if use_openai_responses_api:
            print(
                "[WARN] --use-openai-responses-api is used: RAGAS may raise "
                "AttributeError('str' object has no attribute 'content'). "
                "Use default (chat completions) if that happens."
            )
            llm_kind = "OpenAI Responses API (ChatOpenAI)"
        elif use_responses_api:
            llm_kind = "Llama Stack native responses API"
        else:
            llm_kind = "chat (OpenAI-compatible)"
        print(f"[LLAMA-STACK] Using LLM: {model_id} ({llm_kind}), embeddings: {embedding_model_id}")
        llm, embeddings = _get_llama_stack_llm_and_embeddings(
            model_id,
            embedding_model_id,
            use_responses_api=use_responses_api,
            use_openai_responses_api=use_openai_responses_api,
        )

    # Build RAGAS EvaluationDataset from SingleTurnSample instances
    samples = [SingleTurnSample(**entry) for entry in ragas_data]
    eval_dataset = EvaluationDataset(samples=samples)
    print("[START] Running RAGAS evaluate()...")

    eval_kw: Dict[str, Any] = {
        "metrics": metric_objects,
        "show_progress": show_progress,
        "batch_size": batch_size,
    }
    if llm is not None:
        eval_kw["llm"] = llm
    if embeddings is not None:
        eval_kw["embeddings"] = embeddings

    result = evaluate_fn(eval_dataset, **eval_kw)

    # result.scores: list of dicts, one per row, e.g. [{"faithfulness": 0.9, ...}, ...]
    scores_rows = result.scores
    if not scores_rows:
        raise RuntimeError("RAGAS evaluate() returned no scores")

    # Aggregate: average per metric
    final_metrics: Dict[str, float] = {}
    individual_scores: Dict[str, List[float]] = {m: [] for m in metrics_list}

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
    formatted = {
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
        "use_responses_api": use_responses_api,
        "use_openai_responses_api": use_openai_responses_api,
    }

    print("\n" + "=" * 70)
    print("[RESULTS] RAGAS EVALUATION (direct)")
    print("=" * 70)
    print(f"Run:      {formatted['benchmark_id']}")
    print(f"Dataset:  {formatted['dataset_size']} entries ({formatted['valid_entries']} evaluated)")
    if formatted["metrics"]:
        print("\n[METRICS]")
        for metric, score in formatted["metrics"].items():
            if isinstance(score, float) and math.isnan(score):
                status, val = "[SKIP]", "N/A"
            else:
                val = f"{score:.4f}"
                status = "[PASS]" if score > 0.8 else "[WARN]" if score > 0.6 else "[FAIL]"
            print(f"  {status} {metric:25s}: {val}")
    print("=" * 70)
    return formatted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on a RAGAS dataset using the RAGAS library directly",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "ragas_dataset",
        nargs="?",
        default=None,
        metavar="PATH",
        help="Path to RAGAS dataset JSON file",
    )
    parser.add_argument(
        "-i", "--input",
        dest="input_path",
        metavar="PATH",
        default=None,
        help="Path to RAGAS dataset JSON (alternative to positional)",
    )
    parser.add_argument(
        "-o", "--output",
        default="ragas_eval_results.json",
        help="Output path for evaluation results JSON",
    )
    parser.add_argument(
        "--metrics",
        default="faithfulness,answer_relevancy,context_precision,context_recall",
        help="Comma-separated RAGAS metrics",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="RAGAS evaluate() batch size (default: no batching)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar during evaluation",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Judge LLM model (recorded in output; direct RAGAS uses default LLM unless configured)",
    )
    parser.add_argument(
        "--embedding-model-id",
        default=None,
        help="Embedding model (recorded in output; direct RAGAS uses default embeddings unless configured)",
    )
    parser.add_argument(
        "--use-openai-responses-api",
        action="store_true",
        help="Use OpenAI Responses API (ChatOpenAI use_responses_api=True). Often fails with RAGAS; prefer default chat completions.",
    )
    parser.add_argument(
        "--use-responses-api",
        action="store_true",
        help="Use Llama Stack native Responses API (client.responses.create). Some RAGAS metrics may hit parser errors.",
    )

    args = parser.parse_args()
    args.input_path = args.input_path or args.ragas_dataset
    return args


def main() -> int:
    args = parse_args()
    if not args.input_path:
        print("Error: Provide a RAGAS dataset path (positional or -i/--input).", file=sys.stderr)
        return 1

    print(f"[LOAD] {args.input_path}")
    ragas_dataset = load_ragas_dataset(args.input_path)
    print(f"[OK] Loaded {len(ragas_dataset)} entries")

    metrics_list = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if not metrics_list:
        print("Error: At least one metric required (--metrics).", file=sys.stderr)
        return 1

    results = run_ragas_evaluation_direct(
        ragas_dataset=ragas_dataset,
        metrics_list=metrics_list,
        batch_size=args.batch_size,
        show_progress=not args.no_progress,
        model_id=args.model_id,
        embedding_model_id=args.embedding_model_id,
        use_responses_api=args.use_responses_api,
        use_openai_responses_api=args.use_openai_responses_api,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Results written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
