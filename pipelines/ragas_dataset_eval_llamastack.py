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
RAGAS Dataset Evaluator (CLI)

Runs RAGAS evaluation on a RAGAS-compatible dataset using the Llama Stack
evaluation API (same logic as run_ragas_evaluation in ragas_pipeline.py).
Reads a RAGAS dataset JSON, converts to evaluation format, registers dataset
and benchmark, runs evaluation via client.alpha.eval.run_eval, and writes
aggregated metrics and per-row results to a JSON file.

Usage:
    # Evaluate a generated RAGAS dataset (default metrics, output ragas_eval_results.json)
    python ragas_dataset_eval.py ./ragas_dataset.json

    # Custom metrics and output path
    python ragas_dataset_eval.py ./ragas_dataset.json -o results.json \\
        --metrics "faithfulness,answer_relevancy,context_precision,context_recall"

    # With explicit model IDs (required for RAGAS)
    python ragas_dataset_eval.py ./ragas_dataset.json \\
        --model-id llama-3-1-8b-w4a16/llama-3-1-8b-w4a16 \\
        --embedding-model-id BAAI/bge-m3

    # Batch evaluation and longer wait
    python ragas_dataset_eval.py ./ragas_dataset.json --batch-size 5 --max-wait-seconds 1200

Requires: LLAMA_STACK_HOST, LLAMA_STACK_PORT (and optionally LLAMA_STACK_SECURE).
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

import httpx
from llama_stack_client import LlamaStackClient


def get_llama_stack_client(timeout: int = 600) -> LlamaStackClient:
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


def load_ragas_dataset(path: str) -> List[Dict[str, Any]]:
    """Load RAGAS dataset from a local JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"RAGAS dataset must be a JSON array of entries; got {type(data)}")
    return data


def convert_to_evaluation_format(ragas_dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert RAGAS dataset entries to Llama Stack evaluation format.
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
        ragas_entry = {
            "user_input": question,
            "response": answer,
            "retrieved_contexts": raw_contexts,
        }
        if "ground_truth" in entry and entry["ground_truth"]:
            ragas_entry["reference"] = entry["ground_truth"]
        ragas_data.append(ragas_entry)
    return ragas_data


def run_ragas_evaluation(
    ragas_dataset: List[Dict[str, Any]],
    client: LlamaStackClient,
    model_id: str,
    embedding_model_id: str,
    metrics_list: List[str],
    mode: str = "inline",
    batch_size: int = 0,
    timeout: int = 600,
    max_wait_seconds: int = 900,
    poll_interval: int = 5,
) -> Dict[str, Any]:
    """
    Run RAGAS evaluation via Llama Stack evaluation API.
    Same logic as run_ragas_evaluation in ragas_pipeline.py (without KFP artifacts).
    """
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

    provider_id = "trustyai_ragas_inline" if mode == "inline" else "trustyai_ragas_remote"
    if batch_size and batch_size > 0:
        batches = [ragas_data[i : i + batch_size] for i in range(0, len(ragas_data), batch_size)]
    else:
        batches = [ragas_data]

    print(f"[START] Evaluation in {len(batches)} batch(es) ({mode.upper()} mode)...")
    base_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    aggregated_scores: Dict[str, List[tuple]] = {m: [] for m in metrics_list}
    all_results: Dict[str, Any] = {
        "metrics": {},
        "individual_scores": {},
        "generations": [],
        "failures": [],
    }

    for batch_idx, batch in enumerate(batches):
        print(f"\n[BATCH] {batch_idx + 1}/{len(batches)} (size {len(batch)})")
        dataset_id = f"ragas_dataset_{datetime.now().isoformat()}_{batch_idx}"
        benchmark_id = f"ragas_benchmark_{datetime.now().isoformat()}_{batch_idx}"

        try:
            print(f"[REGISTER] Dataset: {dataset_id}")
            client.datasets.register(
                dataset_id=dataset_id,
                purpose="eval/question-answer",
                source={"type": "rows", "rows": batch},
            )
            print(f"[REGISTER] Benchmark: {benchmark_id}")
            client.benchmarks.register(
                benchmark_id=benchmark_id,
                dataset_id=dataset_id,
                scoring_functions=metrics_list,
                provider_id=provider_id,
            )

            eval_candidate = {
                "type": "model",
                "model": model_id,
                "sampling_params": {"strategy": {"type": "greedy"}, "temperature": 0.1, "max_tokens": 2048},
            }
            scoring_params = {
                m: {"type": "basic", "aggregation_functions": ["average"]} for m in metrics_list
            }
            benchmark_config = {
                "eval_candidate": eval_candidate,
                "scoring_params": scoring_params,
                # "num_examples": 1,
            }
            extra_body = {
                "provider_id": provider_id,
                "judge_model": model_id,
                "model": model_id,
                "embedding_model": embedding_model_id,
                "embeddings": embedding_model_id,
            }

            print("[RUN] Running evaluation...")
            job = client.alpha.eval.run_eval(
                benchmark_id=benchmark_id,
                benchmark_config=benchmark_config,
                extra_body=extra_body,
            )
            print(f"[OK] Job started: {job.job_id}")

            print("[WAIT] Waiting for results...")
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
                    print(f"   status: {status_value} ({waited}s)")

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
                        details = error_msg or error_detail or error_info or "No error details"
                        raise RuntimeError(
                            f"Job {job.job_id} failed: {status_value}. Details: {details}"
                        )
                except RuntimeError:
                    raise
                except Exception as e:
                    print(f"   Waiting... ({waited}s) - {e}")

            if batch_result is None:
                raise RuntimeError(f"Job {job.job_id} did not complete within {max_wait_seconds}s")

            scores = getattr(batch_result, "scores", None)
            if scores:
                scores_dict = scores if isinstance(scores, dict) else dict(scores)
                for metric, score_data in scores_dict.items():
                    score_val = 0.0
                    if hasattr(score_data, "aggregated_results"):
                        agg = score_data.aggregated_results
                        score_val = (
                            agg.get("average", agg.get(metric, 0.0))
                            if isinstance(agg, dict)
                            else (float(agg) if agg else 0.0)
                        )
                    elif isinstance(score_data, dict) and "aggregated_results" in score_data:
                        agg = score_data["aggregated_results"]
                        score_val = agg.get("average", agg.get(metric, 0.0))
                    elif isinstance(score_data, (int, float)):
                        score_val = float(score_data)
                    aggregated_scores.setdefault(metric, []).append((score_val, len(batch)))

                    score_rows = None
                    if hasattr(score_data, "score_rows"):
                        score_rows = score_data.score_rows
                    elif isinstance(score_data, dict) and "score_rows" in score_data:
                        score_rows = score_data["score_rows"]
                    if score_rows:
                        if metric not in all_results["individual_scores"]:
                            all_results["individual_scores"][metric] = []
                        for row in score_rows:
                            s = row.get("score", 0.0) if isinstance(row, dict) else getattr(row, "score", 0.0)
                            all_results["individual_scores"][metric].append(s)

            generations = getattr(batch_result, "generations", None)
            if generations:
                all_results["generations"].extend(list(generations))

        except Exception as e:
            print(f"[ERROR] Batch {batch_idx + 1}: {e}")
            all_results["failures"].append({"batch_index": batch_idx + 1, "error": str(e)})

    total_batches = len(batches)
    failed_batches = len(all_results["failures"])
    if failed_batches == total_batches:
        raise ValueError(
            f"All {total_batches} batch(es) failed. "
            f"First: {all_results['failures'][0]['error'] if all_results['failures'] else 'Unknown'}"
        )
    if failed_batches > total_batches / 2:
        raise ValueError(
            f"Too many failures: {failed_batches}/{total_batches}. "
            f"Errors: {[f['error'] for f in all_results['failures'][:3]]}"
        )

    print("\n[SUM] Aggregating...")
    final_metrics = {}
    for metric, values in aggregated_scores.items():
        if not values:
            continue
        total_score = sum(v[0] * v[1] for v in values)
        total_count = sum(v[1] for v in values)
        final_metrics[metric] = total_score / total_count if total_count else 0.0

    formatted = {
        "benchmark_id": f"ragas_eval_{base_timestamp}",
        "timestamp": datetime.now().isoformat(),
        "metrics": final_metrics,
        "individual_scores": all_results["individual_scores"],
        "generations": all_results["generations"],
        "failures": all_results["failures"],
        "dataset_size": len(ragas_dataset),
        "valid_entries": len(ragas_data),
        "mode": mode,
        "model_id": model_id,
        "embedding_model_id": embedding_model_id,
    }

    print("\n" + "=" * 70)
    print("[RESULTS] RAGAS EVALUATION")
    print("=" * 70)
    print(f"Benchmark: {formatted['benchmark_id']}")
    print(f"Dataset:   {formatted['dataset_size']} entries ({formatted['valid_entries']} evaluated)")
    if formatted.get("failures"):
        for f in formatted["failures"]:
            print(f"  Failure batch {f.get('batch_index')}: {f.get('error')}")
    if formatted["metrics"]:
        print("\n[METRICS]")
        for metric, score in formatted["metrics"].items():
            status = "[PASS]" if score > 0.8 else "[WARN]" if score > 0.6 else "[FAIL]"
            print(f"  {status} {metric:25s}: {score:.4f}")
    print("=" * 70)
    return formatted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on a RAGAS dataset via Llama Stack evaluation API",
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
        "--model-id",
        default="llama-3-1-8b-w4a16/llama-3-1-8b-w4a16",
        help="Judge LLM model for RAGAS scoring",
    )
    parser.add_argument(
        "--embedding-model-id",
        default="BAAI/bge-m3",
        help="Embedding model for RAGAS (e.g. answer_relevancy)",
    )
    parser.add_argument(
        "--metrics",
        default="faithfulness,answer_relevancy,context_precision,context_recall",
        help="Comma-separated RAGAS metrics",
    )
    parser.add_argument(
        "--mode",
        choices=["inline", "remote"],
        default="inline",
        help="Evaluation mode (trustyai_ragas_inline vs trustyai_ragas_remote)",
    )
    parser.add_argument("--batch-size", type=int, default=0, help="Batch size (0 = all at once)")
    parser.add_argument("--timeout", type=int, default=600, help="HTTP timeout (seconds)")
    parser.add_argument("--max-wait-seconds", type=int, default=900, help="Max wait for evaluation job")
    parser.add_argument("--poll-interval", type=int, default=5, help="Job status poll interval (seconds)")

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
    print(f"[METRICS] {', '.join(metrics_list)}")

    if not (args.model_id and str(args.model_id).strip()):
        print("Error: model_id (judge LLM) is required for RAGAS.", file=sys.stderr)
        return 1
    if not (args.embedding_model_id and str(args.embedding_model_id).strip()):
        print("Error: embedding_model_id is required for RAGAS.", file=sys.stderr)
        return 1

    client = get_llama_stack_client(timeout=args.timeout)
    host = os.environ.get("LLAMA_STACK_HOST", "")
    port = os.environ.get("LLAMA_STACK_PORT", "")
    print(f"[CONFIG] Llama Stack: {host}:{port}")

    results = run_ragas_evaluation(
        ragas_dataset=ragas_dataset,
        client=client,
        model_id=args.model_id,
        embedding_model_id=args.embedding_model_id,
        metrics_list=metrics_list,
        mode=args.mode,
        batch_size=args.batch_size,
        timeout=args.timeout,
        max_wait_seconds=args.max_wait_seconds,
        poll_interval=args.poll_interval,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Results written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
