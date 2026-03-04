**Base dataset** (e.g. `base_dataset_small.json`): JSON array of `{ "id", "question", "ground_truth" }`. Optional: `difficulty`, `expected_tool`, `expected_tool_parameters` (for tool-only questions: which tool and params you expect; stored in the generated RAGAS dataset for reference).

**Generated RAGAS dataset**: each entry has `id`, `question`, `answer`, `contexts` (or `["No context retrieved"]` when none), `ground_truth`. If the model used tools, `tool_calls` is added: list of `{ "tool_name", "arguments", "response" }`. Optional `expected_tool` / `expected_tool_parameters` are copied from the base dataset when present. Entries with no contexts (e.g. tool-only answers) are no longer skipped at evaluation; context metrics may be 0 for those rows.

**RAGAS evaluation (e.g. in-pipeline run)**: Needs both a **judge LLM** (`model_id`) and an **embedding model** (`embedding_model_id`). If you see `'NoneType' object has no attribute 'generate'`, the judge model was not set or not found—ensure `model_id` is set and the LLM is deployed and available to the Llama Stack server. For `answer_relevancy`, the server must have `TRUSTYAI_EMBEDDING_MODEL` set (e.g. `ragDefaults.enableRagas: true` and `ragDefaults.embeddingModel` in Helm). Pipeline params `model_id` and `embedding_model_id` must match models available to the server.

Set project:

```sh
export PROJECT=llama-stack-demo

export APPS_DOMAIN=$(oc get ingresses.config.openshift.io cluster -o jsonpath='{.spec.domain}')

```

```sh
export EMBEDDING_MODEL_ID="sentence-transformers/nomic-ai/nomic-embed-text-v1.5"
# export MODEL_ID="llama-4-scout-17b-16e-w4a16/Llama-4-Scout-17B-16E-W4A16"
export MODEL_ID="qwen3-8b-fp8-dynamic/qwen3-8b-fp8-dynamic"
export VECTOR_STORE_NAME="rag-store-pgvector"
export RANKER="default"
export SCORE_THRESHOLD="0.0"
export FILE_SEARCH_MAX_CHUNKS="10"
export FILE_SEARCH_MAX_TOKENS_PER_CHUNK="0"
export LLAMA_STACK_HOST="llama-stack-demo-route-${PROJECT}.${APPS_DOMAIN}"
export LLAMA_STACK_PORT="443"
export LLAMA_STACK_SECURE="True"
export LOG_LEVEL="DEBUG"
export SEARCH_MODE="vector"
export INSTRUCTIONS="You are a helpful AI assistant that uses tools to help citizens of the Republic of Lysmark. Answers should be concise and human readable. AVOID references to tools or function calling nor show any JSON. Infer parameters for function calls or instead use default values or request the needed information from the user."
export TOOLS="compatibility-engine,cluster-insights,eligibility-engine"

```

Run:

```sh
AGENT_TYPE=default
AGENT_PATTERN=plan_execute
SUFFIX="${AGENT_TYPE}-${AGENT_PATTERN}"
# Generate RAGAS dataset (base dataset from file; Llama Stack connection and options from env)
uv run python ragas_dataset_generator.py ../materials/datasets/base_dataset_small.json \
  --vector-store-name "$VECTOR_STORE_NAME" \
  --model-id "$MODEL_ID" \
  --instructions "${INSTRUCTIONS}" \
  --tools "$TOOLS" \
  --retrieval-mode ${SEARCH_MODE} \
  --file-search-max-chunks "${FILE_SEARCH_MAX_CHUNKS:-10}" \
  --file-search-score-threshold "${SCORE_THRESHOLD:-0.0}" \
  --file-search-max-tokens-per-chunk "${FILE_SEARCH_MAX_TOKENS_PER_CHUNK:-512}" \
  --agent-type ${AGENT_TYPE} --pattern simple \
  -o ragas_dataset-${SUFFIX}.json
```

Optional: `--agent-type default|lang-graph` (default: `default`), `--pattern simple|plan_execute` (default: `simple`). Use a specific base dataset path if needed (e.g. `materials/datasets/base_dataset_small.json`). `LLAMA_STACK_HOST`, `LLAMA_STACK_PORT`, and `LLAMA_STACK_SECURE` are read from the environment by the script.

# Run RAGAS evaluation of a dataset

```sh
export EMBEDDING_MODEL_ID="sentence-transformers/nomic-ai/nomic-embed-text-v1.5"
export MODEL_ID="llama-4-scout-17b-16e-w4a16/Llama-4-Scout-17B-16E-W4A16"
export METRICS="faithfulness,answer_relevancy,context_precision,context_recall"
uv run ragas_dataset_eval.py ./scratch/ragas_dataset.json \
  --metrics $METRICS \
  --model-id $MODEL_ID \
  --embedding-model-id $EMBEDDING_MODEL_ID
```