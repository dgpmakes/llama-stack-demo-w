Set project:

```sh
export PROJECT=llama-stack-demo

oc new-project ${PROJECT}
oc label namespace ${PROJECT} modelmesh-enabled=false opendatahub.io/dashboard=true

export APPS_DOMAIN=$(oc get ingresses.config.openshift.io cluster -o jsonpath='{.spec.domain}')

```

```sh
# export EMBEDDING_MODEL="granite-embedding-125m"
export EMBEDDING_MODEL="sentence-transformers/nomic-ai/nomic-embed-text-v1.5"
export EMBEDDING_DIMENSION="768"
export EMBEDDING_MODEL_PROVIDER="sentence-transformers"
export VECTOR_STORE_NAME="rag-store"
export VECTOR_STORE_PROVIDER_ID="milvus"
export RANKER="default"
export SCORE_THRESHOLD="0.8"
export MAX_NUM_RESULTS="10"
export TEST_QUERY="Tell me about taxes in Lysmark."
export DOCS_FOLDER="./docs"
export CHUNK_SIZE_IN_TOKENS="800"
export CHUNK_OVERLAP_IN_TOKENS="400"
export LLAMA_STACK_HOST="llama-stack-demo-route-${PROJECT}.${APPS_DOMAIN}"
export LLAMA_STACK_PORT="443"
export LLAMA_STACK_SECURE="True"
export LOG_LEVEL="DEBUG"
```

Run as code:

```sh
cd client
uv sync
source .venv/bin/activate

./run.sh load

./run.sh agent --input 'My mother had an accident, can I get access to the unpaid leave aid in Lysmark? How much would I get monthly?' --model llama-3-1-8b-w4a16/llama-3-1-8b-w4a16 --agent-type default
```

**Streamlit Chat (app2 – portazgo):**

```sh
cd client
uv sync
uv run streamlit run app2.py
```

Run as an image:

```sh
./image.sh cmd load

./image.sh cmd agent --input 'My mother had an accident, can I get access to the unpaid leave aid in Lysmark? How much would I get monthly?' --model llama-3-1-8b-w4a16/llama-3-1-8b-w4a16 --agent-type default
```



```sh
SYSTEM_INSTRUCTIONS="You are a helpful AI assistant that uses tools to help citizens of the Republic of Lysmark. Answers should be concise and human readable. AVOID references to tools or function calling nor show any JSON. Infer parameters for function calls or instead use default values or request the needed information from the user. Call the RAG tool first if unsure. Parameter single_parent_family only is necessary if birth/adoption/foster_care otherwise use false."
MODEL_NAME="llama-3-1-8b-w4a16"
AGENT_TYPE="lang_chain"

curl -X 'POST' \
  'http://0.0.0.0:8700/agents/execute' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "agent_type": "'"${AGENT_TYPE}"'",
  "input_text": "Tell me how much taxes do I have to pay if my yearly income is 43245€",
  "model_name": "'"${MODEL_NAME}"'",
  "system_instructions": "'"${SYSTEM_INSTRUCTIONS}"'",
  "tools": [
      { 
        "type": "mcp",
        "server_label": "dmcp",
        "server_url": "https://compatibility-engine-llama-stack-demo.apps.ocp.sandbox3322.opentlc.com/sse",
        "transport": "sse",
        "require_approval": "never"
      }
  ]
}'
```