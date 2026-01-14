# Deploying LLM-D Service on OpenShift AI - Complete Guide

This guide walks you through deploying a Large Language Model inference service using Red Hat OpenShift AI's LLM-D (Distributed Inference) framework with proper networking, TLS certificates, and GPU resources.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Architecture Overview](#architecture-overview)
- [Step 1: Verify Prerequisites](#step-1-verify-prerequisites)
- [Step 2: Create Namespace and Kueue Resources](#step-2-create-namespace-and-kueue-resources)
- [Step 3: Create GatewayClass](#step-3-create-gatewayclass)
- [Step 4: Create Let's Encrypt Certificate](#step-4-create-lets-encrypt-certificate)
- [Step 5: Create Gateway](#step-5-create-gateway)
- [Step 6: Deploy LLMInferenceService](#step-6-deploy-llminferenceservice)
- [Step 7: Verify Deployment](#step-7-verify-deployment)
- [Step 8: Test Inference Endpoints](#step-8-test-inference-endpoints)
- [Step 9: Access from Browser](#step-9-access-from-browser)
- [Troubleshooting Guide](#troubleshooting-guide)
- [Cleanup](#cleanup)
- [Performance Tuning](#performance-tuning)
- [Additional Resources](#additional-resources)

---

## Prerequisites

Before starting, ensure you have:

- OpenShift cluster (version 4.19.9+) with OpenShift AI installed
- GPU nodes available in the cluster
- cert-manager operator installed
- Kueue installed for GPU queue management
- Administrative access to the cluster
- `oc` CLI tool configured and logged in

---

## Architecture Overview

The deployment consists of the following components:

| Component | Purpose |
|-----------|---------|
| **Kueue LocalQueue** | Manages GPU resource allocation and queueing |
| **LLMInferenceService** | Deploys the Qwen3-8B model with 2 replicas |
| **Gateway API** | Provides ingress routing with TLS termination |
| **Let's Encrypt Certificate** | Automated TLS certificate management |
| **HTTPRoute** | Path-based routing to inference service |

```
┌─────────────────────────────────────────────────────────────┐
│                    Internet/Users                            │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTPS
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              Gateway (TLS Termination)                       │
│         inference.apps.<cluster-domain>                      │
│           (Let's Encrypt Certificate)                        │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                    HTTPRoute                                 │
│      /llm-d-demo/qwen3-8b-fp8-dynamic-llmis                 │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              LLMInferenceService                             │
│         (2 replicas with GPU resources)                      │
│              Qwen3-8B-FP8-dynamic                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Step 1: Verify Prerequisites

### 1.1 Check cert-manager Installation

Verify that cert-manager is installed and running:

```bash
# Verify cert-manager is running
oc get pods -n cert-manager

# Expected output: 3 pods running
# NAME                                       READY   STATUS    RESTARTS   AGE
# cert-manager-697fdccc44-xxxxx              1/1     Running   0          2d
# cert-manager-cainjector-7dbf76d5c8-xxxxx   1/1     Running   0          2d
# cert-manager-webhook-7894b5b9b4-xxxxx      1/1     Running   0          2d
```

### 1.2 Check ClusterIssuers

Verify that Let's Encrypt ClusterIssuers are configured:

```bash
# List available ClusterIssuers
oc get clusterissuer

# Expected output:
# NAME                  READY   AGE
# letsencrypt           True    2d
# letsencrypt-staging   True    2d
```

> **Note**: If ClusterIssuers don't exist, you'll need to create them first. Refer to cert-manager documentation.

### 1.3 Check GPU Nodes

Verify GPU nodes are available and properly labeled:

```bash
# Verify GPU nodes are available
oc get nodes -l nvidia.com/gpu.present=true

# Check GPU resources on a specific node
oc describe node <gpu-node-name> | grep nvidia.com/gpu

# Expected output should show:
#   nvidia.com/gpu: 1 (or more)
```

### 1.4 Get Cluster Domain

Get your OpenShift cluster's application domain:

```bash
# Get your OpenShift cluster's apps domain
CLUSTER_DOMAIN=$(oc get ingresses.config.openshift.io cluster -o jsonpath='{.spec.domain}')
echo "Cluster domain: ${CLUSTER_DOMAIN}"

# Example output: apps.ocp.sandbox1278.opentlc.com
```

> **Important**: Save this value - you'll need it throughout the deployment.

---

## Step 2: Create Namespace and Kueue Resources

### 2.1 Create Namespace

Create a dedicated namespace for your LLM service:

```bash
# Create the namespace
oc create namespace llm-d-demo

# Verify namespace creation
oc get namespace llm-d-demo

# Expected output:
# NAME         STATUS   AGE
# llm-d-demo   Active   5s
```

### 2.2 Create LocalQueue

The LocalQueue manages GPU resource allocation for your workloads.

Create a file named `local-queue.yaml`:

```yaml
apiVersion: kueue.x-k8s.io/v1beta1
kind: LocalQueue
metadata:
  name: gpu-queue
  namespace: llm-d-demo
spec:
  clusterQueue: cluster-queue
```

Apply the LocalQueue:

```bash
# Apply the LocalQueue
oc apply -f local-queue.yaml

# Verify LocalQueue creation
oc get localqueue -n llm-d-demo

# Expected output:
# NAME        CLUSTERQUEUE    PENDING WORKLOADS   ADMITTED WORKLOADS
# gpu-queue   cluster-queue   0                   0
```

> **Note**: Ensure the `cluster-queue` ClusterQueue exists and has GPU resources allocated. Check with:
> ```bash
> oc get clusterqueue cluster-queue -o yaml
> ```

**Testing Checkpoint**: ✅ Verify LocalQueue is created and references a valid ClusterQueue.

---

## Step 3: Create GatewayClass

### 3.1 Create GatewayClass for AI Inference

The GatewayClass defines a class of Gateway resources for OpenShift AI inference services.

Create a file named `gatewayclass.yaml`:

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: GatewayClass
metadata:
  name: openshift-ai-inference
spec:
  controllerName: openshift.io/gateway-controller/v1
  description: GatewayClass for OpenShift AI LLM inference services
```

Apply the GatewayClass:

```bash
# Create the GatewayClass
oc apply -f gatewayclass.yaml

# Verify GatewayClass is accepted
oc get gatewayclass openshift-ai-inference

# Expected output:
# NAME                     CONTROLLER                           ACCEPTED   AGE
# openshift-ai-inference   openshift.io/gateway-controller/v1   True       10s
```

**Testing Checkpoint**: ✅ Verify `ACCEPTED` column shows `True`.

---

## Step 4: Create Let's Encrypt Certificate

### 4.1 Create Certificate Resource

The Certificate resource requests a TLS certificate from Let's Encrypt.

Create a file named `certificate.yaml`:

> **Important**: Replace `<CLUSTER_DOMAIN>` with your actual cluster domain from Step 1.4.

```yaml
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: openshift-ai-inference-cert
  namespace: openshift-ingress
spec:
  secretName: openshift-ai-inference-tls
  issuerRef:
    name: letsencrypt  # Use 'letsencrypt-staging' for testing
    kind: ClusterIssuer
  dnsNames:
  - "inference.apps.<CLUSTER_DOMAIN>"  # Replace with your domain
  duration: 2160h  # 90 days
  renewBefore: 360h  # Renew 15 days before expiration
```

**Or** create it dynamically using your cluster domain:

```bash
# Create certificate with actual cluster domain
cat <<EOF | oc apply -f -
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: openshift-ai-inference-cert
  namespace: openshift-ingress
spec:
  secretName: openshift-ai-inference-tls
  issuerRef:
    name: letsencrypt
    kind: ClusterIssuer
  dnsNames:
  - "inference.apps.${CLUSTER_DOMAIN}"
  duration: 2160h
  renewBefore: 360h
EOF
```

### 4.2 Monitor Certificate Issuance

Wait for the certificate to be issued (typically 1-2 minutes):

```bash
# Watch certificate status (Ctrl+C to exit)
oc get certificate -n openshift-ingress -w

# Expected output when ready:
# NAME                          READY   SECRET                       AGE
# openshift-ai-inference-cert   True    openshift-ai-inference-tls   2m
```

### 4.3 Verify Certificate Details

```bash
# Check certificate details
oc describe certificate openshift-ai-inference-cert -n openshift-ingress

# Look for:
#   Status:
#     Conditions:
#       Type:    Ready
#       Status:  True

# Verify the secret was created
oc get secret openshift-ai-inference-tls -n openshift-ingress

# Check certificate DNS names
oc get certificate openshift-ai-inference-cert -n openshift-ingress \
  -o jsonpath='{.spec.dnsNames}' | jq
```

**Testing Checkpoint**: ✅ Verify `READY` column shows `True` and secret exists.

> **Troubleshooting**: If certificate fails to issue, check:
> - ClusterIssuer is ready: `oc describe clusterissuer letsencrypt`
> - cert-manager logs: `oc logs -n cert-manager deployment/cert-manager`
> - CertificateRequest: `oc get certificaterequest -n openshift-ingress`

---

## Step 5: Create Gateway

### 5.1 Create Gateway Resource

The Gateway provides the ingress point with TLS termination for your inference service.

Create a file named `gateway.yaml`:

> **Important**: Replace `<CLUSTER_DOMAIN>` with your actual cluster domain.

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: openshift-ai-inference
  namespace: openshift-ingress
spec:
  gatewayClassName: openshift-ai-inference
  listeners:
  - name: http
    hostname: "inference.apps.<CLUSTER_DOMAIN>"
    port: 80
    protocol: HTTP
    allowedRoutes:
      namespaces:
        from: All
  - name: https
    hostname: "inference.apps.<CLUSTER_DOMAIN>"
    port: 443
    protocol: HTTPS
    allowedRoutes:
      namespaces:
        from: All
    tls:
      mode: Terminate
      certificateRefs:
      - name: openshift-ai-inference-tls
        kind: Secret
```

**Or** create it dynamically:

```bash
# Create gateway with actual cluster domain
cat <<EOF | oc apply -f -
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: openshift-ai-inference
  namespace: openshift-ingress
spec:
  gatewayClassName: openshift-ai-inference
  listeners:
  - name: http
    hostname: "inference.apps.${CLUSTER_DOMAIN}"
    port: 80
    protocol: HTTP
    allowedRoutes:
      namespaces:
        from: All
  - name: https
    hostname: "inference.apps.${CLUSTER_DOMAIN}"
    port: 443
    protocol: HTTPS
    allowedRoutes:
      namespaces:
        from: All
    tls:
      mode: Terminate
      certificateRefs:
      - name: openshift-ai-inference-tls
        kind: Secret
EOF
```

### 5.2 Verify Gateway Status

Wait for the Gateway to be programmed (typically 30-60 seconds):

```bash
# Check gateway status
oc get gateway openshift-ai-inference -n openshift-ingress

# Expected output:
# NAME                     CLASS                    ADDRESS                               PROGRAMMED   AGE
# openshift-ai-inference   openshift-ai-inference   a448732...amazonaws.com              True         45s
```

### 5.3 Verify Gateway Details

```bash
# Get detailed gateway information
oc describe gateway openshift-ai-inference -n openshift-ingress

# Look for:
#   Status:
#     Conditions:
#       Type:     Programmed
#       Status:   True
#       Type:     Accepted
#       Status:   True

# Verify listeners are ready
oc get gateway openshift-ai-inference -n openshift-ingress \
  -o jsonpath='{.status.listeners}' | jq
```

**Testing Checkpoint**: ✅ Verify `PROGRAMMED` column shows `True` and both HTTP/HTTPS listeners are ready.

---

## Step 6: Deploy LLMInferenceService

### 6.1 Create LLMInferenceService

This creates the actual inference service with the Qwen3-8B model.

Create a file named `llm-inference-service.yaml`:

```yaml
apiVersion: serving.kserve.io/v1alpha1
kind: LLMInferenceService
metadata:
  annotations:
    opendatahub.io/connections: qwen3-8b-fp8-dynamic-connection
    opendatahub.io/hardware-profile-name: ''
    opendatahub.io/hardware-profile-namespace: ''
    openshift.io/display-name: qwen3-8b-fp8-dynamic-llmis
    security.opendatahub.io/enable-auth: 'false'
  name: qwen3-8b-fp8-dynamic-llmis

  namespace: llm-d-demo
  labels:
    kueue.x-k8s.io/queue-name: gpu-queue
    opendatahub.io/dashboard: 'true'
    opendatahub.io/genai-asset: 'true'
spec:
  model:
    name: RedHatAI/Qwen3-8B-FP8-dynamic
    uri: 'oci://registry.redhat.io/rhelai1/modelcar-qwen3-8b-fp8-dynamic:1.5'
  replicas: 2
  router:
    gateway: {}
    route: {}
    scheduler: {}
  template:
    containers:
      - env:
          - name: VLLM_ADDITIONAL_ARGS
            value: '--enable-auto-tool-choice --tool-call-parser hermes'
        image: 'registry.redhat.io/rhaiis/vllm-cuda-rhel9:3.2.1'
        name: main
        resources:
          limits:
            cpu: '4'
            memory: 32Gi
            nvidia.com/gpu: '1'
          requests:
            cpu: '2'
            memory: 16Gi
            nvidia.com/gpu: '1'
    nodeSelector:
      group: ntt-data
    tolerations:
      - effect: NoSchedule
        key: nvidia.com/gpu
        operator: Exists
```

> **Note**: Adjust `nodeSelector` to match your GPU node labels. Check with:
> ```bash
> oc get nodes --show-labels | grep nvidia
> ```

Apply the LLMInferenceService:

```bash
# Deploy the LLM service
oc apply -f llm-inference-service.yaml

# Watch the deployment (Ctrl+C to exit)
oc get llminferenceservice -n llm-d-demo -w
```

### 6.2 Monitor Deployment Progress

The deployment typically takes 3-5 minutes:

```bash
# Check LLMInferenceService status
oc get llminferenceservice qwen3-8b-fp8-dynamic-llmis -n llm-d-demo

# Expected final output:
# NAME                         URL                                              READY   AGE
# qwen3-8b-fp8-dynamic-llmis   https://inference.apps.<domain>/llm-d-demo/...   True    5m
```

### 6.3 Check Component Status

```bash
# Get detailed status with conditions
oc describe llminferenceservice qwen3-8b-fp8-dynamic-llmis -n llm-d-demo

# Look for these conditions to be True:
#   Type: Ready
#   Type: WorkloadsReady
#   Type: RouterReady
#   Type: HTTPRoutesReady

# Check pods are running
oc get pods -n llm-d-demo

# Expected output: 2 model pods + 1 scheduler pod
# NAME                                                           READY   STATUS
# qwen3-8b-fp8-dynamic-llmis-kserve-xxxxx                        2/2     Running
# qwen3-8b-fp8-dynamic-llmis-kserve-xxxxx                        2/2     Running
# qwen3-8b-fp8-dynamic-llmis-kserve-router-scheduler-xxxxx       1/1     Running
```

### 6.4 Verify HTTPRoute Creation

```bash
# Check HTTPRoute was created automatically
oc get httproute -n llm-d-demo

# Expected output:
# NAME                                      HOSTNAMES   AGE
# qwen3-8b-fp8-dynamic-llmis-kserve-route               5m

# Verify HTTPRoute is accepted
oc describe httproute qwen3-8b-fp8-dynamic-llmis-kserve-route -n llm-d-demo

# Look for:
#   Status:
#     Parents:
#       Conditions:
#         Type:    Accepted
#         Status:  True
#         Type:    ResolvedRefs
#         Status:  True
```

### 6.5 Monitor Model Loading

Watch the model loading process:

```bash
# Watch pod logs (Ctrl+C to exit)
oc logs -n llm-d-demo \
  -l app.kubernetes.io/name=qwen3-8b-fp8-dynamic-llmis \
  -c main -f

# Look for these log messages:
# "Loading model from oci://..."
# "Model loaded successfully"
# "vLLM server started on port 8000"
```

**Testing Checkpoint**: ✅ Verify:
- LLMInferenceService shows `READY: True`
- All pods are in `Running` status
- HTTPRoute shows `Accepted: True`

---

## Step 7: Verify Deployment

### 7.1 Get Service URL

Retrieve the inference service URL:

```bash
# Get the inference service URL
SERVICE_URL=$(oc get llminferenceservice qwen3-8b-fp8-dynamic-llmis \
  -n llm-d-demo -o jsonpath='{.status.url}')
echo "Service URL: ${SERVICE_URL}"

# Example output:
# https://inference.apps.ocp.sandbox1278.opentlc.com/llm-d-demo/qwen3-8b-fp8-dynamic-llmis
```

### 7.2 Test Health Endpoint

```bash
# Test health endpoint (use -k if certificate validation fails)
curl -k https://inference.apps.${CLUSTER_DOMAIN}/llm-d-demo/qwen3-8b-fp8-dynamic-llmis/health

# Expected output: HTTP 200 (empty response is normal for health endpoint)

# Check HTTP status code
curl -k -o /dev/null -s -w "%{http_code}\n" \
  https://inference.apps.${CLUSTER_DOMAIN}/llm-d-demo/qwen3-8b-fp8-dynamic-llmis/health

# Expected output: 200
```

### 7.3 Test OpenAPI Schema

```bash
# Get OpenAPI schema
curl -k https://inference.apps.${CLUSTER_DOMAIN}/llm-d-demo/qwen3-8b-fp8-dynamic-llmis/openapi.json | jq

# This should return a JSON schema with all available endpoints:
# {
#   "openapi": "3.1.0",
#   "info": {
#     "title": "FastAPI",
#     "version": "0.1.0"
#   },
#   "paths": {
#     "/health": {...},
#     "/v1/models": {...},
#     "/v1/completions": {...},
#     "/v1/chat/completions": {...}
#   }
# }
```

### 7.4 Test List Models

```bash
# List available models
curl -k https://inference.apps.${CLUSTER_DOMAIN}/llm-d-demo/qwen3-8b-fp8-dynamic-llmis/v1/models | jq

# Expected output:
# {
#   "object": "list",
#   "data": [
#     {
#       "id": "RedHatAI/Qwen3-8B-FP8-dynamic",
#       "object": "model",
#       "created": 1234567890,
#       "owned_by": "organization"
#     }
#   ]
# }
```

**Testing Checkpoint**: ✅ All endpoints return HTTP 200 and valid responses.

---

## Step 8: Test Inference Endpoints

### 8.1 Test Completions API

Test the text completion endpoint:

```bash
# Test text completion
curl -k https://inference.apps.${CLUSTER_DOMAIN}/llm-d-demo/qwen3-8b-fp8-dynamic-llmis/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "RedHatAI/Qwen3-8B-FP8-dynamic",
    "prompt": "What is Kubernetes?",
    "max_tokens": 100,
    "temperature": 0.7
  }' | jq
```

**Expected Response Structure:**

```json
{
  "id": "cmpl-78edb991-ee3b-43d0-bd1e-bb2549d3f1a7",
  "object": "text_completion",
  "created": 1768417364,
  "model": "RedHatAI/Qwen3-8B-FP8-dynamic",
  "choices": [
    {
      "text": "Kubernetes is an open-source container orchestration platform that automates deploying, scaling, and managing containerized applications...",
      "index": 0,
      "finish_reason": "length",
      "logprobs": null
    }
  ],
  "usage": {
    "prompt_tokens": 4,
    "completion_tokens": 100,
    "total_tokens": 104
  }
}
```

### 8.2 Test Chat Completions API

Test the chat completion endpoint with system and user messages:

```bash
# Test chat completion
curl -k https://inference.apps.${CLUSTER_DOMAIN}/llm-d-demo/qwen3-8b-fp8-dynamic-llmis/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "RedHatAI/Qwen3-8B-FP8-dynamic",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain OpenShift in one sentence."}
    ],
    "max_tokens": 50,
    "temperature": 0.7
  }' | jq
```

**Expected Response Structure:**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "RedHatAI/Qwen3-8B-FP8-dynamic",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "OpenShift is Red Hat's enterprise Kubernetes platform that adds developer and operational tools on top of Kubernetes for building, deploying, and managing containerized applications."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 35,
    "total_tokens": 60
  }
}
```

### 8.3 Test Streaming Response

Test streaming completions for real-time response generation:

```bash
# Test streaming completion
curl -k https://inference.apps.${CLUSTER_DOMAIN}/llm-d-demo/qwen3-8b-fp8-dynamic-llmis/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "RedHatAI/Qwen3-8B-FP8-dynamic",
    "messages": [
      {"role": "user", "content": "Count from 1 to 10"}
    ],
    "max_tokens": 50,
    "stream": true
  }'

# Expected output: Server-Sent Events (SSE) stream
# data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk",...}
# data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk",...}
# data: [DONE]
```

### 8.4 Test with Different Parameters

Experiment with various inference parameters:

```bash
# Test with higher temperature (more creative)
curl -k https://inference.apps.${CLUSTER_DOMAIN}/llm-d-demo/qwen3-8b-fp8-dynamic-llmis/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "RedHatAI/Qwen3-8B-FP8-dynamic",
    "prompt": "Write a haiku about clouds",
    "max_tokens": 50,
    "temperature": 1.0,
    "top_p": 0.95
  }' | jq

# Test with multiple completions
curl -k https://inference.apps.${CLUSTER_DOMAIN}/llm-d-demo/qwen3-8b-fp8-dynamic-llmis/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "RedHatAI/Qwen3-8B-FP8-dynamic",
    "prompt": "Complete this sentence: The best thing about",
    "max_tokens": 20,
    "n": 3
  }' | jq '.choices'
```

**Testing Checkpoint**: ✅ All inference endpoints return valid completions with correct token usage.

---

## Step 9: Access from Browser

### 9.1 Service URL

Your inference service is accessible at:

```
https://inference.apps.<CLUSTER_DOMAIN>/llm-d-demo/qwen3-8b-fp8-dynamic-llmis
```

### 9.2 Available Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/v1/models` | GET | List available models |
| `/v1/completions` | POST | Text completion |
| `/v1/chat/completions` | POST | Chat completion |
| `/openapi.json` | GET | OpenAPI specification |

### 9.3 Using Postman or Browser Tools

**Postman Configuration:**

1. Method: `POST`
2. URL: `https://inference.apps.<CLUSTER_DOMAIN>/llm-d-demo/qwen3-8b-fp8-dynamic-llmis/v1/chat/completions`
3. Headers:
   - `Content-Type: application/json`
4. Body (raw JSON):
   ```json
   {
     "model": "RedHatAI/Qwen3-8B-FP8-dynamic",
     "messages": [
       {"role": "user", "content": "Hello!"}
     ],
     "max_tokens": 50
   }
   ```

### 9.4 Example Python Client

```python
import requests
import json

url = "https://inference.apps.<CLUSTER_DOMAIN>/llm-d-demo/qwen3-8b-fp8-dynamic-llmis/v1/chat/completions"

payload = {
    "model": "RedHatAI/Qwen3-8B-FP8-dynamic",
    "messages": [
        {"role": "user", "content": "What is OpenShift?"}
    ],
    "max_tokens": 100
}

headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers, verify=False)
print(json.dumps(response.json(), indent=2))
```

### 9.5 Example JavaScript Client

```javascript
const url = 'https://inference.apps.<CLUSTER_DOMAIN>/llm-d-demo/qwen3-8b-fp8-dynamic-llmis/v1/chat/completions';

const payload = {
  model: 'RedHatAI/Qwen3-8B-FP8-dynamic',
  messages: [
    { role: 'user', content: 'What is Kubernetes?' }
  ],
  max_tokens: 100
};

fetch(url, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify(payload)
})
  .then(response => response.json())
  .then(data => console.log(data))
  .catch(error => console.error('Error:', error));
```

---

## Troubleshooting Guide

### Issue 1: Gateway Not Programmed

**Symptoms:**
- Gateway shows `PROGRAMMED: False`
- HTTPRoute cannot attach to Gateway

**Diagnosis:**

```bash
# Check Gateway status
oc describe gateway openshift-ai-inference -n openshift-ingress

# Check Gateway controller logs
oc get pods -n openshift-gateway-system
oc logs -n openshift-gateway-system deployment/gateway-controller
```

**Solutions:**
- Verify GatewayClass exists and is accepted
- Check that certificate secret exists in openshift-ingress namespace
- Ensure Gateway controller is running

---

### Issue 2: Certificate Not Ready

**Symptoms:**
- Certificate shows `READY: False`
- TLS handshake failures

**Diagnosis:**

```bash
# Check certificate status
oc describe certificate openshift-ai-inference-cert -n openshift-ingress

# Check cert-manager logs
oc logs -n cert-manager deployment/cert-manager --tail=100

# Check CertificateRequest
oc get certificaterequest -n openshift-ingress
oc describe certificaterequest -n openshift-ingress

# Check challenges (for HTTP-01)
oc get challenges -n openshift-ingress
```

**Solutions:**
- Verify ClusterIssuer is ready: `oc get clusterissuer letsencrypt`
- Ensure DNS is correctly configured for the hostname
- For wildcard certificates, verify DNS-01 challenge is configured
- Check if rate limits are hit (switch to `letsencrypt-staging`)

---

### Issue 3: LLMInferenceService Not Ready

**Symptoms:**
- LLMInferenceService shows `READY: False`
- Pods not starting or stuck in Pending

**Diagnosis:**

```bash
# Check detailed status
oc describe llminferenceservice qwen3-8b-fp8-dynamic-llmis -n llm-d-demo

# Check pods
oc get pods -n llm-d-demo

# Check pod details
oc describe pod -n llm-d-demo -l app.kubernetes.io/name=qwen3-8b-fp8-dynamic-llmis

# Check pod logs
oc logs -n llm-d-demo -l app.kubernetes.io/name=qwen3-8b-fp8-dynamic-llmis -c main

# Check events
oc get events -n llm-d-demo --sort-by='.lastTimestamp' | tail -20
```

**Common Issues and Solutions:**

1. **GPU Resources Unavailable:**
   ```bash
   # Check GPU allocation
   oc describe localqueue gpu-queue -n llm-d-demo
   oc get clusterqueue cluster-queue -o yaml
   
   # Verify node GPU resources
   oc describe node <gpu-node> | grep nvidia.com/gpu
   ```

2. **Image Pull Failures:**
   ```bash
   # Check image pull secrets
   oc get secrets -n llm-d-demo
   
   # Verify registry access
   oc run test --image=registry.redhat.io/rhaiis/vllm-cuda-rhel9:3.2.1 --rm -it
   ```

3. **Node Selector Mismatch:**
   ```bash
   # Check node labels
   oc get nodes --show-labels | grep group=ntt-data
   
   # Update nodeSelector in YAML if needed
   ```

---

### Issue 4: HTTPRoute Not Working

**Symptoms:**
- Cannot access inference endpoint
- 404 or 503 errors

**Diagnosis:**

```bash
# Check HTTPRoute
oc get httproute -n llm-d-demo
oc describe httproute qwen3-8b-fp8-dynamic-llmis-kserve-route -n llm-d-demo

# Check if HTTPRoute is accepted
oc get httproute -n llm-d-demo -o jsonpath='{.items[0].status.parents[0].conditions}' | jq

# Check service endpoints
oc get endpoints -n llm-d-demo
```

**Solutions:**
- Verify Gateway exists and is programmed
- Check that HTTPRoute references correct Gateway
- Verify service has active endpoints
- Check path matching in HTTPRoute rules

---

### Issue 5: TLS Certificate Hostname Mismatch

**Symptoms:**
```
SSL: no alternative certificate subject name matches target host name
```

**Diagnosis:**

```bash
# Check certificate DNS names
oc get certificate openshift-ai-inference-cert -n openshift-ingress \
  -o jsonpath='{.spec.dnsNames}' | jq

# Check actual certificate
oc get secret openshift-ai-inference-tls -n openshift-ingress \
  -o jsonpath='{.data.tls\.crt}' | base64 -d | \
  openssl x509 -noout -text | grep -A2 "Subject Alternative Name"
```

**Solution:**

```bash
# Delete and recreate certificate with correct hostname
oc delete certificate openshift-ai-inference-cert -n openshift-ingress

# Recreate with correct hostname (see Step 4)
cat <<EOF | oc apply -f -
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: openshift-ai-inference-cert
  namespace: openshift-ingress
spec:
  secretName: openshift-ai-inference-tls
  issuerRef:
    name: letsencrypt
    kind: ClusterIssuer
  dnsNames:
  - "inference.apps.${CLUSTER_DOMAIN}"
  duration: 2160h
  renewBefore: 360h
EOF
```

---

### Issue 6: Model Loading Failures

**Symptoms:**
- Pods restart repeatedly
- "Model failed to load" errors in logs

**Diagnosis:**

```bash
# Check pod logs for errors
oc logs -n llm-d-demo -l app.kubernetes.io/name=qwen3-8b-fp8-dynamic-llmis -c main --tail=100

# Check resource usage
oc top pods -n llm-d-demo

# Check GPU status in pod
oc exec -n llm-d-demo <pod-name> -- nvidia-smi
```

**Solutions:**
- Increase memory if OOM errors occur
- Verify model URI is accessible
- Check GPU compatibility with model
- Verify sufficient disk space for model cache

---

### Issue 7: Slow Response Times

**Symptoms:**
- High latency on inference requests
- Timeout errors

**Diagnosis:**

```bash
# Check pod metrics
oc top pods -n llm-d-demo

# Check scheduler logs
oc logs -n llm-d-demo -l app.kubernetes.io/component=llminferenceservice-router

# Test with verbose timing
time curl -k https://inference.apps.${CLUSTER_DOMAIN}/llm-d-demo/qwen3-8b-fp8-dynamic-llmis/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"RedHatAI/Qwen3-8B-FP8-dynamic","prompt":"Hi","max_tokens":10}'
```

**Solutions:**
- Increase replica count
- Optimize resource requests/limits
- Check network latency
- Enable caching if available

---

## Cleanup

To remove all deployed resources:

### Complete Cleanup

```bash
# Delete LLMInferenceService (this will delete pods, services, HTTPRoute)
oc delete llminferenceservice qwen3-8b-fp8-dynamic-llmis -n llm-d-demo

# Delete LocalQueue
oc delete localqueue gpu-queue -n llm-d-demo

# Delete namespace (removes all resources within)
oc delete namespace llm-d-demo

# Delete Gateway
oc delete gateway openshift-ai-inference -n openshift-ingress

# Delete Certificate
oc delete certificate openshift-ai-inference-cert -n openshift-ingress

# Delete GatewayClass (only if not used by other services)
oc delete gatewayclass openshift-ai-inference
```

### Selective Cleanup (Keep Infrastructure)

If you want to keep the Gateway and Certificate for future deployments:

```bash
# Only delete the LLM service
oc delete llminferenceservice qwen3-8b-fp8-dynamic-llmis -n llm-d-demo

# Only delete the namespace
oc delete namespace llm-d-demo
```

---

## Performance Tuning

### Adjusting Replicas

Scale your service up or down based on demand:

```bash
# Scale up to 4 replicas
oc patch llminferenceservice qwen3-8b-fp8-dynamic-llmis -n llm-d-demo \
  --type='json' \
  -p='[{"op": "replace", "path": "/spec/replicas", "value": 4}]'

# Scale down to 1 replica
oc patch llminferenceservice qwen3-8b-fp8-dynamic-llmis -n llm-d-demo \
  --type='json' \
  -p='[{"op": "replace", "path": "/spec/replicas", "value": 1}]'

# Verify scaling
oc get pods -n llm-d-demo -l app.kubernetes.io/name=qwen3-8b-fp8-dynamic-llmis
```

### Adjusting Resources

For larger models or better performance, increase resources:

```yaml
resources:
  limits:
    cpu: '8'           # Increase for better CPU performance
    memory: 64Gi       # Increase for larger models
    nvidia.com/gpu: '2' # Multi-GPU per pod
  requests:
    cpu: '4'
    memory: 32Gi
    nvidia.com/gpu: '2'
```

Apply changes:

```bash
# Edit the LLMInferenceService
oc edit llminferenceservice qwen3-8b-fp8-dynamic-llmis -n llm-d-demo

# Or patch directly
oc patch llminferenceservice qwen3-8b-fp8-dynamic-llmis -n llm-d-demo \
  --type='json' \
  -p='[
    {"op": "replace", "path": "/spec/template/containers/0/resources/limits/memory", "value": "64Gi"},
    {"op": "replace", "path": "/spec/template/containers/0/resources/requests/memory", "value": "32Gi"}
  ]'
```

### Inference Parameters Optimization

Optimize inference parameters for your use case:

**For Lower Latency:**
```json
{
  "temperature": 0.1,
  "top_p": 0.9,
  "max_tokens": 50,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0
}
```

**For Creative Responses:**
```json
{
  "temperature": 0.9,
  "top_p": 0.95,
  "max_tokens": 200,
  "presence_penalty": 0.6,
  "frequency_penalty": 0.3
}
```

**For Deterministic Responses:**
```json
{
  "temperature": 0.0,
  "seed": 42,
  "max_tokens": 100
}
```

---

## Additional Resources

### Documentation

- **Red Hat OpenShift AI**: [docs.redhat.com/openshift-ai](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed)
- **KServe LLM-D**: [kserve.github.io/llmisvc](https://kserve.github.io/website/docs/model-serving/generative-inference/llmisvc/)
- **Gateway API**: [gateway-api.sigs.k8s.io](https://gateway-api.sigs.k8s.io/)
- **cert-manager**: [cert-manager.io/docs](https://cert-manager.io/docs/)
- **vLLM**: [docs.vllm.ai](https://docs.vllm.ai/)
- **Kueue**: [kueue.sigs.k8s.io](https://kueue.sigs.k8s.io/)

### Community

- **OpenShift AI Community**: [github.com/opendatahub-io](https://github.com/opendatahub-io)
- **KServe Community**: [github.com/kserve/kserve](https://github.com/kserve/kserve)

### Support

- **Red Hat Support**: [access.redhat.com/support](https://access.redhat.com/support)
- **OpenShift AI Forum**: [access.redhat.com/discussions](https://access.redhat.com/discussions)

---

## Summary

Congratulations! You have successfully deployed a production-ready LLM inference service with:

✅ **Infrastructure Components:**
- Kueue LocalQueue for GPU resource management
- Let's Encrypt TLS certificate with automatic renewal
- Gateway API with custom GatewayClass
- HTTPRoute for path-based routing

✅ **Inference Service:**
- LLMInferenceService with Qwen3-8B-FP8-dynamic model
- 2 replicas for high availability
- vLLM runtime for optimized inference
- GPU acceleration with NVIDIA GPUs

✅ **Security:**
- TLS termination at Gateway
- Automated certificate management
- Secure HTTPS endpoints

✅ **Endpoints:**
```
Base URL: https://inference.apps.<CLUSTER_DOMAIN>/llm-d-demo/qwen3-8b-fp8-dynamic-llmis

- GET  /health                 - Health check
- GET  /v1/models              - List models
- POST /v1/completions         - Text completion
- POST /v1/chat/completions    - Chat completion
- GET  /openapi.json           - API documentation
```

Your LLM inference service is now production-ready and accessible via HTTPS! 🎉

---

## Quick Reference Card

### Essential Commands

```bash
# Check service status
oc get llminferenceservice -n llm-d-demo

# Check pods
oc get pods -n llm-d-demo

# Check Gateway
oc get gateway -n openshift-ingress

# Check certificate
oc get certificate -n openshift-ingress

# Test health
curl -k https://inference.apps.${CLUSTER_DOMAIN}/llm-d-demo/qwen3-8b-fp8-dynamic-llmis/health

# Test inference
curl -k https://inference.apps.${CLUSTER_DOMAIN}/llm-d-demo/qwen3-8b-fp8-dynamic-llmis/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"RedHatAI/Qwen3-8B-FP8-dynamic","prompt":"Hello","max_tokens":50}'

# View logs
oc logs -n llm-d-demo -l app.kubernetes.io/name=qwen3-8b-fp8-dynamic-llmis -c main -f

# Scale replicas
oc patch llminferenceservice qwen3-8b-fp8-dynamic-llmis -n llm-d-demo \
  --type='json' -p='[{"op":"replace","path":"/spec/replicas","value":3}]'
```

---

**Document Version**: 1.0  
**Last Updated**: January 2026  
**Tested On**: OpenShift 4.19.9+ with OpenShift AI 3.0

