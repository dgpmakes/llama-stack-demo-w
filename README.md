# LLAMA STACK DEMO

Welcome to this Llama Stack Demo you can run easily on Red Hat OpenShift AI (RHOAI). It's a comprehensive demo that deploys everything you need to test agents with MCP tools on RHOAI using Llama Stack:
- Models: IBM Granite 3.3 and Llama 3.1 deployed using vLLM Serving Runtime and automatically added to Llama Stack
- MCP Servers: deployed as normal deployments and automatically added to Llama Stack
- Llama Stack Server: Llama Stack Server is deployed using the Llama Stack Operator included in Red Hat OpenShift AI

## Trip Report

I started with an MCP Server using TypeScript generated using Cursor.ai straight from legal documents. It tried to cover all the legal elements, it was complex and the logic tricky, several iterations to fine tune the descriptions to make it kind of work in a small model (SLM) like IBM Granite and Llama 3.2 (~7B both of them). I changed my strategy and also the programing language, from TypeScript to Rust and from the full logic generated from the legal documents to a decision table generated with Cursor.ai but with much more guidance (and simplification) from my side. This time it worked pretty well with Claude Desktop but not quite with the SLMs, the latest iteration had to do with the names of some variables, is_single_parent_family and number_of_children_after...

The inner to outer loop:
- decision table
- code
- unit testing
- MCP Inspector
- Claude Desktop
- Llama Stack Local with models in RHOAI
- Llama Stack on RHOAI


## Detailed description 

Eligibility Assessment System powered by Llama Stack and Model Context Protocol (MCP)!
This system helps assess eligibility for Family Care Unpaid Leave Support based on the Republic of Lysmark's Act No. 2025/47-SA.
To get started quickly, jump straight to [installation](#install).


The Eligibility Assessment MCP Llama Stack system is an intelligent solution for evaluating eligibility for Family Care Unpaid Leave Support based on the Republic of Lysmark's legislation (Act No. 2025/47-SA and related regulations). The system combines the power of Llama Stack with Model Context Protocol (MCP) servers and Retrieval Augmented Generation (RAG) to provide accurate, context-aware assessments.

This system is designed to help users understand their eligibility for unpaid leave assistance in various family care situations, including care for sick/injured family members, childcare for multiple children, adoption cases, and single-parent family scenarios.

This deployment includes a Helm chart for setting up:

- An OpenShift AI Project with all necessary components
- Llama Stack distribution with Llama 3.1 8B model for natural language processing  
- MCP server for the eligibility engine with specialized knowledge processing
- Vector database (Milvus) with embedded legal documents for RAG capabilities
- Document loader service to populate the knowledge base with Act No. 2025/47-SA and regulations
- Llama Stack Playground interface for interactive eligibility consultations

Use this project to quickly deploy an intelligent eligibility assessment system that provides accurate, legally-informed guidance on Family Care Unpaid Leave Support eligibility. 🏛️

### See it in action

Experience the Eligibility Assessment System through the Llama Stack Playground interface. After deployment, you can interact with the system to:
- Ask questions about eligibility requirements for different family care scenarios
- Get detailed assessments based on the legal framework of Act No. 2025/47-SA
- Understand the documentation needed for applications
- Learn about financial assistance amounts and duration limits

### Architecture diagrams

![architecture.png](images/architecture.png)

### References 

- [Llama Stack Distribution](https://llama-stack.readthedocs.io/en/latest/) - The foundational platform for LLM applications
- [Model Context Protocol (MCP)](https://spec.modelcontextprotocol.io/) - Protocol for integrating context sources with language models  
- [Eligibility Engine MCP Server](https://github.com/alpha-hack-program/eligibility-engine-mcp-rs) - Rust-based MCP server for eligibility processing
- Llama Stack Docker images: [quay.io/opendatahub/llama-stack:odh](https://quay.io/opendatahub/llama-stack:odh)
- Model images from Red Hat AI Services ModelCar Catalog
- Based on Republic of Lysmark's Act No. 2025/47-SA: Family Care Unpaid Leave Support Act

## Requirements 

### Recommended hardware requirements 

- **GPU**: 1x NVIDIA A10G or equivalent (for optimal LLM performance)
- **CPU**: 8 cores 
- **Memory**: 24 Gi (to handle Llama Stack, MCP servers, and vector database)
- **Storage**: 20Gi (for models, vector database, and document storage)

Note: The system uses quantized Llama 3.1 8B model (w4a16) for efficient GPU utilization while maintaining good performance.

### Minimum hardware requirements 

- **GPU**: 1x NVIDIA A10G-SHARED (shared GPU allocation)
- **CPU**: 6 cores 
- **Memory**: 16 Gi 
- **Storage**: 10Gi 

### Required software  

- Red Hat OpenShift 4.16+ 
- Red Hat OpenShift AI 2.16+ 
- Dependencies for [Single-model server](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/2.16/html/installing_and_uninstalling_openshift_ai_self-managed/installing-the-single-model-serving-platform_component-install#configuring-automated-installation-of-kserve_component-install):
    - Red Hat OpenShift Service Mesh
    - Red Hat OpenShift Serverless
- **Llama Stack Components**:
    - Llama Stack Distribution with vLLM runtime
    - Model Context Protocol (MCP) server support
    - Vector database capabilities (Milvus)
- **Container Images**:
    - Llama Stack: `quay.io/opendatahub/llama-stack:odh`
    - vLLM Runtime: `quay.io/modh/vllm:rhoai-2.23-cuda`
    - Eligibility Engine MCP: `quay.io/atarazana/eligibility-engine-mcp-rs:latest`

### Required permissions

- Standard user. No elevated cluster permissions required 

## Install

**Please note before you start**

This system was tested on Red Hat OpenShift 4.16.24 & Red Hat OpenShift AI v2.16.2.  
Ensure you have access to GPU resources and the required container registries.

### Clone

```bash
TODO git clone https://github.com/alpha-hack-program/.git && \
    cd eligibility-mcp-llamastack/
```

### Log in OpenShift

```bash
oc login ...
```

### Make sure that at least a GPU enabled worked has this label:

```yaml
group: llama-stack-demo
```

### Create the project

```bash
PROJECT="llama-stack-demo"

oc new-project ${PROJECT}
``` 

Label project with
- modelmesh-enabled: 'false'
- opendatahub.io/dashboard: 'true'

```bash
oc label namespace ${PROJECT} modelmesh-enabled=false opendatahub.io/dashboard=true
```

### Install with Helm

This default delployment deploys one model... TODO.

```bash
helm install llama-stack-demo helm/ --namespace ${PROJECT} --timeout 20m
```

If you have access to Intel Gaudi accelerators you could use this command which uses `helm/intel.values` instead:

```bash
helm install llama-stack-demo helm/ --namespace ${PROJECT} --values helm/intel.yaml --timeout 20m
```

If you want an NVIDIA deployment with two models run this. TODO explain which models... bla.

```bash
helm install llama-stack-demo helm/ --namespace ${PROJECT} --values helm/nvidia.yaml --timeout 20m
```

### Wait for pods

```bash
oc -n ${PROJECT} get pods -w
```

Expected pods (may take 5-10 minutes to start):
```
(Output)
NAME                                              READY   STATUS    RESTARTS   AGE
eligibility-lsd-0                                1/1     Running   0          8m
eligibility-lsd-playground-0                     1/1     Running   0          8m
eligibility-engine-0                             1/1     Running   0          7m
loader-0                                         0/1     Completed 0          6m
llama-3-1-8b-w4a16-predictor-df76b56d6-fw8fp    2/2     Running   0          10m
```

### Test

You can access the system in multiple ways:

#### Option 1: OpenShift AI Dashboard
Get the OpenShift AI Dashboard URL:
```bash
oc get routes rhods-dashboard -n redhat-ods-applications
```

Navigate to Data Science Projects -> llama-stack-demo. You'll see the deployed models and workbenches.

#### Option 2: Direct Access to Llama Stack Playground
Get the Llama Stack Playground URL:
```bash
oc get routes eligibility-lsd-playground -n ${PROJECT}
```

Access the playground directly to interact with the eligibility assessment system.

#### Option 3: API Access
For programmatic access, get the Llama Stack API endpoint:
```bash
oc get routes eligibility-lsd -n ${PROJECT}
```

Use this endpoint to integrate the eligibility assessment capabilities into your applications.

## Business Rules

### Unpaid Leave Evaluation Data

| Family relationship | Situation | Single-parent family | Number of children | Potentially eligible | Monthly benefit | Case | Description | Output | DESCRIPTION | Rule ID |
|---------------------|-----------|---------------------|-------------------|---------------------|----------------|------|-------------|--------|-------------|---------|
| true | delivery, birth | true | | true | 500 | E | Single-parent family with newborn | The single-parent status must be documented | Case E: Single-parent family with any child | regla-005 |
| true | delivery, birth | | >=3 | true | 500 | B | Third child or more with newborn | The number of children must be 3 or more, the ages of at least 2 of the minors must be less than 6, if there is disability greater than 33% then the limit is 9 years | Case B: Third child or more with newborn | regla-002 |
| true | delivery, birth | | | false | 0 | B | The number of children must be 3 or more, must consult with administration | | The number of children must be 3 or more, must consult with administration | 9ec43eb2-484f-4fcf-9dd7-6510da30850c |
| true | illness, accident | | | true | 725 | A | First-degree family care sick or accident victim | The person must have been hospitalized and the care of the person must be continued | Case A: First-degree family care sick/injured | regla-001 |
| true | adoption, foster_care | | | true | 500 | C | Adoption or foster care | In the foster care case the duration must be longer than one year | Case C: Adoption or foster care | regla-003 |
| true | multiple_birth, multiple_delivery, multiple_adoption, multiple_foster_care | | | true | 500 | D | Delivery, adoption or foster care multiple | | Case D: Delivery, adoption or foster care multiple | regla-004 |
| true | | | | false | 0 | NONE | No case applies | | No case applies | 515afd1f-43cc-44ed-971c-fefb273840b2 |
| false | | | | false | 0 | NONE | Not applicable by relationship (first degree) | | Only father, mother, son, daughter, spouse or partner are accepted | 058dd988-90dd-46da-8478-ee458aacde6f |
| | | | | false | 0 | NONE | UNKNOWN_ERROR | | | f32bfb0f-801d-4d6c-b5bd-13a1edd0eaca |

### Summary

This table contains the evaluation criteria and outcomes for unpaid leave assistance eligibility. The data shows different cases (A through E) with varying monthly benefits:

- **Case A**: First-degree family care sick/injured - 725€
- **Case B**: Third child or more with newborn - 500€  
- **Case C**: Adoption or foster care - 500€
- **Case D**: Multiple delivery/adoption/foster care - 500€
- **Case E**: Single-parent family with any child - 500€
- **NONE**: Cases where no assistance applies - 0€

The table includes input parameters (family relationship, situation, single-parent status, number of children) and corresponding outputs (eligibility, benefit amount, case classification, descriptions, and rule IDs).

## Example queries

- My mother had an accident and she's at the hospital. I have to take care of her, can I get access to the unpaid leave aid?
- My mother had an accident and she's at the hospital. I have to take care of her, tell me if I can get access to the unpaid leave aid and the requirements I have to meet.
- I have just adopted two children, at the same time, aged 3 and 5, am I elegible for the unpaid leave aid? How much?
- I have just adopted two children, at the same time, aged 3 and 5, tell me if I'm elegible for the unpaid leave aid and which requirements I should meet.
- I'm a single mom and I just had a baby, may I get access to the unpaid leave aid?
- Enumerate the legal requirements to get the aid for unpaid leave.

## Example System Prompt

You are a helpful AI assistant that uses tools to help citizens of the Republic of Lysmark. Answers should be concise and human readable. AVOID references to tools or function calling nor show any JSON. Infer parameters for function calls or instead use default values or request the needed information from the user. Call the RAG tool first if unsure. Parameter single_parent_family only is necessary if birth/adoption/foster_care otherwise use false.

## Uninstall

Unistall the helm chart.

```bash
helm uninstall llama-stack-demo --namespace ${PROJECT}
```

Delete all remaining objects like jobs created in hooks.

```bash
oc delete jobs -l "app.kubernetes.io/part-of=llama-stack-demo"
```

Finally remove the project:

```bash
oc delete project ${PROJECT}
```

# Monitoring

```yaml
apiVersion: dscinitialization.opendatahub.io/v2
kind: DSCInitialization
metadata:
  name: default-dsci
spec:
  applicationsNamespace: redhat-ods-applications
  monitoring:
    alerting: {}
    managementState: Managed
    metrics:
      replicas: 1
      storage:
        retention: 90d
        size: 50Gi
    namespace: redhat-ods-monitoring
    traces:
      sampleRatio: '1.0'
      storage:
        backend: pv
        retention: 2160h0m0s
        size: 100Gi
  trustedCABundle:
    customCABundle: ''
    managementState: Managed
```

Just make sure that in your default-dsci spec->monitoring is coherent with:

```yaml
spec:
  ...
  monitoring:
    alerting: {}
    managementState: Managed
    metrics:
      replicas: 1
      storage:
        retention: 90d
        size: 50Gi
    namespace: redhat-ods-monitoring
    traces:
      sampleRatio: '1.0'
      storage:
        backend: pv
        retention: 2160h0m0s
        size: 100Gi
...
```

### BUG collector service

Create this service...

```yaml
kind: Service
apiVersion: v1
metadata:
  name: data-science-collector
  namespace: redhat-ods-monitoring
  labels:
    app.kubernetes.io/component: opentelemetry-collector
    app.kubernetes.io/instance: redhat-ods-monitoring.data-science-collector
    app.kubernetes.io/part-of: opentelemetry
spec:
  ipFamilies:
    - IPv4
  ports:
    - name: otlp-grpc
      protocol: TCP
      appProtocol: grpc
      port: 4317
      targetPort: 4317
    - name: otlp-http
      protocol: TCP
      appProtocol: http
      port: 4318
      targetPort: 4318
    - name: prometheus
      protocol: TCP
      port: 8889
      targetPort: 8889
  internalTrafficPolicy: Cluster

  type: ClusterIP
  ipFamilyPolicy: SingleStack
  sessionAffinity: None
  selector:
    app.kubernetes.io/component: opentelemetry-collector
    app.kubernetes.io/instance: redhat-ods-monitoring.data-science-collector
    app.kubernetes.io/managed-by: opentelemetry-operator
    app.kubernetes.io/part-of: opentelemetry
```