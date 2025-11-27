#!/usr/bin/env python3
"""
REST API server for LlamaStack operations.

Provides endpoints for:
- Models: List and retrieve model information
- Tools: List tool groups and tools
- Context: Retrieve context from vector stores
- Agents: Execute agent queries and list agent types
"""

import os
import asyncio
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from llama_stack_client import LlamaStackClient
from llama_stack_client.types.model import Model
from llama_stack_client.types import VectorStore

from utils import create_client, list_models, get_rag_context
from commands.model_command import list_command as model_list_command
from commands.tool_command import groups_command as tool_groups_command
from commands.agent_command import agent_command


# Initialize FastAPI app with enhanced OpenAPI configuration
app = FastAPI(
    title="LlamaStack REST API",
    description="""
    ## LlamaStack REST API
    
    A comprehensive REST API for interacting with LlamaStack, providing access to:
    
    * **Models** - List and retrieve available language models
    * **Tools** - Access tool groups and individual tools for agent execution
    * **Context** - Retrieve context from vector stores using RAG (Retrieval-Augmented Generation)
    * **Agents** - Execute different types of AI agents with custom tools and instructions
    
    ### Getting Started
    
    1. Check the `/health` endpoint to verify the service is running
    2. Use `/models` to see available models
    3. Use `/tools/groups` and `/tools` to explore available tools
    4. Execute agents with `/agents/execute` endpoint
    
    ### Interactive Documentation
    
    * **Swagger UI**: Available at `/docs` (this page)
    * **ReDoc**: Alternative documentation at `/redoc`
    """,
    version="1.0.0",
    contact={
        "name": "LlamaStack Demo",
        "url": "https://github.com/meta-llama/llama-stack"
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html"
    },
    openapi_tags=[
        {
            "name": "health",
            "description": "Health check and status endpoints"
        },
        {
            "name": "models",
            "description": "Operations for listing and retrieving model information"
        },
        {
            "name": "tools",
            "description": "Operations for managing and listing tools and tool groups"
        },
        {
            "name": "context",
            "description": "RAG operations for retrieving context from vector stores"
        },
        {
            "name": "agents",
            "description": "Execute AI agents and manage agent configurations"
        }
    ]
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response with examples for Swagger UI
class ModelResponse(BaseModel):
    identifier: str = Field(..., description="Unique identifier for the model", examples=["llama-3.3-70b-instruct"])
    provider_id: str = Field(..., description="Provider ID (e.g., meta, ollama)", examples=["meta"])
    api_model_type: str = Field(..., description="Type of model (llm, embedding)", examples=["llm"])
    provider_resource_id: Optional[str] = Field(None, description="Provider-specific resource ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional model metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "identifier": "llama-3.3-70b-instruct",
                "provider_id": "meta",
                "api_model_type": "llm",
                "provider_resource_id": "meta-llama/Llama-3.3-70B-Instruct",
                "metadata": {"max_tokens": 4096}
            }
        }


class ToolGroupResponse(BaseModel):
    identifier: str = Field(..., description="Unique identifier for the tool group", examples=["builtin::memory"])
    provider_id: Optional[str] = Field(None, description="Provider ID for the tool group")
    tool_count: Optional[int] = Field(None, description="Number of tools in this group", examples=[5])
    
    class Config:
        json_schema_extra = {
            "example": {
                "identifier": "builtin::memory",
                "provider_id": "meta-reference",
                "tool_count": 5
            }
        }


class ToolResponse(BaseModel):
    name: str = Field(..., description="Name of the tool", examples=["get_current_weather"])
    description: Optional[str] = Field(None, description="Description of what the tool does")
    type: Optional[str] = Field(None, description="Type of the tool")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Tool parameters schema")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "get_current_weather",
                "description": "Get the current weather for a location",
                "type": "function",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city and state"}
                    }
                }
            }
        }


class ContextRequest(BaseModel):
    query: str = Field(..., description="The search query for context retrieval", examples=["What is RAG?"])
    vector_store_name: Optional[str] = Field(None, description="Name of the vector store to search", examples=["my-docs"])
    max_results: int = Field(10, description="Maximum number of results to return", ge=1, le=100)
    score_threshold: float = Field(0.8, description="Minimum score threshold for results", ge=0.0, le=1.0)
    ranker: str = Field("default", description="Ranker to use for scoring")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is Retrieval-Augmented Generation?",
                "vector_store_name": "my-docs",
                "max_results": 10,
                "score_threshold": 0.8,
                "ranker": "default"
            }
        }


class ContextResponse(BaseModel):
    context: str = Field(..., description="Retrieved context from the vector store")
    vector_store_id: Optional[str] = Field(None, description="ID of the vector store used")
    query: str = Field(..., description="Original query string")
    
    class Config:
        json_schema_extra = {
            "example": {
                "context": "RAG (Retrieval-Augmented Generation) is a technique that combines information retrieval with text generation...",
                "vector_store_id": "vs_123456",
                "query": "What is RAG?"
            }
        }


class AgentRequest(BaseModel):
    agent_type: str = Field(..., description="Type of agent to execute", examples=["default", "lang_chain", "lang_graph"])
    input_text: str = Field(..., description="Input text/prompt for the agent", examples=["What is the weather in San Francisco?"])
    model_name: Optional[str] = Field(None, description="Model name to use", examples=["llama-3.3-70b-instruct"])
    system_instructions: Optional[str] = Field(None, description="System instructions for the agent")
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="List of tools/MCP servers to use")
    
    class Config:
        json_schema_extra = {
            "example": {
                "agent_type": "default",
                "input_text": "What is the weather in San Francisco today?",
                "model_name": "llama-3.3-70b-instruct",
                "system_instructions": "You are a helpful weather assistant.",
                "tools": [{"name": "weather", "mcp_server": "weather-server"}]
            }
        }


class AgentResponse(BaseModel):
    response: str = Field(..., description="Agent's response to the input")
    agent_type: str = Field(..., description="Type of agent that was executed")
    model_name: str = Field(..., description="Model name that was used")
    
    class Config:
        json_schema_extra = {
            "example": {
                "response": "The current weather in San Francisco is sunny with a temperature of 72°F.",
                "agent_type": "default",
                "model_name": "llama-3.3-70b-instruct"
            }
        }


class AgentTypeResponse(BaseModel):
    agent_types: List[str] = Field(..., description="List of available agent types")
    descriptions: Dict[str, str] = Field(..., description="Descriptions for each agent type")
    
    class Config:
        json_schema_extra = {
            "example": {
                "agent_types": ["default", "lang_chain", "lang_graph"],
                "descriptions": {
                    "default": "Uses Llama Stack's native MCP integration",
                    "lang_chain": "LangChain 1.0 agent with MCP adapters",
                    "lang_graph": "LangGraph ReAct agent with MCP"
                }
            }
        }


# Helper function to get client
def get_client() -> LlamaStackClient:
    """Get LlamaStack client with connection parameters from environment."""
    host = os.environ.get("LLAMA_STACK_HOST", "localhost")
    port = int(os.environ.get("LLAMA_STACK_PORT", "8080"))
    secure = os.environ.get("LLAMA_STACK_SECURE", "false").lower() in ["true", "1", "yes"]
    return create_client(host=host, port=port, secure=secure)


# Helper function for async agent execution
async def execute_agent_async(
    agent_type: str,
    input_text: str,
    model_name: str,
    system_instructions: str,
    tools: Optional[List[Dict[str, Any]]] = None
) -> tuple[str, List[str], Dict[str, Any]]:
    """
    Execute an agent asynchronously, handling both sync and async agent types.
    
    This function is designed to be called from FastAPI async endpoints to avoid
    the "asyncio.run() cannot be called from a running event loop" error.
    """
    from commands.agent_command import default_agent, langchain_agent, langgraph_agent
    
    vector_store_name = os.environ.get("VECTOR_STORE_NAME", None)
    
    if agent_type == "default":
        # Default agent is synchronous
        response, structured_data = default_agent(
            input_text=input_text,
            model_name=model_name,
            system_instructions=system_instructions,
            tools=tools,
            vector_store_name=vector_store_name
        )
        return response, [], structured_data
    
    elif agent_type == "lang_chain":
        # LangChain agent is async - call directly (we're already in async context)
        response, structured_data = await langchain_agent(
            input_text=input_text,
            model_name=model_name,
            system_instructions=system_instructions,
            tools=tools,
            vector_store_name=vector_store_name
        )
        return response, [], structured_data
    
    elif agent_type == "lang_graph":
        # LangGraph agent is async - call directly (we're already in async context)
        response, structured_data = await langgraph_agent(
            input_text=input_text,
            model_name=model_name,
            system_instructions=system_instructions,
            tools=tools,
            vector_store_name=vector_store_name
        )
        return response, [], structured_data
    
    else:
        raise ValueError(f"Unsupported agent_type: {agent_type}")



# Health check endpoint
@app.get(
    "/health",
    tags=["health"],
    summary="Health Check",
    description="Check if the API service is running and healthy",
    response_description="Service status information"
)
async def health_check():
    """
    Health check endpoint.
    
    Returns a simple status message to verify the service is running.
    This is a lightweight check suitable for liveness probes.
    """
    return {"status": "healthy", "service": "LlamaStack REST API"}


# Readiness check endpoint
@app.get(
    "/ready",
    tags=["health"],
    summary="Readiness Check",
    description="Check if the service is ready to handle requests by verifying LlamaStack connectivity",
    response_description="Service readiness status"
)
async def readiness_check():
    """
    Readiness check endpoint for Kubernetes probes.
    
    This endpoint verifies that:
    1. The LlamaStack client can be created
    2. The LlamaStack service is reachable
    3. Basic API calls (like listing models) work
    
    Returns:
    - 200: Service is ready to handle requests
    - 503: Service is not ready (dependency unavailable)
    
    **Use in Kubernetes:**
    ```yaml
    readinessProbe:
      httpGet:
        path: /ready
        port: 8700
      initialDelaySeconds: 5
      periodSeconds: 10
      timeoutSeconds: 5
      failureThreshold: 3
    ```
    """
    try:
        # Get connection parameters
        host = os.environ.get("LLAMA_STACK_HOST", "localhost")
        port = int(os.environ.get("LLAMA_STACK_PORT", "8080"))
        secure = os.environ.get("LLAMA_STACK_SECURE", "false").lower() in ["true", "1", "yes"]
        
        # Attempt to create client
        client = create_client(host=host, port=port, secure=secure)
        
        # Verify connectivity by making a simple API call
        try:
            models = list(client.models.list())
            model_count = len(models)
        except Exception as e:
            # If we can't list models, service is not ready
            raise HTTPException(
                status_code=503,
                detail=f"LlamaStack service not ready: Unable to list models - {str(e)}"
            )
        
        # Service is ready
        return {
            "status": "ready",
            "service": "LlamaStack REST API",
            "llama_stack": {
                "host": host,
                "port": port,
                "secure": secure,
                "connected": True,
                "models_available": model_count
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        # Any other error means service is not ready
        raise HTTPException(
            status_code=503,
            detail=f"Service not ready: {str(e)}"
        )


# Models endpoints
@app.get(
    "/models",
    response_model=List[ModelResponse],
    tags=["models"],
    summary="List All Models",
    description="Retrieve a list of all available models, with optional filtering by provider and type",
    response_description="List of available models"
)
async def get_models(
    provider: Optional[str] = Query(None, description="Filter by provider ID (e.g., 'meta', 'ollama')", examples=["meta"]),
    type: Optional[str] = Query(None, description="Filter by model type (e.g., 'llm', 'embedding')", examples=["llm"])
):
    """
    Get all available models.
    
    This endpoint returns a list of all models available in the LlamaStack instance.
    You can filter results by provider or model type.
    
    **Examples:**
    - Get all models: `GET /models`
    - Get Meta models only: `GET /models?provider=meta`
    - Get LLM models only: `GET /models?type=llm`
    
    **Parameters:**
    - **provider**: Filter by provider ID (e.g., 'meta', 'ollama')
    - **type**: Filter by model type (e.g., 'llm', 'embedding')
    
    **Returns:**
    - List of model objects with identifier, provider, type, and metadata
    """
    try:
        client = get_client()
        models: List[Model] = list_models(client)
        
        # Apply filters
        if provider:
            models = [m for m in models if m.provider_id == provider]
        if type:
            models = [m for m in models if m.api_model_type == type]
        
        # Convert to response model
        result = []
        for model in models:
            result.append(ModelResponse(
                identifier=model.identifier,
                provider_id=model.provider_id,
                api_model_type=model.api_model_type,
                provider_resource_id=getattr(model, 'provider_resource_id', None),
                metadata=getattr(model, 'metadata', None)
            ))
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching models: {str(e)}")


@app.get(
    "/models/{model_identifier}",
    response_model=ModelResponse,
    tags=["models"],
    summary="Get Model Details",
    description="Retrieve detailed information about a specific model by its identifier (e.g., llama-3.3-70b-instruct)",
    response_description="Detailed model information"
)
async def get_model_info(model_identifier: str):
    """
    Get detailed information about a specific model.
    
    This endpoint returns comprehensive information about a single model,
    including its provider, type, resource ID, and metadata.
    
    **Example:**
    - `GET /models/llama-3.3-70b-instruct`
    
    **Parameters:**
    - **model_identifier**: The unique identifier of the model
    
    **Returns:**
    - Model object with full details
    
    **Errors:**
    - 404: Model not found
    """
    try:
        client = get_client()
        models: List[Model] = list_models(client)
        
        # Find the specific model
        model = next((m for m in models if m.identifier == model_identifier), None)
        
        if not model:
            raise HTTPException(status_code=404, detail=f"Model '{model_identifier}' not found")
        
        return ModelResponse(
            identifier=model.identifier,
            provider_id=model.provider_id,
            api_model_type=model.api_model_type,
            provider_resource_id=getattr(model, 'provider_resource_id', None),
            metadata=getattr(model, 'metadata', None)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching model info: {str(e)}")


# Tools endpoints
@app.get(
    "/tools/groups",
    response_model=List[ToolGroupResponse],
    tags=["tools"],
    summary="List Tool Groups",
    description="Retrieve all available tool groups with their identifiers and tool counts",
    response_description="List of tool groups"
)
async def get_tool_groups():
    """
    Get all available tool groups.
    
    Tool groups are collections of related tools that can be used by agents.
    This endpoint returns a list of all available tool groups along with
    the number of tools in each group.
    
    **Example:**
    - `GET /tools/groups`
    
    **Returns:**
    - List of tool group objects with identifier, provider, and tool count
    """
    try:
        client = get_client()
        tool_groups = list(client.toolgroups.list())
        
        # Convert to response model
        result = []
        for group in tool_groups:
            # Try to get tool count
            tool_count = None
            try:
                tools_response = client.tools.list(toolgroup_id=group.identifier)
                tool_count = len(list(tools_response)) if tools_response else 0
            except Exception:
                pass
            
            result.append(ToolGroupResponse(
                identifier=group.identifier if hasattr(group, 'identifier') else 'N/A',
                provider_id=getattr(group, 'provider_id', None),
                tool_count=tool_count
            ))
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching tool groups: {str(e)}")


@app.get(
    "/tools",
    response_model=List[ToolResponse],
    tags=["tools"],
    summary="List Tools",
    description="Retrieve all available tools, optionally filtered by tool group",
    response_description="List of tools"
)
async def get_tools(
    group: Optional[str] = Query(None, description="Filter by specific tool group identifier", examples=["builtin::memory"]),
    all: bool = Query(True, description="Get tools from all groups if no group specified")
):
    """
    Get all available tools.
    
    This endpoint returns a list of all tools available for use by agents.
    You can filter by specific tool group or get all tools at once.
    
    **Examples:**
    - Get all tools: `GET /tools`
    - Get tools from specific group: `GET /tools?group=builtin::memory`
    
    **Parameters:**
    - **group**: Filter by specific tool group identifier
    - **all**: Get tools from all groups (default: True)
    
    **Returns:**
    - List of tool objects with name, description, type, and parameters
    """
    try:
        client = get_client()
        tool_groups = list(client.toolgroups.list())
        
        if not tool_groups:
            return []
        
        # Filter tool groups if needed
        if group:
            selected_groups = [g for g in tool_groups if 
                              (hasattr(g, 'identifier') and g.identifier == group)]
            if not selected_groups:
                raise HTTPException(status_code=404, detail=f"Tool group '{group}' not found")
        elif all:
            selected_groups = tool_groups
        else:
            return []
        
        # Collect all tools
        result = []
        for tg in selected_groups:
            group_identifier = tg.identifier if hasattr(tg, 'identifier') else None
            if not group_identifier:
                continue
            
            try:
                tools_response = client.tools.list(toolgroup_id=group_identifier)
                tools = list(tools_response) if tools_response else []
                
                for tool in tools:
                    result.append(ToolResponse(
                        name=getattr(tool, 'name', 'Unknown'),
                        description=getattr(tool, 'description', None),
                        type=getattr(tool, 'type', None),
                        parameters=getattr(tool, 'parameters', None) if hasattr(tool, 'parameters') else None
                    ))
            except Exception:
                continue
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching tools: {str(e)}")


# Context endpoint
@app.post(
    "/context",
    response_model=ContextResponse,
    tags=["context"],
    summary="Retrieve Context",
    description="Retrieve relevant context from a vector store using RAG (Retrieval-Augmented Generation)",
    response_description="Retrieved context and metadata"
)
async def get_context(request: ContextRequest):
    """
    Retrieve context from a vector store for a given query.
    
    This endpoint performs RAG (Retrieval-Augmented Generation) by searching
    a vector store and returning relevant context based on semantic similarity.
    
    **How it works:**
    1. Takes your query and converts it to an embedding
    2. Searches the vector store for similar content
    3. Returns the most relevant context based on score threshold
    
    **Example Request:**
    ```json
    {
      "query": "What is RAG?",
      "vector_store_name": "my-docs",
      "max_results": 10,
      "score_threshold": 0.8
    }
    ```
    
    **Parameters:**
    - **query**: The search query
    - **vector_store_name**: Name of the vector store (or use VECTOR_STORE_NAME env var)
    - **max_results**: Maximum number of results (1-100, default: 10)
    - **score_threshold**: Minimum similarity score (0.0-1.0, default: 0.8)
    - **ranker**: Ranker algorithm (default: "default")
    
    **Returns:**
    - Retrieved context text
    - Vector store ID
    - Original query
    
    **Errors:**
    - 400: Missing vector store name
    - 404: Vector store not found
    """
    try:
        client = get_client()
        
        # Use provided vector store name or default from environment
        vector_store_name = request.vector_store_name or os.environ.get("VECTOR_STORE_NAME")
        
        if not vector_store_name:
            raise HTTPException(
                status_code=400, 
                detail="Vector store name must be provided either in request or VECTOR_STORE_NAME environment variable"
            )
        
        # Get context from vector store
        context = get_rag_context(
            client=client,
            vector_store_name=vector_store_name,
            query=request.query,
            max_results=request.max_results,
            score_threshold=request.score_threshold,
            ranker=request.ranker
        )
        
        # Get vector store ID for reference
        from vector_stores import list_vector_stores
        vector_stores = list_vector_stores(client, name=vector_store_name)
        vector_store_id = vector_stores[0].id if vector_stores else None
        
        return ContextResponse(
            context=context,
            vector_store_id=vector_store_id,
            query=request.query
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving context: {str(e)}")


# Agents endpoints
@app.get(
    "/agents/types",
    response_model=AgentTypeResponse,
    tags=["agents"],
    summary="List Agent Types",
    description="Get a list of all available agent types and their descriptions",
    response_description="Available agent types and descriptions"
)
async def get_agent_types():
    """
    Get list of available agent types and their descriptions.
    
    **Available Agent Types:**
    
    1. **default**: Uses Llama Stack's native MCP integration with server-side tool calling
       - Best for: General purpose use with MCP servers
       - Tool execution: Server-side
    
    2. **lang_chain**: LangChain 1.0 agent with MCP adapters for client-side tool execution
       - Best for: Complex workflows with LangChain ecosystem
       - Tool execution: Client-side
    
    3. **lang_graph**: LangGraph ReAct agent with MCP adapters using LangChain
       - Best for: Multi-step reasoning and complex agent workflows
       - Tool execution: Client-side with ReAct pattern
    
    **Returns:**
    - List of agent type names
    - Descriptions for each agent type
    """
    return AgentTypeResponse(
        agent_types=["default", "lang_chain", "lang_graph"],
        descriptions={
            "default": "Uses Llama Stack's native MCP integration with server-side tool calling",
            "lang_chain": "LangChain 1.0 agent with MCP adapters for client-side tool execution",
            "lang_graph": "LangGraph ReAct agent with MCP adapters using LangChain"
        }
    )


@app.post(
    "/agents/execute",
    response_model=AgentResponse,
    tags=["agents"],
    summary="Execute Agent",
    description="Execute an AI agent with specified configuration and input",
    response_description="Agent execution result"
)
async def execute_agent(request: AgentRequest):
    """
    Execute an agent query.
    
    This endpoint runs an AI agent with the specified configuration and returns
    the agent's response. The agent can use tools, access context, and follow
    custom instructions.
    
    **Example Request:**
    ```json
    {
      "agent_type": "default",
      "input_text": "What is the weather in San Francisco?",
      "model_name": "llama-3.3-70b-instruct",
      "system_instructions": "You are a helpful assistant.",
      "tools": [
        {"name": "weather", "mcp_server": "weather-server"}
      ]
    }
    ```
    
    **Parameters:**
    - **agent_type**: Type of agent (default, lang_chain, lang_graph)
    - **input_text**: The prompt/question for the agent
    - **model_name**: Model to use (or use MODEL_NAME env var)
    - **system_instructions**: Custom system instructions (optional)
    - **tools**: List of tools/MCP servers to make available (optional)
    
    **Returns:**
    - Agent's response text
    - Agent type used
    - Model name used
    
    **Errors:**
    - 400: Invalid agent type or missing model name
    - 500: Agent execution error
    """
    try:
        # Validate agent type
        if request.agent_type not in ["default", "lang_chain", "lang_graph"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid agent_type: '{request.agent_type}'. Must be one of: default, lang_chain, lang_graph"
            )
        
        # Get model name from request or environment
        model_name = request.model_name or os.environ.get("MODEL_NAME")
        if not model_name:
            raise HTTPException(
                status_code=400,
                detail="Model name must be provided either in request or MODEL_NAME environment variable"
            )
        
        # Get system instructions from request or environment
        system_instructions = request.system_instructions or os.environ.get(
            "SYSTEM_INSTRUCTIONS",
            "You are a helpful AI assistant. Answer questions accurately and concisely."
        )
        
        # Execute agent command asynchronously
        response, logs, structured_data = await execute_agent_async(
            agent_type=request.agent_type,
            input_text=request.input_text,
            model_name=model_name,
            system_instructions=system_instructions,
            tools=request.tools
        )
        
        return AgentResponse(
            response=response,
            agent_type=request.agent_type,
            model_name=model_name
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing agent: {str(e)}")


# Main entry point
if __name__ == "__main__":
    import uvicorn
    
    # Get server configuration from environment
    host = os.environ.get("SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("SERVER_PORT", "8700"))
    
    print("=" * 80)
    print(f"🚀 Starting LlamaStack REST API Server")
    print("=" * 80)
    print(f"📍 Server URL: http://{host}:{port}")
    print(f"📚 Swagger UI (Interactive API Docs): http://{host}:{port}/docs")
    print(f"📖 ReDoc (Alternative Docs): http://{host}:{port}/redoc")
    print(f"📋 OpenAPI JSON Schema: http://{host}:{port}/openapi.json")
    print("=" * 80)
    print()
    print("Available Endpoints:")
    print("  - GET  /health              - Health check (liveness probe)")
    print("  - GET  /ready               - Readiness check (readiness probe)")
    print("  - GET  /models              - List all models")
    print("  - GET  /models/{id}         - Get model details")
    print("  - GET  /tools/groups        - List tool groups")
    print("  - GET  /tools               - List tools")
    print("  - POST /context             - Retrieve RAG context")
    print("  - GET  /agents/types        - List agent types")
    print("  - POST /agents/execute      - Execute an agent")
    print("=" * 80)
    print()
    print("💡 Tip: Visit the Swagger UI to test all endpoints interactively!")
    print()
    print("Kubernetes Probes:")
    print("  - Liveness:  GET /health")
    print("  - Readiness: GET /ready")
    print()
    
    uvicorn.run(app, host=host, port=port)

