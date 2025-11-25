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


# Initialize FastAPI app
app = FastAPI(
    title="LlamaStack REST API",
    description="REST API for LlamaStack operations including models, tools, context, and agents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class ModelResponse(BaseModel):
    identifier: str
    provider_id: str
    api_model_type: str
    provider_resource_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ToolGroupResponse(BaseModel):
    identifier: str
    provider_id: Optional[str] = None
    tool_count: Optional[int] = None


class ToolResponse(BaseModel):
    name: str
    description: Optional[str] = None
    type: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class ContextRequest(BaseModel):
    query: str = Field(..., description="The search query for context retrieval")
    vector_store_name: Optional[str] = Field(None, description="Name of the vector store to search")
    max_results: int = Field(10, description="Maximum number of results to return")
    score_threshold: float = Field(0.8, description="Minimum score threshold for results")
    ranker: str = Field("default", description="Ranker to use for scoring")


class ContextResponse(BaseModel):
    context: str
    vector_store_id: Optional[str] = None
    query: str


class AgentRequest(BaseModel):
    agent_type: str = Field(..., description="Type of agent (default, lang_chain, lang_graph)")
    input_text: str = Field(..., description="Input text/prompt for the agent")
    model_name: Optional[str] = Field(None, description="Model name to use")
    system_instructions: Optional[str] = Field(None, description="System instructions for the agent")
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="List of tools/MCP servers to use")


class AgentResponse(BaseModel):
    response: str
    agent_type: str
    model_name: str


class AgentTypeResponse(BaseModel):
    agent_types: List[str]
    descriptions: Dict[str, str]


# Helper function to get client
def get_client() -> LlamaStackClient:
    """Get LlamaStack client with connection parameters from environment."""
    host = os.environ.get("LLAMA_STACK_HOST", "localhost")
    port = int(os.environ.get("LLAMA_STACK_PORT", "8080"))
    secure = os.environ.get("LLAMA_STACK_SECURE", "false").lower() in ["true", "1", "yes"]
    return create_client(host=host, port=port, secure=secure)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "LlamaStack REST API"}


# Models endpoints
@app.get("/models", response_model=List[ModelResponse])
async def get_models(
    provider: Optional[str] = Query(None, description="Filter by provider ID"),
    type: Optional[str] = Query(None, description="Filter by model type")
):
    """
    Get all available models.
    
    Parameters:
    - provider: Filter by provider ID (e.g., 'meta', 'ollama')
    - type: Filter by model type (e.g., 'llm', 'embedding')
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


@app.get("/models/{model_identifier}", response_model=ModelResponse)
async def get_model_info(model_identifier: str):
    """
    Get detailed information about a specific model.
    
    Parameters:
    - model_identifier: The identifier of the model
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
@app.get("/tools/groups", response_model=List[ToolGroupResponse])
async def get_tool_groups():
    """
    Get all available tool groups.
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


@app.get("/tools", response_model=List[ToolResponse])
async def get_tools(
    group: Optional[str] = Query(None, description="Filter by tool group identifier"),
    all: bool = Query(True, description="Get tools from all groups if no group specified")
):
    """
    Get all available tools.
    
    Parameters:
    - group: Filter by specific tool group identifier
    - all: Get tools from all groups (default: True)
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
@app.post("/context", response_model=ContextResponse)
async def get_context(request: ContextRequest):
    """
    Retrieve context from a vector store for a given query.
    
    This endpoint performs RAG (Retrieval-Augmented Generation) by searching
    a vector store and returning relevant context.
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
@app.get("/agents/types", response_model=AgentTypeResponse)
async def get_agent_types():
    """
    Get list of available agent types and their descriptions.
    """
    return AgentTypeResponse(
        agent_types=["default", "lang_chain", "lang_graph"],
        descriptions={
            "default": "Uses Llama Stack's native MCP integration with server-side tool calling",
            "lang_chain": "LangChain 1.0 agent with MCP adapters for client-side tool execution",
            "lang_graph": "LangGraph ReAct agent with MCP adapters using LangChain"
        }
    )


@app.post("/agents/execute", response_model=AgentResponse)
async def execute_agent(request: AgentRequest):
    """
    Execute an agent query.
    
    This endpoint runs an agent with the specified configuration and returns the response.
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
        
        # Execute agent command
        response = agent_command(
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
    port = int(os.environ.get("SERVER_PORT", "8000"))
    
    print(f"Starting LlamaStack REST API server on {host}:{port}")
    print(f"API documentation available at http://{host}:{port}/docs")
    
    uvicorn.run(app, host=host, port=port)

