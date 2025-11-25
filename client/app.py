#!/usr/bin/env python3
"""
Streamlit Chat Application for LlamaStack.

Provides a chat interface with:
- Sidebar for agent and tool configuration
- Chat panel for typing queries and viewing responses
"""

import os
import streamlit as st
from typing import List, Dict, Any, Optional

from llama_stack_client.types.model import Model

from utils import create_client, list_models
from commands.agent_command import agent_command


# Page configuration
st.set_page_config(
    page_title="LlamaStack Chat",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_type" not in st.session_state:
    st.session_state.agent_type = "default"

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

if "selected_tools" not in st.session_state:
    st.session_state.selected_tools = []

if "system_instructions" not in st.session_state:
    st.session_state.system_instructions = os.environ.get(
        "SYSTEM_INSTRUCTIONS",
        "You are a helpful AI assistant. Answer questions accurately and concisely."
    )

if "use_rag" not in st.session_state:
    st.session_state.use_rag = False

if "vector_store_name" not in st.session_state:
    st.session_state.vector_store_name = os.environ.get("VECTOR_STORE_NAME", "")


# Helper functions
@st.cache_resource
def get_llama_client():
    """Get cached LlamaStack client."""
    host = os.environ.get("LLAMA_STACK_HOST", "localhost")
    port = int(os.environ.get("LLAMA_STACK_PORT", "8080"))
    secure = os.environ.get("LLAMA_STACK_SECURE", "false").lower() in ["true", "1", "yes"]
    return create_client(host=host, port=port, secure=secure)


@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_models() -> List[Dict[str, str]]:
    """Fetch available models from LlamaStack."""
    try:
        client = get_llama_client()
        models: List[Model] = list_models(client)
        return [
            {
                "identifier": m.identifier,
                "provider": m.provider_id,
                "type": m.api_model_type
            }
            for m in models
        ]
    except Exception as e:
        st.error(f"Error fetching models: {e}")
        return []


@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_tool_groups() -> List[Dict[str, Any]]:
    """Fetch available tool groups from LlamaStack."""
    try:
        client = get_llama_client()
        tool_groups = list(client.toolgroups.list())
        
        result = []
        for group in tool_groups:
            # Get tools for this group
            tools = []
            try:
                tools_response = client.tools.list(toolgroup_id=group.identifier)
                tools = [
                    {
                        "name": t.name if hasattr(t, 'name') else 'Unknown',
                        "description": t.description if hasattr(t, 'description') else None,
                        "type": t.type if hasattr(t, 'type') else None
                    }
                    for t in list(tools_response)
                ] if tools_response else []
            except Exception:
                pass
            
            result.append({
                "identifier": group.identifier if hasattr(group, 'identifier') else 'N/A',
                "provider": getattr(group, 'provider_id', None),
                "tools": tools
            })
        
        return result
    except Exception as e:
        st.error(f"Error fetching tool groups: {e}")
        return []


def get_default_mcp_servers() -> List[Dict[str, Any]]:
    """Get default MCP server configuration."""
    return [
        {
            "type": "mcp",
            "server_label": "compatibility-engine-MCP-Server",
            "server_url": "https://compatibility-engine-llama-stack-demo.apps.ocp.sandbox3322.opentlc.com/sse"
        },
        {
            "type": "mcp",
            "server_label": "eligibility-engine-MCP-Server",
            "server_url": "https://eligibility-engine-llama-stack-demo.apps.ocp.sandbox3322.opentlc.com/sse"
        },
        {
            "type": "mcp",
            "server_label": "finance-engine-MCP-Server",
            "server_url": "https://finance-engine-llama-stack-demo.apps.ocp.sandbox3322.opentlc.com/sse"
        },
    ]


def execute_agent_query(
    agent_type: str,
    input_text: str,
    model_name: str,
    system_instructions: str,
    tools: Optional[List[Dict[str, Any]]] = None
) -> str:
    """Execute an agent query and return the response."""
    try:
        response = agent_command(
            agent_type=agent_type,
            input_text=input_text,
            model_name=model_name,
            system_instructions=system_instructions,
            tools=tools
        )
        return response
    except Exception as e:
        return f"Error: {str(e)}"


# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Agent type selection
    st.subheader("Agent Type")
    agent_type = st.selectbox(
        "Select Agent",
        options=["default", "lang_chain", "lang_graph"],
        index=["default", "lang_chain", "lang_graph"].index(st.session_state.agent_type),
        help="""
        - **default**: Native Llama Stack with server-side MCP
        - **lang_chain**: LangChain 1.0 with client-side MCP
        - **lang_graph**: LangGraph ReAct agent with MCP
        """
    )
    st.session_state.agent_type = agent_type
    
    # Agent type description
    agent_descriptions = {
        "default": "🚀 Uses Llama Stack's native MCP integration with server-side tool calling",
        "lang_chain": "🔗 LangChain 1.0 agent with MCP adapters for client-side tool execution",
        "lang_graph": "📊 LangGraph ReAct agent with MCP adapters using LangChain"
    }
    st.info(agent_descriptions[agent_type])
    
    st.divider()
    
    # Model selection
    st.subheader("Model")
    models = fetch_models()
    
    if models:
        # Filter for LLM models
        llm_models = [m for m in models if m["type"] == "llm"]
        
        if llm_models:
            model_options = [f"{m['identifier']} ({m['provider']})" for m in llm_models]
            
            # Get default model from environment or use first available
            default_model = os.environ.get("MODEL_NAME")
            default_index = 0
            if default_model:
                try:
                    default_index = next(
                        i for i, m in enumerate(llm_models)
                        if m['identifier'] == default_model
                    )
                except StopIteration:
                    pass
            
            selected_model_display = st.selectbox(
                "Select Model",
                options=model_options,
                index=default_index,
                help="Choose the LLM model to use for generating responses"
            )
            
            # Extract model identifier
            st.session_state.selected_model = llm_models[model_options.index(selected_model_display)]["identifier"]
        else:
            st.warning("No LLM models available")
            st.session_state.selected_model = None
    else:
        st.warning("Unable to fetch models")
        st.session_state.selected_model = None
    
    st.divider()
    
    # Tools configuration
    st.subheader("Tools")
    
    use_default_tools = st.checkbox(
        "Use Default MCP Servers",
        value=True,
        help="Use the default configured MCP servers"
    )
    
    if use_default_tools:
        default_servers = get_default_mcp_servers()
        st.session_state.selected_tools = default_servers
        
        with st.expander("Default MCP Servers", expanded=False):
            for server in default_servers:
                st.write(f"**{server['server_label']}**")
                st.caption(f"URL: {server['server_url']}")
    else:
        st.session_state.selected_tools = []
        st.info("Agent will run without tools")
    
    # Show available tool groups
    tool_groups = fetch_tool_groups()
    if tool_groups:
        with st.expander("Available Tool Groups", expanded=False):
            for group in tool_groups:
                st.write(f"**{group['identifier']}**")
                if group['tools']:
                    for tool in group['tools']:
                        st.caption(f"  • {tool['name']}: {tool['description'] or 'No description'}")
                else:
                    st.caption("  (No tools)")
    
    st.divider()
    
    # System instructions
    st.subheader("System Instructions")
    system_instructions = st.text_area(
        "Instructions",
        value=st.session_state.system_instructions,
        height=150,
        help="Provide instructions for the agent's behavior"
    )
    st.session_state.system_instructions = system_instructions
    
    st.divider()
    
    # RAG Configuration (optional)
    st.subheader("RAG (Optional)")
    use_rag = st.checkbox(
        "Enable RAG",
        value=st.session_state.use_rag,
        help="Enable Retrieval-Augmented Generation using vector stores"
    )
    st.session_state.use_rag = use_rag
    
    if use_rag:
        vector_store_name = st.text_input(
            "Vector Store Name",
            value=st.session_state.vector_store_name,
            help="Name of the vector store to use for RAG"
        )
        st.session_state.vector_store_name = vector_store_name
        
        if not vector_store_name:
            st.warning("Please specify a vector store name")
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Main chat interface
st.title("🦙 LlamaStack Chat")

# Display connection info
col1, col2, col3 = st.columns(3)
with col1:
    host = os.environ.get("LLAMA_STACK_HOST", "localhost")
    port = os.environ.get("LLAMA_STACK_PORT", "8080")
    st.caption(f"🔗 Connected to: {host}:{port}")
with col2:
    if st.session_state.selected_model:
        st.caption(f"🤖 Model: {st.session_state.selected_model.split('/')[-1]}")
    else:
        st.caption("🤖 Model: Not selected")
with col3:
    st.caption(f"🔧 Agent: {st.session_state.agent_type}")

st.divider()

# Display chat messages
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Check if model is selected
    if not st.session_state.selected_model:
        st.error("Please select a model from the sidebar before chatting.")
        st.stop()
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Prepare tools
            tools = st.session_state.selected_tools if st.session_state.selected_tools else []
            
            # Execute agent
            response = execute_agent_query(
                agent_type=st.session_state.agent_type,
                input_text=prompt,
                model_name=st.session_state.selected_model,
                system_instructions=st.session_state.system_instructions,
                tools=tools
            )
            
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})


# Footer
st.divider()
st.caption("Built with Streamlit and LlamaStack | Use the sidebar to configure your agent")


# Display environment info in expander
with st.expander("🔍 Environment Information", expanded=False):
    st.write("**Environment Variables:**")
    env_vars = {
        "LLAMA_STACK_HOST": os.environ.get("LLAMA_STACK_HOST", "Not set"),
        "LLAMA_STACK_PORT": os.environ.get("LLAMA_STACK_PORT", "Not set"),
        "LLAMA_STACK_SECURE": os.environ.get("LLAMA_STACK_SECURE", "Not set"),
        "MODEL_NAME": os.environ.get("MODEL_NAME", "Not set"),
        "VECTOR_STORE_NAME": os.environ.get("VECTOR_STORE_NAME", "Not set"),
    }
    for key, value in env_vars.items():
        st.code(f"{key}={value}")

