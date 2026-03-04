#!/usr/bin/env python3
"""
Streamlit Chat Application for LlamaStack using portazgo Agent.

A chatbot that uses portazgo.Agent.invoke_stream with configuration from
LLAMA_STACK_HOST, LLAMA_STACK_PORT, LLAMA_STACK_SECURE.

Health checks run in a separate process (health-server.py) - Flask and Streamlit
conflict when embedded together (port binding fails on Streamlit reload).
"""

import html
import inspect
import json
import os
from typing import Any, Dict, List

import streamlit as st

from portazgo import (
    Agent,
    discover_mcp_tools,
    list_vector_store_names,
    resolve_vector_store_id,
)
from llama_stack_client import LlamaStackClient

from utils import create_client, list_models


st.set_page_config(
    page_title="LlamaStack Chat (portazgo)",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_llama_client() -> LlamaStackClient:
    """Create LlamaStack client from environment variables."""
    host = os.environ.get("LLAMA_STACK_HOST", "localhost")
    port = int(os.environ.get("LLAMA_STACK_PORT", "8080"))
    secure = os.environ.get("LLAMA_STACK_SECURE", "false").lower() in ["true", "1", "yes"]
    return create_client(host=host, port=port, secure=secure)


@st.cache_resource
def get_cached_client() -> LlamaStackClient:
    """Get cached LlamaStack client."""
    return get_llama_client()


@st.cache_data(ttl=300)
def fetch_models() -> List[Dict[str, str]]:
    """Fetch available LLM models from LlamaStack."""
    try:
        client = get_cached_client()
        models = list_models(client)
        meta = lambda m: m.custom_metadata or {}
        return [
            {"identifier": m.id, "provider": meta(m).get("provider_id", ""), "type": meta(m).get("model_type", "")}
            for m in models
            if meta(m).get("model_type") == "llm"
        ]
    except Exception as e:
        st.error(f"Error fetching models: {e}")
        return []


@st.cache_data(ttl=300)
def fetch_vector_store_names() -> List[str]:
    """Fetch vector store names via portazgo (for dropdown)."""
    try:
        client = get_cached_client()
        vector_stores_names = list_vector_store_names(client)
        print(f"Vector stores names: {vector_stores_names}")
        return vector_stores_names
    except Exception as e:
        st.error(f"Error fetching vector stores: {e}")
        return []


@st.cache_data(ttl=300)
def fetch_mcp_tool_names() -> List[str]:
    """Fetch available MCP tool group names from LlamaStack."""
    try:
        client = get_cached_client()
        tool_groups = list(client.toolgroups.list())
        return [
            g.identifier.split("::", 1)[1] if "::" in g.identifier else g.identifier
            for g in tool_groups
            if getattr(g, "identifier", "").startswith("mcp::")
        ]
    except Exception:
        return []


# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_type" not in st.session_state:
    st.session_state.agent_type = "default"
if "instructions" not in st.session_state:
    st.session_state.instructions = os.environ.get(
        "SYSTEM_INSTRUCTIONS",
        "You are a helpful AI assistant. Answer questions accurately and concisely.",
    )
if "debug_tools" not in st.session_state:
    st.session_state.debug_tools = True
if "show_think_tokens" not in st.session_state:
    st.session_state.show_think_tokens = True


# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")

    st.subheader("Agent Type")
    agent_type = st.selectbox(
        "Select Agent",
        options=["default", "lang_graph"],
        index=["default", "lang_graph"].index(st.session_state.agent_type),
        help="default: Llama Stack Responses API | lang_graph: LangGraph Agent",
    )
    st.session_state.agent_type = agent_type
    st.info(
        "default: portazgo (Llama Stack Responses API)"
        if agent_type == "default"
        else "lang_graph: LangGraph Agent"
    )

    st.divider()
    st.subheader("Model")
    models = fetch_models()
    if models:
        model_options = [f"{m['identifier']} ({m['provider']})" for m in models]
        default_model = os.environ.get("MODEL_NAME")
        default_index = 0
        if default_model:
            try:
                default_index = next(i for i, m in enumerate(models) if m["identifier"] == default_model)
            except StopIteration:
                pass
        # Use user's previous selection if available and still valid
        if "selected_model_id" in st.session_state and st.session_state.selected_model_id:
            try:
                default_index = next(i for i, m in enumerate(models) if m["identifier"] == st.session_state.selected_model_id)
            except StopIteration:
                pass
        selected_model_display = st.selectbox(
            "Select Model",
            options=model_options,
            index=default_index,
            key="model_selectbox",
        )
        selected_model_id = models[model_options.index(selected_model_display)]["identifier"]
    else:
        st.warning("No LLM models available")
        selected_model_id = None

    if "selected_model_id" in st.session_state and st.session_state.selected_model_id != selected_model_id:
        st.session_state.messages = []
    st.session_state.selected_model_id = selected_model_id

    st.divider()
    st.subheader("Vector Store")
    vs_names = fetch_vector_store_names()
    if vs_names:
        default_vs = os.environ.get("VECTOR_STORE_NAME")
        default_index = 0
        if default_vs and default_vs in vs_names:
            default_index = vs_names.index(default_vs) + 1
        selected_name = st.selectbox(
            "Select Vector Store",
            options=["(None)"] + vs_names,
            index=default_index,
        )
        if selected_name == "(None)":
            vector_store_id = ""
        else:
            try:
                vector_store_id = resolve_vector_store_id(get_cached_client(), selected_name)
            except ValueError as e:
                st.warning(str(e))
                vector_store_id = ""
    else:
        st.warning("No vector stores available")
        vector_store_id = ""

    st.divider()
    st.subheader("MCP Tools")
    mcp_names = fetch_mcp_tool_names()
    mcp_options = ["none", "all"] + mcp_names
    mcp_selection = st.selectbox("MCP Tools", options=mcp_options, index=0)
    mcp_tools_str = "none" if mcp_selection == "none" else ("all" if mcp_selection == "all" else mcp_selection)

    st.divider()
    st.subheader("Advanced")
    force_file_search = st.checkbox("Force file search", value=False)
    retrieval_mode = st.selectbox("Retrieval mode", options=["vector", "hybrid"], index=0)
    file_search_max_chunks = st.slider("File search max chunks", 1, 10, 5)
    file_search_score_threshold = st.slider("File search score threshold", 0.0, 1.0, 0.7, 0.05)
    file_search_max_tokens_per_chunk = st.number_input(
        "File search max tokens per chunk", 128, 2048, 512, 64
    )

    st.divider()
    st.subheader("System Instructions")
    instructions = st.text_area("Instructions", value=st.session_state.instructions, height=120)
    st.session_state.instructions = instructions

    st.divider()
    st.subheader("Debug")
    show_think_tokens = st.checkbox(
        "Show think tokens (<think>...</think>)",
        value=st.session_state.show_think_tokens,
        help="Keep thinking tokens in the final answer. When off, they appear during streaming then are removed.",
    )
    st.session_state.show_think_tokens = show_think_tokens
    debug_tools = st.checkbox(
        "Show tool/context debug info",
        value=st.session_state.debug_tools,
        help="Display tool_calls and contexts counts (and raw data) under each response.",
    )
    st.session_state.debug_tools = debug_tools

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Main chat
st.title("🦙 LlamaStack Chat (portazgo)")

host = os.environ.get("LLAMA_STACK_HOST", "localhost")
port = os.environ.get("LLAMA_STACK_PORT", "8080")
secure = os.environ.get("LLAMA_STACK_SECURE", "false")
st.caption(f"🔗 {host}:{port} (secure={secure}) | 🤖 {selected_model_id or 'Not selected'} | 🔧 {agent_type}")

st.divider()

def _parse_segments(text: str) -> List[tuple[str, str]]:
    """Parse text into ordered segments. Returns list of (type, content) where type is 'answer' or 'think'."""
    if not text:
        return []
    if "<think>" not in text:
        return [("answer", text)]

    segments: List[tuple[str, str]] = []
    i = 0
    while i < len(text):
        think_start = text.find("<think>", i)
        if think_start == -1:
            segments.append(("answer", text[i:]))
            break
        if think_start > i:
            segments.append(("answer", text[i:think_start]))
        content_start = think_start + len("<think>")
        think_end = text.find("</think>", content_start)
        if think_end == -1:
            segments.append(("think", text[content_start:]))
            break
        segments.append(("think", text[content_start:think_end]))
        i = think_end + len("</think>")
    return segments


def _render_assistant_content(content: str, *, show_think: bool = True) -> None:
    """Render assistant content in order: answer and think segments, each think in its own grayed expander."""
    segments = _parse_segments(content)
    for seg_type, seg_content in segments:
        seg_content = seg_content.strip()
        if not seg_content:
            continue
        if seg_type == "answer":
            st.markdown(seg_content)
        elif show_think:
            with st.expander("💭 Thinking", expanded=False):
                escaped = html.escape(seg_content).replace("\n", "<br>")
                st.markdown(
                    f'<div style="color: #6b7280; font-size: 0.9em;">{escaped}</div>',
                    unsafe_allow_html=True,
                )


def _render_tool_section(
    tool_calls: List[Dict],
    contexts: List[str],
    *,
    debug: bool = False,
) -> None:
    """Render a collapsible 'Tools Used' section under a response."""
    n_tools = len(tool_calls)
    n_ctx = len(contexts)
    if debug:
        st.caption(
            f"🔍 Debug: tool_calls={n_tools}, contexts={n_ctx}"
            + (" (streaming path may return empty)" if n_tools == 0 and n_ctx == 0 else "")
        )
    if not tool_calls and not contexts:
        return
    with st.expander("🔧 Tools used", expanded=False):
        if tool_calls:
            for i, tc in enumerate(tool_calls):
                name = tc.get("tool_name", "unknown")
                args = tc.get("arguments", {})
                resp = tc.get("response")
                st.markdown(f"**{name}**")
                if args:
                    st.code(json.dumps(args, indent=2), language="json")
                if resp is not None:
                    if isinstance(resp, list):
                        for r in resp:
                            if isinstance(r, str):
                                st.text(r[:300] + ("..." if len(r) > 300 else ""))
                            else:
                                st.json(r)
                    elif isinstance(resp, dict):
                        st.json(resp)
                    else:
                        txt = str(resp)
                        st.text(txt[:500] + ("..." if len(txt) > 500 else ""))
                if i < len(tool_calls) - 1:
                    st.divider()
        if contexts and not tool_calls:
            for i, ctx in enumerate(contexts):
                st.markdown(f"**Context {i + 1}**")
                st.text(ctx[:500] + ("..." if len(ctx) > 500 else ""))
                if i < len(contexts) - 1:
                    st.divider()
        if debug and (tool_calls or contexts):
            st.divider()
            st.markdown("**Raw (debug)**")
            st.json({"tool_calls": tool_calls, "contexts": contexts})


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("role") == "assistant":
            _render_assistant_content(
                message["content"],
                show_think=st.session_state.show_think_tokens,
            )
        else:
            st.markdown(message["content"])
        if message.get("role") == "assistant":
            _render_tool_section(
                message.get("tool_calls", []),
                message.get("contexts", []),
                debug=st.session_state.debug_tools,
            )

if prompt := st.chat_input("Type your message here..."):
    if not selected_model_id:
        st.error("Please select a model from the sidebar before chatting.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        messages_for_history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]
        client = get_cached_client()
        mcp_tools = discover_mcp_tools(client, mcp_tools_str)

        agent_type_for_portazgo = "default" if agent_type == "default" else "lang-graph"
        agent = Agent(type=agent_type_for_portazgo)

        invoke_kwargs: Dict[str, Any] = dict(
            client=client,
            input_text=prompt,
            model_id=selected_model_id,
            vector_store_id=vector_store_id or "",
            mcp_tools=mcp_tools,
            messages=messages_for_history if messages_for_history else None,
            instructions=instructions,
            force_file_search=force_file_search,
            ranker="default",
            retrieval_mode=retrieval_mode,
            file_search_max_chunks=file_search_max_chunks,
            file_search_score_threshold=file_search_score_threshold,
            file_search_max_tokens_per_chunk=file_search_max_tokens_per_chunk,
        )
        if "strip_think_blocks" in inspect.signature(Agent.invoke_stream).parameters:
            invoke_kwargs["strip_think_blocks"] = not st.session_state.show_think_tokens

        full_response = ""
        tool_calls: List[Dict[str, Any]] = []
        contexts: List[str] = []
        response_placeholder = st.empty()

        try:
            stream = agent.invoke_stream(**invoke_kwargs)
            for event in stream:
                if event.get("type") == "content_delta":
                    full_response += event.get("delta", "")
                    with response_placeholder.container():
                        _render_assistant_content(
                            full_response + "▌",
                            show_think=st.session_state.show_think_tokens,
                        )
                elif event.get("type") == "done":
                    full_response = event.get("answer", full_response)
                    tool_calls = event.get("tool_calls", [])
                    contexts = event.get("contexts", [])
                    break
        except NotImplementedError:
            st.error(f"Agent type '{agent_type}' is not yet implemented.")
            full_response = ""
        except Exception as e:
            st.error(str(e))
            full_response = ""

        if not full_response:
            try:
                result = agent.invoke(**invoke_kwargs)
                full_response = result.get("answer", "") or ""
                tool_calls = result.get("tool_calls", [])
                contexts = result.get("contexts", [])
            except Exception as e:
                st.error(str(e))
                full_response = ""

        with response_placeholder.container():
            _render_assistant_content(
                full_response,
                show_think=st.session_state.show_think_tokens,
            )
            _render_tool_section(
                tool_calls,
                contexts,
                debug=st.session_state.debug_tools,
            )

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "tool_calls": tool_calls,
            "contexts": contexts,
        })

st.divider()
st.caption("Built with Streamlit and portazgo | Use the sidebar to configure")
