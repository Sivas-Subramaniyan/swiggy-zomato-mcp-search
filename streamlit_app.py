import os
import asyncio
import uuid

import streamlit as st
from dotenv import load_dotenv
import mlflow

from mcp_client import MCPServerManager
from agent import create_food_aggregator_agent

load_dotenv()

# ─── MLflow ───
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
try:
    mlflow.set_experiment("Food_Aggregator_Agent")
except Exception:
    pass
mlflow.langchain.autolog()

# ─── Page config ───
st.set_page_config(
    page_title="🍔 Food Aggregator Agent",
    page_icon="🍔",
    layout="centered",
)

# ─── Custom styling ───
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .main-header h1 {
        background: linear-gradient(90deg, #f7971e, #ffd200);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
    }
    .main-header p {
        color: #a0a0b8;
        font-size: 0.95rem;
    }
    div[data-testid="stChatMessage"] {
        border-radius: 12px;
        margin-bottom: 0.5rem;
    }
    .sidebar-info {
        background: rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .sidebar-info h4 {
        color: #ffd200;
        margin-bottom: 0.4rem;
    }
    .sidebar-info p, .sidebar-info li {
        color: #c0c0d0;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Async helpers ───
# Streamlit runs in its own event loop; we need a helper to bridge async code.
def run_async(coro):
    """Run an async coroutine from sync Streamlit code."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def initialize_agent():
    """Connect to MCP servers and build the LangGraph agent."""
    manager = MCPServerManager()

    swiggy_url = os.getenv("SWIGGY_MCP_URL", "https://mcp.swiggy.com/food")
    zomato_url = os.getenv("ZOMATO_MCP_URL", "https://mcp-server.zomato.com/mcp")

    connected = []
    try:
        await manager.connect_server("swiggy", "npx", ["-y", "mcp-remote", swiggy_url])
        connected.append("Swiggy ✅")
    except Exception as e:
        connected.append(f"Swiggy ❌ ({e})")

    try:
        await manager.connect_server("zomato", "npx", ["-y", "mcp-remote", zomato_url])
        connected.append("Zomato ✅")
    except Exception as e:
        connected.append(f"Zomato ❌ ({e})")

    agent = create_food_aggregator_agent(manager.all_tools) if manager.all_tools else None
    return manager, agent, connected


async def ask_agent(agent, query: str, thread_id: str):
    """Send a single message to the agent and return the text reply."""
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [("user", query)]}
    result = await agent.ainvoke(inputs, config=config)

    raw = result["messages"][-1].content
    # Gemini sometimes returns list-of-dicts 
    if isinstance(raw, list) and len(raw) > 0:
        if isinstance(raw[0], dict) and "text" in raw[0]:
            return raw[0]["text"]
        if isinstance(raw[0], str):
            return "".join(raw)
    return str(raw)


# ─── Session state init ───
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None
    st.session_state.manager = None
    st.session_state.connection_status = []


# ─── Sidebar ───
with st.sidebar:
    st.markdown("""
    <div class="sidebar-info">
        <h4>🍔 Food Aggregator</h4>
        <p>Compare food prices across <b>Swiggy</b> &amp; <b>Zomato</b> instantly.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔌 Connect to MCP Servers", use_container_width=True):
        with st.spinner("Connecting to Swiggy & Zomato MCP servers..."):
            manager, agent, status = run_async(initialize_agent())
            st.session_state.manager = manager
            st.session_state.agent = agent
            st.session_state.connection_status = status
        st.rerun()

    if st.session_state.connection_status:
        for s in st.session_state.connection_status:
            st.markdown(f"- {s}")

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.markdown("""
    <div class="sidebar-info">
        <h4>💡 Try asking</h4>
        <ul>
            <li>I want to order Narmada Chicken Biriyani</li>
            <li>Find me the cheapest pizza near me</li>
            <li>Show me my saved addresses</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.caption(f"Session: `{st.session_state.thread_id[:8]}...`")


# ─── Header ───
st.markdown("""
<div class="main-header">
    <h1>🍔 Food Aggregator Agent</h1>
    <p>Compare Swiggy &amp; Zomato — find the best deals on your favourite food</p>
</div>
""", unsafe_allow_html=True)


# ─── Chat history ───
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ─── Chat input ───
if prompt := st.chat_input("What would you like to eat today?"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check agent is connected
    if not st.session_state.agent:
        with st.chat_message("assistant"):
            st.warning("⚠️ Please click **Connect to MCP Servers** in the sidebar first!")
        st.stop()

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Searching Swiggy & Zomato..."):
            try:
                response = run_async(
                    ask_agent(st.session_state.agent, prompt, st.session_state.thread_id)
                )
            except Exception as e:
                response = f"❌ Error: {e}"
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
