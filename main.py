import os
import asyncio
import argparse
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import mlflow
from langchain_core.messages import HumanMessage

from mcp_client import MCPServerManager
from agent import create_food_aggregator_agent

load_dotenv()

# ─── MLflow ───
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(mlflow_uri)
try:
    mlflow.set_experiment("Food_Aggregator_Agent")
except Exception:
    pass
try:
    mlflow.langchain.autolog()
except Exception:
    pass

# ─── Global state ───
manager = MCPServerManager()
app_agent = None
connection_status = []

# ─── MCP connection logic ───
async def connect_to_mcp_servers():
    """Connect to Swiggy and Zomato MCP servers."""
    global app_agent, connection_status
    connection_status = []

    swiggy_url = os.getenv("SWIGGY_MCP_URL", "https://mcp.swiggy.com/food")
    zomato_url = os.getenv("ZOMATO_MCP_URL", "https://mcp-server.zomato.com/mcp")

    # Swiggy
    try:
        await manager.connect_server("swiggy", "npx", ["-y", "mcp-remote", swiggy_url])
        swiggy_tools = len([t for t in manager.all_tools])  # Count after connect
        connection_status.append({"name": "Swiggy", "status": "ok", "tools": swiggy_tools})
    except Exception as e:
        connection_status.append({"name": "Swiggy", "status": "error", "detail": str(e)})

    # Zomato
    pre_count = len(manager.all_tools)
    try:
        await manager.connect_server("zomato", "npx", ["-y", "mcp-remote", zomato_url])
        zomato_tools = len(manager.all_tools) - pre_count
        connection_status.append({"name": "Zomato", "status": "ok", "tools": zomato_tools})
    except Exception as e:
        connection_status.append({"name": "Zomato", "status": "error", "detail": str(e)})

    if manager.all_tools:
        app_agent = create_food_aggregator_agent(manager.all_tools)
    
    return connection_status

# ─── FastAPI ───
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    print("Shutting down MCP connections...")
    await manager.cleanup()

app = FastAPI(lifespan=lifespan)

# ─── API Models ───
class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str

# ─── API Endpoints ───
@app.post("/api/connect")
async def api_connect():
    """Connect to MCP servers (triggered by UI button)."""
    status = await connect_to_mcp_servers()
    return {
        "connected": app_agent is not None,
        "servers": status,
        "total_tools": len(manager.all_tools),
    }

@app.get("/api/status")
async def api_status():
    """Check connection health."""
    return {
        "connected": app_agent is not None,
        "servers": connection_status,
        "total_tools": len(manager.all_tools),
    }

@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(request: ChatRequest):
    """Send a message to the agent."""
    if not app_agent:
        return ChatResponse(response="⚠️ Not connected. Please click 'Connect to MCP Servers' first.")

    config = {"configurable": {"thread_id": request.session_id}}
    inputs = {"messages": [("user", request.query)]}

    try:
        result = await app_agent.ainvoke(inputs, config=config)
        raw = result["messages"][-1].content

        # Normalize Gemini's response format
        if isinstance(raw, list) and len(raw) > 0:
            if isinstance(raw[0], dict) and "text" in raw[0]:
                text = raw[0]["text"]
            elif isinstance(raw[0], str):
                text = "".join(raw)
            else:
                text = str(raw)
        else:
            text = str(raw)

        return ChatResponse(response=text)
    except Exception as e:
        return ChatResponse(response=f"❌ Agent error: {e}")

# ─── Serve React frontend (production) ───
frontend_dist = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.isdir(frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_react(full_path: str):
        """Serve React app for all non-API routes."""
        file_path = os.path.join(frontend_dist, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(frontend_dist, "index.html"))

# ─── CLI mode ───
async def run_cli(initial_query: str = None):
    """Run the agent in conversational CLI mode."""
    print("Starting up and connecting to MCP servers...")
    await connect_to_mcp_servers()

    if not app_agent:
        print("Failed to initialize agent.")
        return

    print("\n=== Food Aggregator Agent Chat CLI ===")
    print("Type 'exit' or 'quit' to stop.\n")

    config = {"configurable": {"thread_id": "cli_session_1"}}
    query = initial_query

    try:
        while True:
            if not query:
                query = input("\nYou: ")
            if query.lower() in ["exit", "quit", "q"]:
                break

            print("\nAgent is reasoning...\n")
            inputs = {"messages": [("user", query)]}
            result = await app_agent.ainvoke(inputs, config=config)

            print("--- Agent ---")
            raw = result["messages"][-1].content
            if isinstance(raw, list) and len(raw) > 0:
                if isinstance(raw[0], dict) and "text" in raw[0]:
                    print(raw[0]["text"])
                elif isinstance(raw[0], str):
                    print("".join(raw))
                else:
                    print(raw)
            else:
                print(raw)

            query = None
    finally:
        await manager.cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Food Aggregator Agent")
    parser.add_argument("--query", "-q", type=str, help="Initial query for CLI mode")

    args = parser.parse_args()

    if args.query or not os.getenv("RUN_SERVER"):
        print("Starting in CLI mode. Set RUN_SERVER=1 to run the FastAPI server.")
        asyncio.run(run_cli(args.query))
    else:
        import uvicorn
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
