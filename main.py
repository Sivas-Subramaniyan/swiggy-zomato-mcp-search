import os
import asyncio
import argparse
import hashlib
import base64
import secrets
from contextlib import asynccontextmanager
from urllib.parse import urlencode, urlparse, parse_qsl
from dotenv import load_dotenv

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import mlflow

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

# ─── OAuth config ───
PLATFORMS = {
    "swiggy": {
        "mcp_url":      os.getenv("SWIGGY_MCP_URL", "https://mcp.swiggy.com/food"),
        "auth_url":     "https://mcp.swiggy.com/auth/authorize",
        "token_url":    "https://mcp.swiggy.com/auth/token",
        "register_url": "https://mcp.swiggy.com/auth/register",
        "redirect_uri": "http://localhost:8000/api/auth/swiggy/callback",
        "scope":        "mcp:tools mcp:resources mcp:prompts",
    },
    "zomato": {
        "mcp_url":      os.getenv("ZOMATO_MCP_URL", "https://mcp-server.zomato.com/mcp"),
        "auth_url":     "https://mcp-server.zomato.com/authorize",
        "token_url":    "https://mcp-server.zomato.com/token",
        "register_url": "https://mcp-server.zomato.com/register",
        # Zomato only whitelists the mcp-remote callback port
        "redirect_uri": "http://127.0.0.1:3334/callback",
        "scope":        "mcp:tools mcp:resources mcp:prompts",
    },
}

# ─── Global state ───
manager = MCPServerManager()
app_agent = None
platform_status: dict = {"swiggy": {}, "zomato": {}}

# OAuth state
_oauth_states: dict = {}          # state -> {platform, code_verifier}
_client_ids: dict = {}            # platform -> client_id
_client_secrets: dict = {}        # platform -> client_secret
_zomato_cb_task: asyncio.Task = None   # running port-3334 server task


def _rebuild_agent():
    global app_agent
    app_agent = create_food_aggregator_agent(manager.all_tools) if manager.all_tools else None


# ─── Shared popup HTML ───
def _popup_html(msg: str, success: bool, platform: str) -> str:
    icon = "✅" if success else "❌"
    event_type = f"{platform}_connected" if success else f"{platform}_error"
    safe_msg = msg.replace("'", "\\'").replace('"', '\\"')
    return f"""<!DOCTYPE html><html><head>
<style>body{{font-family:sans-serif;display:flex;align-items:center;justify-content:center;
height:100vh;margin:0;background:#1a1a2e;color:#eee}}
.box{{text-align:center;padding:2rem;border-radius:12px;background:#16213e}}</style>
</head><body><div class="box"><p style="font-size:2rem">{icon}</p><p>{msg}</p></div>
<script>window.opener?.postMessage({{type:'{event_type}',message:"{safe_msg}"}}, '*');
setTimeout(()=>window.close(),1800);</script></body></html>"""


# ─── Zomato: one-shot callback server on port 3334 ───
async def _run_zomato_callback_server(state: str, code_verifier: str, client_id: str):
    """
    Starts a minimal HTTP server on 127.0.0.1:3334.
    Waits for exactly one OAuth callback from Zomato, exchanges the code
    for a token, connects to Zomato MCP, then shuts down.
    This mirrors what mcp-remote does internally.
    """
    CALLBACK_PORT = 3334
    REDIRECT_URI = f"http://127.0.0.1:{CALLBACK_PORT}/callback"

    done_future: asyncio.Future = asyncio.get_event_loop().create_future()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            raw = await asyncio.wait_for(reader.read(8192), timeout=10)
            first_line = raw.decode(errors="replace").split("\r\n")[0]
            parts = first_line.split(" ")
            path = parts[1] if len(parts) >= 2 else "/"
            params = dict(parse_qsl(urlparse(path).query))

            error = params.get("error")
            recv_state = params.get("state")
            code = params.get("code")

            if error:
                html = _popup_html(f"Auth cancelled: {error}", False, "zomato")
            elif recv_state != state:
                html = _popup_html("Invalid state — please try again.", False, "zomato")
            elif code:
                try:
                    async with httpx.AsyncClient() as hc:
                        resp = await hc.post(
                            PLATFORMS["zomato"]["token_url"],
                            data={
                                "grant_type":    "authorization_code",
                                "code":          code,
                                "redirect_uri":  REDIRECT_URI,
                                "client_id":     client_id,
                                "code_verifier": code_verifier,
                            },
                            timeout=10,
                        )
                        token_data = resp.json()
                    access_token = token_data.get("access_token")
                    if access_token:
                        status = await _connect_platform("zomato", access_token)
                        if status.get("status") == "ok":
                            html = _popup_html(f"Zomato connected — {status['tools']} tools!", True, "zomato")
                        else:
                            html = _popup_html(f"MCP connect failed: {status.get('detail','')}", False, "zomato")
                    else:
                        html = _popup_html(f"No token: {token_data}", False, "zomato")
                except Exception as e:
                    html = _popup_html(f"Token exchange failed: {e}", False, "zomato")
            else:
                html = _popup_html("No auth code received.", False, "zomato")

            body = html.encode()
            response = (
                f"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n"
                f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n"
            ).encode() + body
            writer.write(response)
            await writer.drain()
        except Exception as e:
            print(f"[zomato callback] Error: {e}")
        finally:
            try:
                writer.close()
            except Exception:
                pass
            if not done_future.done():
                done_future.set_result(True)

    server = await asyncio.start_server(handle, "127.0.0.1", CALLBACK_PORT)
    print(f"[zomato] Callback server listening on port {CALLBACK_PORT}")
    try:
        await asyncio.wait_for(done_future, timeout=300)
    except asyncio.TimeoutError:
        print("[zomato] OAuth timeout — closing callback server")
    finally:
        server.close()
        await server.wait_closed()
        print(f"[zomato] Callback server on port {CALLBACK_PORT} closed")


# ─── OAuth helpers ───
async def _get_or_register_client(platform: str):
    if platform in _client_ids:
        return _client_ids[platform], _client_secrets.get(platform, "")

    cfg = PLATFORMS[platform]
    async with httpx.AsyncClient() as client:
        resp = await client.post(cfg["register_url"], json={
            "client_name": f"Food Aggregator Agent ({platform.title()})",
            "redirect_uris": [cfg["redirect_uri"]],
            "grant_types": ["authorization_code"],
            "response_types": ["code"],
            "scope": cfg["scope"],
            "token_endpoint_auth_method": "none",
        }, timeout=10)
        data = resp.json()

    _client_ids[platform] = data["client_id"]
    _client_secrets[platform] = data.get("client_secret", "")
    print(f"[{platform}] Registered OAuth client: {data['client_id']}")
    return _client_ids[platform], _client_secrets[platform]


async def _connect_platform(platform: str, token: str):
    """Connect to a platform's MCP server using a Bearer token."""
    global platform_status
    cfg = PLATFORMS[platform]
    bearer = token if token.lower().startswith("bearer ") else f"Bearer {token}"
    try:
        await manager.connect_server_http(platform, cfg["mcp_url"], headers={"Authorization": bearer})
        tools_count = len(manager._server_tools.get(platform, []))
        platform_status[platform] = {"name": platform.title(), "status": "ok", "tools": tools_count}
    except Exception as e:
        platform_status[platform] = {"name": platform.title(), "status": "error", "detail": str(e)}
    _rebuild_agent()
    return platform_status[platform]


# ─── FastAPI ───
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await manager.cleanup()

app = FastAPI(lifespan=lifespan)


# ─── OAuth start (shared for both platforms) ───
def _make_auth_start_handler(platform: str):
    async def handler():
        # Swiggy: always re-register (no token caching)
        if platform == "swiggy":
            _client_ids.pop("swiggy", None)
            _client_secrets.pop("swiggy", None)
        try:
            client_id, _ = await _get_or_register_client(platform)
        except Exception as e:
            return HTMLResponse(f"<p>Registration failed: {e}</p>", status_code=500)

        cfg = PLATFORMS[platform]
        code_verifier = secrets.token_urlsafe(64)
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode()).digest()
        ).rstrip(b"=").decode()
        state = secrets.token_urlsafe(16)
        _oauth_states[state] = {"platform": platform, "code_verifier": code_verifier}

        params = urlencode({
            "client_id":             client_id,
            "response_type":         "code",
            "redirect_uri":          cfg["redirect_uri"],
            "scope":                 cfg["scope"],
            "state":                 state,
            "code_challenge":        code_challenge,
            "code_challenge_method": "S256",
        })
        return RedirectResponse(url=f"{cfg['auth_url']}?{params}")
    return handler


# ─── OAuth callback (shared) ───
def _make_auth_callback_handler(platform: str):
    async def handler(code: str = None, state: str = None, error: str = None):
        def popup_response(msg: str, success: bool):
            icon = "✅" if success else "❌"
            event_type = f"{platform}_connected" if success else f"{platform}_error"
            escaped = msg.replace("'", "\\'").replace('"', '\\"')
            return HTMLResponse(f"""<!DOCTYPE html><html><head>
<style>body{{font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;background:#1a1a2e;color:#eee}}.box{{text-align:center;padding:2rem;border-radius:12px;background:#16213e}}</style>
</head><body><div class="box"><p style="font-size:2rem">{icon}</p><p>{msg}</p></div>
<script>window.opener?.postMessage({{type:'{event_type}',message:"{escaped}"}}, '*');setTimeout(()=>window.close(),1800);</script>
</body></html>""")

        if error:
            return popup_response(f"Auth cancelled: {error}", success=False)
        if not state or state not in _oauth_states:
            return popup_response("Invalid OAuth state — please try again.", success=False)

        state_data = _oauth_states.pop(state)
        if state_data["platform"] != platform:
            return popup_response("State mismatch — please try again.", success=False)

        code_verifier = state_data["code_verifier"]
        client_id, _ = await _get_or_register_client(platform)
        cfg = PLATFORMS[platform]

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(cfg["token_url"], data={
                    "grant_type":    "authorization_code",
                    "code":          code,
                    "redirect_uri":  cfg["redirect_uri"],
                    "client_id":     client_id,
                    "code_verifier": code_verifier,
                }, timeout=10)
                token_data = resp.json()
        except Exception as e:
            return popup_response(f"Token exchange failed: {e}", success=False)

        access_token = token_data.get("access_token")
        if not access_token:
            return popup_response(f"No token received: {token_data}", success=False)

        status = await _connect_platform(platform, access_token)
        if status.get("status") == "ok":
            return popup_response(f"{platform.title()} connected — {status['tools']} tools!", success=True)
        else:
            return popup_response(f"MCP connect failed: {status.get('detail', '')}", success=False)
    return handler


# ─── Zomato: custom start (launches port-3334 callback server) ───
@app.get("/api/auth/zomato/start")
async def zomato_auth_start():
    global _zomato_cb_task
    try:
        client_id, _ = await _get_or_register_client("zomato")
    except Exception as e:
        return HTMLResponse(f"<p>Registration failed: {e}</p>", status_code=500)

    code_verifier = secrets.token_urlsafe(64)
    code_challenge = base64.urlsafe_b64encode(
        hashlib.sha256(code_verifier.encode()).digest()
    ).rstrip(b"=").decode()
    state = secrets.token_urlsafe(16)

    # Cancel any existing callback server (e.g. previous failed attempt)
    if _zomato_cb_task and not _zomato_cb_task.done():
        _zomato_cb_task.cancel()
    _zomato_cb_task = asyncio.create_task(
        _run_zomato_callback_server(state, code_verifier, client_id)
    )

    cfg = PLATFORMS["zomato"]
    params = urlencode({
        "client_id":             client_id,
        "response_type":         "code",
        "redirect_uri":          cfg["redirect_uri"],
        "scope":                 cfg["scope"],
        "state":                 state,
        "code_challenge":        code_challenge,
        "code_challenge_method": "S256",
    })
    return RedirectResponse(url=f"{cfg['auth_url']}?{params}")


# ─── Register routes (Swiggy only — Zomato uses port 3334) ───
app.add_api_route("/api/auth/swiggy/start",    _make_auth_start_handler("swiggy"),    methods=["GET"])
app.add_api_route("/api/auth/swiggy/callback", _make_auth_callback_handler("swiggy"), methods=["GET"])


# ─── Status endpoint ───
@app.get("/api/status")
async def api_status():
    return {
        "agent_ready": app_agent is not None,
        "swiggy":      platform_status["swiggy"],
        "zomato":      platform_status["zomato"],
        "total_tools": len(manager.all_tools),
    }


# ─── Chat endpoint ───
class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str

@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(request: ChatRequest):
    if not app_agent:
        return ChatResponse(response="⚠️ Not connected. Please log in to Swiggy or Zomato first.")

    config = {"configurable": {"thread_id": request.session_id}}
    inputs = {"messages": [("user", request.query)]}
    try:
        result = await app_agent.ainvoke(inputs, config=config)
        raw = result["messages"][-1].content
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


# ─── Serve React frontend ───
frontend_dist = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.isdir(frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_react(full_path: str):
        file_path = os.path.join(frontend_dist, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(frontend_dist, "index.html"))


# ─── CLI mode ───
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", "-q", type=str)
    args = parser.parse_args()

    if not os.getenv("RUN_SERVER"):
        print("Set RUN_SERVER=1 to start the web server.")
        print("OAuth login is required — use the web UI at http://localhost:8000")
    else:
        import uvicorn
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
