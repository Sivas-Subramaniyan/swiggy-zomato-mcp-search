import os
import sys
import json
import asyncio
import hashlib
import base64
import secrets
import random
from contextlib import asynccontextmanager
from datetime import date
from urllib.parse import urlencode, urlparse, parse_qsl
from dotenv import load_dotenv

# Force UTF-8 output on Windows to handle emoji and non-ASCII characters
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import mlflow

from mcp_client import MCPServerManager, filter_tool_result, pending_payment_qr
from agent import create_food_aggregator_agent
from agent_logger import AgentLogger

load_dotenv(override=True)  # always prefer .env values over any pre-set env vars

# ─── MLflow (non-blocking setup — experiment/autolog deferred to lifespan) ───
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))

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
_access_tokens: dict = {}         # platform -> current access_token
_refresh_tokens: dict = {}        # platform -> refresh_token (for auto-refresh on 403)
_zomato_cb_task: asyncio.Task = None   # running port-3334 server task

# Cached address data per platform (fetched automatically after login)
_address_cache: dict = {}         # platform -> raw address text from get_addresses tool

# Daily Gemini request budget (free tier = 1500 req/day; buffer to stay safely under)
_daily_requests: dict = {"date": None, "count": 0}
DAILY_REQUEST_LIMIT = 1400


def _rebuild_agent():
    global app_agent
    if not manager.all_tools:
        app_agent = None
        return
    connected = [p for p in ("swiggy", "zomato") if manager.is_connected(p)]

    # Parse literal address IDs from cache — injected verbatim into system prompt
    # so the LLM never needs to guess which field to use.
    address_ids: dict[str, str] = {}
    for p in connected:
        raw = _address_cache.get(p, "")
        if raw:
            try:
                data = json.loads(raw)
                if isinstance(data, dict):
                    aid = data.get("addressId") or data.get("address_id")
                    if aid:
                        address_ids[p] = str(aid)
            except Exception:
                pass
    for p, aid in address_ids.items():
        print(f"[{p}] Resolved addressId for agent: {aid}")

    # Compare addresses in pure Python (no LLM tokens spent)
    addr_cmp = None
    if "swiggy" in connected and "zomato" in connected:
        s_raw = _address_cache.get("swiggy", "")
        z_raw = _address_cache.get("zomato", "")
        if s_raw and z_raw:
            from agent import compare_addresses_by_pincode
            addr_cmp = compare_addresses_by_pincode(s_raw, z_raw)
            safe = addr_cmp['summary'].encode('utf-8', errors='replace').decode('utf-8')
            print(f"[address] Pincode comparison: {safe}")

    app_agent = create_food_aggregator_agent(
        manager.all_tools,
        connected_platforms=connected,
        address_context=_address_cache,
        address_ids=address_ids,
        address_comparison=addr_cmp,
        payment_types={p: manager.get_payment_types(p) for p in connected},
    )


def _extract_first_address(raw: str) -> str | None:
    """
    Parse the address tool JSON response and return a compact dict containing
    only the first saved address's ID and a short human label.

    Handles both Swiggy and Zomato response shapes:
      Swiggy: {"success": true, "data": {"addresses": [{id, addressLine, ...}]}}
      Zomato: {"addresses": [{address_id, location_name, ...}]}

    Returns a compact JSON string like:
      {"addressId": "d01m57g...", "label": "20th Floor, Prestige, 560001"}
    or None if parsing fails (caller falls back to raw text).
    """
    try:
        data = json.loads(raw)

        # Unwrap common envelope shapes to reach the list
        addr_list = None
        if isinstance(data, list):
            addr_list = data
        elif isinstance(data, dict):
            # Walk known nesting paths
            for path in (
                ["addresses"],
                ["data", "addresses"],
                ["data"],
                ["results"],
                ["savedAddresses"],
            ):
                node = data
                for key in path:
                    node = node.get(key) if isinstance(node, dict) else None
                    if node is None:
                        break
                if isinstance(node, list) and node:
                    addr_list = node
                    break

        if not addr_list:
            return None

        first = addr_list[0]
        if not isinstance(first, dict):
            return None

        # Extract address ID — platform-specific names take priority over generic "id"
        # Swiggy uses "addressId"; Zomato uses "address_id"; fallback to generic "id"
        addr_id = (
            first.get("addressId") or first.get("address_id") or
            first.get("_id") or first.get("id") or "unknown"
        )

        # Build a short human label (street + pincode)
        label_parts = []
        for field in ("addressLine", "address_line1", "line1", "address",
                      "location_name", "area", "locality", "flatNo"):
            val = first.get(field)
            if val:
                label_parts.append(str(val)[:50])
                break
        for field in ("pincode", "pin_code", "zipcode", "zip", "areaCode"):
            val = first.get(field)
            if val:
                label_parts.append(str(val))
                break

        label = ", ".join(label_parts) or "saved address"
        return json.dumps({"addressId": str(addr_id), "label": label},
                          ensure_ascii=False)
    except Exception:
        return None


async def _fetch_address(platform: str):
    """
    Call the platform's address tool directly (no LLM) right after login.
    Extracts the FIRST saved address and caches a compact JSON snippet
    (addressId + label) instead of the full raw response — reduces the
    address section of the system prompt from ~300 chars to ~80 chars.
    """
    try:
        session = manager.sessions.get(platform)
        if not session:
            return
        tools = manager._tools.get(platform, [])
        addr_tool = next(
            (t for t in tools if "address" in getattr(t, "name", "").lower()),
            None,
        )
        if not addr_tool:
            print(f"[{platform}] No address tool found — skipping address pre-fetch")
            return

        result = await session.call_tool(addr_tool.name, {})
        text = ""
        if result and result.content:
            text = " | ".join(
                c.text for c in result.content if hasattr(c, "text") and c.text
            )
        if not text:
            print(f"[{platform}] Address tool returned empty result")
            return

        # Log the full raw response so we can verify the correct addressId field
        print(f"[{platform}] Raw address response (first 600 chars):")
        print(f"  {text[:600].encode('utf-8', errors='replace').decode('utf-8')}")

        # Extract first address compactly (no LLM, pure Python)
        compact = _extract_first_address(text)
        _address_cache[platform] = compact if compact else text

        safe = (_address_cache[platform])[:200].encode('utf-8', errors='replace').decode('utf-8')
        print(f"[{platform}] Extracted address → {safe}")

    except Exception as e:
        print(f"[{platform}] Could not pre-fetch address: {e}")


# ─── Daily request budget ───
def _check_daily_limit():
    """Raise a clear error when the free-tier daily Gemini request cap is hit."""
    today = date.today().isoformat()
    if _daily_requests["date"] != today:
        _daily_requests["date"] = today
        _daily_requests["count"] = 0
    _daily_requests["count"] += 1
    if _daily_requests["count"] > DAILY_REQUEST_LIMIT:
        raise Exception(
            f"Daily Gemini request limit reached ({DAILY_REQUEST_LIMIT} req/day on free tier). "
            "Try again tomorrow or switch to a paid API key."
        )


# ─── Exponential backoff on Gemini 429 rate limits ───
async def _invoke_with_backoff(agent, inputs: dict, config: dict, max_retries: int = 4):
    """Retry agent.ainvoke with exponential back-off when Gemini returns 429."""
    for attempt in range(max_retries):
        try:
            return await agent.ainvoke(inputs, config=config)
        except Exception as e:
            err = str(e)
            is_rate_limit = "429" in err or "RESOURCE_EXHAUSTED" in err
            if not is_rate_limit:
                raise
            # Daily cap — retrying won't help
            if "requests" in err.lower() and ("day" in err.lower() or "daily" in err.lower()):
                raise
            if attempt == max_retries - 1:
                raise
            wait = (2 ** attempt) + random.uniform(0, 1)
            print(f"[backoff] 429 rate limit on attempt {attempt + 1} — retrying in {wait:.1f}s")
            await asyncio.sleep(wait)


# ─── Sanitize orphaned tool-call messages in LangGraph checkpoint ───
async def _sanitize_thread(agent, config: dict):
    """
    Remove AIMessages whose tool_calls have no matching ToolMessage from the
    LangGraph checkpoint. Prevents the 'orphaned tool calls' validation error.
    """
    try:
        from langchain_core.messages import AIMessage, ToolMessage
        from langgraph.graph.message import RemoveMessage

        state = await agent.aget_state(config)
        messages = list(state.values.get("messages", []))
        if not messages:
            return

        # IDs of tool calls that already have a response
        responded_ids = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}

        orphans = [
            msg for msg in messages
            if isinstance(msg, AIMessage)
            and msg.tool_calls
            and not {tc["id"] for tc in msg.tool_calls}.issubset(responded_ids)
        ]

        if orphans:
            thread = config["configurable"]["thread_id"]
            print(f"[sanitize] Removing {len(orphans)} orphaned AIMessage(s) from thread '{thread}'")
            await agent.aupdate_state(
                config,
                {"messages": [RemoveMessage(id=m.id) for m in orphans]},
            )
    except Exception as e:
        print(f"[sanitize] History cleanup failed (non-fatal): {e}")


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
                        token_data = json.loads(resp.content.decode('utf-8'))
                    access_token = token_data.get("access_token")
                    if access_token:
                        print(f"[zomato] Token received — length={len(access_token)}, prefix={access_token[:12]}...")
                        print(f"[zomato] Full token (for list_mcp_tools.py): {access_token}")
                        # Store refresh token for auto-refresh on 403
                        if token_data.get("refresh_token"):
                            _refresh_tokens["zomato"] = token_data["refresh_token"]
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
        data = json.loads(resp.content.decode('utf-8'))

    _client_ids[platform] = data["client_id"]
    _client_secrets[platform] = data.get("client_secret", "")
    print(f"[{platform}] Registered OAuth client: {data['client_id']}")
    return _client_ids[platform], _client_secrets[platform]


async def _connect_platform(platform: str, token: str):
    """Connect to a platform's MCP server using a Bearer token."""
    global platform_status
    cfg = PLATFORMS[platform]
    bearer = token if token.lower().startswith("bearer ") else f"Bearer {token}"
    _access_tokens[platform] = token
    try:
        await manager.connect_server_http(platform, cfg["mcp_url"], headers={"Authorization": bearer})
        tools_count = len(manager._tools.get(platform, []))
        platform_status[platform] = {"name": platform.title(), "status": "ok", "tools": tools_count}
        # Pre-fetch the user's saved address so the agent never needs to ask for it
        await _fetch_address(platform)
    except Exception as e:
        platform_status[platform] = {"name": platform.title(), "status": "error", "detail": str(e)}
    _rebuild_agent()
    return platform_status[platform]


async def _refresh_platform_token(platform: str) -> bool:
    """
    Exchange the stored refresh_token for a new access_token, then reconnect.
    Returns True if the refresh succeeded and MCP reconnected successfully.
    """
    refresh_token = _refresh_tokens.get(platform)
    client_id = _client_ids.get(platform)
    if not refresh_token or not client_id:
        print(f"[{platform}] No refresh token available — cannot auto-refresh")
        return False

    cfg = PLATFORMS[platform]
    print(f"[{platform}] Access token expired (403). Attempting token refresh...")
    try:
        async with httpx.AsyncClient() as hc:
            resp = await hc.post(cfg["token_url"], data={
                "grant_type":    "refresh_token",
                "refresh_token": refresh_token,
                "client_id":     client_id,
            }, timeout=10)
            token_data = json.loads(resp.content.decode('utf-8'))

        new_access = token_data.get("access_token")
        if not new_access:
            print(f"[{platform}] Token refresh failed: {token_data}")
            return False

        # Update stored tokens
        _access_tokens[platform] = new_access
        if token_data.get("refresh_token"):
            _refresh_tokens[platform] = token_data["refresh_token"]

        _address_cache.pop(platform, None)  # clear stale address; _connect_platform will re-fetch
        status = await _connect_platform(platform, new_access)
        if status.get("status") == "ok":
            print(f"[{platform}] Token refreshed and reconnected — {status['tools']} tools")
            return True
        else:
            print(f"[{platform}] Reconnect after refresh failed: {status.get('detail')}")
            return False
    except Exception as e:
        print(f"[{platform}] Token refresh error: {e}")
        return False


# ─── FastAPI ───
@asynccontextmanager
async def lifespan(app: FastAPI):
    # MLflow setup — runs after MLflow server is already started
    try:
        mlflow.set_experiment("Food_Aggregator_Agent")
    except Exception:
        pass
    try:
        mlflow.langchain.autolog()
    except Exception:
        pass
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

        auth_params = {
            "client_id":             client_id,
            "response_type":         "code",
            "redirect_uri":          cfg["redirect_uri"],
            "scope":                 cfg["scope"],
            "state":                 state,
            "code_challenge":        code_challenge,
            "code_challenge_method": "S256",
        }
        # Force re-login every time for Swiggy (don't reuse browser session)
        if platform == "swiggy":
            auth_params["prompt"] = "login"

        return RedirectResponse(url=f"{cfg['auth_url']}?{urlencode(auth_params)}")
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
                token_data = json.loads(resp.content.decode('utf-8'))
        except Exception as e:
            return popup_response(f"Token exchange failed: {e}", success=False)

        access_token = token_data.get("access_token")
        if not access_token:
            print(f"[{platform}] Token exchange response: {token_data}")
            return popup_response(f"No token received: {token_data}", success=False)

        print(f"[{platform}] Token received — length={len(access_token)}, prefix={access_token[:12]}...")
        print(f"[{platform}] Full token (for list_mcp_tools.py): {access_token}")
        if token_data.get("refresh_token"):
            _refresh_tokens[platform] = token_data["refresh_token"]
            print(f"[{platform}] Refresh token stored")

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

@app.get("/api/payment/qr/{order_id}")
async def api_payment_qr(order_id: str, download: bool = False):
    """Serve the UPI QR PNG for the given order_id.

    ?download=1  → adds Content-Disposition: attachment so the browser saves
                   the file instead of opening it inline.
    """
    from fastapi.responses import Response as FastAPIResponse, JSONResponse
    png_bytes = pending_payment_qr.get(order_id)
    if not png_bytes:
        return JSONResponse(
            status_code=404,
            content={"error": "QR not found", "order_id": order_id},
        )
    headers = {}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="upi-qr-{order_id}.png"'
    return FastAPIResponse(content=png_bytes, media_type="image/png", headers=headers)


@app.get("/api/status")
async def api_status():
    # Verify actual MCP connection — don't report stale status if connection dropped
    def live_status(platform: str) -> dict:
        if not manager.is_connected(platform):
            return {}
        return platform_status.get(platform, {})

    return {
        "agent_ready": app_agent is not None,
        "swiggy":      live_status("swiggy"),
        "zomato":      live_status("zomato"),
        "total_tools": len(manager.all_tools),
    }


# ─── Chat endpoint ───
class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    qr_order_id: str | None = None        # set when checkout_cart generated a QR this turn
    status_changed: bool = False           # True when a session expired this turn
    expired_platforms: list[str] = []     # platforms whose session expired


def _extract_text(result) -> str:
    raw = result["messages"][-1].content
    if isinstance(raw, list) and raw:
        first = raw[0]
        if isinstance(first, dict) and "text" in first:
            return first["text"]
        if isinstance(first, str):
            return "".join(raw)
        return str(raw)
    return str(raw)


def _friendly_error(err: str) -> str:
    """Convert raw exception strings into user-readable messages."""
    # MCP returns errors as JSON: {"success": false, "error": {"message": "..."}}
    try:
        data = json.loads(err)
        if isinstance(data, dict) and data.get("success") is False:
            msg = (data.get("error") or {}).get("message", err)
            if "403" in msg or "Forbidden" in msg:
                return (
                    "⚠️ Your session has expired (403 Forbidden). "
                    "Please click **Re-login** in the sidebar to reconnect."
                )
            if "500" in msg or "Internal Server Error" in msg:
                return (
                    "⚠️ The food platform returned a server error (500). "
                    "This is a temporary issue on their end — please try again in a moment."
                )
            return f"⚠️ Platform error: {msg}"
    except Exception:
        pass

    if "429" in err or "RESOURCE_EXHAUSTED" in err:
        if "day" in err.lower() or "daily" in err.lower():
            return "⚠️ Daily Gemini request limit reached. Resets at midnight."
        return "⚠️ Gemini rate limit hit — please wait a moment and try again."

    return f"❌ Error: {err}"


@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(request: ChatRequest):
    if not app_agent:
        return ChatResponse(response="⚠️ Not connected. Please log in to Swiggy or Zomato first.")

    try:
        _check_daily_limit()
    except Exception as e:
        return ChatResponse(response=f"⚠️ {e}")

    logger = AgentLogger(session_id=request.session_id)
    config = {
        "configurable": {"thread_id": request.session_id},
        "recursion_limit": 25,
        "callbacks": [logger],
    }
    inputs = {"messages": [("user", request.query)]}

    # Clean up orphaned tool-call messages from the checkpoint before invoking
    await _sanitize_thread(app_agent, config)

    # Snapshot QR store before the turn so we can detect newly generated QRs
    pre_turn_qr_ids = set(pending_payment_qr.keys())

    try:
        result = await _invoke_with_backoff(app_agent, inputs, config)

        # Detect any QR generated during this turn — return it directly so the
        # frontend can display the QR panel without parsing the agent's text.
        new_qr_ids = set(pending_payment_qr.keys()) - pre_turn_qr_ids
        qr_order_id = next(iter(new_qr_ids), None)
        if qr_order_id:
            print(f"[chat] QR generated for order {qr_order_id} — returning in response")

        response_text = _extract_text(result)

        # Detect expired sessions from the agent's response text
        expired = []
        if "SESSION_EXPIRED_403(swiggy)" in response_text or "Swiggy session expired" in response_text:
            expired.append("swiggy")
        if "SESSION_EXPIRED_403(zomato)" in response_text or "Zomato session expired" in response_text:
            expired.append("zomato")

        return ChatResponse(
            response=response_text,
            qr_order_id=qr_order_id,
            status_changed=bool(expired),
            expired_platforms=expired,
        )

    except asyncio.CancelledError:
        # Happens when a parallel MCP tool call (asyncio.gather) is cancelled
        # because a sibling call on another platform raised an exception.
        return ChatResponse(response=(
            "⚠️ One platform's search was interrupted. Please try your query again."
        ))

    except Exception as e:
        err = str(e)

        # Corrupt thread history — retry once on a fresh thread
        if "tool_calls that do not have" in err or "orphaned" in err.lower():
            print("[chat] Corrupt thread — retrying on fresh thread")
            try:
                fresh = {
                    "configurable": {"thread_id": request.session_id + "_fresh"},
                    "recursion_limit": 25,
                }
                result = await _invoke_with_backoff(app_agent, inputs, fresh)
                return ChatResponse(response=_extract_text(result))
            except Exception as e2:
                return ChatResponse(response=_friendly_error(str(e2)))

        return ChatResponse(response=_friendly_error(err))


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


# ─── Entry point ───
if __name__ == "__main__":
    import socket
    import subprocess
    import time
    import threading
    import webbrowser
    import uvicorn

    HOST = "0.0.0.0"
    PORT = 8000
    APP_URL = f"http://localhost:{PORT}"

    def _port_in_use(port: int) -> bool:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            return False

    # ── Auto-start MLflow if not running ──────────────────────────────────
    if not _port_in_use(5000):
        print("Starting MLflow server on http://127.0.0.1:5000 ...")
        subprocess.Popen(
            [sys.executable, "-m", "mlflow", "server", "--host", "127.0.0.1", "--port", "5000"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(3)
        print("MLflow ready  →  http://127.0.0.1:5000")
    else:
        print("MLflow already running  →  http://127.0.0.1:5000")

    # ── Open browser once the server is ready ─────────────────────────────
    def _open_browser():
        for _ in range(20):          # wait up to ~5 s
            time.sleep(0.25)
            if _port_in_use(PORT):
                webbrowser.open(APP_URL)
                return
    threading.Thread(target=_open_browser, daemon=True).start()

    # ── Start server ──────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  🍔  Food Aggregator Agent")
    print(f"  App  →  {APP_URL}")
    print(f"{'─'*50}\n")
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)