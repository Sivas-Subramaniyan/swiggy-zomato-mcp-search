import io
import os
import json
import base64
import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.sse import sse_client
from langchain_mcp_adapters.tools import load_mcp_tools

# ═══════════════════════════════════════════════════════════════════════════════
# BIDIRECTIONAL TOOL FILTER
#
# INPUT  (LLM → MCP)  ── applied at tool-call time via _make_input_filter()
#   Each tool's coroutine is wrapped once at connect time.  When the LLM calls
#   a tool the wrapper:
#     1. Strips LangGraph runtime objects  (runtime / config / store / writer).
#     2. Strips any key NOT declared in the tool's own args_schema — catches
#        extra fields Gemini hallucinates that would confuse or error the MCP.
#     3. Logs everything stripped so failures are visible in the console.
#
# OUTPUT (MCP → LLM)  ── applied in agent.py's state modifier via filter_tool_result()
#   Before every LLM invocation, every ToolMessage content is run through
#   filter_tool_result which:
#     1. Truncates large list responses to N items.
#     2. For checkout_cart: extracts QR/UPI payment data, generates a QR PNG,
#        stores it in pending_payment_qr, purges binary blobs, appends an
#        image markdown link for the agent to pass to the user.
#     3. Caps total chars so the context window is not blown.
# ═══════════════════════════════════════════════════════════════════════════════


# ── QR payment store ──────────────────────────────────────────────────────────
# order_id → raw PNG bytes.
# Populated by filter_tool_result; served by /api/payment/qr/{order_id}.
pending_payment_qr: dict[str, bytes] = {}

_UPI_FIELDS      = ["upi_string", "upi_url", "upi_intent_url", "upi_deep_link",
                     "payment_url", "payment_string", "intent_url"]
_QR_B64_FIELDS   = ["qr_code", "qr_image", "upi_qr_image", "payment_qr",
                     "qr_data", "qr_base64", "payment_qr_code"]
_ORDER_ID_FIELDS = ["order_id", "orderId", "order_number", "orderNumber", "id"]


# ─────────────────────────────────────────────────────────────────────────────
# INPUT FILTER  (LLM → MCP)
# ─────────────────────────────────────────────────────────────────────────────

# Keys LangGraph injects into every tool call; never part of any MCP schema.
_LANGGRAPH_INTERNALS = frozenset({"runtime", "config", "store", "writer", "metadata"})


def _get_schema_keys(tool) -> frozenset | None:
    """
    Read the tool's args_schema and return the set of declared parameter names.
    Returns None when the schema is unavailable (caller strips only internals).
    """
    schema = getattr(tool, "args_schema", None)
    if not schema:
        return None
    try:
        props = schema.schema().get("properties", {})
        return frozenset(props.keys()) if props else None
    except Exception:
        return None


def _make_input_filter(tool):
    """
    Build and return a coroutine wrapper for `tool` that:
      Pass 1 — removes LangGraph internal keys unconditionally.
      Pass 2 — if the tool has a schema, removes any remaining key not in it.
    The original coroutine is captured by closure before being replaced.
    """
    orig         = tool.coroutine          # captured before reassignment
    name         = getattr(tool, "name", "?")
    allowed_keys = _get_schema_keys(tool)  # frozenset or None, captured once

    async def _filtered(**kw):
        stripped = []

        # Pass 1: strip LangGraph internals
        for k in list(kw):
            if k in _LANGGRAPH_INTERNALS:
                del kw[k]
                stripped.append(k)

        # Pass 2: strip keys absent from schema (halluciations / unknown fields)
        if allowed_keys is not None:
            for k in list(kw):
                if k not in allowed_keys:
                    del kw[k]
                    stripped.append(k)

        if stripped:
            print(f"[input-filter] {name}: stripped {stripped}")

        return await orig(**kw)

    return _filtered


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT FILTER helpers  (MCP → LLM)
# ─────────────────────────────────────────────────────────────────────────────

def _deep_find(data, fields: list, depth: int = 0):
    """Return the first non-empty string value whose key is in fields."""
    if depth > 4:
        return None
    if isinstance(data, dict):
        for f in fields:
            v = data.get(f)
            if v and isinstance(v, str):
                return v
        for v in data.values():
            r = _deep_find(v, fields, depth + 1)
            if r:
                return r
    elif isinstance(data, list):
        for item in data:
            r = _deep_find(item, fields, depth + 1)
            if r:
                return r
    return None


def _strip_keys(data, keys: list, depth: int = 0):
    """Return a deep copy of data with all keys in `keys` removed."""
    if depth > 4:
        return data
    if isinstance(data, dict):
        return {k: _strip_keys(v, keys, depth + 1)
                for k, v in data.items() if k not in keys}
    if isinstance(data, list):
        return [_strip_keys(i, keys, depth + 1) for i in data]
    return data


def _make_qr_png_b64(payload: str) -> str:
    """Generate a QR code PNG from `payload`; return as base64 string."""
    import qrcode  # lazy import — only needed at checkout time
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=12,
        border=4,
    )
    qr.add_data(payload)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT FILTER  (MCP → LLM)
# ─────────────────────────────────────────────────────────────────────────────

_MAX_LIST_ITEMS: dict[str, int] = {
    "search_restaurants":                10,
    "get_restaurants_for_keyword":       10,
    "search_menu":                       10,
    "get_menu_items_listing":            10,
    "get_restaurant_menu":               10,
    "get_restaurant_menu_by_categories": 10,
    "fetch_food_coupons":                3,
    "get_cart_offers":                   3,
}

_MAX_CHARS: dict[str, int] = {
    "search_restaurants":                8000,
    "get_restaurants_for_keyword":       8000,
    "search_menu":                       10000,
    "get_menu_items_listing":            10000,
    "get_restaurant_menu":               10000,
    "get_restaurant_menu_by_categories": 10000,
    "fetch_food_coupons":                2000,
    "get_cart_offers":                   2000,
}

_WRAPPER_KEYS = (
    "restaurants", "cards", "items", "results",
    "data", "menu_items", "offers", "coupons", "dishes",
)


def _decode_png_b64(b64_str: str) -> bytes | None:
    """Decode a base64 string and return bytes if it is a valid PNG, else None."""
    b64_clean = b64_str.split(",", 1)[-1] if "," in b64_str else b64_str
    b64_clean += "=" * (-len(b64_clean) % 4)
    try:
        decoded = base64.b64decode(b64_clean)
        if decoded[:8] == b'\x89PNG\r\n\x1a\n':
            return decoded
    except Exception:
        pass
    return None


def _extract_checkout_qr(tool_name: str, data) -> tuple[bytes | None, str | None]:
    """
    Given a parsed checkout tool response (dict or MCP content-block list),
    return (png_bytes, order_id) if a QR PNG can be found/generated.

    Handles two response shapes from Zomato's checkout_cart:
      A) List of MCP content blocks:
           [{'type': 'image', 'base64': '<png_b64>', 'mime_type': 'image/png', 'id': '...'}, ...]
      B) Dict with nested UPI / QR fields (legacy / Swiggy):
           {'order_id': '...', 'upi_string': 'upi://...', ...}
    """
    # ── Shape A: MCP content-block list ──────────────────────────────────
    if isinstance(data, list):
        png_bytes = None
        order_id  = None
        for block in data:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")

            if btype == "image":
                b64 = block.get("base64") or block.get("data") or ""
                if b64 and png_bytes is None:
                    png_bytes = _decode_png_b64(b64)
                    if png_bytes:
                        # Use the block id (strip lc_ prefix) as the order key
                        raw_id = block.get("id", "")
                        order_id = raw_id.removeprefix("lc_") if raw_id else None
                        print(f"[output-filter] {tool_name}: image block PNG found, id={raw_id}")

            elif btype == "text":
                # Text blocks may carry order metadata
                try:
                    text_data = json.loads(block.get("text", ""))
                    oid = _deep_find(text_data, _ORDER_ID_FIELDS)
                    if oid:
                        order_id = oid
                except Exception:
                    pass

        if png_bytes and not order_id:
            import hashlib
            order_id = hashlib.md5(png_bytes[:64]).hexdigest()
            print(f"[output-filter] {tool_name}: no order_id in blocks, using hash {order_id}")

        return png_bytes, order_id

    # ── Shape B: dict response ────────────────────────────────────────────
    if isinstance(data, dict):
        print(f"[output-filter] {tool_name}: dict keys = {list(data.keys())[:20]}")
        order_id  = _deep_find(data, _ORDER_ID_FIELDS)
        png_bytes = None

        if order_id:
            # Try pre-built QR image fields first
            qr_b64 = _deep_find(data, _QR_B64_FIELDS)
            if qr_b64:
                png_bytes = _decode_png_b64(qr_b64)
                if png_bytes:
                    print(f"[output-filter] {tool_name}: pre-built PNG for order {order_id}")
                else:
                    # Might be a UPI string encoded in base64
                    try:
                        decoded_str = base64.b64decode(qr_b64 + "==").decode("utf-8").strip()
                        if decoded_str.startswith("upi://") or "pa=" in decoded_str:
                            png_bytes = base64.b64decode(_make_qr_png_b64(decoded_str))
                    except Exception:
                        pass

            # Fallback: generate QR from UPI string
            if png_bytes is None:
                upi_str = _deep_find(data, _UPI_FIELDS)
                if upi_str:
                    try:
                        png_bytes = base64.b64decode(_make_qr_png_b64(upi_str) + "==")
                        print(f"[output-filter] {tool_name}: generated QR from UPI string for order {order_id}")
                    except Exception as e:
                        print(f"[output-filter] {tool_name}: QR generation failed: {e}")
                else:
                    print(f"[output-filter] {tool_name}: no UPI/QR data for order {order_id}")

        return png_bytes, order_id

    return None, None


def filter_tool_result(tool_name: str, raw) -> str:
    """
    Clean a raw MCP tool result before it enters the LLM context window.

    checkout_cart / place_food_order
      ├─ Detects MCP image content blocks OR dict UPI/QR fields.
      ├─ Reconstructs and stores PNG in pending_payment_qr[order_id].
      ├─ Strips all binary data so the LLM never sees base64 blobs.
      └─ Appends  ![UPI QR Payment — Order <id>](/api/payment/qr/<id>)

    List-returning tools
      ├─ Truncates to _MAX_LIST_ITEMS items.
      └─ Caps total serialised length at _MAX_CHARS.
    """
    max_items = _MAX_LIST_ITEMS.get(tool_name)
    max_chars = _MAX_CHARS.get(tool_name, 3000)
    qr_suffix = ""

    # ── Parse ──────────────────────────────────────────────────────────────
    if isinstance(raw, (dict, list)):
        data = raw
    else:
        try:
            data = json.loads(str(raw))
        except Exception:
            try:
                import ast
                data = ast.literal_eval(str(raw))
            except Exception:
                return str(raw)[:max_chars]

    # ── checkout / place-order: reconstruct QR from response ─────────────
    if tool_name in ("checkout_cart", "place_food_order"):
        png_bytes, order_id = _extract_checkout_qr(tool_name, data)

        if png_bytes and order_id:
            pending_payment_qr[order_id] = png_bytes
            qr_suffix = (
                f"\n\n![UPI QR Payment — Order {order_id}]"
                f"(/api/payment/qr/{order_id})"
            )
            print(f"[output-filter] {tool_name}: QR stored → order_id={order_id}")
        elif not png_bytes:
            print(f"[output-filter] {tool_name}: no PNG found in response")

        # Purge binary blobs so the LLM text is clean
        if isinstance(data, list):
            # Replace image blocks with a short note; keep text blocks
            text_parts = []
            for block in data:
                if isinstance(block, dict):
                    if block.get("type") == "image":
                        text_parts.append("[QR image captured server-side]")
                    elif block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    else:
                        text_parts.append(json.dumps({k: v for k, v in block.items()
                                                       if k not in ("base64", "data")},
                                                      ensure_ascii=False))
            result = "\n".join(text_parts) if text_parts else "Checkout complete."
            return (result[:max_chars] + qr_suffix)

        elif isinstance(data, dict):
            data = _strip_keys(data, _QR_B64_FIELDS + _UPI_FIELDS)

    # ── List truncation ────────────────────────────────────────────────────
    if max_items is not None:
        if isinstance(data, list):
            data = data[:max_items]
        elif isinstance(data, dict):
            for key in _WRAPPER_KEYS:
                if key in data and isinstance(data[key], list):
                    data = dict(data)
                    data[key] = data[key][:max_items]
                    break

    # ── Serialise & cap ────────────────────────────────────────────────────
    try:
        result = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        result = str(data)

    if len(result) > max_chars:
        result = result[:max_chars] + f"…[truncated, {len(str(raw))} chars total]"

    return result + qr_suffix


# ═══════════════════════════════════════════════════════════════════════════════
# MCP SERVER MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class MCPServerManager:
    """
    Manages persistent MCP connections.  Each server runs in its own asyncio
    Task (required by anyio cancel-scope semantics).

    At connect time, for every tool:
      • Input filter is applied  (_make_input_filter) so only schema-valid
        fields ever reach the remote MCP server.
      • Tool list is filtered to ESSENTIAL_KEYWORDS and EXCLUDED_TOOLS.
    """

    ESSENTIAL_KEYWORDS = frozenset({
        "address", "search", "restaurant", "menu", "item",
        "coupon", "cart", "offer", "discount", "price",
        "order", "track", "checkout",
    })

    EXCLUDED_TOOLS = frozenset({
        "bind_user_number", "bind_user_number_verify_code", "report_error",
    })

    def __init__(self):
        self._tools: dict[str, list]                = {}
        self._tasks: dict[str, asyncio.Task]        = {}
        self._stop_events: dict[str, asyncio.Event] = {}
        self._ready_events: dict[str, asyncio.Event]= {}
        self._errors: dict[str, str | None]         = {}
        self.sessions: dict[str, ClientSession]     = {}

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def all_tools(self) -> list:
        out = []
        for t in self._tools.values():
            out.extend(t)
        return out

    def is_connected(self, server_name: str) -> bool:
        task = self._tasks.get(server_name)
        return task is not None and not task.done()

    def get_payment_types(self, server_name: str) -> list[str]:
        """Return the payment_type enum from create_cart's schema, read live."""
        for tool in self._tools.get(server_name, []):
            if getattr(tool, "name", "") == "create_cart":
                schema = getattr(tool, "args_schema", None)
                if schema:
                    try:
                        props     = schema.schema().get("properties", {})
                        enum_vals = props.get("payment_type", {}).get("enum", [])
                        if enum_vals:
                            return [str(v) for v in enum_vals]
                    except Exception:
                        pass
        return []

    async def connect_server_http(self, server_name: str, url: str, headers: dict = None):
        """
        Connect to an MCP server at `url`.  Tries streamable-HTTP first,
        falls back to SSE.  Blocks until tools are loaded or raises on error.
        """
        await self._disconnect(server_name)

        ready = asyncio.Event()
        stop  = asyncio.Event()
        self._ready_events[server_name] = ready
        self._stop_events[server_name]  = stop
        self._errors[server_name]       = None

        async def _run():
            try:
                try:
                    async with streamablehttp_client(
                        url, headers=headers or {}, timeout=30
                    ) as (r, w, _):
                        await self._init_session(server_name, r, w, ready, stop)
                except Exception:
                    if ready.is_set():
                        raise
                    async with sse_client(url, headers=headers or {}) as (r, w):
                        await self._init_session(server_name, r, w, ready, stop)
            except Exception as e:
                self._errors[server_name] = str(e)
                ready.set()
            finally:
                self._tools.pop(server_name, None)
                self.sessions.pop(server_name, None)

        self._tasks[server_name] = asyncio.create_task(_run())

        try:
            await asyncio.wait_for(ready.wait(), timeout=40)
        except asyncio.TimeoutError:
            stop.set()
            raise Exception(f"Connection to {server_name} timed out after 40 s")

        if self._errors.get(server_name):
            raise Exception(self._errors[server_name])

        print(f"[{server_name}] Connected — {len(self._tools.get(server_name, []))} tools loaded")

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _init_session(self, server_name, read, write, ready, stop):
        async with ClientSession(read, write) as session:
            await session.initialize()
            self.sessions[server_name] = session

            all_tools = await load_mcp_tools(session)

            # ── Keep only food-ordering tools ──────────────────────────────
            tools = [
                t for t in all_tools
                if (
                    any(kw in getattr(t, "name", "").lower()
                        for kw in self.ESSENTIAL_KEYWORDS)
                    and getattr(t, "name", "") not in self.EXCLUDED_TOOLS
                )
            ] or all_tools  # fall back if filter is too aggressive

            print(f"[{server_name}] {len(tools)}/{len(all_tools)} tools after filter:")

            # ── Apply input filter + log schema ────────────────────────────
            for tool in tools:
                name   = getattr(tool, "name", "?")
                schema = getattr(tool, "args_schema", None)
                params = ""
                if schema:
                    try:
                        props    = schema.schema().get("properties", {})
                        required = set(schema.schema().get("required", []))
                        params   = ", ".join(
                            f"{k}{'*' if k in required else ''}" for k in props
                        )
                    except Exception:
                        pass
                print(f"  {name}({params})")

                # Wrap coroutine — input filter is now active for this tool
                if getattr(tool, "coroutine", None):
                    tool.coroutine = _make_input_filter(tool)

            self._tools[server_name] = tools
            ready.set()
            await stop.wait()   # hold connection open until disconnect()

    async def _disconnect(self, server_name: str):
        if server_name in self._stop_events:
            self._stop_events[server_name].set()
        task = self._tasks.get(server_name)
        if task and not task.done():
            try:
                await asyncio.wait_for(task, timeout=5)
            except (asyncio.TimeoutError, Exception):
                task.cancel()
        for store in (self._tasks, self._stop_events, self._ready_events):
            store.pop(server_name, None)

    async def cleanup(self):
        for name in list(self._tasks.keys()):
            await self._disconnect(name)
        self._tools.clear()
        self.sessions.clear()
