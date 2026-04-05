"""
list_mcp_tools.py
-----------------
Connects to Swiggy and Zomato MCP servers, enumerates all available tools
with descriptions and parameter schemas, and saves the results as:
  - mcp_tools_report.txt   (formatted table, human-readable)
  - mcp_tools_report.json  (machine-readable, full schema)

HOW TO GET YOUR ACCESS TOKENS
------------------------------
1. Start the main server:  python main.py
2. Open http://localhost:8000 and log into both Swiggy and Zomato.
3. The server prints the access tokens to the console, e.g.:
      [swiggy] Access token acquired: eyJ...
4. Copy those tokens and set them in your environment (or .env file):
      SWIGGY_ACCESS_TOKEN=eyJ...
      ZOMATO_ACCESS_TOKEN=eyJ...
5. Then run:  python list_mcp_tools.py

Alternatively, pass tokens via CLI:
      python list_mcp_tools.py --swiggy-token eyJ... --zomato-token eyJ...
"""

import os
import sys
import json
import asyncio
import argparse
from datetime import datetime

from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.sse import sse_client

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv(override=True)

PLATFORM_URLS = {
    "swiggy": os.getenv("SWIGGY_MCP_URL", "https://mcp.swiggy.com/food"),
    "zomato": os.getenv("ZOMATO_MCP_URL", "https://mcp-server.zomato.com/mcp"),
}

OUTPUT_TXT  = "mcp_tools_report.txt"
OUTPUT_JSON = "mcp_tools_report.json"


# ── MCP connection helpers ────────────────────────────────────────────────────

async def _fetch_tools_from_url(url: str, token: str | None) -> list[dict]:
    """
    Open an MCP session (streamable-HTTP → SSE fallback),
    call list_tools, and return tool dicts with full schema.
    """
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    async def _init(read, write) -> list[dict]:
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            tools = []
            for t in result.tools:
                schema = {}
                if hasattr(t, "inputSchema") and t.inputSchema:
                    schema = t.inputSchema if isinstance(t.inputSchema, dict) else {}
                tools.append({
                    "name":        t.name,
                    "description": (t.description or "").strip(),
                    "schema":      schema,
                })
            return tools

    # Try streamable-HTTP first, fall back to SSE
    try:
        async with streamablehttp_client(url, headers=headers, timeout=30) as (r, w, _):
            return await _init(r, w)
    except Exception as e1:
        print(f"  streamable-HTTP failed, trying SSE …")
        try:
            async with sse_client(url, headers=headers) as (r, w):
                return await _init(r, w)
        except Exception as e2:
            raise RuntimeError(
                f"Both transports failed.\n  HTTP: {e1}\n  SSE:  {e2}"
            )


# ── Table formatter ───────────────────────────────────────────────────────────

def _param_summary(schema: dict) -> str:
    """Return 'param*(type), …' from a JSON Schema."""
    props    = schema.get("properties", {})
    required = set(schema.get("required", []))
    if not props:
        return "—"
    parts = []
    for name, info in props.items():
        ptype = info.get("type", "any")
        star  = "*" if name in required else ""
        parts.append(f"{name}{star}({ptype})")
    return ", ".join(parts)


def _wrap(text: str, width: int) -> list[str]:
    """Word-wrap text to width, return list of lines."""
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            if cur:
                lines.append(cur.strip())
            cur = w + " "
        else:
            cur += w + " "
    if cur.strip():
        lines.append(cur.strip())
    return lines or ["—"]


def _wrap_csv(text: str, width: int) -> list[str]:
    """Wrap comma-separated params to width."""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    lines, cur = [], ""
    for p in parts:
        chunk = p + ", "
        if len(cur) + len(chunk) > width:
            lines.append(cur.rstrip(", "))
            cur = chunk
        else:
            cur += chunk
    if cur.strip().rstrip(","):
        lines.append(cur.strip().rstrip(","))
    return lines or ["—"]


def _build_table(platform: str, tools: list[dict]) -> str:
    """Build a human-readable ASCII table for one platform."""
    if not tools:
        return f"\n{'='*60}\n  PLATFORM: {platform.upper()}  — no tools\n{'='*60}"

    col_name  = max(len("Tool Name"),  max(len(t["name"]) for t in tools))
    col_desc  = 55
    col_param = 55

    sep  = f"+{'-'*(col_name+2)}+{'-'*(col_desc+2)}+{'-'*(col_param+2)}+"
    head = (
        f"| {'Tool Name':<{col_name}} "
        f"| {'Description':<{col_desc}} "
        f"| {'Parameters  (* = required)':<{col_param}} |"
    )

    lines = [
        "",
        "=" * len(sep),
        f"  PLATFORM: {platform.upper()}  ({len(tools)} tools)",
        "=" * len(sep),
        sep, head, sep,
    ]

    for t in tools:
        desc_lines  = _wrap(t["description"] or "—", col_desc)
        param_lines = _wrap_csv(_param_summary(t["schema"]), col_param)
        row_count   = max(len(desc_lines), len(param_lines))

        for i in range(row_count):
            n_col = t["name"]       if i == 0 else ""
            d_col = desc_lines[i]   if i < len(desc_lines)  else ""
            p_col = param_lines[i]  if i < len(param_lines) else ""
            lines.append(
                f"| {n_col:<{col_name}} | {d_col:<{col_desc}} | {p_col:<{col_param}} |"
            )
        lines.append(sep)

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(tokens: dict[str, str | None]):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    all_data  = {}
    report_sections = [
        f"MCP Tools Report",
        f"Generated : {timestamp}",
        f"Platforms : {', '.join(PLATFORM_URLS.keys())}",
        "",
    ]

    for platform, url in PLATFORM_URLS.items():
        token = tokens.get(platform)
        auth_note = "with token" if token else "NO TOKEN — may fail"
        print(f"\n[{platform}] Connecting to {url} ({auth_note}) …")
        try:
            tools = await _fetch_tools_from_url(url, token)
            print(f"[{platform}] {len(tools)} tools found.")
            all_data[platform] = tools
            report_sections.append(_build_table(platform, tools))
        except Exception as e:
            print(f"[{platform}] FAILED: {e}")
            all_data[platform] = []
            report_sections.append(
                f"\n{'='*60}\n  PLATFORM: {platform.upper()}  — CONNECTION FAILED\n  {e}\n{'='*60}"
            )

    # ── Write text report ─────────────────────────────────────────
    txt_content = "\n".join(report_sections) + "\n"
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(txt_content)
    print(f"\nText report  → {OUTPUT_TXT}")

    # ── Write JSON report ─────────────────────────────────────────
    json_payload = {
        "generated_at": timestamp,
        "platforms": {
            p: {
                "url":        PLATFORM_URLS[p],
                "tool_count": len(tools),
                "tools":      tools,
            }
            for p, tools in all_data.items()
        },
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False)
    print(f"JSON report  → {OUTPUT_JSON}")

    # ── Console summary ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("CONSOLE SUMMARY")
    print("=" * 70)
    total = 0
    for platform, tools in all_data.items():
        total += len(tools)
        print(f"\n{platform.upper()} ({len(tools)} tools):")
        for t in tools:
            params = _param_summary(t["schema"])
            print(f"  {t['name']:<45} {params}")
    print(f"\nTotal tools across all platforms: {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List MCP tools for Swiggy and Zomato")
    parser.add_argument("--swiggy-token", default=os.getenv("SWIGGY_ACCESS_TOKEN"), help="Swiggy OAuth access token")
    parser.add_argument("--zomato-token", default=os.getenv("ZOMATO_ACCESS_TOKEN"), help="Zomato OAuth access token")
    args = parser.parse_args()

    if not args.swiggy_token and not args.zomato_token:
        print(
            "\nWARNING: No access tokens provided.\n"
            "Both Swiggy and Zomato require OAuth authentication.\n\n"
            "Steps to get tokens:\n"
            "  1. Run: python main.py\n"
            "  2. Open http://localhost:8000 and log in to both platforms.\n"
            "  3. The console prints: [swiggy] Access token acquired: eyJ...\n"
            "  4. Set SWIGGY_ACCESS_TOKEN and ZOMATO_ACCESS_TOKEN in .env\n"
            "     or pass --swiggy-token / --zomato-token here.\n"
        )

    tokens = {
        "swiggy": args.swiggy_token,
        "zomato": args.zomato_token,
    }
    asyncio.run(main(tokens))
