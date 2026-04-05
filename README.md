# 🍔 Food Options Aggregator Agent

An AI-powered food search and ordering assistant that connects to **Swiggy** and **Zomato** simultaneously through their official **Model Context Protocol (MCP)** servers. It compares prices across both platforms, recommends the best option, and walks you through the full ordering flow — all through a conversational chat interface.

![React](https://img.shields.io/badge/Frontend-React_19-61DAFB?logo=react)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi)
![LangGraph](https://img.shields.io/badge/Agent-LangGraph_ReAct-4A90D9)
![Gemini](https://img.shields.io/badge/LLM-Google_Gemini-4285F4?logo=google)
![MCP](https://img.shields.io/badge/Protocol-MCP-orange)
![MLflow](https://img.shields.io/badge/Tracing-MLflow-0194E2?logo=mlflow)

---

## What it does

You type a food query like *"Find chicken biryani options near me"* and the agent:

1. Searches Swiggy and Zomato **simultaneously** in a single LLM turn
2. Presents up to **10 options per platform** sorted by price, with ratings and delivery times
3. Gives a **smart recommendation** (highlights preferred restaurants like Sangam if present)
4. Takes your selection (e.g. *"Zomato 1"* or *"Swiggy 3"*)
5. Confirms the order details and asks for a **payment method**
6. Builds the cart, **applies available coupons**, and checks out
7. Displays a **bill breakdown** (item total, delivery fee, taxes, final amount)
8. For UPI payments on Zomato — shows a **scannable QR code** and a download button

All of this happens through a dark-themed React chat interface. The agent remembers context across turns within a session.

---

## Architecture

```
Browser (React Chat UI)
        │  HTTP / WebSocket
        ▼
FastAPI server  ─── port 8000 ────────────────────────────┐
        │                                                  │
        │  LangGraph ReAct Agent (Gemini)                  │
        │  ┌─────────────────────────────────────────┐     │
        │  │  State Modifier (runs before every LLM) │     │
        │  │  ├── Input filter: strip LangGraph       │     │
        │  │  │   internals + hallucinated params     │     │
        │  │  ├── Output filter: truncate lists,      │     │
        │  │  │   extract QR codes from MCP blobs     │     │
        │  │  └── Context compression (3-tier)        │     │
        │  └──────────────┬──────────────────────────┘     │
        │                 │ tool calls                      │
        │       ┌─────────┴──────────┐                     │
        │       ▼                    ▼                      │
        │  Swiggy MCP           Zomato MCP                  │
        │  mcp.swiggy.com       mcp-server.zomato.com       │
        │  (OAuth 2.0 + PKCE)   (OAuth 2.0 + PKCE)         │
        │                                                   │
        ├── /api/payment/qr/{id}  → serves UPI QR PNG       │
        ├── /api/auth/*/start     → initiates OAuth         │
        └── MLflow  ─── port 5000 (auto-started)            │
                    (traces all LLM + tool calls)  ─────────┘
```

### Key design decisions

| Decision | Rationale |
|---|---|
| MCP over direct API scraping | Official, authenticated, schema-validated tool calls — no brittle HTML parsing |
| LangGraph `create_react_agent` | Built-in ReAct loop, `MemorySaver` checkpointer for multi-turn sessions |
| State modifier pattern | Runs before every LLM call — guarantees tool outputs are filtered/compressed without modifying LangGraph internals |
| Gemini via `langchain-google-genai` | Supports parallel tool calls in a single turn (searches both platforms simultaneously) |
| Three-tier context compression | Keeps the context window lean without losing critical ordering IDs |

---

## Features

### Agent capabilities
- **Parallel search**: calls `search_restaurants`, `search_menu` (Swiggy) and `get_restaurants_for_keyword`, `get_menu_items_listing` (Zomato) in a single LLM turn
- **Cross-platform comparison table**: identical column format for both platforms, sorted cheapest first
- **Smart recommendation**: picks the best price/rating/delivery balance; highlights Sangam Restaurant when present for biryani
- **Turn-to-turn continuity**: after presenting the table, the LLM emits a `📋 ID Reference` block containing all restaurant/item IDs — this block survives context compression so the correct IDs are always available when the user makes a selection
- **Automatic coupon application**: fetches available coupons and applies the first valid one silently; if coupon fails, skips without blocking the flow
- **Bill breakdown display**: after cart creation shows `item_total`, platform fee, delivery charge (with discount), taxes, `final_amount`, and estimated delivery time
- **UPI QR payment**: for Zomato `upi_qr` payment — decodes the base64 PNG returned by `checkout_cart`, stores it server-side, returns the QR panel to the frontend with a download button
- **Session expiry handling**: detects `SESSION_EXPIRED_403` from either platform mid-flow and automatically switches to the other platform, completing the full order there
- **Token refresh**: when a 403 is received, attempts to exchange the stored `refresh_token` for a new access token before marking the session as expired

### Infrastructure
- **Bidirectional tool filter**: input side strips LangGraph runtime internals and any parameters Gemini hallucinated that aren't in the tool's schema; output side truncates large list responses and caps total characters
- **Exponential backoff**: retries on Gemini 429/RESOURCE_EXHAUSTED with jitter (up to 4 attempts)
- **Daily request budget**: soft cap of 1400 requests/day on the free Gemini tier with a clear user-facing error when hit
- **Agent logging**: every LLM call and MCP tool invocation logged to a daily JSONL file with token counts and latency
- **MLflow tracing**: auto-starts MLflow and logs all agent traces under the `Food_Aggregator_Agent` experiment
- **Browser auto-open**: `python main.py` starts the server and opens `http://localhost:8000` automatically

---

## Prerequisites

| Requirement | Version | How to check |
|---|---|---|
| Python | 3.11+ | `python --version` |
| Node.js | 18+ | `node --version` |
| npm | 9+ | `npm --version` |
| Gemini API key | — | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) — free tier available |
| Swiggy account | — | Must be the account you want to order from |
| Zomato account | — | Must be the account you want to order from |

> **Note on `npx`**: The app uses `npx mcp-remote` internally to connect to the MCP servers. This is bundled with Node.js — no separate install needed.

---

## Installation

### 1. Clone

```bash
git clone https://github.com/your-username/swiggy-zomato-mcp-search.git
cd swiggy-zomato-mcp-search
```

### 2. Python environment

```bash
python -m venv venv

# Activate
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows (cmd)
# or: venv\Scripts\Activate.ps1 for PowerShell

pip install -r requirements.txt
```

> `requirements.txt` installs: `langchain`, `langgraph`, `langchain-google-genai`, `langchain-mcp-adapters`, `mcp`, `mlflow`, `fastapi`, `uvicorn`, `python-dotenv`, `qrcode[pil]`, `httpx`

### 3. React frontend

```bash
cd frontend
npm install
npm run build   # outputs to frontend/dist/ — served by FastAPI
cd ..
```

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Required
GEMINI_API_KEY="your_gemini_api_key_here"

# Optional — defaults shown
GEMINI_MODEL="gemini-2.5-flash"
GEMINI_TEMPERATURE="0.1"
GEMINI_MAX_TOKENS="4096"
SWIGGY_MCP_URL="https://mcp.swiggy.com/food"
ZOMATO_MCP_URL="https://mcp-server.zomato.com/mcp"
MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
```

---

## Running the app

```bash
source venv/bin/activate    # Windows: venv\Scripts\activate
python main.py
```

This single command:
1. Auto-starts MLflow on port 5000 (if not already running)
2. Starts the FastAPI server on port 8000
3. Opens `http://localhost:8000` in your default browser

**For frontend hot-reload during development:**

```bash
# Terminal 1 — backend
python main.py

# Terminal 2 — React dev server with hot reload
cd frontend
npm run dev
# Open http://localhost:5173
```

Vite proxies all `/api/*` requests to the FastAPI backend automatically.

---

## Connecting to Swiggy and Zomato

Both platforms require OAuth 2.0 authentication. The sidebar in the UI has individual **Connect** buttons for each platform.

### How OAuth works in this app

**Swiggy:**
1. Click **Connect Swiggy** → app registers an OAuth client with Swiggy's auth server, then redirects your browser to Swiggy's login page
2. Log in and authorize → Swiggy redirects back to `http://localhost:8000/api/auth/swiggy/callback`
3. The app exchanges the code for an access token, connects to the Swiggy MCP server, and pre-fetches your saved delivery address
4. A small popup confirms connection; the sidebar shows the tool count

**Zomato:**
1. Click **Connect Zomato** → the app starts a temporary HTTP server on port 3334 (Zomato's whitelisted callback port), then redirects your browser to Zomato's login page
2. Log in and authorize → Zomato redirects to `http://127.0.0.1:3334/callback`
3. The port-3334 server captures the code, exchanges it for a token, connects to the Zomato MCP server, and pre-fetches your address
4. The temporary server shuts down automatically

> **Why port 3334?** Zomato's OAuth server only allows `http://127.0.0.1:3334/callback` as a valid redirect URI (the same URI used by `mcp-remote`). This app replicates that callback flow directly to avoid needing `mcp-remote` as a subprocess.

### Token refresh
If a session expires mid-conversation (403 from MCP), the app automatically attempts to refresh the token using the stored `refresh_token`. If refresh fails, the agent tells you to re-login and completes the order on the other platform if possible.

---

## The ordering flow

Once both platforms are connected, start a conversation:

```
You:   Find chicken biryani near me

Agent: [searches Swiggy + Zomato in parallel]

       ## 🟠 Swiggy Options
       | # | Restaurant        | Area          | Item             | Price | Rating | Delivery |
       |---|-------------------|---------------|------------------|-------|--------|----------|
       | 1 | Madurai Pandi Mess| Kalyan Nagar  | Chicken Biryani  | ₹199  | ⭐4.3  | ~52 min  |
       | 2 | Meghana Foods     | Kalyan Nagar  | Chicken Biryani  | ₹230  | ⭐4.5  | ~41 min  |
       ...

       ## 🔴 Zomato Options
       | # | Restaurant  | Area | Item            | Price | Rating | Delivery  |
       |---|-------------|------|-----------------|-------|--------|-----------|
       | 1 | Sangam Mess | —    | Chicken Biryani | ₹230  | ⭐4.2  | ~35 min   |
       ...

       💡 Recommendation: Sangam Mess on Zomato at ₹230 is the preferred choice...

       📋 ID Reference (preserved for ordering):
       Zomato: 1. Chicken Biryani | res_id=57482 | variant_id=v_230514425 | price=₹230 | restaurant=Sangam Mess
       ...

       Reply with Swiggy <number> or Zomato <number> to place the order.

You:   Zomato 1

Agent: Selected: Chicken Biryani from Sangam Mess on Zomato
       restaurant_id = 57482 | variant_id = v_230514425 | price = ₹230

       Payment options:
         1. UPI QR
         2. Pay Later
       Choose your payment method.

You:   1

Agent: [calls create_cart → applies coupons → calls checkout_cart]

       🧾 Order Bill:
       | Item Total      | ₹230     |
       | Platform Fee    | ₹17.58   |
       | Delivery Fee    | Free     |
       | Taxes           | ₹14.18   |
       | Total Payable   | ₹261.76  |
       | Estimated Delivery | 35-40 min |

       Order Placed Successfully!
       Order ID: 122db02b-7d7e-4fd3-b483-...

       [QR code panel appears in the UI — scan to complete UPI payment]
```

---

## Context compression

The agent uses a **three-tier strategy** to keep the context window lean without losing critical information:

| Tier | Condition | Action |
|---|---|---|
| **1 — Action phase** | Any cart / checkout / coupon tool has been called | Keep ALL messages intact — every tool result is needed |
| **2 — Post-selection** | User has replied after the last search result AND a `📋 ID Reference` block exists in the chat history | Stub the raw search payloads (often 30–80k chars) with one-line placeholders. IDs remain accessible via the reference block |
| **3 — Search in progress** | Search tools ran but user hasn't replied yet, OR no `📋 ID Reference` block exists | Keep everything — the LLM needs the data to build the table, or hasn't output the ID block yet |

The `📋 ID Reference` block is the safety gate for Tier 2. If the LLM omits it (unlikely but possible), compression is blocked and the full search data is preserved. If IDs are genuinely unavailable, the system prompt instructs the agent to re-run the search rather than guess.

---

## Tool filtering

### Input filter (LLM → MCP)
Every MCP tool is wrapped once at connection time. Before each call the wrapper:
1. Removes LangGraph runtime keys (`runtime`, `config`, `store`, `writer`) that are injected by the framework but not part of any MCP schema
2. Removes any additional key that Gemini hallucinated and that is not declared in the tool's `args_schema`

### Output filter (MCP → LLM)
Before every LLM call, the state modifier runs `filter_tool_result` on each `ToolMessage`:
- **List truncation**: search and menu tools capped at 10 items; coupon tools capped at 3
- **Character cap**: per-tool limits (8 000 chars for restaurant search, 10 000 for menu data)
- **QR extraction**: `checkout_cart` returns a list of MCP content blocks, one of which is a base64 PNG image. The filter decodes the PNG, stores it in `pending_payment_qr[order_id]`, strips the binary blob from the tool result, and replaces it with an image markdown link for the agent to forward to the user

---

## Agent logging

Every LLM call and MCP tool invocation is appended to `logs/agent_log_YYYYMMDD.jsonl` (one JSON object per line, UTF-8, daily rotation).

| `type` field | Contents |
|---|---|
| `session_start` | `session_id`, `timestamp` |
| `llm_input` | All messages sent to Gemini, `message_count`, `estimated_input_tokens` (rough 4-char/token estimate) |
| `llm_output` | Generated text, `tool_calls`, actual `token_usage` (input/output/total), `duration_ms` |
| `tool_input` | `tool_name`, parsed `args` |
| `tool_output` | `tool_name`, truncated `output` (first 4000 chars), `output_length`, `duration_ms` |
| `tool_error` | `tool_name`, `error` string, `duration_ms` |
| `llm_error` | `error` string |

**Reading logs:**

```bash
# Pretty-print the last 5 entries (Python)
python -c "
import json, sys
sys.stdout.reconfigure(encoding='utf-8')
lines = open('logs/agent_log_$(date +%Y%m%d).jsonl', encoding='utf-8').readlines()
for l in lines[-5:]:
    d = json.loads(l)
    print(d.get('type'), '|', d.get('tool_name',''), d.get('duration_ms',''), 'ms')
"
```

---

## Project structure

```
.
├── main.py              # FastAPI server
│                        #  ├── OAuth 2.0 + PKCE for Swiggy and Zomato
│                        #  ├── Port-3334 callback server for Zomato
│                        #  ├── Address pre-fetch after login
│                        #  ├── Token refresh on 403
│                        #  ├── /api/chat — agent endpoint
│                        #  ├── /api/payment/qr/{id} — serves UPI QR PNG
│                        #  ├── /api/status — connection health
│                        #  └── Serves React frontend (frontend/dist/)
│
├── agent.py             # LangGraph ReAct agent
│                        #  ├── _compress_old_tool_messages — 3-tier compression
│                        #  ├── _make_prompt — state modifier (runs before every LLM call)
│                        #  ├── _build_system_prompt — phase-by-phase tool guide,
│                        #  │   mandatory address/payment params, ID Reference instruction,
│                        #  │   bill breakdown display, QR conditional output rule
│                        #  ├── compare_addresses_by_pincode — pure-Python address matching
│                        #  └── Gemini integer enum schema patch
│
├── mcp_client.py        # MCP connection manager + bidirectional tool filter
│                        #  ├── MCPServerManager — connects via streamable-HTTP or SSE
│                        #  ├── _make_input_filter — strips LangGraph internals + hallucinations
│                        #  ├── filter_tool_result — truncates output, extracts QR PNG
│                        #  └── pending_payment_qr — in-memory order_id → PNG store
│
├── agent_logger.py      # LangChain BaseCallbackHandler → JSONL log
│                        #  One instance per /api/chat request
│                        #  Captures LLM input/output, tool input/output/error, token usage
│
├── list_mcp_tools.py    # Standalone utility: list all MCP tools with schemas
│                        #  Run after logging in to get a full tool reference
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx      # Chat UI
│   │   │                #  ├── renderMarkdown — parses bold, code, links, images
│   │   │                #  ├── renderTable — renders comparison tables with overflow scroll
│   │   │                #  ├── QR panel — shows scannable image + download button
│   │   │                #  ├── Platform connect buttons with OAuth popup handling
│   │   │                #  └── Session expiry detection + auto re-connect prompt
│   │   ├── index.css    # Dark gradient theme, table layout, QR image styles
│   │   └── main.jsx     # React 19 entry point
│   ├── index.html
│   ├── vite.config.js   # Proxy /api/* → http://localhost:8000
│   └── package.json     # React 19, Vite 6
│
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
├── .gitignore
├── Dockerfile           # Multi-stage: Node 20 builds React → Python 3.11 serves all
├── render.yaml          # Render Blueprint (one-click deploy)
└── README.md
```

---

## API reference

### `GET /api/auth/{platform}/start`
Initiates OAuth for `swiggy` or `zomato`. Redirects the browser to the platform's login page.

### `GET /api/auth/swiggy/callback`
Swiggy OAuth callback — exchanges the code for a token, connects to MCP, pre-fetches address.

### `POST /api/chat`

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find chicken biryani near me",
    "session_id": "my-session-abc123"
  }'
```

**Request body:**

| Field | Type | Description |
|---|---|---|
| `query` | string | The user's message |
| `session_id` | string | Unique session identifier — reuse across turns to maintain conversation history |

**Response:**

```json
{
  "response": "## 🟠 Swiggy Options\n...",
  "qr_order_id": "122db02b-7d7e-4fd3-b483-e97fc4402ab3",
  "status_changed": false,
  "expired_platforms": []
}
```

| Field | Description |
|---|---|
| `response` | Markdown-formatted agent reply |
| `qr_order_id` | Non-null when a QR was generated this turn — use with `/api/payment/qr/{id}` |
| `status_changed` | `true` when a platform session expired during this turn |
| `expired_platforms` | List of platform names (`"swiggy"`, `"zomato"`) whose session expired |

### `GET /api/payment/qr/{order_id}`

Returns the UPI QR code as a PNG image.

```bash
# View inline
curl http://localhost:8000/api/payment/qr/122db02b-... -o qr.png

# Download (triggers browser save dialog)
curl "http://localhost:8000/api/payment/qr/122db02b-...?download=1" -o qr.png
```

### `GET /api/status`

```json
{
  "agent_ready": true,
  "swiggy": {"name": "Swiggy", "status": "ok", "tools": 13},
  "zomato": {"name": "Zomato", "status": "ok", "tools": 11},
  "total_tools": 24
}
```

---

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | ✅ | — | Google Gemini API key. Get one free at [aistudio.google.com](https://aistudio.google.com/apikey) |
| `GEMINI_MODEL` | | `gemini-2.5-flash` | Gemini model. `gemini-2.5-flash` recommended for cost and speed |
| `GEMINI_TEMPERATURE` | | `0.1` | Lower = more deterministic tool calls. Keep at 0.1 or below |
| `GEMINI_MAX_TOKENS` | | `4096` | Max output tokens per LLM turn |
| `SWIGGY_MCP_URL` | | `https://mcp.swiggy.com/food` | Swiggy MCP endpoint |
| `ZOMATO_MCP_URL` | | `https://mcp-server.zomato.com/mcp` | Zomato MCP endpoint |
| `MLFLOW_TRACKING_URI` | | `http://127.0.0.1:5000` | MLflow server. Auto-started by `python main.py` |

---

## Deployment

### Render (recommended — free tier)

The repo includes a `render.yaml` Blueprint:

1. Push to GitHub
2. Go to [render.com](https://render.com) → **New** → **Blueprint**
3. Connect your repo
4. Set the `GEMINI_API_KEY` environment variable in the Render dashboard
5. Click **Apply** — Render builds the Docker image and deploys

> **OAuth redirect URIs**: For cloud deployment, update `redirect_uri` in `PLATFORMS` in `main.py` to use your Render domain (e.g. `https://your-app.onrender.com/api/auth/swiggy/callback`). Zomato's port-3334 callback only works on localhost.

### Docker

```bash
# Build (compiles React + installs Python)
docker build -t food-aggregator .

# Run
docker run -p 8000:8000 \
  -e GEMINI_API_KEY="your_key_here" \
  food-aggregator

# Open http://localhost:8000
```

The Dockerfile is multi-stage:
- Stage 1: Node 20 — runs `npm install && npm run build`
- Stage 2: Python 3.11-slim — installs Python deps, copies the built React dist and all `.py` files, installs Node.js runtime (needed for `npx mcp-remote` at runtime)

---

## MLflow tracing

MLflow is started automatically when you run `python main.py`. Open **http://127.0.0.1:5000** to see all agent traces.

Under the **Food_Aggregator_Agent** experiment you can inspect:
- Every LLM call with its full input/output and token counts
- Every MCP tool invocation with arguments and responses
- The complete ReAct reasoning chain across all turns

---

## Listing available MCP tools

After logging in, you can enumerate all tools with their parameter schemas:

```bash
# The server prints the access token to the console after login, e.g.:
# [swiggy] Full token (for list_mcp_tools.py): eyJ...

python list_mcp_tools.py \
  --swiggy-token "eyJ..." \
  --zomato-token "eyJ..."
```

This writes `mcp_tools_report.txt` (human-readable table) and `mcp_tools_report.json` (full schema).

---

## Tech stack

| Layer | Technology | Notes |
|---|---|---|
| **LLM** | Google Gemini (`langchain-google-genai`) | Parallel tool calls in a single turn |
| **Agent framework** | LangGraph `create_react_agent` | ReAct loop + `MemorySaver` per session |
| **MCP client** | `langchain-mcp-adapters` + `mcp` | Streamable-HTTP with SSE fallback |
| **Backend** | FastAPI + Uvicorn | OAuth flows, file serving, QR endpoint |
| **Frontend** | React 19 + Vite 6 | Custom markdown/table renderer, no UI library |
| **QR generation** | `qrcode[pil]` | Generated server-side from UPI string/intent URL |
| **Tracing** | MLflow `langchain.autolog` | Automatic LangChain trace capture |
| **Logging** | Custom `BaseCallbackHandler` | JSONL file, one entry per event |

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError` | venv not activated | `source venv/bin/activate` (or `venv\Scripts\activate` on Windows) |
| `GEMINI_API_KEY not set` | `.env` not created | `cp .env.example .env` then add your key |
| `frontend/dist not found` | React not built | `cd frontend && npm install && npm run build` |
| OAuth window doesn't open | Pop-up blocker | Allow pop-ups for `localhost:8000` in your browser |
| Swiggy/Zomato login keeps looping | Stale OAuth state | Click **Connect** again — each click registers a fresh OAuth client |
| Port 3334 in use (Zomato) | Another process | Kill it: `lsof -ti:3334 \| xargs kill` (macOS/Linux) or `netstat -ano \| findstr 3334` (Windows) |
| `QR not found` error | Checkout tool result wasn't captured | Check `logs/` — look for `tool_error` on `checkout_cart` |
| Agent says "I encountered an error" | ID Reference block missing from prior turn | Start a fresh session — the new system prompt generates the block automatically |
| `429 RESOURCE_EXHAUSTED` | Gemini rate limit | The app retries automatically with backoff; if it persists, wait ~1 minute |
| Daily limit reached | 1400 requests/day on free Gemini tier | Upgrade to a paid key or wait until midnight |
| MLflow not accessible | Port 5000 in use | `python main.py` will skip auto-start; run `mlflow ui --port 5001` manually and update `MLFLOW_TRACKING_URI` |
| Agent skips coupon step | Coupon API returned an error | Expected — the agent is designed to skip silently and proceed to checkout |

---

## Known limitations

- **MemorySaver is in-memory only**: conversation history is lost when the server restarts. Each restart requires re-logging in to Swiggy/Zomato as well (access tokens are also in-memory).
- **Zomato port-3334 callback only works on localhost**: cloud deployments cannot complete Zomato OAuth without running `mcp-remote` as a sidecar. Swiggy works on any domain.
- **Single delivery address**: the app uses the first saved address returned by each platform's address API. Multiple addresses are not currently supported.
- **Gemini free tier**: `gemini-2.5-flash` on the free tier has a 1500 requests/day limit and occasional 429 errors during peak hours. A paid API key removes these constraints.

---

## License

MIT
