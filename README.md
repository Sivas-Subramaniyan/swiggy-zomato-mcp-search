# 🍔 Food Aggregator Agent

Compare food prices across **Swiggy** and **Zomato** using AI — powered by Google Gemini, LangGraph, and the Model Context Protocol (MCP).

![React](https://img.shields.io/badge/Frontend-React-61DAFB?logo=react)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi)
![LangGraph](https://img.shields.io/badge/Agent-LangGraph-blue)
![Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-4285F4?logo=google)
![MLflow](https://img.shields.io/badge/Tracing-MLflow-0194E2?logo=mlflow)

---

## ✨ Features

- 🔍 **Cross-platform search** — queries both Swiggy and Zomato simultaneously
- 🎟️ **Coupon comparison** — fetches and applies available discounts on both platforms
- 💬 **React chat UI** — dark-themed conversational interface with multi-turn memory
- 📊 **MLflow tracing** — full observability of agent reasoning, tool calls, and LLM responses
- ⚡ **Token-optimized** — tool descriptions trimmed to reduce API costs
- 🐳 **Deployable** — multi-stage Dockerfile + Render Blueprint included

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────┐
│                 Render (Docker)                  │
│                                                  │
│  ┌────────────────────────────────────────────┐  │
│  │  FastAPI (port 8000)                       │  │
│  │  ├── GET  /           → React static app   │  │
│  │  ├── POST /api/connect → connect MCP       │  │
│  │  ├── POST /api/chat   → agent endpoint     │  │
│  │  └── GET  /api/status → health check       │  │
│  └──────────┬─────────────────────────────────┘  │
│             │                                    │
│  ┌──────────▼─────────────────────────────────┐  │
│  │  LangGraph ReAct Agent (Gemini)            │  │
│  │  ├── Swiggy MCP (via npx mcp-remote)       │  │
│  │  └── Zomato MCP (via npx mcp-remote)       │  │
│  └────────────────────────────────────────────┘  │
│                                                  │
│  ┌────────────────────────────────────────────┐  │
│  │  MLflow Tracking (localhost:5000)           │  │
│  └────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
.
├── frontend/                # React chat frontend (Vite)
│   ├── src/
│   │   ├── App.jsx          # Chat UI component
│   │   ├── index.css        # Dark gradient theme
│   │   └── main.jsx         # React entry point
│   ├── index.html           # HTML template
│   ├── vite.config.js       # Vite config with API proxy
│   ├── package.json         # Frontend dependencies
│   └── dist/                # Built static files (after npm run build)
│
├── main.py                  # FastAPI backend (API + serves React)
├── agent.py                 # LangGraph ReAct agent with Gemini
├── mcp_client.py            # MCP server connection manager
├── streamlit_app.py         # Streamlit frontend (alternative)
│
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
├── .gitignore               # Git ignore rules
│
├── Dockerfile               # Multi-stage: Node builds React → Python serves all
├── render.yaml              # Render Blueprint for one-click deploy
└── README.md                # This file
```

---

## 🚀 Quick Start (Step-by-Step)

### Prerequisites

| Requirement | Version | Check Command |
|---|---|---|
| Python | 3.11+ | `python --version` |
| Node.js | 18+ | `node --version` |
| npm | 9+ | `npm --version` |
| Gemini API Key | — | [Get one here](https://aistudio.google.com/apikey) |

---

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd "Swiggy MCP POC"
```

---

### Step 2: Setup Python Backend

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate          # Windows

# Install Python dependencies
pip install -r requirements.txt
```

**Verify:** You should see packages installing without errors. Run this to confirm:
```bash
python -c "from agent import create_food_aggregator_agent; print('✅ Backend imports OK')"
```

---

### Step 3: Setup React Frontend

```bash
# Navigate to frontend
cd frontend

# Install Node dependencies
npm install

# Build the React app
npm run build

# Go back to project root
cd ..
```

**Verify:** Check that the build output exists:
```bash
ls frontend/dist/
# Should show: index.html  assets/
```

---

### Step 4: Configure Environment Variables

```bash
# Copy the template
cp .env.example .env
```

Now edit `.env` with your editor and add your Gemini API key:

```env
# ─── REQUIRED ───
GEMINI_API_KEY="your_gemini_api_key_here"

# ─── OPTIONAL (defaults shown) ───
GEMINI_MODEL="gemini-3-flash-preview"
SWIGGY_MCP_URL="https://mcp.swiggy.com/food"
ZOMATO_MCP_URL="https://mcp-server.zomato.com/mcp"
MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
MAX_TOOL_DESC_LENGTH="200"
```

---

### Step 5: Run the Application

You have **3 ways** to run the app. Choose one:

#### Option A: React + FastAPI (Recommended)

This is the production setup. FastAPI serves both the API and the React UI on a single port.

**Terminal 1** — Start the server:
```bash
source venv/bin/activate
RUN_SERVER=1 uvicorn main:app --host 0.0.0.0 --port 8000
```

**Open:** http://localhost:8000

**Usage:**
1. Click **"Connect to MCP Servers"** in the sidebar
2. A browser window will open for Swiggy OAuth — authorize it
3. Another browser window will open for Zomato OAuth — authorize it
4. Once connected, type your food query and chat!

---

#### Option B: React Dev Mode (for frontend development)

Run the React dev server with hot-reload, proxied to the FastAPI backend.

**Terminal 1** — Start FastAPI backend:
```bash
source venv/bin/activate
RUN_SERVER=1 uvicorn main:app --host 0.0.0.0 --port 8000
```

**Terminal 2** — Start React dev server:
```bash
cd frontend
npm run dev
```

**Open:** http://localhost:5173 (Vite dev server with hot-reload)

---

#### Option C: CLI Mode (no UI needed)

For quick testing directly in the terminal.

```bash
source venv/bin/activate

# Interactive chat mode
python main.py

# Or single query mode
python main.py -q "I want to order Narmada Chicken Biriyani"
```

---

#### Option D: Streamlit (alternative UI)

```bash
source venv/bin/activate
streamlit run streamlit_app.py --server.port 8501
```

**Open:** http://localhost:8501

---

### Step 6: (Optional) Start MLflow for Tracing

Open a **separate terminal**:

```bash
source venv/bin/activate
mlflow ui --port 5000
```

**Open:** http://localhost:5000

All agent traces will appear under the **Food_Aggregator_Agent** experiment. You can see:
- Every LLM call and its tokens
- Every MCP tool invocation and its arguments/results
- The full ReAct reasoning chain

---

## 🔌 API Reference

The FastAPI backend exposes these endpoints:

### `POST /api/connect`

Connects to Swiggy and Zomato MCP servers. Call this once before chatting.

```bash
curl -X POST http://localhost:8000/api/connect
```

**Response:**
```json
{
  "connected": true,
  "servers": [
    {"name": "Swiggy", "status": "ok", "tools": 13},
    {"name": "Zomato", "status": "ok", "tools": 11}
  ],
  "total_tools": 24
}
```

### `POST /api/chat`

Send a message to the agent.

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "I want to order Narmada Chicken Biriyani", "session_id": "my-session-1"}'
```

**Response:**
```json
{
  "response": "I found several options for Narmada Chicken Biriyani..."
}
```

> **Note:** Use the same `session_id` across requests to maintain conversation history (multi-turn).

### `GET /api/status`

Check if MCP servers are connected.

```bash
curl http://localhost:8000/api/status
```

---

## 🚢 Deployment

### Option 1: Render (Free — Recommended)

Render offers a free Docker-based web service tier.

1. Push your code to GitHub
2. Go to [render.com](https://render.com) → **New** → **Web Service**
3. Connect your GitHub repo
4. Select **Docker** as the environment
5. Add environment variable: `GEMINI_API_KEY` = your key
6. Click **Deploy**

The included `render.yaml` file also supports **Blueprint deploys** — just click "New Blueprint Instance" on Render and point to your repo.

### Option 2: Docker (anywhere)

```bash
# Build the image (this builds React + installs Python)
docker build -t food-aggregator .

# Run it
docker run -p 8000:8000 \
  -e GEMINI_API_KEY="your_key_here" \
  -e GEMINI_MODEL="gemini-3-flash-preview" \
  food-aggregator
```

**Open:** http://localhost:8000

### Option 3: Google Cloud Run

```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT/food-aggregator

gcloud run deploy food-aggregator \
  --image gcr.io/YOUR_PROJECT/food-aggregator \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars "GEMINI_API_KEY=your_key"
```

---

## ⚙️ Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | ✅ | — | Google Gemini API key |
| `GEMINI_MODEL` | | `gemini-3-flash-preview` | Gemini model to use |
| `SWIGGY_MCP_URL` | | `https://mcp.swiggy.com/food` | Swiggy MCP server endpoint |
| `ZOMATO_MCP_URL` | | `https://mcp-server.zomato.com/mcp` | Zomato MCP server endpoint |
| `MLFLOW_TRACKING_URI` | | `http://127.0.0.1:5000` | MLflow tracking server URL |
| `MAX_TOOL_DESC_LENGTH` | | `200` | Max characters for MCP tool descriptions (token optimization) |
| `RUN_SERVER` | | — | Set to `1` to start FastAPI server (otherwise starts CLI) |

---

## 📋 How It Works

1. **Connect** — Click the button → FastAPI spawns `npx mcp-remote` processes that connect to Swiggy/Zomato remote HTTP MCP servers via OAuth
2. **Load Tools** — MCP tools (search, menu, cart, coupons, orders, tracking) are loaded; descriptions truncated to save tokens
3. **Chat** — Your message enters a LangGraph ReAct loop powered by Gemini with `MemorySaver` for multi-turn conversations
4. **Search** — The agent calls MCP tools on both platforms, compares results, applies available coupons
5. **Recommend** — The agent formats a response comparing prices and recommends the cheapest option

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Google Gemini (`langchain-google-genai`) |
| **Agent** | LangGraph ReAct + `MemorySaver` checkpointer |
| **MCP** | `langchain-mcp-adapters` + `npx mcp-remote` |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | React + Vite |
| **Tracing** | MLflow |
| **Deployment** | Docker / Render / Cloud Run |

---

## 🐛 Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` | Make sure venv is activated: `source venv/bin/activate` |
| `npx: command not found` | Install Node.js 18+: `brew install node` (macOS) |
| OAuth window doesn't open | Check your browser's pop-up blocker |
| `GEMINI_API_KEY not set` | Add your key to `.env` — see Step 4 |
| `frontend/dist not found` | Run `cd frontend && npm run build` — see Step 3 |
| MLflow connection error | Start MLflow first: `mlflow ui --port 5000` |
| Token limit errors | Reduce `MAX_TOOL_DESC_LENGTH` in `.env` (default: 200) |

---

## 📄 License

MIT
