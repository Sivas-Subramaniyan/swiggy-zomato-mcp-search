"""
agent_logger.py — LangChain callback handler that records every LLM call and
MCP tool call to a daily append-only JSONL file.

Log location:  logs/agent_log_YYYYMMDD.jsonl  (one JSON object per line)

Each entry has a "type" field:
  session_start  — emitted once per /api/chat request
  llm_input      — messages sent to Gemini (full context window snapshot)
  llm_output     — Gemini response + token usage + latency
  tool_input     — MCP tool name + arguments
  tool_output    — raw MCP response + latency
  tool_error     — exception from a tool call
  llm_error      — exception from an LLM call
"""

import os
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Union

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_core.outputs import LLMResult

# ── Log directory ──────────────────────────────────────────────────────────────
_LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)


def _log_path() -> str:
    return os.path.join(_LOG_DIR, f"agent_log_{datetime.now().strftime('%Y%m%d')}.jsonl")


def _write(entry: dict) -> None:
    """Append one JSON line to today's log file (thread-safe for single-process use)."""
    entry.setdefault("timestamp", datetime.now().isoformat(timespec="milliseconds"))
    try:
        with open(_log_path(), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:
        print(f"[logger] write failed: {exc}")


# ── Message serialiser ─────────────────────────────────────────────────────────

def _serialise_message(msg: BaseMessage) -> dict:
    """Convert a LangChain message to a JSON-safe dict for logging."""
    role = type(msg).__name__.replace("Message", "").lower()  # human / ai / tool / system
    content = msg.content

    # Truncate very long content but keep it meaningful
    if isinstance(content, str):
        content_log = content[:4000] + ("…" if len(content) > 4000 else "")
    elif isinstance(content, list):
        # MCP content blocks — summarise image blocks, keep text
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "image":
                    parts.append(f"[image block, mime={block.get('mime_type','?')}, "
                                 f"b64_len={len(block.get('base64',''))}]")
                elif block.get("type") == "text":
                    parts.append(block.get("text", "")[:1000])
                else:
                    parts.append(json.dumps({k: v for k, v in block.items()
                                             if k not in ("base64", "data")},
                                            ensure_ascii=False)[:500])
        content_log = "\n".join(parts)
    else:
        content_log = str(content)[:4000]

    entry = {"role": role, "content": content_log}

    # Attach tool-call metadata when present
    if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
        entry["tool_calls"] = [
            {"id": tc.get("id"), "name": tc.get("name"), "args": tc.get("args")}
            for tc in msg.tool_calls
        ]
    if isinstance(msg, ToolMessage):
        entry["tool_call_id"] = msg.tool_call_id
        entry["tool_name"] = getattr(msg, "name", "")

    return entry


# ── Callback handler ───────────────────────────────────────────────────────────

class AgentLogger(BaseCallbackHandler):
    """
    Plugs into a LangGraph agent via the invocation config's "callbacks" key.
    One instance per /api/chat request; resets turn counter per request.
    """

    raise_error = False   # don't let logging failures crash the agent

    def __init__(self, session_id: str = ""):
        super().__init__()
        self.session_id = session_id
        self._turn: int = 0
        self._llm_t0: dict[str, float] = {}   # run_id → start time
        self._tool_t0: dict[str, float] = {}
        self._tool_name: dict[str, str] = {}

        _write({
            "type": "session_start",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        })

    # ── LLM ───────────────────────────────────────────────────────────────────

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        **kwargs,
    ) -> None:
        run_id = str(kwargs.get("run_id", ""))
        self._llm_t0[run_id] = time.time()
        self._turn += 1

        flat = [msg for batch in messages for msg in batch]
        token_estimate = sum(len(str(m.content)) for m in flat) // 4  # rough 4-char/token

        _write({
            "type": "llm_input",
            "session_id": self.session_id,
            "turn": self._turn,
            "run_id": run_id,
            "message_count": len(flat),
            "estimated_input_tokens": token_estimate,
            "messages": [_serialise_message(m) for m in flat],
        })

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        run_id = str(kwargs.get("run_id", ""))
        elapsed_ms = round((time.time() - self._llm_t0.pop(run_id, time.time())) * 1000)

        # ── Token usage ──────────────────────────────────────────────────────
        # ChatGoogleGenerativeAI may surface usage in llm_output or on the
        # generation's message.usage_metadata (LangChain ≥ 0.2).
        token_usage: dict = {}
        if response.llm_output:
            raw = (response.llm_output.get("token_usage")
                   or response.llm_output.get("usage_metadata")
                   or response.llm_output.get("usage")
                   or {})
            token_usage = {
                "input_tokens":  raw.get("input_tokens")  or raw.get("prompt_tokens")     or 0,
                "output_tokens": raw.get("output_tokens") or raw.get("completion_tokens") or 0,
                "total_tokens":  raw.get("total_tokens")  or 0,
            }

        # Fallback: try generation.message.usage_metadata (Gemini-specific path)
        if not any(token_usage.values()):
            for batch in response.generations:
                for gen in batch:
                    um = getattr(getattr(gen, "message", None), "usage_metadata", None)
                    if um:
                        token_usage = {
                            "input_tokens":  um.get("input_tokens", 0),
                            "output_tokens": um.get("output_tokens", 0),
                            "total_tokens":  um.get("total_tokens", 0),
                        }
                        break

        # ── Response content ─────────────────────────────────────────────────
        generations_log = []
        for batch in response.generations:
            for gen in batch:
                text = getattr(gen, "text", "") or ""
                tool_calls = []
                msg = getattr(gen, "message", None)
                if msg and getattr(msg, "tool_calls", None):
                    tool_calls = [
                        {"id": tc.get("id"), "name": tc.get("name"), "args": tc.get("args")}
                        for tc in msg.tool_calls
                    ]
                generations_log.append({
                    "text": text[:3000] + ("…" if len(text) > 3000 else ""),
                    "tool_calls": tool_calls,
                })

        _write({
            "type": "llm_output",
            "session_id": self.session_id,
            "turn": self._turn,
            "run_id": run_id,
            "duration_ms": elapsed_ms,
            "token_usage": token_usage,
            "generations": generations_log,
        })

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs) -> None:
        run_id = str(kwargs.get("run_id", ""))
        self._llm_t0.pop(run_id, None)
        _write({
            "type": "llm_error",
            "session_id": self.session_id,
            "turn": self._turn,
            "run_id": run_id,
            "error": str(error)[:2000],
        })

    # ── Tools ─────────────────────────────────────────────────────────────────

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs,
    ) -> None:
        run_id = str(kwargs.get("run_id", ""))
        tool_name = serialized.get("name", "") or kwargs.get("name", "unknown")
        self._tool_t0[run_id] = time.time()
        self._tool_name[run_id] = tool_name

        # input_str is JSON-encoded args
        try:
            args = json.loads(input_str)
        except Exception:
            args = input_str

        _write({
            "type": "tool_input",
            "session_id": self.session_id,
            "turn": self._turn,
            "run_id": run_id,
            "tool_name": tool_name,
            "args": args,
        })

    def on_tool_end(self, output: Any, **kwargs) -> None:
        run_id = str(kwargs.get("run_id", ""))
        elapsed_ms = round((time.time() - self._tool_t0.pop(run_id, time.time())) * 1000)
        tool_name = self._tool_name.pop(run_id, "")

        raw = str(output)
        _write({
            "type": "tool_output",
            "session_id": self.session_id,
            "turn": self._turn,
            "run_id": run_id,
            "tool_name": tool_name,
            "duration_ms": elapsed_ms,
            "output_length": len(raw),
            "output": raw[:4000] + ("…" if len(raw) > 4000 else ""),
        })

    def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        **kwargs,
    ) -> None:
        run_id = str(kwargs.get("run_id", ""))
        elapsed_ms = round((time.time() - self._tool_t0.pop(run_id, time.time())) * 1000)
        tool_name = self._tool_name.pop(run_id, "")
        _write({
            "type": "tool_error",
            "session_id": self.session_id,
            "turn": self._turn,
            "run_id": run_id,
            "tool_name": tool_name,
            "duration_ms": elapsed_ms,
            "error": str(error)[:2000],
        })
