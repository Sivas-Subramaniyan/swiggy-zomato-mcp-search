import os
import asyncio
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.sse import sse_client
from langchain_mcp_adapters.tools import load_mcp_tools


class MCPServerManager:
    """
    Manages persistent HTTP connections to remote MCP servers.

    Each server connection runs in its own asyncio.Task so that anyio
    cancel scopes (used internally by streamablehttp_client) are always
    entered and exited within the same task — avoiding the
    'cancel scope in a different task' RuntimeError.
    """

    def __init__(self):
        self._server_tools: dict[str, list] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._stop_events: dict[str, asyncio.Event] = {}
        self._ready_events: dict[str, asyncio.Event] = {}
        self._errors: dict[str, str] = {}
        self.sessions: dict = {}

    @property
    def all_tools(self):
        tools = []
        for t in self._server_tools.values():
            tools.extend(t)
        return tools

    def is_connected(self, server_name: str) -> bool:
        task = self._tasks.get(server_name)
        return task is not None and not task.done()

    async def connect_server_http(self, server_name: str, url: str, headers: dict = None):
        """
        Start a background task that opens and holds the HTTP MCP connection.
        Blocks until tools are loaded (or an error occurs).
        """
        # Disconnect existing connection for this server
        await self._disconnect(server_name)

        ready = asyncio.Event()
        stop = asyncio.Event()
        self._ready_events[server_name] = ready
        self._stop_events[server_name] = stop
        self._errors[server_name] = None

        async def _run():
            try:
                try:
                    # Try streamable HTTP (newer MCP protocol)
                    async with streamablehttp_client(url, headers=headers or {}, timeout=30) as (read, write, _):
                        await self._init_session(server_name, read, write, ready, stop)
                except Exception as e1:
                    if ready.is_set():
                        raise  # error after init — propagate
                    # Fall back to SSE transport
                    try:
                        async with sse_client(url, headers=headers or {}) as (read, write):
                            await self._init_session(server_name, read, write, ready, stop)
                    except Exception as e2:
                        self._errors[server_name] = str(e2)
                        ready.set()
            except Exception as e:
                self._errors[server_name] = str(e)
                ready.set()
            finally:
                self._server_tools.pop(server_name, None)
                self.sessions.pop(server_name, None)

        task = asyncio.create_task(_run())
        self._tasks[server_name] = task

        # Wait for session to be ready (tools loaded) or for failure
        try:
            await asyncio.wait_for(ready.wait(), timeout=40)
        except asyncio.TimeoutError:
            stop.set()
            raise Exception(f"Connection to {server_name} timed out after 40 s")

        if self._errors.get(server_name):
            raise Exception(self._errors[server_name])

        print(f"[{server_name}] Connected — {len(self._server_tools.get(server_name, []))} tools")

    async def _init_session(self, server_name, read, write, ready: asyncio.Event, stop: asyncio.Event):
        """Initialise the MCP session, load tools, then hold until stop is set."""
        async with ClientSession(read, write) as session:
            await session.initialize()
            self.sessions[server_name] = session

            tools = await load_mcp_tools(session)
            max_len = int(os.getenv("MAX_TOOL_DESC_LENGTH", "200"))
            for tool in tools:
                if hasattr(tool, "description") and tool.description and len(tool.description) > max_len:
                    tool.description = tool.description[:max_len].rstrip() + "..."

            self._server_tools[server_name] = tools
            ready.set()  # Signal: tools are ready

            # Keep the connection alive until disconnect() is called
            await stop.wait()

    async def _disconnect(self, server_name: str):
        """Stop the background task for a server if it exists."""
        if server_name in self._stop_events:
            self._stop_events[server_name].set()
        task = self._tasks.get(server_name)
        if task and not task.done():
            try:
                await asyncio.wait_for(task, timeout=5)
            except (asyncio.TimeoutError, Exception):
                task.cancel()
        self._tasks.pop(server_name, None)
        self._stop_events.pop(server_name, None)
        self._ready_events.pop(server_name, None)

    async def cleanup(self):
        """Disconnect all servers."""
        for name in list(self._tasks.keys()):
            await self._disconnect(name)
        self._server_tools.clear()
        self.sessions.clear()
