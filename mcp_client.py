import os
from contextlib import AsyncExitStack
from mcp import StdioServerParameters, ClientSession
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

class MCPServerManager:
    """Manages connections to multiple local MCP servers."""
    def __init__(self):
        self.exit_stack = AsyncExitStack()
        self.sessions = {}
        self.all_tools = []

    async def connect_server(self, server_name: str, command: str, args: list[str], env: dict = None):
        """Connects to a single MCP server and loads its tools."""
        server_env = os.environ.copy() 
        if env:
            server_env.update(env)

        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=server_env
        )
        
        # Enter stdio client context
        read, write = await self.exit_stack.enter_async_context(stdio_client(server_params))
        
        # Enter session context
        session = await self.exit_stack.enter_async_context(ClientSession(read, write))
        
        await session.initialize()
        
        self.sessions[server_name] = session
        
        # Load tools for LangChain
        tools = await load_mcp_tools(session)
        
        # Token optimization: trim verbose MCP tool descriptions to save ~7k tokens/call.
        max_desc_len = int(os.getenv("MAX_TOOL_DESC_LENGTH", "200"))
        for tool in tools:
            if hasattr(tool, 'description') and tool.description and len(tool.description) > max_desc_len:
                tool.description = tool.description[:max_desc_len].rstrip() + "..."
        
        self.all_tools.extend(tools)
        
        print(f"[{server_name}] Connected and loaded {len(tools)} tools:")
        for t in tools:
            print(f"  - {getattr(t, 'name', 'Unknown Tool')}: {getattr(t, 'description', '')[:80]}")
        
    async def cleanup(self):
        """Close all connections."""
        await self.exit_stack.aclose()
