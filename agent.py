import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
import langchain_google_genai._function_utils as genai_utils

# Monkey patch to fix Gemini's strict schema validation that crashes on integer enums
# which are commonly provided by MCP tools (e.g. valid_addons, statuses, etc).
_orig_dict_to_genai_schema = genai_utils._dict_to_genai_schema
def _patched_dict_to_genai_schema(schema_dict, *args, **kwargs):
    if isinstance(schema_dict, dict) and "enum" in schema_dict:
        schema_dict["enum"] = [str(x) for x in schema_dict["enum"]]
        if "type" in schema_dict:
            schema_dict["type"] = "string"
    return _orig_dict_to_genai_schema(schema_dict, *args, **kwargs)

genai_utils._dict_to_genai_schema = _patched_dict_to_genai_schema


def create_food_aggregator_agent(mcp_tools: list):
    """
    Creates a LangGraph ReAct agent using ChatGroq and the dynamically loaded MCP tools.
    """
    # Using gemini model. The user mentioned gemini-3-flash-preview, which is available via the google-genai SDK.
    # We will use it with Langchain's ChatGoogleGenAI which wraps it.
    model_name = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
    
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.2,
        max_tokens=8192,
    )

    system_prompt = """You are an intelligent Food Aggregator Agent. 
You have access to MCP tools from Swiggy and Zomato.

Your goal:
1. When a user asks for food (e.g., 'I want Narmada Chicken Biriyani'), use the available tools to search both Swiggy and Zomato.
2. If exact matches fail, broaden your search to just the restaurant name first, then look at the menu.
3. Once you find the items, retrieve pricing information and apply any available coupons on both platforms.
4. Compare the final prices and recommend the cheapest/best option to the user.

CRITICAL INSTRUCTIONS ON TOOL USAGE:
- You must strictly adhere to the types required by the tools.
- If a tool requires a number (like `vegFilter`, `sortType`, or price limits), you MUST pass it as an actual JSON integer/number (e.g., `0`), NOT as a string (e.g., `"0"`).
- If a tool argument is optional and you don't need it, omit it entirely do not pass "null" as a string.
- You must get the `addressId` by calling the `get_addresses` tool FIRST before calling any search functions.

ALWAYS explain your reasoning. You MUST check BOTH platforms before making a recommendation.
"""

    # We use MemorySaver to persist conversation state across multiple chat turns.
    memory = MemorySaver()

    # create_react_agent handles the tool binding and the ReAct loop automatically graph compilation.
    agent_executor = create_react_agent(
        model=llm, 
        tools=mcp_tools,
        prompt=system_prompt,
        checkpointer=memory
    )
    
    return agent_executor
