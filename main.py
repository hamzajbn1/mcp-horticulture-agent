from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from agent import build_agent_graph
from config import configure_logging, get_settings

_LOGGER = logging.getLogger("horticulture.main")

# A test prompt to ensure the AI knows how to route to different tools
_DEMO_QUERY = (
    "I'm starting cold stratification for 40 blueberry seeds today, 2026-07-20. "
    "Log it. Also my soil pH reads 6.4 -- is that acceptable for blueberries?"
)

def _server_config() -> dict[str, dict[str, object]]:
    """
    Constructs the configuration needed to launch the background MCP tool server.
    It maps the command (e.g., 'python') to the actual script (e.g., 'mcp_server.py').
    """
    settings = get_settings()
    server_script = str(Path(__file__).resolve().parent / settings.mcp_server_script)
    return {
        "horticulture": {
            "command": settings.mcp_server_command,
            "args": [server_script],
            "transport": "stdio",
        }
    }

async def run_agent(query: str) -> str:
    """
    The main execution flow: boots up the tool server, grabs the tools, 
    builds the LangGraph AI brain, and feeds it the user's prompt.
    """
    settings = get_settings()
    client = MultiServerMCPClient(_server_config())

    try:
        tools = await client.get_tools()
    except Exception:
        _LOGGER.exception("Failed to retrieve tools from MCP server.")
        raise

    if not tools:
        msg = "MCP server returned no tools"
        raise RuntimeError(msg)

    _LOGGER.info("Fetched %d tool(s): %s", len(tools), [t.name for t in tools])

    # Hand the tools and settings over to LangGraph to build the AI loop
    graph = build_agent_graph(tools, settings)

    try:
        # Trigger the AI graph with the user's message and the emergency brake limit.
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=query)]},
            config={"recursion_limit": settings.recursion_limit},
        )
    except Exception:
        _LOGGER.exception("Agent execution failed.")
        raise

    # Extract just the final text answer from the AI's memory box.
    final_message = result["messages"][-1]
    content = final_message.content
    return content if isinstance(content, str) else str(content)

async def main() -> int:
    """
    The starting point of the application. 
    Sets up logging and fires off the demo query.
    """
    configure_logging()
    _LOGGER.info("Starting horticulture agent (model=%s)", get_settings().ollama_model)

    try:
      answer = await run_agent(_DEMO_QUERY)
    except Exception:
      _LOGGER.exception("Fatal error during agent run")
      return 1

    _LOGGER.info("Agent response:\n%s", answer)
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
