from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Annotated, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, BaseMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from config import Settings

_LOGGER = logging.getLogger(__name__)

# The hidden master prompt that dictates exactly how the AI should behave.
_SYSTEM_PROMPT = (
    "You are an indoor-horticulture assistant. Use the available tools to log "
    "cold-stratification timelines, validate ericaceous soil pH, and track "
    "transplant-shock recovery. Prefer calling a tool over guessing. When a tool "
    "returns structured data, summarise it clearly for the grower."
)

class AgentState(TypedDict):
    """
    The memory box of the agent. 
    'add_messages' ensures that new messages are appended to the list, 
    creating a continuous chat history rather than overwriting it.
    """
    messages: Annotated[list[AnyMessage], add_messages]

def build_llm(settings: Settings) -> BaseChatModel:
    """
    Initializes the connection to the local Ollama daemon using parameters
    defined in the centralized config.py (Settings).
    """
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.ollama_temperature,
        num_ctx=settings.ollama_num_ctx,
    )

def build_agent_graph(    
    tools: Sequence[BaseTool], settings: Settings
) -> CompiledStateGraph[AgentState, AgentState, AgentState]:
    """
    Constructs the LangGraph state machine. This defines the workflow loop 
    where the AI alternates between thinking (reason) and acting (tools).
    """
    if not tools:
        msg="No tools available to bind; check the MCP server connection."
        raise ValueError(msg)

    model = build_llm(settings).bind_tools(tools)

    async def reason(state: AgentState) -> dict[str, list[BaseMessage]]:
        """
        The Decision Engine. 
        It reads the system prompt, stacks the current chat history, 
        and decides whether to answer the user or trigger a tool.
        """
        messages: list[BaseMessage] = [SystemMessage(content=_SYSTEM_PROMPT)]
        messages.extend(state["messages"])
        response = await  model.ainvoke(messages)
        return {"messages": [response]}

    # --- Graph Construction ---
    builder = StateGraph(AgentState)

    builder.add_node("reason", reason)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "reason")

    builder.add_conditional_edges("reason", tools_condition)
    builder.add_edge("tools", "reason")

    compiled = builder.compile()
    _LOGGER.info("Compiled agent graph with %d tool(s)", len(tools))
    return compiled