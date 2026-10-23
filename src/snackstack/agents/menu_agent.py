"""Menu agent: RAG over the menu via search_menu_catalog."""

from typing import Literal

from langchain_core.messages import SystemMessage
from langgraph.types import Command

from snackstack.agents._loop import run_tool_loop
from snackstack.agents.prompts import MENU_AGENT_PROMPT
from snackstack.state import StackState
from snackstack.tools.menu_tools import search_menu_catalog

HISTORY_WINDOW = 6


def menu_agent_node(state: StackState) -> Command[Literal["synthesizer_node"]]:
    history = state.get("messages", [])[-HISTORY_WINDOW:]
    answer = run_tool_loop(
        [SystemMessage(MENU_AGENT_PROMPT), *history],
        [search_menu_catalog],
        agent_name="menu_agent",
    )
    return Command(goto="synthesizer_node", update={"menu_response": answer})
