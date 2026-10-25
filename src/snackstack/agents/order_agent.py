"""Order agent: ask for an ID if the message has none (HITL), then get_order_status."""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command, interrupt

from snackstack.agents._loop import run_tool_loop
from snackstack.agents.prompts import ORDER_AGENT_PROMPT
from snackstack.logger import setup_logger
from snackstack.state import StackState
from snackstack.tools.order_tools import get_order_status, has_identifier

logger = setup_logger(__name__)

ASK_FOR_ID = "I can help track your order! Please share your order ID (e.g. ORD-201), tracking ID (e.g. SS201TRK), or email."


def order_agent_node(state: StackState) -> Command[Literal["synthesizer_node"]]:
    query = state["user_query"]

    # Ask before any LLM call: on resume, this node re-runs from the top.
    if not has_identifier(query):
        reply = interrupt(ASK_FOR_ID)
        query = f"{query}\nUser's reply: {reply}"
        logger.info("Resumed with reply %r", reply)

    # The LLM reads the message and passes the ID to get_order_status itself.
    answer = run_tool_loop(
        [SystemMessage(ORDER_AGENT_PROMPT), HumanMessage(query)],
        [get_order_status],
        agent_name="order_agent",
    )
    return Command(goto="synthesizer_node", update={"order_response": answer})
