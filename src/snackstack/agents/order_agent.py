"""Order agent: regex identifier extraction, HITL interrupt, then get_order_status."""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command, interrupt

from snackstack.agents._loop import run_tool_loop
from snackstack.agents.prompts import ORDER_AGENT_PROMPT
from snackstack.logger import setup_logger
from snackstack.state import StackState
from snackstack.tools.order_tools import extract_identifier, get_order_status

logger = setup_logger(__name__)

ASK_FOR_ID = "I can help track your order! Please share your order ID (e.g. ORD-201), tracking ID (e.g. SS201TRK), or email."


def order_agent_node(state: StackState) -> Command[Literal["synthesizer_node"]]:
    query = state["user_query"]

    # The interrupt must come before any LLM call: on resume this node re-runs from the top.
    identifier = extract_identifier(query)
    if identifier is None:
        reply = interrupt(ASK_FOR_ID)
        identifier = extract_identifier(str(reply)) or str(reply).strip()
        logger.info("Resumed with identifier %r", identifier)

    answer = run_tool_loop(
        [
            SystemMessage(ORDER_AGENT_PROMPT),
            HumanMessage(f"User message: {query}\nOrder identifier: {identifier}"),
        ],
        [get_order_status],
        agent_name="order_agent",
    )
    return Command(goto="synthesizer_node", update={"order_response": answer})
