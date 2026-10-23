"""Orchestrator: structured-output routing, then Send() fan-out to the agents."""

from typing import Literal

from langchain_core.messages import SystemMessage
from langgraph.types import Command, Send
from pydantic import BaseModel, Field

from snackstack.agents.prompts import ORCHESTRATOR_PROMPT
from snackstack.config import llm
from snackstack.logger import setup_logger
from snackstack.state import StackState

logger = setup_logger(__name__)

AgentName = Literal["menu_agent", "order_agent"]
HISTORY_WINDOW = 6


class RouteDecision(BaseModel):
    agents: list[AgentName] = Field(description="Agent(s) that should handle the message.")
    reasoning: str = Field(description="One short sentence explaining the choice.")


router_llm = llm.with_structured_output(RouteDecision)


def orchestrator_node(state: StackState) -> Command[AgentName]:
    query = state["user_query"]
    history = state.get("messages", [])[-HISTORY_WINDOW:]

    decision: RouteDecision = router_llm.invoke([SystemMessage(ORCHESTRATOR_PROMPT), *history])
    agents = list(dict.fromkeys(decision.agents)) or ["menu_agent"]
    logger.info("Route %s: %s", agents, decision.reasoning)

    payload = {"user_query": query, "messages": state.get("messages", [])}
    return Command(
        # Clear last turn's outputs so they don't leak into this turn's synthesis.
        update={
            "route": agents,
            "menu_response": None,
            "order_response": None,
            "final_answer": None,
        },
        goto=[Send(agent, payload) for agent in agents],
    )
