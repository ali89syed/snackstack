"""Shared graph state."""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class StackState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    user_query: str
    route: list[str]
    # Parallel branches write different keys, so they never conflict.
    menu_response: str | None
    order_response: str | None
    final_answer: str | None
