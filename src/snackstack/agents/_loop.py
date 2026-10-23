"""Shared manual tool-calling loop used by the specialist agents."""

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool

from snackstack.config import MAX_TOOL_ITERATIONS, llm
from snackstack.logger import setup_logger

logger = setup_logger(__name__)

FALLBACK_ANSWER = "Sorry, I couldn't finish looking that up. Could you rephrase your question?"


def run_tool_loop(messages: list[BaseMessage], tools: list[BaseTool], agent_name: str) -> str:
    """Let the LLM call tools until it answers in text, capped at MAX_TOOL_ITERATIONS."""
    llm_with_tools = llm.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}
    messages = list(messages)

    for iteration in range(1, MAX_TOOL_ITERATIONS + 1):
        response: AIMessage = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return response.content

        for call in response.tool_calls:
            logger.info("[%s] iter %d -> %s(%s)", agent_name, iteration, call["name"], call["args"])
            tool = tools_by_name.get(call["name"])
            if tool is None:
                result = f"Unknown tool '{call['name']}'."
            else:
                try:
                    result = tool.invoke(call["args"])
                except Exception as exc:  # surface tool errors to the LLM instead of crashing
                    result = f"Tool error: {exc}"
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    logger.warning("[%s] hit MAX_TOOL_ITERATIONS=%d, using fallback", agent_name, MAX_TOOL_ITERATIONS)
    return FALLBACK_ANSWER
