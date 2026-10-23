"""Synthesizer: merges one or both agent outputs into the final answer."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from snackstack.agents.prompts import SYNTHESIZER_PROMPT
from snackstack.config import llm
from snackstack.state import StackState


def synthesizer_node(state: StackState) -> dict:
    parts = []
    if state.get("order_response"):
        parts.append(f"[Order agent]\n{state['order_response']}")
    if state.get("menu_response"):
        parts.append(f"[Menu agent]\n{state['menu_response']}")
    if not parts:
        answer = "Sorry, I couldn't come up with an answer. Could you try again?"
    else:
        prompt = f"User message: {state['user_query']}\n\n" + "\n\n".join(parts)
        answer = llm.invoke([SystemMessage(SYNTHESIZER_PROMPT), HumanMessage(prompt)]).content
    return {"final_answer": answer, "messages": [AIMessage(answer)]}
