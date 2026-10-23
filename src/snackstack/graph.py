"""StateGraph assembly: START -> orchestrator -> (Send) agents -> synthesizer -> END."""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from snackstack.agents import menu_agent_node, orchestrator_node, order_agent_node, synthesizer_node
from snackstack.state import StackState


def build_graph():
    builder = StateGraph(StackState)
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("menu_agent", menu_agent_node)
    builder.add_node("order_agent", order_agent_node)
    builder.add_node("synthesizer_node", synthesizer_node)

    # Orchestrator -> agents and agents -> synthesizer are routed with Command/Send.
    builder.add_edge(START, "orchestrator")
    builder.add_edge("synthesizer_node", END)

    # interrupt() needs a checkpointer; it also gives multi-turn memory per thread_id.
    return builder.compile(checkpointer=MemorySaver())


graph = build_graph()
