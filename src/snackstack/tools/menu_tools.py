"""search_menu_catalog: semantic search over the menu."""

from langchain_core.tools import tool

from snackstack.tools.rag import retriever


@tool
def search_menu_catalog(query: str) -> str:
    """Search the SnackStack menu for dishes matching a natural-language query.

    Use for questions about dishes, cuisines, dietary needs (vegan, vegetarian,
    gluten-free), prices, or ratings. Returns the closest matching dishes.
    """
    docs = retriever.invoke(query)
    if not docs:
        return "No matching dishes found."
    lines = []
    for doc in docs:
        m = doc.metadata
        status = "available" if m["available"] else "unavailable"
        lines.append(
            f"- {m['name']} ({m['cuisine']} {m['category']}) | Rs {m['price']} | "
            f"rating {m['rating']} | dietary: {m['dietary_tags']} | {status}\n"
            f"  {doc.page_content}"
        )
    return "\n".join(lines)
