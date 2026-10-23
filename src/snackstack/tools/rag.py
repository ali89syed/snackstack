"""In-memory ChromaDB vector store over the menu catalog. Rebuilt on every start."""

from langchain_chroma import Chroma
from langchain_core.documents import Document

from snackstack.config import embeddings
from snackstack.data import MENU_ITEMS
from snackstack.logger import setup_logger

logger = setup_logger(__name__)


def _dish_to_document(dish: dict) -> Document:
    tags = ", ".join(dish["dietary_tags"]) or "none"
    # Cuisine first: it noticeably improves ranking for queries like "Indian food under 300".
    text = (
        f"{dish['cuisine']} cuisine: {dish['name']} ({dish['category']}). "
        f"{dish['description']}. Dietary: {tags}. "
        f"Price: Rs {dish['price']}. Rating: {dish['rating']}."
    )
    metadata = {
        "id": dish["id"],
        "name": dish["name"],
        "category": dish["category"],
        "cuisine": dish["cuisine"],
        "price": dish["price"],
        "rating": dish["rating"],
        "dietary_tags": tags,
        "available": dish["available"],
    }
    return Document(page_content=text, metadata=metadata, id=dish["id"])


def build_vector_store() -> Chroma:
    docs = [_dish_to_document(d) for d in MENU_ITEMS]
    store = Chroma(collection_name="snackstack_menu", embedding_function=embeddings)
    store.add_documents(docs, ids=[d.id for d in docs])
    logger.info("Built menu vector store with %d dishes", len(docs))
    return store


vector_store = build_vector_store()
retriever = vector_store.as_retriever(search_kwargs={"k": 6})
