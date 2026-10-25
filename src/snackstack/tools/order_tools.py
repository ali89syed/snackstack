"""get_order_status: look up an order by order ID, tracking ID, or email."""

from langchain_core.tools import tool

from snackstack.data import ORDERS


def has_identifier(text: str) -> bool:
    """True if the message looks like it contains an order ID, tracking ID, or email."""
    text = text.upper()
    return "ORD-" in text or "TRK" in text or "@" in text


@tool
def get_order_status(identifier: str) -> str:
    """Look up an order by order ID (ORD-201), tracking ID (SS201TRK), or email."""
    wanted = identifier.strip().lower()

    for order in ORDERS.values():
        if wanted in (order["order_id"].lower(),
                      order["tracking_id"].lower(),
                      order["customer_email"].lower()):
            return (f"Order {order['order_id']}: {order['item_name']} (Rs {order['price']}) "
                    f"is {order['status']}. Estimated delivery: {order['estimated_delivery']}.")

    return f"No order found for '{identifier}'."
