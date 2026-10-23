"""get_order_status: look up an order by order ID, tracking ID, or email."""

import re

from langchain_core.tools import tool

from snackstack.data import ORDERS

# Separators are optional so transcribed speech ("ORD 201", "SS 201 TRK") still matches.
ORDER_ID_RE = re.compile(r"\bORD[-\s]?(\d+)\b", re.IGNORECASE)
TRACKING_ID_RE = re.compile(r"\bSS[-\s]?(\d+)[-\s]?TRK\b", re.IGNORECASE)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]*\w")


def extract_identifier(text: str) -> str | None:
    """Return the first order ID, tracking ID, or email found in text, in canonical form."""
    if match := ORDER_ID_RE.search(text):
        return f"ORD-{match.group(1)}"
    if match := TRACKING_ID_RE.search(text):
        return f"SS{match.group(1)}TRK"
    if match := EMAIL_RE.search(text):
        return match.group(0).lower()
    return None


def _find_order(identifier: str) -> dict | None:
    ident = identifier.strip()
    if order := ORDERS.get(ident.upper()):
        return order
    for order in ORDERS.values():
        if order["tracking_id"].upper() == ident.upper():
            return order
        if order["customer_email"].lower() == ident.lower():
            return order
    return None


@tool
def get_order_status(identifier: str) -> str:
    """Look up a SnackStack order and return its status.

    identifier: an order ID (e.g. ORD-201), a tracking ID (e.g. SS201TRK),
    or the customer's email address.
    """
    order = _find_order(identifier)
    if order is None:
        return f"No order found for '{identifier}'."
    return (
        f"Order {order['order_id']} (tracking {order['tracking_id']})\n"
        f"Customer: {order['customer_name']} <{order['customer_email']}>\n"
        f"Item: {order['item_name']} (Rs {order['price']})\n"
        f"Status: {order['status']}\n"
        f"Ordered: {order['order_date']} | Estimated delivery: {order['estimated_delivery']}"
    )
