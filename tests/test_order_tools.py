import pytest

from snackstack.tools.order_tools import get_order_status, has_identifier


@pytest.mark.parametrize("text, expected", [
    ("Track order ORD-201", True),
    ("track ord-203 please", True),
    ("tracking SS204TRK", True),
    ("Check order by email priya@example.com", True),
    ("Where is my order?", False),
])
def test_has_identifier(text, expected):
    assert has_identifier(text) == expected


@pytest.mark.parametrize("identifier, expected", [
    ("ORD-201", "Out for Delivery"),
    ("ord-201", "Out for Delivery"),
    ("SS203TRK", "Preparing"),
    ("Priya@Example.com", "ORD-201"),
    ("ORD-999", "No order found"),
])
def test_get_order_status(identifier, expected):
    assert expected in get_order_status.invoke({"identifier": identifier})
