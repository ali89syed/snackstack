import pytest

from snackstack.tools.order_tools import extract_identifier, get_order_status


@pytest.mark.parametrize("text, expected", [
    ("Track order ORD-201", "ORD-201"),
    ("track ord-203 please", "ORD-203"),
    ("Where is ORD 205?", "ORD-205"),             # STT often drops the hyphen
    ("tracking SS204TRK", "SS204TRK"),
    ("tracking SS 204 TRK", "SS204TRK"),
    ("Check order by email Priya@Example.com.", "priya@example.com"),
    ("Where is my order?", None),
])
def test_extract_identifier(text, expected):
    assert extract_identifier(text) == expected


@pytest.mark.parametrize("identifier, expected", [
    ("ORD-201", "Out for Delivery"),
    ("SS203TRK", "Preparing"),
    ("priya@example.com", "ORD-201"),
    ("ORD-999", "No order found"),
])
def test_get_order_status(identifier, expected):
    assert expected in get_order_status.invoke({"identifier": identifier})
