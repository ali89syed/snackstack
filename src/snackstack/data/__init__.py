"""Menu catalog and order database, loaded from the bundled JSON files."""

import json
from importlib.resources import files

_DATA = files("snackstack.data")

MENU_ITEMS: list[dict] = json.loads(_DATA.joinpath("menu.json").read_text(encoding="utf-8"))
ORDERS: dict[str, dict] = json.loads(_DATA.joinpath("orders.json").read_text(encoding="utf-8"))
