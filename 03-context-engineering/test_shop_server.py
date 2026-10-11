import json

import pytest

from mcp import Client

import shop_server
from shop_server import mcp


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    # In memory: no subprocess, no port. Reset the fake data for each test.
    shop_server.ORDERS["ORD-10043"]["status"] = "processing"
    async with Client(mcp) as c:
        yield c


@pytest.mark.anyio
async def test_lists_five_tools(client):
    tools = {tool.name: tool for tool in (await client.list_tools()).tools}
    assert set(tools) == {"get_order_status", "search_products", "check_stock", "get_policy", "cancel_order"}
    assert all(tools[name].annotations.read_only_hint for name in set(tools) - {"cancel_order"})
    assert tools["cancel_order"].annotations.destructive_hint is True
    assert tools["get_policy"].input_schema["properties"]["topic"]["enum"] == ["returns", "shipping", "warranty"]


@pytest.mark.anyio
async def test_search_returns_only_the_fields_the_agent_needs(client):
    result = await client.call_tool("search_products", {"query": "USB-C charger"})
    data = json.loads(result.content[0].text)
    assert data["total_matches"] == 3 and len(data["products"]) == 3
    assert set(data["products"][0]) == {"name", "price_usd", "units_in_stock"}
    # The full catalog records of the same products are about 5 times bigger
    names = {p["name"] for p in data["products"]}
    full = json.dumps({"total_matches": 3, "products": [p for p in shop_server.CATALOG if p["name"] in names]}, indent=2)
    assert len(full) > 4 * len(result.content[0].text)


@pytest.mark.anyio
async def test_search_limit_is_validated(client):
    result = await client.call_tool("search_products", {"query": "charger", "limit": 10})
    assert len(json.loads(result.content[0].text)["products"]) == 7
    too_many = await client.call_tool("search_products", {"query": "charger", "limit": 500})
    assert too_many.is_error is True


@pytest.mark.anyio
async def test_get_policy(client):
    result = await client.call_tool("get_policy", {"topic": "warranty"})
    assert result.content[0].text.startswith("# Warranty policy")
    bad = await client.call_tool("get_policy", {"topic": "everything"})
    assert bad.is_error is True


@pytest.mark.anyio
async def test_order_status_and_unknown_order(client):
    result = await client.call_tool("get_order_status", {"order_id": "ord-10042"})
    assert '"tracking_number": "JD014600003SE"' in result.content[0].text
    missing = await client.call_tool("get_order_status", {"order_id": "A12"})
    assert missing.is_error is True and "Order IDs look like 'ORD-10042'" in missing.content[0].text


@pytest.mark.anyio
async def test_cancel_once_then_already_cancelled(client):
    first = await client.call_tool("cancel_order", {"order_id": "ORD-10043", "reason": "customer_request"})
    assert first.is_error is False
    second = await client.call_tool("cancel_order", {"order_id": "ORD-10043", "reason": "customer_request"})
    assert second.is_error is True and "already cancelled" in second.content[0].text
