import sys

import pytest

from mcp import Client, StdioServerParameters

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
async def test_lists_three_tools(client):
    result = await client.list_tools()
    tools = {tool.name: tool for tool in result.tools}
    assert set(tools) == {"get_order_status", "check_stock", "cancel_order"}
    assert tools["get_order_status"].annotations.read_only_hint is True
    assert tools["cancel_order"].annotations.destructive_hint is True
    reason = tools["cancel_order"].input_schema["properties"]["reason"]
    assert reason["enum"] == ["customer_request", "duplicate_order", "wrong_item"]


@pytest.mark.anyio
async def test_order_status(client):
    result = await client.call_tool("get_order_status", {"order_id": "ord-10042"})
    assert result.is_error is False
    assert '"tracking_number": "JD014600003SE"' in result.content[0].text


@pytest.mark.anyio
async def test_unknown_order_is_a_readable_error(client):
    result = await client.call_tool("get_order_status", {"order_id": "A12"})
    assert result.is_error is True
    assert "Order IDs look like 'ORD-10042'" in result.content[0].text


@pytest.mark.anyio
async def test_invalid_reason_is_rejected_before_the_tool_runs(client):
    result = await client.call_tool("cancel_order", {"order_id": "ORD-10043", "reason": "bored"})
    assert result.is_error is True
    assert shop_server.ORDERS["ORD-10043"]["status"] == "processing"


@pytest.mark.anyio
async def test_cancel_once_then_already_cancelled(client):
    first = await client.call_tool("cancel_order", {"order_id": "ORD-10043", "reason": "customer_request"})
    assert first.is_error is False
    second = await client.call_tool("cancel_order", {"order_id": "ORD-10043", "reason": "customer_request"})
    assert second.is_error is True and "already cancelled" in second.content[0].text


@pytest.mark.anyio
async def test_shipped_order_cannot_be_cancelled(client):
    result = await client.call_tool("cancel_order", {"order_id": "ORD-10042", "reason": "wrong_item"})
    assert result.is_error is True and "Offer the customer a return" in result.content[0].text


@pytest.mark.anyio
async def test_resource_and_prompt(client):
    policy = await client.read_resource("shop://policies/returns")
    assert "30 days" in policy.contents[0].text
    prompt = await client.get_prompt("support_reply", {"customer_message": "Where is ORD-10042?"})
    assert prompt.messages[0].role == "user"
    assert "Where is ORD-10042?" in prompt.messages[0].content.text


@pytest.mark.anyio
async def test_works_over_stdio():
    # Start the real server as a subprocess, like Claude Desktop or Cursor would.
    server = StdioServerParameters(command=sys.executable, args=["shop_server.py"])
    async with Client(server) as c:
        result = await c.list_tools()
        assert len(result.tools) == 3
        stock = await c.call_tool("check_stock", {"product_name": "USB-C Hub"})
        assert '"units_in_stock": 0' in stock.content[0].text
