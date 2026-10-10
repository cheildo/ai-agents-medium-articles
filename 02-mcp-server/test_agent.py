"""Runs the agents against a fake OpenAI Responses API and the real shop MCP server (no API key needed)."""
import builtins
import json
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")  # the clients are created at import time

import agents
import httpx2
import openai
import pytest

from mcp import Client

import agent
import runner_agent
import shop_server

requests_seen, script = [], []


def response(output, status="completed", reason=None):
    return {"id": f"resp_{len(requests_seen)}", "object": "response", "created_at": 0, "model": "gpt-6.1-sol",
            "status": status, "incomplete_details": {"reason": reason} if reason else None,
            "output": output, "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
            "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20,
                      "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}}


def call(call_id, name, arguments):
    return {"type": "function_call", "id": f"fc_{call_id}", "call_id": call_id, "name": name,
            "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments), "status": "completed"}


def text(t):
    return {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": t, "annotations": []}]}


def handler(request):
    assert request.url.path.endswith("/responses"), request.url
    requests_seen.append(json.loads(request.content))
    return httpx2.Response(200, json=script.pop(0))


fake_llm = openai.AsyncOpenAI(
    api_key="test", max_retries=0,
    http_client=openai.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
)
agent.llm = fake_llm
agents.set_default_openai_client(fake_llm, use_for_tracing=False)
agents.set_tracing_disabled(True)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def shop():
    requests_seen.clear()
    script.clear()
    shop_server.ORDERS["ORD-10043"]["status"] = "processing"
    async with Client(shop_server.mcp) as c:
        yield c


async def load_tools(shop):
    return {tool.name: tool for tool in (await shop.list_tools()).tools}


def outputs(request):
    return [item for item in request["input"] if isinstance(item, dict) and item.get("type") == "function_call_output"]


@pytest.mark.anyio
async def test_parallel_read_only_calls_need_no_approval(shop, monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda prompt: pytest.fail("should not ask"))
    script.extend([
        response([call("c1", "get_order_status", {"order_id": "ORD-10042"}),
                  call("c2", "check_stock", {"product_name": "usb-c hub"})]),
        response([text("Shipped with DHL. The hub is sold out.")]),
    ])
    answer = await agent.run_agent(shop, await load_tools(shop), "instructions", "Where is my order?")
    assert answer == "Shipped with DHL. The hub is sold out."
    first, second = requests_seen
    assert first["model"] == "gpt-6.1-sol" and first["instructions"] == "instructions"
    sent = {t["name"]: t for t in first["tools"]}
    assert sent["cancel_order"]["type"] == "function" and sent["cancel_order"]["strict"] is False
    assert sent["cancel_order"]["parameters"]["properties"]["reason"]["enum"][0] == "customer_request"
    results = outputs(second)
    assert [r["call_id"] for r in results] == ["c1", "c2"]
    assert '"carrier": "DHL"' in results[0]["output"] and not results[0]["output"].startswith("Error")
    # the model's own function_call items were sent back too
    assert [i.get("call_id") for i in second["input"] if i.get("type") == "function_call"] == ["c1", "c2"]


@pytest.mark.anyio
@pytest.mark.parametrize("typed, order_id, expected, status", [
    ("n", "ORD-10043", "declined", "processing"),
    ("y", "ORD-10043", '"status": "cancelled"', "cancelled"),
    ("y", "ORD-10042", "can no longer be cancelled", None),
])
async def test_cancel_asks_a_human(shop, monkeypatch, typed, order_id, expected, status):
    asked = []
    monkeypatch.setattr(builtins, "input", lambda prompt: asked.append(prompt) or typed)
    script.extend([
        response([call("c3", "cancel_order", {"order_id": order_id, "reason": "customer_request"})]),
        response([text("Done.")]),
    ])
    await agent.run_agent(shop, await load_tools(shop), "instructions", f"Cancel {order_id}")
    result = outputs(requests_seen[1])[0]["output"]
    assert len(asked) == 1 and expected in result
    if status:
        assert shop_server.ORDERS[order_id]["status"] == status


@pytest.mark.anyio
async def test_errors_reach_the_model_as_text(shop):
    script.extend([
        response([call("c4", "get_order_status", {"order_id": "A12"}),
                  call("c5", "delete_everything", {}),
                  call("c6", "check_stock", "{not json")]),
        response([text("Could you check the order ID?")]),
    ])
    await agent.run_agent(shop, await load_tools(shop), "instructions", "Where is A12?")
    bad_id, unknown, bad_json = (r["output"] for r in outputs(requests_seen[1]))
    assert bad_id.startswith("Error:") and "Order IDs look like" in bad_id
    assert unknown == "Error: unknown tool 'delete_everything'."
    assert bad_json.startswith("Error: the arguments were not a valid JSON object")


@pytest.mark.anyio
async def test_stops_early_and_at_the_step_limit(shop):
    script.append(response([], status="incomplete", reason="max_output_tokens"))
    assert "max_output_tokens" in await agent.run_agent(shop, await load_tools(shop), "i", "stock?")
    script.extend(response([call(f"s{i}", "check_stock", {"product_name": "hdmi cable"})]) for i in range(agent.MAX_STEPS))
    assert "step limit" in await agent.run_agent(shop, await load_tools(shop), "i", "loop")


@pytest.mark.anyio
async def test_main_connects_over_stdio_and_loads_the_policy(monkeypatch, capsys):
    monkeypatch.setattr(builtins, "input", lambda prompt: "quit")
    await agent.main()
    assert "Connected to the shop server. Tools: get_order_status, check_stock, cancel_order" in capsys.readouterr().out


@pytest.mark.anyio
async def test_agents_sdk_version_gets_only_read_only_tools(shop):
    script.extend([
        response([call("r1", "check_stock", {"product_name": "usb-c hub"})]),
        response([text("Sold out.")]),
    ])
    assert await runner_agent.ask("Is the hub in stock?") == "Sold out."
    first, second = requests_seen
    assert first["model"] == "gpt-6.1-sol"
    assert sorted(t["name"] for t in first["tools"]) == ["check_stock", "get_order_status"]
    assert '"units_in_stock": 0' in outputs(second)[0]["output"][0]["text"]


@pytest.mark.anyio
async def test_failed_response_and_protocol_errors(shop, monkeypatch):
    script.append(response([], status="failed"))
    assert "reason: failed" in await agent.run_agent(shop, await load_tools(shop), "i", "stock?")

    async def broken_call_tool(name, arguments):
        raise agent.MCPError(-32603, "server went away")
    monkeypatch.setattr(shop, "call_tool", broken_call_tool)
    tools = await load_tools(shop)
    output = await agent.run_tool(shop, tools["check_stock"], {"product_name": "hdmi cable"})
    assert output.startswith("Error: the tool call failed")
