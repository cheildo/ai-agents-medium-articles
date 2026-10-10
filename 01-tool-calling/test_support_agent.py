"""Runs support_agent.py against a fake Messages API (no network, no API key).

The fake server checks the exact JSON the SDK sends, so this also verifies that
the request shapes in the article (tools, tool_result blocks, history) are valid.
"""
import builtins
import json

import anthropic
import httpx2

import support_agent as agent

USAGE = {"input_tokens": 10, "output_tokens": 10}
requests_seen = []
script = []


def msg(content, stop_reason):
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": USAGE,
    }


def handler(request):
    body = json.loads(request.content)
    requests_seen.append(body)
    return httpx2.Response(200, json=script.pop(0))


agent.client = anthropic.Anthropic(
    api_key="test",
    max_retries=0,
    http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
)


def reset(*responses):
    requests_seen.clear()
    script.clear()
    script.extend(responses)


def test_support_agent():
    # 1. Two tools in parallel, then a final answer
    reset(
        msg(
            [
                {"type": "thinking", "thinking": "", "signature": "sig"},
                {"type": "text", "text": "Let me check both."},
                {"type": "tool_use", "id": "toolu_1", "name": "get_order_status", "input": {"order_id": "ORD-10042"}},
                {"type": "tool_use", "id": "toolu_2", "name": "check_stock", "input": {"product_name": "USB-C hub"}},
            ],
            "tool_use",
        ),
        msg([{"type": "text", "text": "Your order shipped with DHL. The hub is sold out."}], "end_turn"),
    )
    answer = agent.run_agent("Where is ORD-10042 and is the USB-C hub in stock?")
    assert answer == "Your order shipped with DHL. The hub is sold out.", answer
    first, second = requests_seen
    assert first["model"] == "claude-opus-5-5" and first["system"].startswith("You are a customer support")
    assert all(t["strict"] is True for t in first["tools"])
    assert [m["role"] for m in second["messages"]] == ["user", "assistant", "user"]
    assert second["messages"][1]["content"][0]["type"] == "thinking"  # full content echoed back
    results = second["messages"][2]["content"]
    assert [r["tool_use_id"] for r in results] == ["toolu_1", "toolu_2"]
    assert json.loads(results[0]["content"])["tracking_number"] == "JD014600003SE"
    assert json.loads(results[1]["content"]) == {"product": "usb-c hub", "units_in_stock": 0}
    assert results[0]["is_error"] is False
    print("ok: parallel tool calls, one user message with both results")

    # 2. Bad order ID becomes an is_error result the model can act on
    reset(
        msg([{"type": "tool_use", "id": "toolu_3", "name": "get_order_status", "input": {"order_id": "A12"}}], "tool_use"),
        msg([{"type": "text", "text": "Could you check the order ID?"}], "end_turn"),
    )
    agent.run_agent("Where is order A12?")
    r = requests_seen[1]["messages"][2]["content"][0]
    assert r["is_error"] is True and "Order IDs look like 'ORD-10042'" in r["content"], r
    print("ok: error result:", r["content"])

    # 3. Cancellation: approved, declined, already shipped
    for typed, order_id, expect in [("n", "ORD-10043", "declined"), ("y", "ORD-10043", "cancelled"), ("y", "ORD-10042", "can no longer be cancelled")]:
        builtins.input = lambda prompt: typed
        reset(
            msg([{"type": "tool_use", "id": "toolu_4", "name": "cancel_order", "input": {"order_id": order_id, "reason": "customer_request"}}], "tool_use"),
            msg([{"type": "text", "text": "Done."}], "end_turn"),
        )
        agent.run_agent(f"Please cancel {order_id}")
        r = requests_seen[1]["messages"][2]["content"][0]
        assert expect in r["content"], (typed, order_id, r)
        print(f"ok: cancel {order_id} with answer '{typed}':", r["content"][:70])

    # 4. max_tokens stops the loop instead of running half-written tool calls
    reset(msg([{"type": "tool_use", "id": "toolu_5", "name": "check_stock", "input": {}}], "max_tokens"))
    assert "stopped early" in agent.run_agent("stock?")
    print("ok: max_tokens stops the loop")

    # 5. Step limit
    reset(*[msg([{"type": "tool_use", "id": f"t{i}", "name": "check_stock", "input": {"product_name": "hdmi cable"}}], "tool_use") for i in range(agent.MAX_STEPS)])
    assert "step limit" in agent.run_agent("loop forever")
    print("ok: step limit")

    # 6. A crashing tool does not crash the agent
    reset(
        msg([{"type": "tool_use", "id": "toolu_6", "name": "check_stock", "input": {"wrong_arg": 1}}], "tool_use"),
        msg([{"type": "text", "text": "Sorry."}], "end_turn"),
    )
    agent.run_agent("stock?")
    assert requests_seen[1]["messages"][2]["content"][0]["is_error"] is True
    print("ok: unexpected exception becomes a short error result")

    # 7. Cancelling an order that is already cancelled
    agent.ORDERS["ORD-10043"]["status"] = "cancelled"
    reset(
        msg([{"type": "tool_use", "id": "toolu_7", "name": "cancel_order", "input": {"order_id": "ORD-10043", "reason": "customer_request"}}], "tool_use"),
        msg([{"type": "text", "text": "Already done."}], "end_turn"),
    )
    agent.run_agent("cancel ORD-10043")
    assert "already cancelled" in requests_seen[1]["messages"][2]["content"][0]["content"]
    print("ok: already-cancelled order gets its own message")

    # 8. Tool runner version: strict mode on, clean ToolError message
    import runner_agent as rs
    rs.client = agent.client
    reset(
        msg([{"type": "tool_use", "id": "r1", "name": "check_stock", "input": {"product_name": "toaster"}}], "tool_use"),
        msg([{"type": "text", "text": "We don't sell toasters."}], "end_turn"),
    )
    assert rs.ask("Do you have toasters?") == "We don't sell toasters."
    assert requests_seen[0]["tools"][0]["strict"] is True
    r = requests_seen[1]["messages"][-1]["content"][0]
    assert r["is_error"] is True and r["content"] == "Unknown product 'toaster'. Products we sell: hdmi cable, usb-c hub, wireless mouse.", r
    print("ok: tool runner sends strict tools and clean ToolError messages")
    print("ALL TESTS PASSED (including runner)")


if __name__ == "__main__":
    test_support_agent()
