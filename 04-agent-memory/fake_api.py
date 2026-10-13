"""A fake OpenAI Responses API for the tests: it records every request and answers from a script."""
import json

import httpx2
import openai

requests_seen, script = [], []


def response(output, status="completed", input_tokens=100):
    return {"id": f"resp_{len(requests_seen)}", "object": "response", "created_at": 0, "model": "gpt-6.1-sol",
            "status": status, "incomplete_details": None if status == "completed" else {"reason": "max_output_tokens"},
            "output": output, "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
            "usage": {"input_tokens": input_tokens, "output_tokens": 10, "total_tokens": input_tokens + 10,
                      "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                      "output_tokens_details": {"reasoning_tokens": 0}}}


def call(call_id, name, arguments):
    return {"type": "function_call", "id": f"fc_{call_id}", "call_id": call_id, "name": name,
            "arguments": json.dumps(arguments), "status": "completed"}


def text(t):
    return {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": t, "annotations": []}]}


def operations(*ops):
    """A structured output answer for the memory extraction: (action, id, kind, text[, is_instruction]) tuples."""
    rows = [{"action": op[0], "memory_id": op[1], "kind": op[2], "text": op[3],
             "is_instruction": op[4] if len(op) > 4 else False} for op in ops]
    return text(json.dumps({"operations": rows}))


def refusal():
    return {"type": "message", "id": "msg_r", "role": "assistant", "status": "completed",
            "content": [{"type": "refusal", "refusal": "I can't help with that."}]}


def handler(request):
    assert request.url.path.endswith("/responses"), request.url
    requests_seen.append(json.loads(request.content))
    return httpx2.Response(200, json=script.pop(0))


def client(transport_handler=handler):
    return openai.AsyncOpenAI(api_key="test", max_retries=0, http_client=openai.DefaultAsyncHttpxClient(
        transport=httpx2.MockTransport(transport_handler)))
